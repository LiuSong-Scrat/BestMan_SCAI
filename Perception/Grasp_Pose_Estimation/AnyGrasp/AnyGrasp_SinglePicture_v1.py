#V1 版本更新
#更新引入了GroundedSAM
#通过GSAM得到物体掩膜，然后用掩膜点云进行指定物体的抓取
 
import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import time
import torch
from graspnetAPI import GraspGroup
 
import pyrealsense2 as rs
import cv2
from matplotlib import pyplot as plt
 
 
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))


#GroundedSAM
GSAM_ROOT_DIR="/home/liusong/ProgramFiles/GroundedSAM/Grounded-Segment-Anything/"
sys.path.append(os.path.join(GSAM_ROOT_DIR))
from my_grounded_sam_inference import inference_mask




from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import GraspNetDataset, minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--num_view', type=int, default=256, help='View Number [default: 300]')

parser.add_argument('--dataset_root', default="../", required=False)
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default="logs/log_kn/checkpoint.tar", required=False)
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default="logs/log_rs/dump_epoch10", required=False)
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=0.01,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--eval', action='store_true', default=False)
cfgs = parser.parse_args()
 
 
def get_net():
    # Init the model
    # net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
    #         cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net
 
def get_and_process_data(data_dir):
    # load data
    img_src = Image.open(os.path.join(data_dir, 'color.png'))
    target_obj_masks = inference_mask(img_src,"chip")
    target_obj_masks=np.logical_or.reduce(target_obj_masks)
    color = np.array(img_src, dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))# Resize depth to match color image resolution while preserving spatial alignment
    # meta["intrinsic_matrix"]=[[909.2156372070312, 0.0, 648.0855712890625],
 	#                           [0.0,  908.82470703125, 381.73773193359375], 
	#                           [0.0, 0.0, 1.0]]
    # scio.savemat("./data3/meta1.mat",meta)
    color_height, color_width = color.shape[:2]
    depth = cv2.resize(depth, (color_width, color_height), interpolation=cv2.INTER_NEAREST)
    intrinsic = meta['intrinsic_matrix']
    factor_depth =meta['factor_depth']
    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    # # get valid points 限制高度0~70cm
    mask = (workspace_mask & (depth > 0 )&( depth <2000)) & target_obj_masks[0]
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
    # # convert data
    # #可视化
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    # geometries = []
    # geometries.append(cloud)
    # o3d.visualization.draw_geometries(geometries)
    end_points = {
        "cloud_colors":color_sampled,
        "point_clouds":cloud_sampled.astype(np.float32),
        "coors":cloud_sampled.astype(np.float32) / cfgs.voxel_size,
        "feats":np.ones_like(cloud_sampled).astype(np.float32)

    }
    return end_points, cloud





def get_grasps(net, end_points):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_data = minkowski_collate_fn([end_points])
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        with torch.no_grad():
            end_points = net(batch_data)
            # end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg
 
def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg
 

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:1]#前几个
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


 
def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))

    
  
    vis_grasps(gg, cloud)
class Camera(object):
    '''
    realsense相机处理类
    '''
    def __init__(self, width=1280, height=720, fps=30):   # 图片格式可根据程序需要进行更改
 
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16,  fps)
        # self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, fps)
        # self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, fps)
        self.pipeline.start(self.config)      # 获取图像视频流
    def get_frame(self):
        frames = self.pipeline.wait_for_frames()   # 获得frame (包括彩色，深度图)
        colorizer = rs.colorizer()                 # 创建伪彩色图对象
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
        # 创建对齐对象
        align_to = rs.stream.color                 # rs.align允许我们执行深度帧与其他帧的对齐
        align = rs.align(align_to)                 # “align_to”是我们计划对齐深度帧的流类型。
        aligned_frames = align.process(frames)
        # 获取对齐的帧
        aligned_depth_frame = aligned_frames.get_depth_frame()      # aligned_depth_frame是对齐的深度图
        color_frame   = aligned_frames.get_color_frame()
        # left_frame  = frames.get_infrared_frame(1)
        # right_frame = frames.get_infrared_frame(2)
        color_image     = np.asanyarray(color_frame.get_data())
        colorizer_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        depthx_image    = np.asanyarray(aligned_depth_frame.get_data())  # 原始深度图
        # left_frame   = np.asanyarray(left_frame.get_data())
        # right_frame  = np.asanyarray(right_frame.get_data())
        return color_image, depthx_image, colorizer_depth
        # left_frame, right_frame
    def release(self):
        self.pipeline.stop()
 
def input1():
     # 视频保存路径
    save_file_dir = "./data3/"
    # os.mkdir(f'~/{int(time.time())}')
    fps, w, h = 30, 1280, 720
    cam = Camera(w, h, fps)
    flag_V = 0
    idx = 0
    id  = 0
    print('截图请按: s, 退出请按：q')
    try:
        while True:
            # 读取图像帧，包括RGB图和深度图
            color_image, depthxy_image, colorizer_depth = cam.get_frame()
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('RealSense1', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('RealSense2', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.imshow('RealSense1', depthxy_image)
            cv2.imshow('RealSense1', colorizer_depth)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') :
                # 保存图像帧
                # wr.write(color_image)                          # 保存RGB图像帧
                # wr_colordepth.write(colorizer_depth)           # 保存相机自身着色深度图
                # wr_left.write(left_image)                      # 保存左帧深度图
                # wr_right.write(right_image)                    # 保存右帧深度图
                # res, depth16_image = cv2.imencode('.png', depthxy_image)  # 深度图解码方式一：点云小，但是出错
                depth16_image = cv2.imencode('.png', depthxy_image)[1]      # 深度图解码方式二：文件较大，测试稳定
                depth_map_name = 'depth.png'
                color_map_name = 'color.png'
                cv2.imwrite(save_file_dir+depth_map_name,depthxy_image)
                cv2.imwrite(save_file_dir+color_map_name,color_image)
                cv2.destroyAllWindows()
                print('.../开始识别...')
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
 
if __name__=='__main__':
    # -------------------GetImage---------------------
    time_start_1 = time.time()
    # input1()
    # print("采图时间："+str(time.time() - time_start_1)+"秒")
    # -------------------Inference---------------------
    data_dir = 'data3'
    demo(data_dir)
    print("推理时间："+str(time.time() - time_start_1)+"秒")