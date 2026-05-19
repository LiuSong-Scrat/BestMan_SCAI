import sys
import torch
import numpy as np 
import cv2
import copy

sys.path.append("/home/liusong/ProgramFiles/REAP/StageGen")
from stagegen.geometry_utils import GeometryUtils
from stagegen.projection_utils import ProjectionUtils
from stagegen.visualization_utils import VisualizationUtils
from stagegen.stage2_editing import Stage2Editing
from stagegen.stage1_segmentation import Stage1Segmentation


sys.path.append("/home/liusong/ProgramFiles/REAP/SAVG/savg/")
sys.path.append("/home/liusong/ProgramFiles/REAP/SAVG/savg/models")
from models.pose_act_cvae import PoseACTCVAE
from utils.rot6d import pose9_to_homo
from preprocess_hdf5 import load_cloud_from_group,maybe_bytes_to_str,remove_outliers_fast
from eval import single_data_inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append("/home/liusong/ProgramFiles/Huggingface/lerobot/src/lerobot/scripts/")
from smolvla_model_inference import SmolVLA_ModelInference



import time
import math
import numpy as np
import cv2
import sys, os
import threading
import select
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Franka3, Pose
from Motion_Planning.Manipulation.Skill_Franka3 import skill_database
from Sensor.Camera_Realsense import Camera_Realsense
from Dataset.scripts.data_collection import RealDataCollection
from franky import Affine
import pytorch3d.ops as torch3d_ops



CONST_POINTS_NUM=640*480


def point_cloud_filter(points):
    WORK_SPACE = [
        [0.0, 1.0],
        [-0.5, 0.5],
        [0, 0.8]
    ]
     # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]
    return points


def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices



def uniform_random_sample_points(xyzrgb: np.ndarray, M: int):
    N = xyzrgb.shape[0]
    if N == 0:
        return np.zeros((M, 6))
    if N >= M:
        idx = np.linspace(0, N - 1, M).astype(np.int64)
        return xyzrgb[idx]
    else:
        extra = np.random.choice(N, M - N, replace=True)
        return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)


def update_cam_extrinsics(bestman,camera_name):
    # #camera_hand2eff
    # camera_hand2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
    # camera_hand2ee_aff = Affine([-0.05,-0.03,-0.04],camera_hand2ee_quat) #x y z
    H_camera_hand2eff = np.array([[-3.79969778e-02,-9.98877888e-01,-2.82700204e-02,-0.0457490352],
                                    [9.99277851e-01,-3.79838244e-02,-1.00233611e-03,-0.0278605145],
                                    [-7.25921195e-05,-2.82876910e-02,9.99599821e-01,-0.0525787876],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00, 1.00000000e+00]])
    cur_eff_pose = bestman.get_current_eef_pose()
    H_eff2base = np.eye(4)
    H_eff2base[:3,:3] = R.from_quat(cur_eff_pose.orientation).as_matrix()
    H_eff2base[:3,3] = np.array(cur_eff_pose.position)
    H_camera_hand2base = H_eff2base@H_camera_hand2eff

    
    if camera_name == "D435I":
        H_camera_extrics = H_camera_hand2base
    elif camera_name == "L515":

        # # [ 1280x720  p[643.178 357.433]  f[898.481 899.104]  Brown Conrady [0.144588 -0.485378 0.0004702 -4.61056e-05 0.44427] ]
        # H_camera_overhead2camera_hand = np.array([[-0.994305,-0.046932,0.095686,0.006064],
        #                                             [0.093147,-0.818933,0.566278,-0.538343],
        #                                             [0.051785,0.571966,0.818641,0.017486],
        #                                             [0.000000,0.000000,0.000000,1.000000]]
        #                                                                                                     )
        # H_camera_hand2base=np.array([[-3.81971794e-02, -9.98831631e-01, -2.96031838e-02,
        #                                 3.04561658e-01],
        #                             [-9.99270213e-01,  3.81767610e-02,  1.25483779e-03,
        #                                 2.79467731e-02],
        #                             [-1.23218001e-04,  2.96295110e-02, -9.99560942e-01,
        #                                 7.02004808e-01],
        #                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #                                 1.00000000e+00]])
        # H_camera_overhead2base = H_camera_hand2base@H_camera_overhead2camera_hand
        # H_camera_extrics = H_camera_overhead2base

        H_camera_extrics = np.array([[-0.05659152,  0.80283684, -0.59350569,  0.84141181],
                                [ 0.9972004 ,  0.01635126, -0.07297025, -0.00164086],
                                [-0.04887985, -0.59597368, -0.80151482,  0.66857453],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]])


    return H_camera_extrics

def interaction_policy_inference(camera_hand,camera_overhead, bestman,stage1segmentation,predictor,model_va,allow_gripper_open_flag,visualize,task="None"):
    model_va.policy_reset()
    model_va.policy.n_action_steps=26 #26
    while True:
        if  len(model_va.predict_action_queue)<model_va.policy.horizon-model_va.policy.n_action_steps+2:
            cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
            
            ###################CLOUD_RGB EXTRACT#############
            
            # Scene CloudRgb 
            masked_scene_cloud_rgb = []
            
            # 添加夹爪点云
            eff_pose_zyx_eular = cur_model_observation['pose_eular']
            eff_gripper_width = cur_model_observation['gripper_width']
            normalize_eff_angular = 0 if eff_gripper_width<0.04 else 1 
            gripper_mesh = VisualizationUtils.update_gripper(normalize_eff_angular, eff_pose_zyx_eular+np.array([0.015, 0, 0,0,0,0]), gripper_len = 0.06)
            gripper_pcd = gripper_mesh.sample_points_uniformly(number_of_points=500)
            gripper_cloud_rgb = GeometryUtils.pcd_to_cloud_rgb(gripper_pcd)
            # ADD CloudRgb to Scene
            masked_scene_cloud_rgb.append(gripper_cloud_rgb)


            # WORKSPACE DOWNSAMPLE  
            overhead_cloud_rgb = cur_model_observation['point_cloud']
            overhead_cloud_rgb_workspace = point_cloud_filter(overhead_cloud_rgb)  
            # Objects Segmentation 
            obj_sets = ["yellow_mug","blue_cube"]  
            img_overhead_rgb = cur_model_observation['overhead']
            obj_masks = get_sam2_obj_masks_fast(predictor,img_overhead_rgb,obj_sets)
            # Objects CloudRgb Extracted 
            scene_pcd = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb_workspace)
            camera_intrics, H_world2image = stage1segmentation.setup_camera_transforms()
            for obj_str, obj_mask in obj_masks.items():
                obj_seg_pcd = ProjectionUtils.get_seg_pcd(scene_pcd, H_world2image, camera_intrics, obj_mask)
                obj_seg_cloud_rgb = np.hstack((np.array(obj_seg_pcd.points), 
                                                np.array(255 * np.array(obj_seg_pcd.colors)).astype(np.uint8)))
                # obj_seg_cloud_rgb = remove_outliers_fast(obj_seg_cloud_rgb, nb_neighbors=50, std_ratio=1)
                
                # ADD CloudRgb to Scene
                masked_scene_cloud_rgb.append(obj_seg_cloud_rgb)
            # Convert Scene CloudRgb to Numpy
            masked_scene_cloud_rgb = np.vstack(masked_scene_cloud_rgb)

            # Align with Collection
            overhead_cloud_rgb_filter = GeometryUtils.random_repeat_sample_points(masked_scene_cloud_rgb,48*64)
            points_xyz = overhead_cloud_rgb_filter[..., :3]
            points_xyz, sample_indices = farthest_point_sampling(points_xyz)
            sample_indices = sample_indices.cpu()
            points_rgb = overhead_cloud_rgb_filter[sample_indices, 3:][0]
            points = np.hstack((points_xyz, points_rgb))
            cur_model_observation["point_cloud"] = points


            ###################STATE ALIGNED WITH DP3#############
            cur_observation_key_list = list(cur_model_observation.keys())
            for idx in range(len(cur_model_observation['pose_eular'])):
                cur_model_observation[cur_observation_key_list[idx]] =cur_model_observation['pose_eular'][idx]
            cur_model_observation['joint_7'] = eff_gripper_width



        inference_action =  model_va.single_inference(cur_model_observation,visualize=visualize,task=task)
        if inference_action is None:
            continue
        
        # Image Visualize
        overhead_pcd_filter = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb_filter)
        depth, rgb = ProjectionUtils.project_pcd_to_image_depth(overhead_pcd_filter, H_world2image, camera_intrics, (480, 640))
        cv2.imwrite("/home/liusong/temp/temp.png",rgb)

        ###################POSE CONTROAL#############
        target_pose_eular_zyx=np.array(inference_action[3:6])
        rotation = R.from_euler('zyx',target_pose_eular_zyx)#( x, y, z, w)
        target_pose_orientation_xyzw = rotation.as_quat()
        target_pose_position =np.array(inference_action[:3])

        move_towards_pose = Pose(target_pose_position, target_pose_orientation_xyzw)

        while True:
            try:
                bestman.move_eef_to_goal_pose(move_towards_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45),asynchronous=False)
                print("SUCCESS------------------")
                break
            except Exception as e:
                bestman.robot.recover_from_errors() 
                print("ERROR------------------")
                time.sleep(0.1)

        # ###################Gripper CONTROAL#############
        # inference_gripper_width=inference_action[-1]*2 #recover the normal scale
        # bestman.open_gripper_width(inference_gripper_width)


        inference_gripper_width=inference_action[-1]*2 #recover the normal scale
        if allow_gripper_open_flag==0 and inference_gripper_width<0.06:
            allow_gripper_open_flag = 1
            bestman.close_gripper() 
            return allow_gripper_open_flag
        if allow_gripper_open_flag==1 and inference_gripper_width>0.07:
            bestman.open_gripper()
            allow_gripper_open_flag = 0
            return allow_gripper_open_flag

        if sys.stdin in select.select([sys.stdin], [], [], 0.01)[0]:
            line = sys.stdin.readline()
            pressed_key = line.strip()
            if line:
                print(f"You pressed: {pressed_key}")
            if pressed_key == 'o':
                print("Open.................")
                bestman.open_gripper()
                allow_gripper_open_flag = 0
                return allow_gripper_open_flag
            if pressed_key == 'c':
                print("Close.................")
                bestman.close_gripper()
                allow_gripper_open_flag = 1
                return allow_gripper_open_flag
def force_move(bestman,pose, maxLinearVel=0.22, maxAngularVel=math.radians(45)):
    while True:
        try:
            bestman.move_eef_to_goal_pose(pose, maxLinearVel=maxLinearVel, maxAngularVel=maxAngularVel)
            print("Grasp SUCCESS------------------")
            break
        except Exception as e:
            bestman.robot.recover_from_errors() 
            print("Grasp ERROR------------------")
            time.sleep(0.1)


def mission_execution(task_name,camera_hand,camera_overhead, bestman,predictor,model_va,visualize=False):
    #4.load stage1 segmentation for camera parameters
    stagegen_config_file = f"/home/liusong/ProgramFiles/REAP/StageGen/config/{task_name}.yaml"
    stagegen_src_hdf5_path = f"/home/liusong/ProgramFiles/REAP/StageGen/source/{task_name}.hdf5"
    stage1segmentation = Stage1Segmentation(stagegen_config_file, stagegen_src_hdf5_path)
    # import pickle
    # with open(f"/home/liusong/ProgramFiles/REAP/StageGen/out/{task_name}/{task_name}_stage1_result.pkl", 'rb') as file:
    #     stage1_result = pickle.load(file)
    # stage2editing = Stage2Editing(stage1_result)

    object_marker_list = object_marker_dict[task_name]
    predictor = predictor_initialize(predictor,task_name,object_marker_list)

    # 6.LOAD SAVG MODEL
    SAVG_PRETRAINED_CKPT_PATH = f"/home/liusong/ProgramFiles/REAP/SAVG/out/checkpoints/{task_name}.pt"
    # model
    model_savg = PoseACTCVAE(
        pc_in_dim=6,
        pc_dim=256,
        pc_grid_size=0.005,
        pc_tokens=256,
        geo_k=256,
        model_dim=256,
        latent_dim=32,
        n_enc_layers=4,
        n_dec_layers=4,
        heads=4,
        ff_dim=1024,
        dropout=0.1,
        pre_norm=True,
    ).to(DEVICE)
    ckpt = torch.load(SAVG_PRETRAINED_CKPT_PATH)
    model_savg.load_state_dict(ckpt["model"])
    model_savg.eval()
    assert os.path.isfile(SAVG_PRETRAINED_CKPT_PATH), f"ckpt not found: {SAVG_PRETRAINED_CKPT_PATH}"



    
    
    ##############Subtask##########
    print("###########Subtask##########")
    subtasks = stage1segmentation.task_config['subtask']
    [print(subtask) for subtask in subtasks]

    allow_gripper_open_flag = 0
    for subtask in subtasks:
        subtask_label = subtask.split(',')
        action = subtask_label[0] 
        target_obj = subtask_label[1].strip() 
        cond_obj = subtask_label[3].strip()
        if action == "move_towards":
            cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
            target_pose_position,target_pose_orientation_xyzw = savg_inference(model_savg,cur_model_observation,predictor,stage1segmentation,target_obj,cond_obj,visualize=visualize)
            move_towards_pose = Pose(target_pose_position, target_pose_orientation_xyzw)
            force_move(bestman,move_towards_pose, maxLinearVel=0.3, maxAngularVel=math.radians(90))
        else:
            allow_gripper_open_flag = interaction_policy_inference(camera_hand,camera_overhead, bestman,stage1segmentation,predictor,model_va,allow_gripper_open_flag,visualize=visualize,task = subtask)
            if action=="place":
                # Lift Up 5cm
                cur_eff_pose = bestman.get_current_eef_pose()
                move_towards_pose = Pose(cur_eff_pose.position+np.array([0,0,0.08]), cur_eff_pose.orientation)
                force_move(bestman,move_towards_pose, maxLinearVel=0.3, maxAngularVel=math.radians(90))
            if action=="pick":
                # Lift Up 5cm
                cur_eff_pose = bestman.get_current_eef_pose()
                move_towards_pose = Pose(cur_eff_pose.position+np.array([0,0,0.05]), cur_eff_pose.orientation)
                force_move(bestman,move_towards_pose, maxLinearVel=0.3, maxAngularVel=math.radians(90))
    # Go Back Home
    bestman.open_gripper()
    bestman.go_home(home_js)




#Visualize With Predict Trajectory
{
            # cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
            # ###################CLOUD_RGB EXTRACT#############
            # # 添加夹爪点云
            # eff_pose_zyx_eular = cur_model_observation['pose_eular']
            # eff_gripper_width = cur_model_observation['gripper_width']
            # normalize_eff_angular = 0 if eff_gripper_width<0.03 else 1
            # gripper_mesh = VisualizationUtils.updaqte_gripper(normalize_eff_angular, eff_pose_zyx_eular, gripper_len = 0.06)
            # gripper_pcd = gripper_mesh.sample_points_uniformly(number_of_points=500)
            # gripper_cloud_rgb = GeometryUtils.pcd_to_cloud_rgb(gripper_pcd)

            # # Objects CloudRgb Extracted 
            # masked_scene_cloud_rgb = []
            # masked_scene_cloud_rgb.append(gripper_cloud_rgb)

            # # Objects Segmentation 
            # obj_sets = ["yellow_mug","blue_cube"]  
            # img_overhead_rgb = cur_model_observation['overhead']
            # obj_masks = get_sam2_obj_masks_fast(predictor,img_overhead_rgb,obj_sets)

            # # Objects CloudRgb Extracted 
            # obj_seg_dict={}
            # overhead_cloud_rgb = cur_model_observation['point_cloud']

            # #WORKSPACE DOWNSAMPLE
            # overhead_cloud_rgb_workspace = point_cloud_filter(overhead_cloud_rgb)  
            # scene_pcd = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb_workspace)
            # camera_intrics, H_world2image = stage1segmentation.setup_camera_transforms()
            # for obj_str, obj_mask in obj_masks.items():
            #     obj_seg_pcd = ProjectionUtils.get_seg_pcd(scene_pcd, H_world2image, camera_intrics, obj_mask)
            #     obj_seg_cloud_rgb = np.hstack((np.array(obj_seg_pcd.points), 
            #                                     np.array(255 * np.array(obj_seg_pcd.colors)).astype(np.uint8)))
            #     masked_scene_cloud_rgb.append(obj_seg_cloud_rgb)
            # masked_scene_cloud_rgb = np.vstack(masked_scene_cloud_rgb)

            # # Align with Collection
            # overhead_cloud_rgb_filter = GeometryUtils.random_repeat_sample_points(masked_scene_cloud_rgb,48*64)
            # points_xyz = overhead_cloud_rgb_filter[..., :3]
            # points_xyz, sample_indices = farthest_point_sampling(points_xyz)
            # sample_indices = sample_indices.cpu()
            # points_rgb = overhead_cloud_rgb_filter[sample_indices, 3:][0]
            # points = np.hstack((points_xyz, points_rgb))
            # cur_model_observation["point_cloud"] = points

            # ###################STATE ALIGNED WITH DP3#############
            # cur_observation_key_list = list(cur_model_observation.keys())
            # for idx in range(len(cur_model_observation['pose_eular'])):
            #     cur_model_observation[cur_observation_key_list[idx]] =cur_model_observation['pose_eular'][idx]
            # cur_model_observation['joint_7'] = cur_model_observation["gripper_width"]
            # inference_action =  model_va.single_inference(cur_model_observation)

            # np_list = [t.detach().cpu().numpy() for t in model_va.predict_action_queue]
            # np_array = np.stack(np_list)  # 可选：拼成一个数组
            # pre_traj_points = np_array[:,:3]
            # pre_traj_colors = np.zeros(pre_traj_points.shape)
            # pre_traj_cloud_rgb = np.concatenate((pre_traj_points,pre_traj_colors),axis=1)
            # pre_traj_pcd = GeometryUtils.cloud_rgb_to_pcd(pre_traj_cloud_rgb)
            # overhead_pcd_filter = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb_filter)
            # import open3d as o3d
            # o3d.visualization.draw_geometries([pre_traj_pcd,overhead_pcd_filter])
            # from collections import deque
            # model_va.predict_action_queue = deque()
        }




def get_cur_model_observation(camera_hand,camera_overhead,bestman):
    gripper_width = bestman.gripper.width

    cur_pos = bestman.get_current_joint_values()
    
    cur_eff_pose = bestman.get_current_eef_pose()
    cur_eff_pose_position = cur_eff_pose.position
    cur_eff_rotation =  R.from_quat(cur_eff_pose.orientation)
    cur_eff_pose_orientation_eular_zyx = cur_eff_rotation.as_euler('zyx')
    

    img_hand_rgb = camera_hand.get_rgb_image()
    points, colors = camera_hand.get_3d_points()
    # points, colors = np.zeros((10,3)),np.zeros((10,3))#-----------temp_use
    H_camera_hand_extrinsic = update_cam_extrinsics(bestman,camera_hand.dev_name)
    world_points = ((H_camera_hand_extrinsic[:3,:3]@points.T).T+H_camera_hand_extrinsic[:3,3].T).astype(np.float32)
    hand_cloud_rgb_var_len = np.hstack((world_points,((colors*255).astype(np.uint8)))) 
    hand_cloud_rgb = uniform_random_sample_points(hand_cloud_rgb_var_len,CONST_POINTS_NUM)
    # camera_hand.visualize_3d_points()


    img_overhead_rgb = camera_overhead.get_rgb_image()
    img_overhead_rgb = cv2.resize(img_overhead_rgb,(640,480),interpolation=cv2.INTER_LINEAR)
    points, colors = camera_overhead.get_3d_points()
    # points, colors = np.zeros((10,3)),np.zeros((10,3))#-----------temp_use
    H_camera_overhead_extrinsic = update_cam_extrinsics(bestman,camera_overhead.dev_name)
    world_points = ((H_camera_overhead_extrinsic[:3,:3]@points.T).T+H_camera_overhead_extrinsic[:3,3].T).astype(np.float32)
    overhead_cloud_rgb_var_len = np.hstack((world_points,((colors*255).astype(np.uint8)))) 
    overhead_cloud_rgb = uniform_random_sample_points(overhead_cloud_rgb_var_len,CONST_POINTS_NUM)
    # camera_overhead.visualize_3d_points()


    # Lerobot-VA
    # cur_model_observation = {'joint_1':None,'joint_2':None,'joint_3': None,'joint_4':None,'joint_5':None,'joint_6': None,'joint_7':None,'gripper_width':None,
    #                         'overhead':img_overhead_rgb,"hand":img_hand_rgb,"point_cloud":{},"pose_eular":{}}
    # cur_observation_key_list = list(cur_model_observation.keys())
    # for idx, value in enumerate(cur_pos):
    #     cur_model_observation[cur_observation_key_list[idx]] =180*value/3.14
    # cur_model_observation['gripper_width'] = gripper_width*1000*0.5
    # cur_model_observation['pose_eular'] = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)
    # cur_model_observation['point_cloud'] = overhead_cloud_rgb



    #DP3
    cur_model_observation = {'joint_1':None,'joint_2':None,'joint_3': None,'joint_4':None,'joint_5':None,'joint_6': None,'joint_7':None,'gripper_width':None,
                            'overhead':img_overhead_rgb,"hand":img_hand_rgb,"point_cloud":{},"pose_eular":{}}
    cur_model_observation['pose_eular'] = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)
    cur_model_observation['gripper_width'] = gripper_width
    cur_model_observation['point_cloud'] = overhead_cloud_rgb

    return cur_model_observation


def get_sam2_obj_masks(predictor,frame,obj_sets):
    out_obj_ids, out_mask_logits = predictor.track(frame)
    all_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    all_mask[..., 1] = 255
    obj_masks = {}
    for obj_index in range(len(out_obj_ids)):
        if obj_index >= len(obj_sets):
            continue
        
        obj_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        obj_mask[..., 1] = 255
        
        out_mask = (out_mask_logits[obj_index] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        hue = (obj_index + 3) / (len(out_obj_ids) + 3) * 255
        
        obj_mask[out_mask[..., 0] == 255, 0] = hue
        obj_mask[out_mask[..., 0] == 255, 2] = 255
        obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_HSV2RGB)
        mask = np.any(obj_mask > 0, axis=-1)
        obj_mask[mask] = [255, 255, 255]

        obj_mask_gray = cv2.cvtColor(obj_mask, cv2.COLOR_BGR2GRAY)
        obj_masks[obj_sets[obj_index]] = obj_mask_gray
    return obj_masks

def get_sam2_obj_masks_fast(predictor, frame, obj_sets):
    out_obj_ids, out_mask_logits = predictor.track(frame)

    H, W = frame.shape[:2]
    num_obj = min(len(out_obj_ids), len(obj_sets))

    # 🔴 1. 一次性 GPU → CPU（最重要）
    # out_mask_logits: [N, 1, H, W] or [N, H, W]
    masks = (out_mask_logits[:num_obj] > 0.0) \
        .squeeze(1) \
        .cpu() \
        .numpy() \
        .astype(np.uint8) * 255  # [N, H, W]

    obj_masks = {}

    # 🔴 2. 只生成 grayscale mask
    for i in range(num_obj):
        # masks[i] already H×W uint8 {0,255}
        obj_masks[obj_sets[i]] = masks[i]

    return obj_masks


def from_trajectory_to_H(trajectory):
    """从轨迹数据转换为齐次矩阵"""
    position = trajectory[:3]
    euler_zyx = trajectory[3:]
    rotation_matrix = R.from_euler('zyx', euler_zyx, degrees=False).as_matrix()
    H = np.eye(4)
    H[:3, :3] = rotation_matrix
    H[:3, 3] = position
    return H
def from_H_to_trajectory(H):
    """从齐次矩阵转换为轨迹数据"""
    position = H[:3, 3]
    rotation_matrix = H[:3, :3]
    euler_zyx = R.from_matrix(rotation_matrix).as_euler('zyx', degrees=False)
    trajectory = np.hstack((position, euler_zyx))
    return trajectory


def get_base_points_from_cam_points(bestman,mouse_get_cam_3d_points,camera_name):
    base_3d_points = []
    for cam_obj_translation in mouse_get_cam_3d_points:
        H_obj2cam = np.eye(4)
        H_obj2cam[:3,3] = np.array(cam_obj_translation)
        
        
        # #camera_hand2eff
        # camera_hand2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
        # camera_hand2ee_aff = Affine([-0.05,-0.03,-0.04],camera_hand2ee_quat) #x y z
        H_camera_hand2eff = np.array([[-3.79969778e-02,-9.98877888e-01,-2.82700204e-02,-0.0457490352],
                                        [9.99277851e-01,-3.79838244e-02,-1.00233611e-03,-0.0278605145],
                                        [-7.25921195e-05,-2.82876910e-02,9.99599821e-01,-0.0525787876],
                                        [0.00000000e+00,0.00000000e+00,0.00000000e+00, 1.00000000e+00]])
        cur_eff_pose = bestman.get_current_eef_pose()
        H_eff2base = np.eye(4)
        H_eff2base[:3,:3] = R.from_quat(cur_eff_pose.orientation).as_matrix()
        H_eff2base[:3,3] = np.array(cur_eff_pose.position)
        H_camera_hand2base = H_eff2base@H_camera_hand2eff

        if camera_name == "hand":
            H_camera_extrics = H_camera_hand2base
        elif camera_name == "overhead":

            # # [ 1280x720  p[643.178 357.433]  f[898.481 899.104]  Brown Conrady [0.144588 -0.485378 0.0004702 -4.61056e-05 0.44427] ]
            # H_camera_overhead2camera_hand = np.array([[-0.994305,-0.046932,0.095686,0.006064],
            #                                             [0.093147,-0.818933,0.566278,-0.538343],
            #                                             [0.051785,0.571966,0.818641,0.017486],
            #                                             [0.000000,0.000000,0.000000,1.000000]]
            #                                                                                                     )
            # H_camera_hand2base=np.array([[-3.81971794e-02, -9.98831631e-01, -2.96031838e-02,
            #                                 3.04561658e-01],
            #                             [-9.99270213e-01,  3.81767610e-02,  1.25483779e-03,
            #                                 2.79467731e-02],
            #                             [-1.23218001e-04,  2.96295110e-02, -9.99560942e-01,
            #                                 7.02004808e-01],
            #                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            #                                 1.00000000e+00]])
            # H_camera_overhead2base = H_camera_hand2base@H_camera_overhead2camera_hand
            # H_camera_extrics = H_camera_overhead2base

            H_camera_extrics = np.array([[-0.05659152,  0.80283684, -0.59350569,  0.84141181],
                                    [ 0.9972004 ,  0.01635126, -0.07297025, -0.00164086],
                                    [-0.04887985, -0.59597368, -0.80151482,  0.66857453],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])


        H_obj2base = H_camera_extrics@H_obj2cam
        base_3d_points.append(H_obj2base[:3,3])


    return base_3d_points



def savg_inference(model_savg,cur_model_observation,predictor,stage1segmentation,target_obj,cond_obj,visualize):
    obj_sets = stage1segmentation.task_config['obj_sets']
    
    cond_name = cond_obj
    if cond_name=="None":
        has_cond=False
    else:
        has_cond=True

    # Objects Segmentation 
    img_overhead_rgb = cur_model_observation['overhead']
    obj_masks = get_sam2_obj_masks_fast(predictor,img_overhead_rgb,obj_sets)

    # Objects CloudRgb Extracted 
    obj_seg_dict={}
    overhead_cloud_rgb = cur_model_observation['point_cloud']
    scene_pcd = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb)
    camera_intrics, H_world2image = stage1segmentation.setup_camera_transforms()
    for obj_str, obj_mask in obj_masks.items():
        obj_seg_pcd = ProjectionUtils.get_seg_pcd(scene_pcd, H_world2image, camera_intrics, obj_mask)
        obj_seg_cloud_rgb = np.hstack((np.array(obj_seg_pcd.points), 
                                        np.array(255 * np.array(obj_seg_pcd.colors)).astype(np.uint8)))
        obj_seg_dict[obj_str] = {"cloud_rgb": obj_seg_cloud_rgb}

    # ApproachMode Switch
    if has_cond:
        cond_raw= obj_seg_dict[cond_obj]['cloud_rgb']
        target_raw= obj_seg_dict[target_obj]['cloud_rgb']
    else:
        cond_raw = np.zeros((0, 6), dtype=np.float32)
        target_raw= obj_seg_dict[target_obj]['cloud_rgb']

    # SAVG Inference
    cur_eff_trajectory = cur_model_observation['pose_eular']
    with torch.no_grad():
        #MakeSure torch.float32 //Avoid SAM2 Collision
        with torch.cuda.amp.autocast(False):
            approaching_raw = cur_eff_trajectory # NO USE
            if has_cond:
                H_pr = np.eye(4)
                for i in range(3):
                    H_pr_slice,target_raw,cond_raw,approaching_raw,cond_name = single_data_inference(model_savg,target_raw,cond_raw,approaching_raw,cond_name,visualize=visualize)
                    H_pr = H_pr_slice@H_pr
            if has_cond != True:
                H_pr_slice,target_raw,cond_raw,approaching_raw,cond_name = single_data_inference(model_savg,target_raw,cond_raw,approaching_raw,cond_name,visualize=visualize)
                H_pr = H_pr_slice
        
    if has_cond:
        H_cur_eff_trajectory = from_trajectory_to_H(cur_eff_trajectory)
        H_next_eff_trajectory = H_pr@H_cur_eff_trajectory
    else:
        H_next_eff_trajectory = H_pr
    #Move Towards Action Execution 
    target_pose_position = H_next_eff_trajectory[:3, 3]
    target_pose_orientation_xyzw = R.from_matrix(H_next_eff_trajectory[:3, :3]).as_quat()
    return target_pose_position,target_pose_orientation_xyzw

def sam2_initialize():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    import time
    import h5py
    sys.path.append("/home/liusong/ProgramFiles/SAM2/sam2")
    from sam2.build_sam import build_sam2_camera_predictor

    # sam2_checkpoint = "/home/liusong/ProgramFiles/SAM2/sam2/checkpoints/sam2.1_hiera_small.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    # Better -->Base_Plus SAM2-----------------
    # Need Be Careful of the order of the object! SAM-Object Order Not Always Correct 

    sam2_checkpoint = "/home/liusong/ProgramFiles/SAM2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    return predictor

def predictor_initialize(predictor,task_name,object_marker_list):

    
    # img_rgb = cv2.resize(camera_overhead.get_rgb_image(),(640,480),cv2.INTER_LINEAR)
    # img_bgr = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)
    # cv2.imshow("img",img_bgr)
    # cv2.waitKey(0)
    # cv2.imwrite("/home/liusong/ProgramFiles/BestMan/Dataset/Images/Desk_TrashSweep.png",img_bgr)


    frame = cv2.imread(f"/home/liusong/ProgramFiles/BestMan/Dataset/Images/{task_name}.png")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    width, height = frame.shape[:2][::-1]
    predictor.load_first_frame(frame)
    if_init = True

    ann_frame_idx = 0  # the frame index we interact with
    # First annotation
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    ##! add points, `1` means positive click and `0` means negative click
    points = np.array(object_marker_list[0], dtype=np.float32)
    labels = np.array([1,1,1], dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    )
    
    ann_obj_id = 2  # give a unique id to each object we inter act with (it can be any integers)
    points = np.array(object_marker_list[1], dtype=np.float32)
    labels = np.array([1,1,1], dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    )

    return predictor


# 1.初始化机器人（原代码逻辑）
bestman = Bestman_Real_Franka3()
if bestman.initialize_robot() is not True:
    exit(-1)
bestman.open_gripper()
#Twist 90 degree
bestman.go_home()
skill_franka3_database = skill_database.SkillFranka3Database()

#2.Camera Initialize
camera_devices = list( bestman.cfg.Camera.keys())
camera_hand,camera_overhead = None,None
for cam_dev in camera_devices:
    if cam_dev == "D435I":
        # camera_hand = Camera_Realsense(bestman.cfg.Camera[cam_dev])
        camera_hand=camera_overhead
    if cam_dev == "L515":
        camera_overhead = Camera_Realsense(bestman.cfg.Camera[cam_dev])
if camera_overhead == None:
    camera_overhead = camera_hand
time.sleep(bestman.cfg.Camera.init_delay)




task_name = "Desk_CubeStacking"
object_marker_dict = {"PutStationeryBox":[[[477,293],[485,320],[494,363]],[[342,306],[340,333],[341,362]]],
                      "CubeStacking":[[[327, 310],[337,323],[332,334]],[[441, 308],[454,321],[447,335]]],
                      "TrashSweep":[[[273, 205],[280,263],[284,286]],[[422, 339],[432,400],[437,457]]],
                      "MugRack":[[[459, 287],[468,344],[434,315]],[[329, 318],[303,337],[321,330]]],
                      "Desk_MugRack":[[[459, 287],[468,344],[434,315]],[[329, 318],[303,337],[321,330]]],
                      "Desk_CubeStacking":[[[308,169],[308,180],[307,186]],[[391,327],[393,333],[390,352]]],
                      "Desk_TrashSweep":[[[490,132],[472,181],[471,203]],[[224,306],[208,358],[193,431]]],
                      }
predictor = sam2_initialize()

x_home_js = np.array([-0.07188314616233507, -0.5007457342122718, 0.07313486429670638, -2.7816527503720883, 0.05476125807473123, 2.2630911769337083, -0.7468963222873954])
y_home_js = np.array([-0.0684262,-0.512061,0.0723129,-2.794,0.0469305,2.28233,0.749217])
home_js = y_home_js

# DP3_PRETRAINED_CKPT_PATH = f"/home/liusong/scp_receive/dp3/franka_real_simple_pose9/checkpoints/{task_name}.ckpt"
# sys.path.append("/home/liusong/ProgramFiles/VA-VLA/DP3/3D-Diffusion-Policy/3D-Diffusion-Policy/")
# from DP3_ModelInference import DP3_ModelInference
# model_va = DP3_ModelInference(DP3_PRETRAINED_CKPT_PATH)

model_va = SmolVLA_ModelInference(
    policy_path="/home/liusong/ProgramFiles/Huggingface/lerobot/outputs/train/my_smolvla_song1/checkpoints/last/pretrained_model",
    policy_repo_id="/home/liusong/scp_receive/smolvla",
    device=DEVICE,
)

mission_execution_flag= False
visualize=False

while True:

    window_name = "Overhead RGB"
    cv2.namedWindow(window_name)
    img_hand_rgb = camera_hand.get_rgb_image()
    img_overhead_rgb = camera_overhead.get_rgb_image()
    cv2.imshow(window_name, cv2.resize(cv2.cvtColor(img_overhead_rgb,cv2.COLOR_RGB2BGR),(640,360)))
    cv2.waitKey(1)

    if mission_execution_flag == True:
        
        #Window Overview
        mission_execution_thread = threading.Thread(
            target=mission_execution,
            args=(task_name,camera_hand,camera_overhead, bestman,predictor,model_va,visualize))
        mission_execution_thread.start()
        mission_execution_flag = False


    if sys.stdin in select.select([sys.stdin], [], [],  0.01)[0]:
        line = sys.stdin.readline()
        pressed_key = line.strip()
        if line:
            print(f"You pressed: {pressed_key}")
        if pressed_key == 'n':
            mission_execution_flag = True
        if pressed_key == 'q':
            print("Program Over!")
            break
        if pressed_key in object_marker_dict.keys():
            task_name = pressed_key
            if task_name == "Desk_CubeStacking":
                home_js = y_home_js
            else:
                home_js = x_home_js
            bestman.go_home(home_js)
            mission_execution_flag = True

bestman.release_robot()
exit(-1)
