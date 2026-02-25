import sys
import torch
import numpy as np 
import cv2
import copy

sys.path.append("/home/liusong/ProgramFiles/Huggingface/lerobot/lerobot/")
from record_song import SmolVLA_ModelInference,ACT_ModelInference,DP_ModelInference
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
from preprocess_hdf5 import load_cloud_from_group,maybe_bytes_to_str
from eval import single_data_inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



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




CONST_POINTS_NUM=640*480


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
    #camera_hand2eff
    camera_hand2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
    camera_hand2ee_aff = Affine([0.05,0.03,-0.04],camera_hand2ee_quat)
    cur_eff_pose = bestman.get_current_eef_pose()
    eff2base_aff = Affine(cur_eff_pose.position,cur_eff_pose.orientation)
    camera_hand2base_aff = eff2base_aff*camera_hand2ee_aff

    if camera_name == "D435I":
        camera_overhead_extrics = camera_hand2base_aff
    elif camera_name == "L515":

        # # [ 1280x720  p[643.178 357.433]  f[898.481 899.104]  Brown Conrady [0.144588 -0.485378 0.0004702 -4.61056e-05 0.44427] ]
        # camera_overhead2camera_hand = np.array([[-0.994942,0.093101,0.037705,0.082036],
        #                                         [-0.044646,-0.746147,0.664282,-0.576643],
        #                                         [0.089979,0.659240,0.746530,-0.352948],
        #                                         [0.000000,0.000000,0.000000,1.000000]])
        # camera_overhead2camera_hand_quat = R.from_matrix(camera_overhead2camera_hand[:3,:3]).as_quat()
        # camera_overhead2camera_hand_aff = Affine(camera_overhead2camera_hand[:3,3],camera_overhead2camera_hand_quat)
        # camera_overhead2base_aff = camera_hand2base_aff*camera_overhead2camera_hand_aff
        # camera_overhead_extrics = camera_overhead2base_aff
        camera_overhead_extrics = Affine(np.array([ 0.85683062, -0.05194819,  0.65741864]), np.array([ 0.68633475,  0.63405731, -0.27509058, -0.22636501]))
    
    H_camera_extrinsic = np.eye(4) 
    Rotation = R.from_quat(camera_overhead_extrics.quaternion).as_matrix()
    H_camera_extrinsic[:3,:3] = Rotation
    H_camera_extrinsic[:3,3] = camera_overhead_extrics.translation

    return H_camera_extrinsic

def main_mission(camera_hand,camera_overhead, bestman,predictor,model_savg,stage1segmentation):
    click_camera_name = "overhead"
    visualize=False

    while True:

        if click_camera_name == "overhead":
            mouse_get_cam_3d_points = camera_overhead.get_cam_3d_points_from_mouse()
        elif click_camera_name == "hand":
            mouse_get_cam_3d_points = camera_hand.get_cam_3d_points_from_mouse()
        mouse_base_3d_points = get_base_points_from_cam_points(bestman,mouse_get_cam_3d_points,click_camera_name)

        ################# Grasp#################
        # Move Towards Phase
        has_cond=False
        cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
        target_pose_position,target_pose_orientation_xyzw = savg_inference(model_savg,cur_model_observation,predictor,stage1segmentation,has_cond,visualize=visualize)
        move_towards_pose = Pose(target_pose_position, target_pose_orientation_xyzw)
        bestman.move_eef_to_goal_pose(move_towards_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
        #Interaction Phase
        standard_quaternion = [0,1,0,0]
        standard_quaternion_twist = [ 0.7071068, 0.7071068, 0, 0 ]
        new_red_gripper = 0.08
        grasp_pose = [mouse_base_3d_points[0]-np.array([0,0,0.02])+np.array([0,0,new_red_gripper]), standard_quaternion] 
        skill_franka3_database.grasp(bestman, grasp_pose, approaching_dir='top', retracting_dir='top', D_pre=0.15, D_ret=0.15)


        #################Place#################
        # Move Towards Phase
        has_cond=True
        cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
        target_pose_position,target_pose_orientation_xyzw = savg_inference(model_savg,cur_model_observation,predictor,stage1segmentation,has_cond,visualize=visualize)
        move_towards_pose = Pose(target_pose_position, target_pose_orientation_xyzw)
        bestman.move_eef_to_goal_pose(move_towards_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
        #Interaction Phase
        move_pose = [mouse_base_3d_points[1]+np.array([0,0,0.04])+np.array([0,0,new_red_gripper]), standard_quaternion] 
        skill_franka3_database.place(bestman, move_pose, retracting_dir='top', D_ret=0.10)

def mission_execution(camera_hand,camera_overhead, bestman,stage1segmentation, mouse_base_3d_points,predictor,model_savg):

    ################# Grasp#################
    # Move Towards Phase
    has_cond=False
    visualize=True
    cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
    target_pose_position,target_pose_orientation_xyzw = savg_inference(model_savg,cur_model_observation,predictor,stage1segmentation,has_cond,visualize=visualize)
    move_towards_pose = Pose(target_pose_position, target_pose_orientation_xyzw)
    bestman.move_eef_to_goal_pose(move_towards_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
    #Interaction Phase
    new_red_gripper = 0.01
    grasp_pose = [mouse_base_3d_points[0]-np.array([0,0,0.02])+np.array([0,0,new_red_gripper]), target_pose_orientation_xyzw] 
    skill_franka3_database.grasp(bestman, grasp_pose, approaching_dir='top', retracting_dir='top', D_pre=0.05, D_ret=0.05)


    #################Place#################
    # Move Towards Phase
    has_cond=True
    cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
    target_pose_position,target_pose_orientation_xyzw = savg_inference(model_savg,cur_model_observation,predictor,stage1segmentation,has_cond,visualize=visualize)
    move_towards_pose = Pose(target_pose_position, target_pose_orientation_xyzw)
    bestman.move_eef_to_goal_pose(move_towards_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
    #Interaction Phase
    move_pose = [mouse_base_3d_points[1]+np.array([0,0,0.03])+np.array([0,0,new_red_gripper]), target_pose_orientation_xyzw] 
    skill_franka3_database.place(bestman, move_pose, retracting_dir='top', D_ret=0.05)



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
    world_points = ((H_camera_hand_extrinsic[:3,:3]@points.T).T+H_camera_hand_extrinsic[:3,3].T)
    hand_cloud_rgb_var_len = np.hstack((world_points,((colors*255).astype(np.uint8)))) 
    hand_cloud_rgb = uniform_random_sample_points(hand_cloud_rgb_var_len,CONST_POINTS_NUM)
    # camera_hand.visualize_3d_points()


    img_overhead_rgb = camera_overhead.get_rgb_image()
    img_overhead_rgb = cv2.resize(img_overhead_rgb,(640,480),interpolation=cv2.INTER_LINEAR)
    points, colors = camera_overhead.get_3d_points()
    # points, colors = np.zeros((10,3)),np.zeros((10,3))#-----------temp_use
    H_camera_overhead_extrinsic = update_cam_extrinsics(bestman,camera_overhead.dev_name)
    world_points = ((H_camera_overhead_extrinsic[:3,:3]@points.T).T+H_camera_overhead_extrinsic[:3,3].T)
    overhead_cloud_rgb_var_len = np.hstack((world_points,((colors*255).astype(np.uint8)))) 
    overhead_cloud_rgb = uniform_random_sample_points(overhead_cloud_rgb_var_len,CONST_POINTS_NUM)
    # camera_overhead.visualize_3d_points()


    cur_model_observation = {'joint_1':None,'joint_2':None,'joint_3': None,'joint_4':None,'joint_5':None,'joint_6': None,'joint_7':None,'gripper_width':None,
                            'overhead':img_overhead_rgb,"hand":img_hand_rgb,"point_cloud":{},"pose_eular":{}}
    cur_observation_key_list = list(cur_model_observation.keys())
    for idx, value in enumerate(cur_pos):
        cur_model_observation[cur_observation_key_list[idx]] =180*value/3.14
    cur_model_observation['gripper_width'] = gripper_width*1000*0.5
    cur_model_observation['pose_eular'] = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)
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
        cam_obj_quaternion = R.from_matrix(np.array([[1,0,0],[0,1,0],[0,0,1]])).as_quat()
        obj2cam_aff = Affine(cam_obj_translation,cam_obj_quaternion)

        #camera_hand2eff
        camera_hand2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
        camera_hand2ee_aff = Affine([0.05,0.03,-0.04],camera_hand2ee_quat)
        cur_eff_pose = bestman.get_current_eef_pose()
        eff2base_aff = Affine(cur_eff_pose.position,cur_eff_pose.orientation)
        camera_hand2base_aff = eff2base_aff*camera_hand2ee_aff

        if camera_name == "hand":
            camera_overhead_extrics = camera_hand2base_aff
        elif camera_name == "overhead":

            # # [ 1280x720  p[643.178 357.433]  f[898.481 899.104]  Brown Conrady [0.144588 -0.485378 0.0004702 -4.61056e-05 0.44427] ]
            # camera_overhead2camera_hand = np.array([[-0.994942,0.093101,0.037705,0.082036],
            #                                         [-0.044646,-0.746147,0.664282,-0.556717],
            #                                         [0.089979,0.659240,0.746530,-0.367878],
            #                                         [0.000000,0.000000,0.000000,1.000000]])
            # camera_overhead2camera_hand_quat = R.from_matrix(camera_overhead2camera_hand[:3,:3]).as_quat()
            # camera_overhead2camera_hand_aff = Affine(camera_overhead2camera_hand[:3,3],camera_overhead2camera_hand_quat)
            # camera_overhead2base_aff = camera_hand2base_aff*camera_overhead2camera_hand_aff
            # camera_overhead_extrics = camera_overhead2base_aff
            camera_overhead_extrics = Affine(np.array([ 0.85683062, -0.05194819,  0.65741864]), np.array([ 0.68633475,  0.63405731, -0.27509058, -0.22636501]))
            

        obj2base_aff = camera_overhead_extrics*obj2cam_aff
        base_3d_points.append(obj2base_aff.translation)


    return base_3d_points


def savg_inference(model_savg,cur_model_observation,predictor,stage1segmentation,has_cond,visualize):
    # Objects Segmentation 
    obj_sets = ["yellow_mug","blue_cube"]  
    img_overhead_rgb = cur_model_observation['overhead']
    obj_masks = get_sam2_obj_masks(predictor,img_overhead_rgb,obj_sets)

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
        cond_name = obj_sets[0]
        cond_raw= obj_seg_dict[cond_name]['cloud_rgb']
        target_raw= obj_seg_dict[obj_sets[1]]['cloud_rgb']
    else:
        cond_name = "None"
        cond_raw = np.zeros((0, 6), dtype=np.float32)
        target_raw= obj_seg_dict[obj_sets[0]]['cloud_rgb']

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

# 1.初始化机器人（原代码逻辑）
bestman = Bestman_Real_Franka3()
if bestman.initialize_robot() is not True:
    exit(-1)
bestman.open_gripper()
bestman.go_home()
skill_franka3_database = skill_database.SkillFranka3Database()

#2.Camera Initialize
camera_devices = list( bestman.cfg.Camera.keys())
camera_hand,camera_overhead = None,None
for cam_dev in camera_devices:
    if cam_dev == "D435I":
        camera_hand = Camera_Realsense(bestman.cfg.Camera[cam_dev])
    if cam_dev == "L515":
        camera_overhead = Camera_Realsense(bestman.cfg.Camera[cam_dev])
if camera_overhead == None:
    camera_overhead = camera_hand
time.sleep(bestman.cfg.Camera.init_delay)



#3.SAM2 Initialize
USE_SAM2 = True
if USE_SAM2:
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    import time
    import h5py
    sys.path.append("/home/liusong/ProgramFiles/SAM2/sam2")
    from sam2.build_sam import build_sam2_camera_predictor

    sam2_checkpoint = "/home/liusong/ProgramFiles/SAM2/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    frame = cv2.imread("/home/liusong/ProgramFiles/BestMan/Dataset/Images/cube_stak.png")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    width, height = frame.shape[:2][::-1]
    # cv2.imshow("overhead_frame", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    predictor.load_first_frame(frame)
    if_init = True
    ann_frame_idx = 0  # the frame index we interact with
    # First annotation
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    ##! add points, `1` means positive click and `0` means negative click
    points = np.array([[327, 310],[337,323],[332,334]], dtype=np.float32)
    labels = np.array([1,1,1], dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    )

    ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
    points = np.array([[441, 308],[454,321],[447,335]], dtype=np.float32)
    labels = np.array([1,1,1], dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    )





#4.load stage1 segmentation for camera parameters
stagegen_task_name = "real_task_simple"
stagegen_config_file = f"/home/liusong/ProgramFiles/REAP/StageGen/config/{stagegen_task_name}.yaml"
stagegen_src_hdf5_path = f"/home/liusong/ProgramFiles/REAP/StageGen/source/{stagegen_task_name}.hdf5"
stage1segmentation = Stage1Segmentation(stagegen_config_file, stagegen_src_hdf5_path)



# 5.Config For Different Policies
# {
#     # ############3.PREPARE THE PARAMETERS
#     # REAL_RANDOM = False 
#     # IMAGE_STANDARD = False
#     # ############4.READY FOR SAM2
#     # USE_CAMERA = "overhead"#hand/overhead/all
#     # USE_SAM2 = True
#     # USE_MARK = "cloud_rgb"#arrowmark/dotmark/overlaymarkn
#     # ############5.Model Mode
#     # inference_model = "act"#dp3/smolvla/act/dp
#     # action_mode = "absolute"#relative/absolute
#     # action_type = "joint"#joint/pose/wait
# }
# gripper_width_thresh = 0.03
# repo_id = "/home/liusong/ProgramFiles/BestMan/Dataset/dataset/test3/src_hdf5_to_lerobot/lerobot_datasets/processed_hdf5_to_lerobot_real_franka3_place_cube_joint_absolute/LiuSong-Scrat/smolvla"
# policy_path = "/home/liusong/ProgramFiles/BestMan/Policy/trained_model/lerobot_act/act_real_franka3_place_cube_joint_absolute_50episodes/pretrained_model"
# model_inference = ACT_ModelInference(repo_id,policy_path)



# 6.LOAD SAVG MODEL
PRETRAINED_CKPT_PATH = "/home/liusong/ProgramFiles/REAP/SAVG/out/checkpoints/last.pt"
# model
model_savg = PoseACTCVAE(
    pc_in_dim=6,
    pc_dim=256,
    pc_grid_size=0.005,
    pc_tokens=512,
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
ckpt = torch.load(PRETRAINED_CKPT_PATH)
model_savg.load_state_dict(ckpt["model"])
model_savg.eval()
assert os.path.isfile(PRETRAINED_CKPT_PATH), f"ckpt not found: {PRETRAINED_CKPT_PATH}"




mission_execution_flag= True

while True:


    window_name = "Overhead RGB"
    cv2.namedWindow(window_name)
    img_hand_rgb = camera_hand.get_rgb_image()
    img_overhead_rgb = camera_overhead.get_rgb_image()
    cv2.imshow(window_name, cv2.resize(cv2.cvtColor(img_overhead_rgb,cv2.COLOR_RGB2BGR),(640,360)))
    cv2.waitKey(1)

    if mission_execution_flag == True:
        click_camera_name = "overhead"
        if click_camera_name == "overhead":
            mouse_get_cam_3d_points = camera_overhead.get_cam_3d_points_from_mouse()
        elif click_camera_name == "hand":
            mouse_get_cam_3d_points = camera_hand.get_cam_3d_points_from_mouse()
        mouse_base_3d_points = get_base_points_from_cam_points(bestman,mouse_get_cam_3d_points,click_camera_name)

        #Window Overview
        mission_execution_thread = threading.Thread(
            target=mission_execution,
            args=(camera_hand,camera_overhead, bestman,stage1segmentation, mouse_base_3d_points,predictor,model_savg))
        mission_execution_thread.start()
        mission_execution_flag = False



    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        pressed_key = line.strip()
        if line:
            print(f"You pressed: {pressed_key}")
        if pressed_key == 'g':
            print("PICKING.................")
            has_cond=True

            ##########GRASP POLICY##############
            # inference_action =  model_inference.single_inference(cur_model_observation).numpy()
            # inference_qpos = np.array(3.14*inference_action[:7]/180)
            # inference_gripper_width = inference_action[-1]/(1000*0.5)
            # bestman.move_arm_to_joint_values(inference_qpos)
            # if inference_gripper_width<gripper_width_thresh:
            #     bestman.close_gripper()

        if pressed_key == 'p':
            print("PLACEING.................")

            ##########PLACE POLICY##############
            # inference_action =  model_inference.single_inference(cur_model_observation).numpy()
            # inference_qpos = np.array(3.14*inference_action[:7]/180)
            # inference_gripper_width = inference_action[-1]/(1000*0.5)
            # bestman.move_arm_to_joint_values(inference_qpos)
            # if inference_gripper_width<gripper_width_thresh:
            #     bestman.close_gripper()

        if pressed_key == 'n':
            mission_execution_flag = True

        if pressed_key == 'q':
            print("Program Over!")
            break
        
bestman.release_robot()
exit(-1)
