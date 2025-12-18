import sys
import torch
import numpy as np 
import cv2
import copy

sys.path.append("/home/liusong/ProgramFiles/Huggingface/lerobot/lerobot/")
from record_song import SmolVLA_ModelInference,ACT_ModelInference,DP_ModelInference

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

static_overhead_frame = []

def get_cur_model_observation(camera,bestman,static_overhead_frame):
        
        gripper_width = bestman.gripper.width

        cur_pos = bestman.get_current_joint_values()
        
        cur_eff_pose = bestman.get_current_eef_pose()
        cur_eff_pose_position = cur_eff_pose.position
        cur_eff_rotation =  R.from_quat(cur_eff_pose.orientation)
        cur_eff_pose_orientation_eular_zyx = cur_eff_rotation.as_euler('zyx')
        
        img_hand_rgb = camera.get_rgb_image()

        if len(static_overhead_frame) == 0:
            static_overhead_frame = img_hand_rgb

        # points, colors = camera.get_3d_points()
        # points, colors = np.zeros((10,3)),np.zeros((10,3))#-----------temp_use
        # hand_cloud_rgb = np.hstack((points,((colors*255).astype(np.uint8)))) 

        cur_model_observation = {'joint_1':None,'joint_2':None,'joint_3': None,'joint_4':None,'joint_5':None,'joint_6': None,'joint_7':None,'gripper_width':None,
                                'overhead':static_overhead_frame,"hand":img_hand_rgb,"point_cloud":{},"pose_eular":{}}
        cur_observation_key_list = list(cur_model_observation.keys())
        for idx, value in enumerate(cur_pos):
            cur_model_observation[cur_observation_key_list[idx]] =180*value/3.14
        cur_model_observation['gripper_width'] = gripper_width*1000*0.5
        cur_model_observation['pose_eular'] = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)
        

        return cur_model_observation



# 1.初始化机器人（原代码逻辑）
bestman = Bestman_Real_Franka3()
if bestman.initialize_robot() is not True:
    exit(-1)
bestman.open_gripper()
bestman.go_home()
skill_franka3_database = skill_database.SkillFranka3Database()

#2.Camera Initialize
camera = Camera_Realsense(bestman.cfg.Camera)
time.sleep(bestman.cfg.Camera.init_delay)

#Config For Different Policies
{
    # ############3.PREPARE THE PARAMETERS
    # REAL_RANDOM = False 
    # IMAGE_STANDARD = False
    # ############4.READY FOR SAM2
    # USE_CAMERA = "overhead"#hand/overhead/all
    # USE_SAM2 = True
    # USE_MARK = "cloud_rgb"#arrowmark/dotmark/overlaymarkn
    # ############5.Model Mode
    # inference_model = "act"#dp3/smolvla/act/dp
    # action_mode = "absolute"#relative/absolute
    # action_type = "joint"#joint/pose/wait
}
    
gripper_width_thresh = 0.03
repo_id = "/home/liusong/ProgramFiles/BestMan/Dataset/dataset/test3/src_hdf5_to_lerobot/lerobot_datasets/processed_hdf5_to_lerobot_real_franka3_place_cube_joint_absolute/LiuSong-Scrat/smolvla"
policy_path = "/home/liusong/ProgramFiles/BestMan/Policy/trained_model/lerobot_act/act_real_franka3_place_cube_joint_absolute_50episodes/pretrained_model"
model_inference = ACT_ModelInference(repo_id,policy_path)

while True:
    cur_model_observation = get_cur_model_observation(camera,bestman,static_overhead_frame)
    inference_action =  model_inference.single_inference(cur_model_observation).numpy()
    inference_qpos = np.array(3.14*inference_action[:7]/180)
    inference_gripper_width = inference_action[-1]/(1000*0.5)

    bestman.move_arm_to_joint_values(inference_qpos)
    if inference_gripper_width<gripper_width_thresh:
        bestman.close_gripper()
        
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        pressed_key = line.strip()
        if line:
            print(f"You pressed: {pressed_key}")
        if pressed_key == 'q':
            print("Program Over!")
            break
        
bestman.release_robot()
exit(-1)
