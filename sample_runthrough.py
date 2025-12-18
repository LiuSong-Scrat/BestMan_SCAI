import time
import math
import numpy as np
import cv2
import sys, os
import threading
# sys.path.insert(1, "../../..")
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Franka3, Pose
from Motion_Planning.Manipulation.Skill_Franka3 import skill_database
from Sensor.Camera_Realsense import Camera_Realsense
from Dataset.scripts.data_collection import RealDataCollection
from franky import Affine
def collection_data_update(camera,bestman,sim_data_collection):
    while True:
        gripper_width = bestman.gripper.width

        cur_pos = bestman.get_current_joint_values()
        
        cur_eff_pose = bestman.get_current_eef_pose()
        cur_eff_pose_position = cur_eff_pose.position
        cur_eff_rotation =  R.from_quat(cur_eff_pose.orientation)
        cur_eff_pose_orientation_eular_zyx = cur_eff_rotation.as_euler('zyx')
        
        img_hand_rgb = camera.get_rgb_image()
        points, colors = camera.get_3d_points()
        points, colors = np.zeros((10,3)),np.zeros((10,3))#-----------temp_use
        hand_cloud_rgb = np.hstack((points,((colors*255).astype(np.uint8)))) 
        # camera.visualize_3d_points()

        sim_data_collection.cur_pos = cur_pos
        sim_data_collection.pose_eular = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)#由于是zxy轴，不能用xyz
        sim_data_collection.hand_cloud_rgb = hand_cloud_rgb
        sim_data_collection.overhead_cloud_rgb = hand_cloud_rgb #-----------temp_use
        sim_data_collection.hand_frame = img_hand_rgb 
        sim_data_collection.overhead_frame = img_hand_rgb #-----------temp_use
        sim_data_collection.eff_angular = np.array([gripper_width]) #xyz/xyzw
        time.sleep(0.2)
def get_base_points_from_cam_points(bestman,mouse_get_cam_3d_points):
    base_3d_points = []
    for cam_obj_translation in mouse_get_cam_3d_points:
        cam_obj_quaternion = R.from_matrix(np.array([[1,0,0],[0,1,0],[0,0,1]])).as_quat()
        obj2cam_aff = Affine(cam_obj_translation,cam_obj_quaternion)

        cam2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
        cam2ee_aff = Affine([0.05,0.03,-0.04],cam2ee_quat)

        cur_eff_pose = bestman.get_current_eef_pose()
        eff2base = Affine(cur_eff_pose.position,cur_eff_pose.orientation)

        obj2eff_aff = cam2ee_aff*obj2cam_aff
        obj2base_aff = eff2base*obj2eff_aff
        base_3d_points.append(obj2base_aff.translation)
    return base_3d_points

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

############4.ADD SIMDATACOLLETION
# Parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="test3")  # open_lid, open_fridge, open_drawer, pick_place_pot
parser.add_argument('--config_file', type=str, default='Config/real_data_collection_config.json')
sim_data_collection = RealDataCollection(parser) 
#Consumer
send_data_thread = threading.Thread(
    target=collection_data_update,
    args=(camera, bestman, sim_data_collection))
send_data_thread.start()


for i in range(50):

    mouse_get_cam_3d_points = camera.get_cam_3d_points_from_mouse()
    mouse_base_3d_points = get_base_points_from_cam_points(bestman,mouse_get_cam_3d_points)

    collect_data_thread = threading.Thread(target=sim_data_collection.data_collection)
    collect_data_thread.start()
    #----------------------TASK EXECUTION----------------------# 
    #Grasp Pose
    standard_quaternion = [0,1,0,0]
    standard_quaternion_twist = [ 0.7071068, 0.7071068, 0, 0 ]
    new_red_gripper = 0.08
    grasp_pose = [mouse_base_3d_points[0]-np.array([0,0,0.02])+np.array([0,0,new_red_gripper]), standard_quaternion] 
    skill_franka3_database.grasp(bestman, grasp_pose, approaching_dir='top', retracting_dir='top', D_pre=0.15, D_ret=0.15)

    #Place Pose
    move_pose = [mouse_base_3d_points[1]+np.array([0,0,0.04])+np.array([0,0,new_red_gripper]), standard_quaternion] 
    skill_franka3_database.place(bestman, move_pose, retracting_dir='top', D_ret=0.10)

    #2.Consumer Save & Restart
    sim_data_collection.episode_finish=True 
    sim_data_collection.data_save_hdf5()
    bestman.open_gripper()
    bestman.go_home()

bestman.release_robot()
exit(-1)
