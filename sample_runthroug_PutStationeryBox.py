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
import copy
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

def collection_data_update(camera_hand,camera_overhead,bestman,sim_data_collection):
    while True:
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



        sim_data_collection.cur_pos = cur_pos
        sim_data_collection.pose_eular = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)#由于是zxy轴，不能用xyz
        sim_data_collection.hand_cloud_rgb = hand_cloud_rgb
        sim_data_collection.overhead_cloud_rgb = overhead_cloud_rgb #-----------temp_use
        sim_data_collection.hand_frame = img_hand_rgb 
        sim_data_collection.overhead_frame = img_overhead_rgb #-----------temp_use
        sim_data_collection.eff_angular = np.array([gripper_width]) #xyz/xyzw
        time.sleep(0.04)
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
# 1.初始化机器人（原代码逻辑）
bestman = Bestman_Real_Franka3()
if bestman.initialize_robot() is not True:
    exit(-1)
bestman.open_gripper()
home_js = np.array([-0.07188314616233507, -0.5007457342122718, 0.07313486429670638, -2.7816527503720883, 0.05476125807473123, 2.2630911769337083, -0.7468963222873954])
bestman.go_home(home_js)
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
    args=(camera_hand,camera_overhead, bestman, sim_data_collection))
send_data_thread.start()

click_camera_name = "overhead"
for i in range(50):

    if click_camera_name == "overhead":
        mouse_get_cam_3d_points = camera_overhead.get_cam_3d_points_from_mouse()
    elif click_camera_name == "hand":
        mouse_get_cam_3d_points = camera_hand.get_cam_3d_points_from_mouse()
    mouse_base_3d_points = get_base_points_from_cam_points(bestman,mouse_get_cam_3d_points,click_camera_name)

    collect_data_thread = threading.Thread(target=sim_data_collection.data_collection)
    collect_data_thread.start()
    #----------------------TASK EXECUTION----------------------# 
    #先点k字母，然后笔中心

    new_red_gripper = 0.01
    standard_quaternion = [0.735709, 0.677228, -0.00406522, -0.00882531] # #[0.442212, 0.896865, 0.00592168, 0.00684362] #[0,1,0,0]
    standard_quaternion_twist = [ 0.7071068, 0.7071068, 0, 0 ]
    ##########################OPEN BOX##########################
    move_pose1 = [mouse_base_3d_points[0]-np.array([0,-0.01,0.010])+np.array([0,0,new_red_gripper]), standard_quaternion] 
    pre_move_pose= copy.deepcopy(move_pose1)
    pre_move_pose[0]+=np.array([0,0.05,0])
    force_move(bestman,Pose(pre_move_pose[0],move_pose1[1]), maxLinearVel=0.22, maxAngularVel=math.radians(45))
    force_move(bestman,Pose(move_pose1[0],move_pose1[1]), maxLinearVel=0.22, maxAngularVel=math.radians(45))

    move_pose2= copy.deepcopy(move_pose1)
    move_pose2[0]+=np.array([0,-0.03,0.03])
    force_move(bestman,Pose(move_pose2[0],move_pose2[1]), maxLinearVel=0.22, maxAngularVel=math.radians(45))
    move_pose3= copy.deepcopy(move_pose2)
    move_pose3[0]+=np.array([0,-0.03,0.0])
    force_move(bestman,Pose(move_pose3[0],move_pose3[1]), maxLinearVel=0.22, maxAngularVel=math.radians(45))
    
    # Lift Up 5cm
    cur_eff_pose = bestman.get_current_eef_pose()
    move_towards_pose = Pose(cur_eff_pose.position+np.array([0,0,0.08]), cur_eff_pose.orientation)
    bestman.move_eef_to_goal_pose(move_towards_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
    bestman.open_gripper()




    standard_quaternion = [0,1,0,0]
    ##########################Grasp Pose##########################
    while True:
        try:
            grasp_pose = [mouse_base_3d_points[1]-np.array([0,0.01,0.01])+np.array([0,0,new_red_gripper]), standard_quaternion] 
            skill_franka3_database.grasp(bestman, grasp_pose, approaching_dir='top', retracting_dir='top', D_pre=0.05, D_ret=0.05,force=0.1)
            print("Grasp SUCCESS------------------")
            break
        except Exception as e:
            bestman.robot.recover_from_errors() 
            print("Grasp ERROR------------------")
            time.sleep(0.1)
            
    ##########################Place Pose##########################
    move_pose = [mouse_base_3d_points[0]+np.array([-0.0,-0.05,0.05])+np.array([0,0,new_red_gripper]), standard_quaternion] 

    D_pre = 0.05
    place_position = move_pose[0]
    place_orientation = move_pose[1]
    X,Y,Z = place_position

    preparation_position = [X - D_pre, 
                            Y - D_pre, 
                            Z + D_pre]
    preparation_pose = Pose(preparation_position, place_orientation)
    force_move(bestman,preparation_pose, maxLinearVel=0.3, maxAngularVel=math.radians(90))
    preparation_position = [X , 
                            Y , 
                            Z]
    preparation_pose = Pose(preparation_position, place_orientation)
    force_move(bestman,preparation_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
    
    bestman.open_gripper()

    after_position = [X , 
                        Y , 
                        Z + D_pre]   
    preparation_pose = Pose(after_position, place_orientation)
    force_move(bestman,preparation_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))


    ##########################Close BOX##########################
    close_pose1= copy.deepcopy(move_pose3)
    close_pose1[0]+=np.array([0,-0.05,0.0])
    close_pose1[1] = place_orientation
    force_move(bestman,Pose(close_pose1[0],close_pose1[1]), maxLinearVel=0.22, maxAngularVel=math.radians(45))


    close_pose2= copy.deepcopy(close_pose1)
    close_pose2[0]+=np.array([0,0.12,0.0])
    force_move(bestman,Pose(close_pose2[0],close_pose2[1]), maxLinearVel=0.22, maxAngularVel=math.radians(45))


    #2.Consumer Save & Restart
    sim_data_collection.episode_finish=True 
    sim_data_collection.data_save_hdf5()
    bestman.open_gripper()
    bestman.go_home(home_js)

bestman.release_robot()
exit(-1)


# #Overhead Extrics Label with CloudCompare
# import open3d as o3d
# camera_hand_points = camera_hand.get_3d_points()
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(camera_hand_points[0])
# pcd.colors = o3d.utility.Vector3dVector(camera_hand_points[1])
# # 保存为ASCII格式的PLY文件
# o3d.io.write_point_cloud("/home/liusong/桌面/temp.ply", pcd, write_ascii=True)

# import open3d as o3d
# camera_overhead_points = camera_overhead.get_3d_points()
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(camera_overhead_points[0])
# pcd.colors = o3d.utility.Vector3dVector(camera_overhead_points[1])
# # 保存为ASCII格式的PLY文件
# o3d.io.write_point_cloud("/home/liusong/桌面/temp1.ply", pcd, write_ascii=True)


