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
        time.sleep(0.02)
def get_base_points_from_cam_points(bestman,mouse_get_cam_3d_points,camera_name):
    base_3d_points = []
    for cam_obj_translation in mouse_get_cam_3d_points:
        H_obj2cam = np.eye(4)
        H_obj2cam[:3,3] = np.array(cam_obj_translation)
        
        
        # #camera_hand2eff
        # camera_hand2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
        # camera_hand2ee_aff = Affine([-0.05,-0.03,-0.04],camera_hand2ee_quat) #x y z
        H_camera_hand2eff = np.array([[-3.79969778e-02,-9.98877888e-01,-2.82700204e-02,-0.0557490352],
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
            # H_camera_overhead2camera_hand = np.array(
            #                                         [[-0.997181,-0.023506,0.071261,0.001421],
            #                                         [0.062925,-0.779280,0.623508,-0.602406],
            #                                         [0.040877,0.626235,0.778562,0.030220],
            #                                         [0.000000,0.000000,0.000000,1.000000]])
            # H_camera_overhead2base = H_camera_hand2base@H_camera_overhead2camera_hand
            # H_camera_extrics = H_camera_overhead2base

            H_camera_extrics = np.array([[-0.02634658,  0.76111293, -0.64808347,  0.89516434],
                                        [ 0.99890868, -0.00496784, -0.04644601,  0.00406642],
                                        [-0.0385712 , -0.64860046, -0.76015112,  0.65447038],
                                        [ 0.        ,  0.        ,  0.        ,  1.        ]])

        H_obj2base = H_camera_extrics@H_obj2cam
        base_3d_points.append(H_obj2base[:3,3])


    return base_3d_points


def update_cam_extrinsics(bestman,camera_name):
    # #camera_hand2eff
    # camera_hand2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
    # camera_hand2ee_aff = Affine([-0.05,-0.03,-0.04],camera_hand2ee_quat) #x y z
    H_camera_hand2eff = np.array([[-3.79969778e-02,-9.98877888e-01,-2.82700204e-02,-0.0557490352],
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
        # H_camera_overhead2camera_hand = np.array(
        #                                         [[-0.997181,-0.023506,0.071261,0.001421],
        #                                         [0.062925,-0.779280,0.623508,-0.602406],
        #                                         [0.040877,0.626235,0.778562,0.030220],
        #                                         [0.000000,0.000000,0.000000,1.000000]])
        # H_camera_overhead2base = H_camera_hand2base@H_camera_overhead2camera_hand
        # H_camera_extrics = H_camera_overhead2base

        H_camera_extrics = np.array([[-0.02634658,  0.76111293, -0.64808347,  0.89516434],
                                            [ 0.99890868, -0.00496784, -0.04644601,  0.00406642],
                                            [-0.0385712 , -0.64860046, -0.76015112,  0.65447038],
                                            [ 0.        ,  0.        ,  0.        ,  1.        ]])
    return H_camera_extrics
# 1.初始化机器人（原代码逻辑）
bestman = Bestman_Real_Franka3()
if bestman.initialize_robot() is not True:
    exit(-1)
bestman.open_gripper()
home_js = np.array([-0.11582,-0.476437,0.0715459,-1.69814,0.0351751,1.22258,0.745147])
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
    #Grasp Pose
    standard_quaternion = [1,0,0,0] # #[0.909507,-0.415611,-0.00766705,-0.00220353] #[1,0,0,0]
    standard_quaternion_twist = [ 0.7071068, 0.7071068, 0, 0 ]
    new_red_gripper = 0.04

    while True:
        try:
            grasp_pose = [mouse_base_3d_points[0]-np.array([0,0,0.02])+np.array([0,0,new_red_gripper]), standard_quaternion] 
            skill_franka3_database.grasp(bestman, grasp_pose, approaching_dir='top', retracting_dir='top', D_pre=0.05, D_ret=0.05)
            print("SUCCESS------------------")
            break
        except Exception as e:
            bestman.robot.recover_from_errors() 
            print("ERROR------------------")
            time.sleep(0.1)

    while True:
        import math
        def quat_to_yaw(quaternion):
            """
            从四元数 (x, y, z, w) 计算 yaw 角度（绕 Z 轴）
            返回值：弧度 [-π, π]
            """
            x,y,z,w = quaternion
            sinr_cosp = 2.0 * (w * z + x * y)
            cosr_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = math.atan2(sinr_cosp, cosr_cosp)
            return yaw
        move_pose = [mouse_base_3d_points[1]+np.array([0,0,0.015])+np.array([0,0,new_red_gripper]), standard_quaternion] 
        D_pre = 0.05
        place_position = move_pose[0]
        place_orientation = move_pose[1]
        yaw = quat_to_yaw(place_orientation)
        c = abs(math.cos(yaw))
        s = abs(math.sin(yaw))
        X,Y,Z = place_position
        preparation_position = [X - np.sign(X) * D_pre * c, 
                                Y + np.sign(Y) * D_pre * s, 
                                Z]
        try:
            preparation_pose = Pose(preparation_position, place_orientation)
            bestman.move_eef_to_goal_pose(preparation_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))

            skill_franka3_database.place(bestman, move_pose, retracting_dir='top', D_ret=0.05)
            print("SUCCESS------------------")
            break
        except Exception as e:
            bestman.robot.recover_from_errors() 
            print("ERROR------------------")
            time.sleep(0.1)
    

    #2.Consumer Save & Restart
    sim_data_collection.episode_finish=True 
    sim_data_collection.data_save_hdf5()
    bestman.open_gripper()
    bestman.go_home()

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


