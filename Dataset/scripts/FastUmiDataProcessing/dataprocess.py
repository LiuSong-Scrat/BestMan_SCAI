'''
step1
将相机系下面的数据转换为基座系下的数据，并且将夹爪的宽度归一化
'''

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

process_mode = None #global variable

# 加载预定义的字典
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

def get_gripper_width(img_list):

    distances = []
    distances_index = []
    current_frame = 0
    frame_count = len(img_list)

    for i in range(img_list.shape[0]):
        gray = cv2.cvtColor(img_list[i, :, :, :], cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # print(f"Detected markers in frame {current_frame}: {ids.flatten()}")
            current_frame += 1

            # 存储标记的中心点
            marker_centers = []
            for i, marker_id in enumerate(ids.flatten()):
                # 获取标记的四个角点
                if marker_id == 0 or marker_id == 1:
                    marker_corners = corners[i][0]
                    # 计算标记的中心点
                    center = np.mean(marker_corners, axis=0).astype(int)
                    marker_centers.append(center)
                # 在图像上绘制标记的ID

            # 如果检测到至少两个标记，计算它们之间的距离
            if len(marker_centers) >= 2:
                # 只取前两个标记计算距离130...
                distance = np.linalg.norm(marker_centers[0] - marker_centers[1])

                distances.append(distance)
                distances_index.append(current_frame)

            elif len(marker_centers) == 1:
                distance = abs(gray.shape[1] / 2 - marker_centers[0][0]) * 2

                distances.append(distance)
                distances_index.append(current_frame)


    distances = np.array(distances)
    distances_index = np.array(distances_index)
    distances = ((distances - 140.0) / (566.0 - 140.0) * 850).astype(np.int16).clip(0, 850)
    new_distances = []
    for i in range(len(distances) - 1):
        #处理第一帧
        if i == 0:
            if distances_index[i] == 1:
                new_distances.append(distances[0])
                continue
            else:
                for _ in range(distances_index[0]):
                    new_distances.append(distances[0])
        else:
            if distances_index[i+1] - distances_index[i]==1:
                new_distances.append(distances[i])
            else:
                for k in range(distances_index[i+1] - distances_index[i]):
                    new_distances.append(int( k * (distances[i+1] - distances[i]) / (distances_index[i+1] - distances_index[i]) + distances[i] ))
    new_distances.append(distances[-1])
    if len(new_distances) < frame_count:
        for _ in range(frame_count - len(new_distances)):
            new_distances.append(distances[-1])
    
    return np.array(new_distances)

def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base_to_local):
    # 创建局部坐标系下的旋转矩阵
    rotation_local = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    # 创建局部坐标系下的齐次变换矩阵 T_local
    T_local = np.eye(4)
    T_local[:3, :3] = rotation_local
    T_local[:3, 3] = [x, y, z]
    # print(T_local)
    # print(T_base_to_local)
    # 计算基座坐标系下的齐次变换矩阵 T_base
    T_base_r = np.dot(T_local[:3, :3] , T_base_to_local[:3, :3] )
    
    # 提取基座坐标系下的位置
    x_base, y_base, z_base = T_base_to_local[:3, 3] + T_local[:3, 3]
    
    # 提取基座坐标系下的旋转矩阵并转换为欧拉角
    rotation_base = R.from_matrix(T_base_r)
    roll_base, pitch_base, yaw_base = rotation_base.as_euler('xyz', degrees=False)
    # print(roll_base * 180 / np.pi, pitch_base * 180 / np.pi, yaw_base * 180 / np.pi)
    qx_base, qy_base, qz_base, qw_base = rotation_base.as_quat()
    
    return x_base, y_base, z_base, qx_base, qy_base, qz_base, qw_base, roll_base, pitch_base, yaw_base


def absolute_to_delta_pose_rpy(current_qpos, next_qpos):
    pos_cur = current_qpos[:3]
    quat_cur = current_qpos[3:7]

    pos_next = next_qpos[:3]
    quat_next = next_qpos[3:7]

    t0 = np.eye(4)
    t0[:3, :3] = R.from_quat(quat_cur).as_matrix()
    t0[:3, 3] = pos_cur

    t1 = np.eye(4)
    t1[:3, :3] = R.from_quat(quat_next).as_matrix()
    t1[:3, 3] = pos_next

    t_rel = np.linalg.inv(t0) @ t1

    dpos = t_rel[:3, 3]
    drot = R.from_matrix(t_rel[:3, :3]).as_euler('xyz', degrees=True)  # roll, pitch, yaw

    return np.concatenate([dpos, drot]) # [dx, dy, dz, droll, dpitch, dyaw]


import json
config_file = '/home/liusong/ProgramFiles/BestMan/Dataset/scripts/FastUmiDataProcessing/config.json'
with open(config_file, 'r') as f:
    config = json.load(f)
TASK_CONFIG = config['task_config']
camera_cfg = TASK_CONFIG


def normalize_and_save_hdf5(args):
    input_file = args[0]
    output_file = args[1]
    
    
    # 打开输入HDF5文件
    with h5py.File(input_file, 'r') as f_in:
        # 读取需要处理的数据
        qpos_data = f_in['observations/qpos'][:].astype(np.float32)

        # image = f_in['observations/images/front'][i, :]
        # print(image.shape)
        # exit()
        
        normalized_qpos = np.copy(qpos_data)
        if len(normalized_qpos)==0:
            print("Bad File: ",input_file)
            return
        gripper_width = f_in['observations/eff_angular'][:].astype(np.float32)
        if f_in['observations/eff_angular'] == []:
            print("BAD File: ",input_file)

        normalized_qpos_with_gripper = np.concatenate((normalized_qpos, gripper_width), axis=1)
        # 计算 action: 相邻两帧在基坐标系下的位姿差
        num_steps = normalized_qpos_with_gripper.shape[0]
        delta_action = np.zeros((num_steps, 7), dtype=np.float32)  # [dx, dy, dz, droll, dpitch, dyaw]

        for i in range(num_steps - 1):
            current_qpos_7d = normalized_qpos_with_gripper[i, :7]
            next_qpos_7d    = normalized_qpos_with_gripper[i+1, :7]
            delta = absolute_to_delta_pose_rpy(current_qpos_7d, next_qpos_7d)  # 得到 6 维 [dx…dyaw]
            # 前 6 维保持原值
            delta_action[i, :6] = delta
            # 第 7 维写当前帧的 gripper_width（normalized_qpos_with_gripper[:,7]）
            delta_action[i, 6]  = (
                normalized_qpos_with_gripper[i+1, 7]
                - normalized_qpos_with_gripper[i, 7]
            )

        # 最后一帧动作的平移/旋转保持 0
        delta_action[-1, 6] = 0.0
        with h5py.File(output_file, 'w') as f_out:
            # 复制原文件的结构并写入归一化数据
            f_out.create_dataset('action', data=delta_action)
            observations_group = f_out.create_group('observations')
            images_group = observations_group.create_group('images')
            
            for cam_name in camera_cfg['camera_names']:
                

                max_timesteps = f_in[f'observations/images/{cam_name}'].shape[0]
                cam_hight, cam_width = f_in[f'observations/images/{cam_name}'].shape[1], f_in[f'observations/images/{cam_name}'].shape[2]
                # 复制 images/cam_name 数据集
                images_group.create_dataset(
                    cam_name,
                    (max_timesteps, cam_hight, cam_width, 3),
                    dtype='uint8',
                    chunks=(1, cam_hight, cam_width, 3),
                    compression='gzip',
                    compression_opts=4
                )
                images_group[cam_name][:] = f_in[f'observations/images/{cam_name}'][:]



            
            # 写入归一化后的 qpos 数据
            observations_group.create_dataset('qpos', data=normalized_qpos_with_gripper)
            
            # 复制原始 qvel 数据（假设不需要处理）
            # observations_group.create_dataset('qvel', data=f_in['observations/qvel'][:])
            
            print(f"Normalized data saved to: {output_file}")



def add_index_to_filename(filename, index):
    prefix, num_str = filename.split('_')
    num = int(num_str.split('.')[0])
    suffix = filename.split('.')[-1]
    return f'{prefix}_{num + index}.{suffix}'









def smolvla_save_hdf5(args):
    input_file = args[0]
    output_file = args[1]
    process_mode = args[2]
    # 打开输入HDF5文件
    with h5py.File(input_file, 'r') as f_in:
        # 读取需要处理的数据
        qpos_data = f_in['observations/qpos'][:].astype(np.float32)*180/3.14  #radin ——> angle
        normalized_qpos = np.copy(qpos_data)
        if len(normalized_qpos)==0:
            print("Bad File: ",input_file)
            return
        gripper_width = f_in['observations/eff_angular'][:].astype(np.float32)*1000*0.5  # 2finger /mm——>middle position /m
        if f_in['observations/eff_angular'] == []:
            print("BAD File: ",input_file)
        normalized_qpos_with_gripper = np.concatenate((normalized_qpos, gripper_width), axis=1)
        
        # process_mode = "relative" #absolute
        delta_action_list = []
        if process_mode == "relative":
            # 计算 action: 相邻两帧在基坐标系下的位姿差
            num_steps = normalized_qpos_with_gripper.shape[0]
            delta_action = np.zeros((num_steps, 7), dtype=np.float32)  # [dx, dy, dz, droll, dpitch, dyaw]
            for i in range(num_steps):
                if i == num_steps-1:
                    delta = normalized_qpos_with_gripper[i]-normalized_qpos_with_gripper[i]  
                    delta_action = delta
                else:
                    current_qpos_7d = normalized_qpos_with_gripper[i]
                    next_qpos_7d    = normalized_qpos_with_gripper[i+1]
                    delta = next_qpos_7d-current_qpos_7d
                    delta_action = delta
                delta_action_list.append(delta_action)
        else:
            delta_action_list = normalized_qpos_with_gripper

        with h5py.File(output_file, 'w') as f_out:
            # 复制原文件的结构并写入归一化数据
            f_out.create_dataset('action', data=np.array(delta_action_list))
            observations_group = f_out.create_group('observations')
            images_group = observations_group.create_group('images')
            for cam_name in camera_cfg['camera_names']:
                max_timesteps = f_in[f'observations/images/{cam_name}'].shape[0]
                cam_hight, cam_width = f_in[f'observations/images/{cam_name}'].shape[1], f_in[f'observations/images/{cam_name}'].shape[2]
                # 复制 images/cam_name 数据集
                images_group.create_dataset(
                    cam_name,
                    (max_timesteps, cam_hight, cam_width, 3),
                    dtype='uint8',
                    chunks=(1, cam_hight, cam_width, 3),
                    compression='gzip',
                    compression_opts=4
                )

                images_group[cam_name][:] = f_in[f'observations/images/{cam_name}'][:]

                # # Make the overhead be the hand first frame
                # if cam_name == 'overhead':
                #     images_group[cam_name][:] = np.tile(f_in[f'observations/images/hand'][0],(max_timesteps,1,1,1))
                # else:
                #     images_group[cam_name][:] = f_in[f'observations/images/{cam_name}'][:]


            
            # 写入归一化后的 qpos 数据
            observations_group.create_dataset('qpos', data=normalized_qpos_with_gripper)
            # 复制原始 qvel 数据（假设不需要处理）
            # observations_group.create_dataset('qvel', data=f_in['observations/qvel'][:])
            print(f"Normalized data saved to: {output_file}")


def smolvla_save_hdf5_pose(args):
    input_file = args[0]
    output_file = args[1]
    process_mode = args[2]
    # 打开输入HDF5文件
    with h5py.File(input_file, 'r') as f_in:
        # 读取需要处理的数据
        qpos_data =f_in['observations']['pose_eular'][:].astype(np.float32)  #translation-m  RPY-radin
        normalized_qpos = np.copy(qpos_data)
        if len(normalized_qpos)==0:
            print("Bad File: ",input_file)
            return
        gripper_width = f_in['observations/eff_angular'][:].astype(np.float32)*0.5  # 2finger /mm——>middle position /m
        gripper_width[gripper_width>0.035] = 1
        gripper_width[gripper_width<=0.035] = 0

        
        if f_in['observations/eff_angular'] == []:
            print("BAD File: ",input_file)
        normalized_qpos_with_gripper = np.concatenate((normalized_qpos, gripper_width), axis=1)
        
        delta_action_list = []
        if process_mode == "relative":
            # 计算 action: 相邻两帧在基坐标系下的位姿差
            num_steps = normalized_qpos_with_gripper.shape[0]
            delta_action = np.zeros((num_steps, 7), dtype=np.float32)  # [dx, dy, dz, droll, dpitch, dyaw]
            for i in range(num_steps):
                if i == num_steps-1:
                    delta = normalized_qpos_with_gripper[i]-normalized_qpos_with_gripper[i]  
                    delta_action = delta
                else:
                    current_qpos_7d = normalized_qpos_with_gripper[i]
                    next_qpos_7d    = normalized_qpos_with_gripper[i+1]
                    delta = next_qpos_7d-current_qpos_7d
                    delta_action = delta
                delta_action_list.append(delta_action)
        else:
            delta_action_list = normalized_qpos_with_gripper

        with h5py.File(output_file, 'w') as f_out:
            # 复制原文件的结构并写入归一化数据
            f_out.create_dataset('action', data=np.array(delta_action_list))
            observations_group = f_out.create_group('observations')
            images_group = observations_group.create_group('images')
            for cam_name in camera_cfg['camera_names']:
                max_timesteps = f_in[f'observations/images/{cam_name}'].shape[0]
                cam_hight, cam_width = f_in[f'observations/images/{cam_name}'].shape[1], f_in[f'observations/images/{cam_name}'].shape[2]
                # 复制 images/cam_name 数据集
                images_group.create_dataset(
                    cam_name,
                    (max_timesteps, cam_hight, cam_width, 3),
                    dtype='uint8',
                    chunks=(1, cam_hight, cam_width, 3),
                    compression='gzip',
                    compression_opts=4
                )
                images_group[cam_name][:] = f_in[f'observations/images/{cam_name}'][:]

            
            # 写入归一化后的 qpos 数据
            observations_group.create_dataset('qpos', data=normalized_qpos_with_gripper)
            # 复制原始 qvel 数据（假设不需要处理）
            # observations_group.create_dataset('qvel', data=f_in['observations/qvel'][:])
            print(f"Normalized data saved to: {output_file}")


if __name__ == "__main__":


    
    input_dir = '/home/liusong/ProgramFiles/BestMan/Dataset/dataset/test3'
    task_name = "real_franka3_place_cube"
    action_type = "joint_absolute"
    data_type = task_name+"_"+action_type




    output_dir = '/home/liusong/ProgramFiles/BestMan/Dataset/dataset/test3/src_hdf5_to_lerobot/src_hdf5_to_processed_hdf5_'+data_type
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = [
        filename for filename in os.listdir(input_dir)
        if filename.endswith('.hdf5')
    ] 

    if "absolute" in data_type:
        process_mode = "absolute"
    if "relative" in data_type: 
        process_mode = "relative"

    args_list = []
    index = 400
    for filename in file_list:
        input_file = os.path.join(input_dir, filename)
        filename = add_index_to_filename(filename, index)
        output_file = os.path.join(output_dir, filename)
        args_list.append((input_file,output_file,process_mode))

    print("开始并行处理...")

    # for i in range(len(args_list)):
    # print(f"Processing file {i+1}/{len(args_list)}: {args_list[0][2]}")
    # normalize_and_save_hdf5(args_list[45])  # 测试单个文件处理

    # 使用所有可用的CPU核心数
    num_processes = 10
    with Pool(num_processes) as pool:
        if "joint" in data_type:
            #Joint Collection
            list(
                tqdm(pool.imap_unordered(smolvla_save_hdf5, args_list),
                        total=len(args_list),
                        desc="Processing files"))
        if "pose" in data_type:
            #Pose Collection
            list(
                tqdm(pool.imap_unordered(smolvla_save_hdf5_pose, args_list),
                        total=len(args_list),
                        desc="Processing files"))




        
    print("所有文件处理完成。")

