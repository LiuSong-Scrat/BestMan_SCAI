import json
from tqdm import tqdm

import cv2
import numpy as np
import os

import argparse
import h5py
import time
import threading


# class SimulationDataCollection:
#     def __init__(self,parser):
#         self.args = parser.parse_args()
#         with open(self.args.config_file, 'r') as f:
#             config = json.load(f)
#         TASK_CONFIG = config['task_config']
#         self.cfg = TASK_CONFIG



#         self.task = self.args.task
#         self.data_path = os.path.join(config['device_settings']["data_dir"], "dataset" ,str(self.task))
#         self.timestamps_list=[]
#         self.trajectory_list=[]
#         self.overhead_frame = None #overhead_frame
#         self.hand_frame = None #hand_frame
#         self.cur_pos = None #xyz/xyzw
#         self.episode_finish=False
#         self.collection_finish=False
#         self.buffer_lock = threading.Lock()

#     def reset_parameters(self):
#         self.timestamps_list=[]
#         self.trajectory_list=[]
#         self.overhead_frame = None #overhead_frame
#         self.hand_frame = None #hand_frame
#         self.cur_pos = None #xyz/xyzw
#         self.eff_angular = None #xyz/xyzw
#         self.episode_finish=False
#         self.collection_finish=False
#         self.frame_id=0


#     # Consumer Receive
#     def data_collection(self):
#         self.frame_id = 0
#         #wait for the producer starts
#         while self.episode_finish is not True:
#             while self.overhead_frame is None or self.hand_frame is None or  self.cur_pos is None:
#                     time.sleep(0.1)
#             with self.buffer_lock:
#                 if self.overhead_frame is None or self.hand_frame is None or  self.cur_pos is None:
#                     continue
#                 #cur_pos
#                 cur_time_stamp = time.time()
#                 timetamp_unit_data = {'Frame Index':self.frame_id,'Timestamp':cur_time_stamp,"overhead":self.overhead_frame,"hand":self.hand_frame}
#                 self.timestamps_list.append(timetamp_unit_data)
#                 trajectory_unit_data={"Pos X":None,"Pos Y":None ,"Pos Z": None,"Q_X": None,"Q_Y":None ,"Q_Z": None,"Q_W": None,"eff_angular":self.eff_angular,"Timestamp":cur_time_stamp}
#                 trajectory_unit_data_keys_list = list(trajectory_unit_data.keys())
#                 for idx, value in enumerate(self.cur_pos):
#                     trajectory_unit_data[trajectory_unit_data_keys_list[idx]] = value
#                 self.trajectory_list.append(trajectory_unit_data)
#                 time.sleep(0.1)
#                 self.frame_id+=1

                
#                 if self.episode_finish is True:
#                     self.collection_finish =  True
#                     break
#     def data_save_hdf5(self):
#         # Data list preparation
#         data_dict = {
#             '/observations/qpos': [],
#             '/action': [],
#             '/observations/eff_angular':[],
#         }
#         for cam_name in self.cfg['camera_names']:
#             data_dict[f'/observations/images/{cam_name}'] = []



#         for idx, row in tqdm(enumerate(self.timestamps_list), desc='Extracting Images'):
#             frame_idx = row['Frame Index']
#             for cam_name in self.cfg['camera_names']:
#                 data_dict[f'/observations/images/{cam_name}'].append(row[cam_name])

#         trajectory_timestamps = np.array([d['Timestamp'] for d in self.trajectory_list])
#         for idx, row in tqdm(enumerate(self.timestamps_list), desc='Extracting States'):
#             closest_idx = (np.abs(trajectory_timestamps - row['Timestamp'])).argmin()
#             closest_row = self.trajectory_list[closest_idx]
#             pos_quat = [
#                 closest_row['Pos X'], closest_row['Pos Y'], closest_row['Pos Z'],
#                 closest_row['Q_X'], closest_row['Q_Y'], closest_row['Q_Z'], closest_row['Q_W']
#             ]
#             data_dict['/observations/qpos'].append(pos_quat)
#             data_dict['/action'].append(pos_quat)
#             data_dict['/observations/eff_angular'].append(closest_row['eff_angular'])
#         idx = len([name for name in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, name))])
#         dataset_path = os.path.join(self.data_path, f'episode_{idx}.hdf5')
#         os.makedirs(os.path.dirname(dataset_path), exist_ok=True)


#         # Save the data
#         with h5py.File(dataset_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
#             root.attrs['sim'] = False
#             obs = root.create_group('observations')
#             image_grp = obs.create_group('images')
#             for cam_name in self.cfg['camera_names']:
#                 image_grp.create_dataset(
#                     cam_name,
#                     data=np.array(data_dict[f'/observations/images/{cam_name}'], dtype=np.uint8),
#                     compression='gzip',
#                     compression_opts=4
#                 )
#             root.create_dataset('observations/qpos', data=np.array(data_dict['/observations/qpos']))
#             root.create_dataset('action', data=np.array(data_dict['/action']))
#             root.create_dataset('observations/eff_angular', data=np.array(data_dict['/observations/eff_angular']))
#         # Reset All Enviornments
#         self.reset_parameters()

    








# # Parse command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--task', type=str, default="test3")  # open_lid, open_fridge, open_drawer, pick_place_pot
# parser.add_argument('--num_episodes', type=int, default=2)
# parser.add_argument('--config_file', type=str, default='/home/liusong/ProgramFiles/IssacSim/Tasks_Wokspace/VisionStacking/config/config.json')

# simulation_data_collection = SimulationDataCollection(parser) 

# for i in range(3):
#     print(f" Verify:  cam in config file: {simulation_data_collection.cfg['camera_names']}  ")

#     #Consumer
#     get_video_thread = threading.Thread(target=simulation_data_collection.data_collection)
#     get_video_thread.start()

#     #Wait for collection_finish/ Trigger on episode_finish
#     while simulation_data_collection.collection_finish is not True: 
#         with simulation_data_collection.buffer_lock:
#             #Producer
#             simulation_data_collection.overhead_frame = cv2.imread("/home/liusong/ProgramFiles/IssacSim/Tasks_Wokspace/temp.png")
#             simulation_data_collection.hand_frame = cv2.imread("/home/liusong/ProgramFiles/IssacSim/Tasks_Wokspace/temp.png")
#             #overhead_frame
#             #hand_frame
#             simulation_data_collection.cur_pos = np.array([1,2,3,4,5,6,7]) #xyz/xyzw
#             simulation_data_collection.eff_angular = np.array([0.04]) #xyz/xyzw

#             simulation_data_collection.episode_finish=True 
#             time.sleep(0.01)
#     simulation_data_collection.data_save_hdf5()







class SimulationDataCollection:
    def __init__(self,parser):
        self.args = parser.parse_args()
        with open(self.args.config_file, 'r') as f:
            config = json.load(f)
        TASK_CONFIG = config['task_config']
        self.cfg = TASK_CONFIG



        self.task = self.args.task
        self.data_path = os.path.join(config['device_settings']["data_dir"], "dataset" ,str(self.task))
        self.timestamps_list=[]
        self.trajectory_list=[]
        self.overhead_frame = None #overhead_frame
        self.hand_frame = None #hand_frame
        self.cur_pos = None #7dof position
        self.pose_eular = None #xyz/xyzw
        self.overhead_cloud_rgb = None 
        self.hand_cloud_rgb = None 
        self.eff_angular = None 
        self.episode_finish=False
        self.collection_finish=False
        self.buffer_lock = threading.Lock()

    def reset_parameters(self):
        self.timestamps_list=[]
        self.trajectory_list=[]
        self.overhead_frame = None #overhead_frame
        self.hand_frame = None #hand_frame
        self.cur_pos = None #7dof position
        self.pose_eular = None #xyz/rgb  m/0-255
        self.overhead_cloud_rgb = None 
        self.hand_cloud_rgb = None 
        self.eff_angular = None #xyz/eular
        self.episode_finish=False
        self.collection_finish=False
        self.frame_id=0


    # Consumer Receive
    def data_collection(self):
        self.frame_id = 0
        #wait for the producer starts
        while self.episode_finish is not True:
            while self.overhead_frame is None or self.hand_frame is None or  self.cur_pos is None:
                    time.sleep(0.1)
            with self.buffer_lock:
                if self.overhead_frame is None or self.hand_frame is None or  self.cur_pos is None:
                    continue
                #cur_pos
                cur_time_stamp = time.time()
                timetamp_unit_data = {'Frame Index':self.frame_id,'Timestamp':cur_time_stamp,"overhead":self.overhead_frame,"hand":self.hand_frame,
                                      "overhead_cloud_rgb":self.overhead_cloud_rgb,"hand_cloud_rgb":self.hand_cloud_rgb}
                self.timestamps_list.append(timetamp_unit_data)
                trajectory_unit_data={"joint_1":None,"joint_2":None ,"joint_3": None,"joint_4": None,"joint_5":None ,"joint_6": None,"joint_7": None,
                                      "eff_angular":self.eff_angular,"pose_eular":self.pose_eular,"Timestamp":cur_time_stamp}
                trajectory_unit_data_keys_list = list(trajectory_unit_data.keys())
                for idx, value in enumerate(self.cur_pos):
                    trajectory_unit_data[trajectory_unit_data_keys_list[idx]] = value
                self.trajectory_list.append(trajectory_unit_data)
                time.sleep(0.2)
                self.frame_id+=1

                
                if self.episode_finish is True:
                    self.collection_finish =  True
                    break
    def data_save_hdf5(self):
        # Data list preparation
        data_dict = {
            '/observations/qpos': [], #joint_position 7dof
            '/observations/pose_eular': [],#xyz-eular
            '/observations/eff_angular':[],# mm
            '/action': [],#xyz
        }
        for cam_name in self.cfg['camera_names']:
            data_dict[f'/observations/images/{cam_name}'] = []
            data_dict[f'/observations/cloud_rgb/{cam_name}'] = []



        for idx, row in tqdm(enumerate(self.timestamps_list), desc='Extracting Images'):
            frame_idx = row['Frame Index']
            for cam_name in self.cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'].append(row[cam_name])
            for cam_name in self.cfg['camera_names']:
                data_dict[f'/observations/cloud_rgb/{cam_name}'].append(row[f'{cam_name}_cloud_rgb'])
        trajectory_timestamps = np.array([d['Timestamp'] for d in self.trajectory_list])
        for idx, row in tqdm(enumerate(self.timestamps_list), desc='Extracting States'):
            closest_idx = (np.abs(trajectory_timestamps - row['Timestamp'])).argmin()
            closest_row = self.trajectory_list[closest_idx]
            arm_pos = [
                closest_row['joint_1'], closest_row['joint_2'], closest_row['joint_3'],
                closest_row['joint_4'], closest_row['joint_5'], closest_row['joint_6'], closest_row['joint_7']
            ]
            pose_eular = closest_row['pose_eular']
            data_dict['/observations/qpos'].append(arm_pos)
            data_dict['/observations/pose_eular'].append(pose_eular)
            data_dict['/action'].append(arm_pos)
            data_dict['/observations/eff_angular'].append(closest_row['eff_angular'])


        idx = len([name for name in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, name))])
        dataset_path = os.path.join(self.data_path, f'episode_{idx}.hdf5')
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)


        # Save the data
        with h5py.File(dataset_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image_grp = obs.create_group('images')
            cloud_rgb_grp = obs.create_group('cloud_rgb')
            for cam_name in self.cfg['camera_names']:
                image_grp.create_dataset(
                    cam_name,
                    data=np.array(data_dict[f'/observations/images/{cam_name}'], dtype=np.uint8),
                    compression='gzip',
                    compression_opts=4
                )
                cloud_rgb_grp.create_dataset(cam_name,data=np.array(data_dict[f'/observations/cloud_rgb/{cam_name}']))
            root.create_dataset('observations/qpos', data=np.array(data_dict['/observations/qpos']))
            root.create_dataset('observations/pose_eular', data=np.array(data_dict['/observations/pose_eular']))
            root.create_dataset('action', data=np.array(data_dict['/action']))
            root.create_dataset('observations/eff_angular', data=np.array(data_dict['/observations/eff_angular']))



        # Reset All Enviornments
        self.reset_parameters()












