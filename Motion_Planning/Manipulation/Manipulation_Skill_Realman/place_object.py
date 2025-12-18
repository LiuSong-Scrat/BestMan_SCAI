#!/usr/bin/env python

import time
import math
import argparse
import numpy as np
import sys, os
# sys.path.insert(1, "../../..")
# sys.path.insert(2, "../../../Robotics_API")
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Realman, Pose  
from Planning_tools.get_auto_grasp_orientation import auto_orientation

def place(bestman, raw_pose, retracting_dir='top', D_ret=0.10, GV=None):

    place_position = raw_pose[0]
    place_orientation = raw_pose[1]
    yaw = math.radians(place_orientation[2]-90)
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    X, Y, Z = place_position
    if bestman.arm == 'left':
        X_w_offset = X + 0.4
    elif bestman.arm == 'right':
        X_w_offset = X - 0.4

    if retracting_dir == 'top':
        retraction_position = [X, Y, Z + D_ret]
    elif retracting_dir == 'front':
        retraction_position = [X - np.sign(X_w_offset) * D_ret * s, 
                               Y - np.sign(Y) * D_ret * c, 
                               Z]
    elif retracting_dir == 'iso':
        D_ret_decom_ver = D_ret * math.sin(math.radians(30))
        D_ret_decom_hor = D_ret * math.cos(math.radians(30))
        # print([D_ret_decom_ver, D_ret_decom_hor])
        retraction_position = [X - np.sign(X_w_offset) * D_ret_decom_hor * s, 
                               Y - np.sign(Y) * D_ret_decom_hor * c]

    # print(place_position)
    # print(place_orientation)
    # print([math.radians(k) for k in place_orientation])
    ##########################################################################################
    ##############################    DEFINE Hyperparameters    ##############################
    ##########################################################################################

    # other features for motion plan and collision check
    frequency = bestman.frequency

    # parameters for force control
    timeout = bestman.cfg.Manipulation.place_object.timeout
    bottom_detect_threshold = bestman.cfg.Manipulation.place_object.bottom_detect_threshold

    ##########################################################################################
    ######################    Associated Evaluations (DON'T Modify)   ########################
    ##########################################################################################

    period = 1.0 / frequency
    loop_counter = 0

    ##########################################################################################
    ####################################    START Main    ####################################
    ##########################################################################################

    # Move robot to preparation pose
    print('MOVING to preparation pose')
    preparation_pose = Pose(place_position, place_orientation)
    bestman.move_eef_to_goal_pose(preparation_pose, constrained_eef=True, v=40)
    print('MOVED to preparation pose')
    time.sleep(0.3)
    
    # START main task
    print('START Descending')
    start_time = time.time()
    current_pose_list = place_position + [math.radians(k) for k in place_orientation]

    while time.time() - start_time < timeout:

        # time.sleep(period)
        current_pose_list[2] = current_pose_list[2] - 0.00100
        wrench_mea = bestman.get_eef_wrench()
        
        ext_force = np.array([wrench_mea[0], wrench_mea[1], wrench_mea[2]])
        if np.linalg.norm(ext_force) > bottom_detect_threshold:
            print("PLACED object with force: " + str(bottom_detect_threshold) + 'N')
            # print(bestman.get_eef_wrench())
            bestman.hold_robot()
            break
        if current_pose_list[2] < 0.05: 
            bestman.hold_robot()
            break
        
        bestman.robot.rm_movep_canfd(current_pose_list, follow=True)
        
        loop_counter += 1

    bestman.open_gripper()
    time.sleep(0.5)
    print(bestman.get_eef_wrench())
    print('MOVING to retracting pose')
    if retracting_dir == 'top':
        retraction_position.append(current_pose_list[2] + D_ret)
    elif retracting_dir == 'front':
        retraction_position.append(current_pose_list[2])
    elif retracting_dir == 'iso':
        retraction_position.append(current_pose_list[2] + D_ret_decom_ver)
    retracting_pose = Pose(retraction_position, place_orientation)
    bestman.move_eef_to_goal_pose(retracting_pose, constrained_eef=True, v=40)
    bestman.wait_for_eef(retracting_pose)
    print('MOVED to retracting pose')
    time.sleep(0.2)

    if GV is not None:
        GV.current_position = retraction_position 
        GV.current_orientation = place_orientation
        GV.last_action = 'place'


if __name__ == "__main__":

    bestman = Bestman_Real_Realman(arm='left')
    bestman.connect_gripper()

    position = [-0.30, 0.40, 0.20]
    orientation = auto_orientation(position, arm=bestman.arm)
    place_pose = [position, orientation]

    place(bestman, place_pose, retracting_dir='iso', D_ret=0.15)
