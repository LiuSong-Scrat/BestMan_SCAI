#!/usr/bin/env python

import time
import math
import numpy as np
import sys, os
# sys.path.insert(1, "../../..")
# sys.path.insert(2, "../../../Robotics_API")
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Realman, Pose  
from Planning_tools.get_auto_grasp_orientation import auto_orientation

def grasp(bestman, raw_pose, approaching_dir='iso', retracting_dir='top', D_pre=0.10, D_ret=0.05, GV=None):

    ##########################################################################################
    ######################    Associated Evaluations (DON'T Modify)   ########################
    ##########################################################################################

    grasp_position = raw_pose[0]
    grasp_orientation = raw_pose[1]
    yaw = math.radians(grasp_orientation[2]-90)
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    X, Y, Z = grasp_position
    if bestman.arm == 'left':
        X_w_offset = X + 0.4
    elif bestman.arm == 'right':
        X_w_offset = X - 0.4

    if approaching_dir == 'top':
        preparation_position = [X, Y, Z + D_pre]
    elif approaching_dir == 'front':
        preparation_position = [X - np.sign(X_w_offset) * D_pre * s, 
                                Y - np.sign(Y) * D_pre * c, 
                                Z]
    elif approaching_dir == 'iso':
        D_pre_decom_ver = D_pre * math.sin(math.radians(30))
        D_pre_decom_hor = D_pre * math.cos(math.radians(30))
        preparation_position = [X - np.sign(X_w_offset) * D_pre_decom_hor * s, 
                                Y - np.sign(Y) * D_pre_decom_hor * c, 
                                Z + D_pre_decom_ver]

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
                               Y - np.sign(Y) * D_ret_decom_hor * c, 
                               Z + D_ret_decom_ver]

    # print(preparation_position)
    # print(grasp_position)
    # print(retraction_position)
    # print(grasp_orientation)
    # print([math.radians(k) for k in grasp_orientation])
    ##########################################################################################
    ####################################    START Main    ####################################
    ##########################################################################################

    # Move robot to preparation pose
    print('MOVING to preparation pose')
    preparation_pose = Pose(preparation_position, grasp_orientation)
    bestman.move_eef_to_goal_pose(preparation_pose)
    print('MOVED to preparation pose')
    time.sleep(0.2)

    # time.sleep(1)

    # Move robot to grasping pose
    print('MOVING to grasping pose')
    grasping_pose = Pose(grasp_position, grasp_orientation)
    bestman.move_eef_to_goal_pose(grasping_pose, constrained_eef = True, v=40)
    print('MOVED to grasping pose')
    time.sleep(0.2)

    bestman.close_gripper(speed=255, force=0)
    time.sleep(0.5)

    # Move robot to retracting pose
    print('MOVING to retracting pose')
    retracting_pose = Pose(retraction_position, grasp_orientation)
    bestman.move_eef_to_goal_pose(retracting_pose, constrained_eef = True, v=40)
    print('MOVED to retracting pose')
    time.sleep(0.2)

    if GV is not None:
        GV.current_position = retraction_position   
        GV.current_orientation = raw_pose[1]
        GV.last_action = 'grasp'
    

if __name__ == "__main__":

    bestman = Bestman_Real_Realman(arm='right')
    bestman.connect_gripper()

    position = [0.20, 0.35, 0.20]
    orientation = auto_orientation(position, arm=bestman.arm)
    grasp_pose = [position, orientation]
    print(position)
    print(orientation)

    grasp(bestman, grasp_pose, approaching_dir='iso', retracting_dir='iso', D_pre=0.10, D_ret=0.10)
