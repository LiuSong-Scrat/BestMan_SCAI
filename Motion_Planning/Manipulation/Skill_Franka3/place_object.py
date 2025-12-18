#!/usr/bin/env python
__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import math
import numpy as np
import sys, os
# sys.path.insert(1, "../../..")
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Franka3, Pose

def place(bestman, raw_pose, retracting_dir='top', D_ret=0.10):

    """
    Description:
    The elementary skill 'place' is implemented with 3 sub-motions:
    1. move to the preparation pose on top of the placing point on platform: 
          - This is to avoid collision with the platform when directly move to placing pose
    2. move to the placing pose:
          - Move the gipper to the desired placing pose vertically
          - The vertical motion is implemented based on high-frequency control and external contact detection
    3. open gripper and move to retreat pose:
          - Move the gripper to the retreat pose defined by user. 

    The function provides flexible grapsing options including:
    1. raw_pose (List[List[float]]): The preparation pose. The format follows [position (xyz in m), orientation (euler in degrees)]
    2. retracting_dir (str): 
          - 'top': retracting pose is on top of the object. 
          - 'front': retracting pose is on the same height as the object
          - 'iso': isometric view. The angle is adjustable. 'top' and 'front' are the spacial case where iso_angle is 90 and 0
    3. D_ret (float): The distance between the placing position and retreat position

    The key parameters of objects and robot will be update at the end of the motion modules, preparing for the next action.
    """

    ##########################################################################################
    ######################    Associated Evaluations (DON'T Modify)   ########################
    ##########################################################################################
    
    place_position = raw_pose[0]
    place_orientation = raw_pose[1]
    yaw = math.radians(place_orientation[2])
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    X, Y, Z = place_position
    iso_angle = 60

    if retracting_dir == 'top':
        retraction_position = [X, Y]
    elif retracting_dir == 'front':
        retraction_position = [X - np.sign(X) * D_ret * c, 
                               Y - np.sign(Y) * D_ret * s]
    elif retracting_dir == 'iso':
        D_ret_decom_ver = D_ret * math.sin(math.radians(iso_angle))
        D_ret_decom_hor = D_ret * math.cos(math.radians(iso_angle))
        retraction_position = [X - np.sign(X) * D_ret_decom_hor * c, 
                               Y - np.sign(Y) * D_ret_decom_hor * s]

    ##########################################################################################
    ####################################    START Main    ####################################
    ##########################################################################################

    # Move robot to preparation pose
    print('MOVING to preparation pose')
    preparation_pose = Pose(place_position, place_orientation)
    bestman.move_eef_to_goal_pose(preparation_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
    print('MOVED to preparation pose')


    print('MOVING to place pose')
    place_pose = preparation_pose
    bestman.move_eef_to_goal_pose(place_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
    bestman.open_gripper()
    print('MOVED to place pose')


    
    current_pose = bestman.get_current_eef_pose()
    # update the retreating pose with respect to the level detected

    print('MOVING to retracting pose')
    if retracting_dir == 'top':
        retraction_position.append(current_pose.position[2] + D_ret)
    elif retracting_dir == 'front':
        retraction_position.append(current_pose.position[2])
    elif retracting_dir == 'iso':
        retraction_position.append(current_pose.position[2] + D_ret_decom_ver)
    retracting_pose = Pose(retraction_position, place_orientation)
    bestman.move_eef_to_goal_pose(retracting_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
    print('MOVED to retracting pose')

    # update robot parameters 
    # TODO: can consider combine with the robot updates in the sameple_runthrough.py, as in the main execution file

    bestman.record_position = retraction_position 
    bestman.record_orientation = place_orientation
    bestman.last_action = 'place'
    bestman.constrained_grasping_pitch = None

if __name__ == "__main__":

   # 1. 实例化 Session 对象
    bestman = Bestman_Real_Franka3()

    # 初始化机器人（原代码逻辑）
    if bestman.initialize_robot() is not True:
        exit(-1)

    # grasp_pose = Pose([0.90, -0.20, 0.20], [0.0, 135.0, 0.0])
    # yaw = 0.0
    position = [0.303344,0.00522804,0.275582]
    orientation = [0,1,0,0]
    move_pose = [position, orientation]


    place(bestman, move_pose, retracting_dir='top', D_ret=0.10)
    bestman.release_robot()
