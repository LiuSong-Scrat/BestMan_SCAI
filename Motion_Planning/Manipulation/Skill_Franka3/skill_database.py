#!/usr/bin/env python
__copyright__ = "Copyright (C) 2016-2021 Flexiv Ltd. All Rights Reserved."
__author__ = "Flexiv"

import time
import math
import numpy as np
import argparse
import sys, os
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Franka3, Pose  

class SkillFranka3Database:
   def grasp(self,bestman, raw_pose, approaching_dir='iso', retracting_dir='top', D_pre=0.10, D_ret=0.05, force=5, grasp_speed=0.1):
        """
        Description:
        The elementary skill 'grasp' is implemented with 3 sub-motions:
        1. move to the preparation pose: 
            - This is to avoid collision with objects when directly move to grapsing pose
            - In preparation pose, the gripper's opening (approaching axis Z) is always facing (pointing towards) the object
        2. move to the grasping pose:
            - Move the gipper to the desired grapsing pose along the gripper's direction Z
        3. grasp and move to retreat pose:
            - Move the gripper to the retreat pose above the platform. This is to avoid scratching the object on the platform. 

        The function provides flexible grapsing options including:
        1. raw_pose (List[List[float]]): The desired gripper center. The format follows [position (xyz in m), orientation (euler in degrees)]
        2. approaching_dir (str): 
            - 'top': preparation pose is on top of the object. 
            - 'front': preparation pose is on the same height as the object
            - 'iso': isometric view. The angle is adjustable. 'top' and 'front' are the spacial case where iso_angle is 90 and 0
        3. retracting_dir (str): same options as approaching_dir
        4. D_pre (float): The distance between the grasping position and preparation position
        5. D_ret (float): The distance between the grasping position and retreat position
        6. force (int): the grapsing force of the gripper
        7. grasp_speed (int): the grasping speed pf the gripper

        The key parameters of objects and robot will be update at the end of the motion modules, preparing for the next action.
        """

        ##########################################################################################
        ######################    Associated Evaluations (DON'T Modify)   ########################
        ##########################################################################################

        grasp_position = raw_pose[0]
        grasp_orientation = raw_pose[1]
        yaw = math.radians(grasp_orientation[2])
        c = abs(math.cos(yaw))
        s = abs(math.sin(yaw))
        X, Y, Z = grasp_position
        iso_angle = 45.0

        if approaching_dir == 'top':
            preparation_position = [X, Y, Z + D_pre]
        elif approaching_dir == 'front':
            preparation_position = [X - np.sign(X) * D_pre * c, 
                                    Y - np.sign(Y) * D_pre * s, 
                                    Z]
        elif approaching_dir == 'iso':
            D_pre_decom_ver = D_pre * math.sin(math.radians(iso_angle))
            D_pre_decom_hor = D_pre * math.cos(math.radians(iso_angle))
            preparation_position = [X - np.sign(X) * D_pre_decom_hor * c, 
                                    Y - np.sign(Y) * D_pre_decom_hor * s, 
                                    Z + D_pre_decom_ver]

        if retracting_dir == 'top':
            retraction_position = [X, Y, Z + D_ret]
        elif retracting_dir == 'front':
            retraction_position = [X - np.sign(X) * D_ret * c, 
                                Y - np.sign(Y) * D_ret * s, 
                                Z]
        elif retracting_dir == 'iso':
            D_ret_decom_ver = D_ret * math.sin(math.radians(iso_angle))
            D_ret_decom_hor = D_ret * math.cos(math.radians(iso_angle))
            print([D_ret_decom_ver, D_ret_decom_hor])
            retraction_position = [X - np.sign(X) * D_ret_decom_hor * c, 
                                Y - np.sign(Y) * D_ret_decom_hor * s, 
                                Z + D_ret_decom_ver]


        ##########################################################################################
        ####################################    START Main    ####################################
        ##########################################################################################

        # safe_pose = [0.50,  0.25, 0.25, 0.0, 180.0, 0.0]
        # # Move robot to safe pose
        # print('MOVING to safe pose')
        # preparation_pose = Pose(safe_pose[0:3], safe_pose[3:6])
        # bestman.move_eef_to_goal_pose(preparation_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
        # bestman.wait_for_eef(preparation_pose)
        # print('MOVED to safe pose')

        bestman.open_gripper()

        # Move robot to preparation pose
        print('MOVING to preparation pose')
        preparation_pose = Pose(preparation_position, grasp_orientation)
        bestman.move_eef_to_goal_pose(preparation_pose, maxLinearVel=0.22, maxAngularVel=math.radians(45))
        print('MOVED to preparation pose')


        # Move robot to target pose
        print('MOVING to grasp pose')
        target_pose = Pose(raw_pose[0], raw_pose[1])
        bestman.move_eef_to_goal_pose(target_pose, maxLinearVel=0.1, maxAngularVel=math.radians(45))
        print('MOVED to grasp pose')


        bestman.close_gripper(speed=grasp_speed, force=force)
        time.sleep(0.4)

        # Move robot to retracting pose
        print('MOVING to retracting pose')
        retracting_pose = Pose(retraction_position, grasp_orientation)
        bestman.move_eef_to_goal_pose(retracting_pose, maxLinearVel=0.1, maxAngularVel=math.radians(45))
        print('MOVED to retracting pose')


        # update robot parameters 
        # TODO: can consider combine with the robot updates in the sameple_runthrough.py, as in the main execution file

        bestman.record_position = retraction_position   
        bestman.record_orientation = raw_pose[1]
        bestman.last_action = 'grasp'
        bestman.constrained_grasping_pitch = raw_pose[1][1]

   def place(self,bestman, raw_pose, retracting_dir='top', D_ret=0.10):
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
        bestman.move_eef_to_goal_pose(preparation_pose, maxLinearVel=0.1, maxAngularVel=math.radians(45))
        print('MOVED to preparation pose')


        print('MOVING to place pose')
        place_pose = preparation_pose
        bestman.move_eef_to_goal_pose(place_pose, maxLinearVel=0.1, maxAngularVel=math.radians(45))
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
        bestman.move_eef_to_goal_pose(retracting_pose, maxLinearVel=0.1, maxAngularVel=math.radians(45))
        print('MOVED to retracting pose')

        # update robot parameters 
        # TODO: can consider combine with the robot updates in the sameple_runthrough.py, as in the main execution file

        bestman.record_position = retraction_position 
        bestman.record_orientation = place_orientation
        bestman.last_action = 'place'
        bestman.constrained_grasping_pitch = None



        