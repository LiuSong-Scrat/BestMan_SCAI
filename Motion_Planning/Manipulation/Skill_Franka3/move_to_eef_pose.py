#!/usr/bin/env python
import time
import math
import argparse
import numpy as np
import sys, os
# sys.path.insert(1, "../../..")
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Franka3, Pose

def move(bestman, raw_pose):

    """
    Description:
    Simply move the eef to a pose

    Input:
    bestman: the robot
    raw_pose (List[List[float]]): The desired gripper center. The format follows [position (xyz in m), orientation (euler in degrees)]

    Output:
    None
    """

    goal_pose = Pose(raw_pose[0], raw_pose[1])
    ##########################################################################################
    ####################################    START Main    ####################################
    ##########################################################################################
    # Move robot to preparation pose
    print('MOVING to preparation pose')

    bestman.move_eef_to_goal_pose(goal_pose, maxLinearVel=0.15, maxAngularVel=math.radians(45))

    print('MOVED to preparation pose')
    time.sleep(0.3)

    # update robot parameters 
    # TODO: can consider combine with the robot updates in the sameple_runthrough.py, as in the main execution file

    bestman.record_position = raw_pose[0] 
    bestman.record_orientation = raw_pose[1] 
    bestman.last_action = 'move'
    bestman.constrained_grasping_pitch = None


if __name__ == "__main__":

    # 1. 实例化 Session 对象
    bestman = Bestman_Real_Franka3()

    # 初始化机器人（原代码逻辑）
    if bestman.initialize_robot() is not True:
        exit(-1)

    # #prepared_pose: go home
    # position = [0.55,-0.05,0.45]
    # orientation = [0.999649,0.00253632,-0.0196627,0.0175826]
    # move_pose = [position, orientation]
    position = [0.55,-0.05,0.45]
    orientation = [0,1,0,0]
    move_pose = [position, orientation]

    move(bestman, move_pose)


    bestman.release_robot()