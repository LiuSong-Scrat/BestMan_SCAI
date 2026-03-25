#!/usr/bin/env python
import time
import math
import argparse
import numpy as np
import sys, os
import cv2
# sys.path.insert(1, "../../..")
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Franka3, Pose
from Sensor.Camera_Realsense import Camera_Realsense

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

def affine(t=[],q=[]):
    return [t,q]

if __name__ == "__main__":

    # 1. 实例化 Session 对象
    bestman = Bestman_Real_Franka3()

    # 初始化机器人（原代码逻辑）
    if bestman.initialize_robot() is not True:
        exit(-1)
    #2.Camera Initialize
    camera_devices = list( bestman.cfg.Camera.keys())
    camera_hand,camera_overhead = None,None
    for cam_dev in camera_devices:
        if cam_dev == "D435I":
            camera_hand = Camera_Realsense(bestman.cfg.Camera[cam_dev])
            time.sleep(1)
    
    # # #prepared_pose: go home
    # # position = [0.55,-0.05,0.45]
    # # orientation = [0.999649,0.00253632,-0.0196627,0.0175826]
    # # move_pose = [position, orientation]
    # position = [0.35,-0.0,0.65] 
    # orientation = [1,0,0,0]
    # move_pose = [position, orientation]

    pose_list = []
    pose_list.append(affine(t=[0.257465,-0.0836753,0.418019],q=[0.955242,0.0158718,0.287519,-0.0677707]))
    pose_list.append(affine(t=[0.225042,0.0674834,0.442378],q=[0.956005,0.000848874,0.28251,0.0790067]))
    pose_list.append(affine(t=[0.341383,0.155315,0.487035],q=[0.970415,0.0302444,0.192019,0.14321]))
    pose_list.append(affine(t=[0.36708,0.00474053,0.53269],q=[0.988014,0.00581013,0.153266,0.0174361]))
    pose_list.append(affine(t=[0.312005,-0.118436,0.456804],q=[0.976582,0.0305622,0.206138,-0.0534896]))
    pose_list.append(affine(t=[0.326286,-0.13641,0.466857],q=[0.956326,0.127441,0.229996,-0.127673]))
    pose_list.append(affine(t=[0.332225,-0.0628791,0.666703],q=[0.967768,-0.0377253,0.248999,-0.00113862]))
    pose_list.append(affine(t=[0.476693,-0.0953353,0.575185],q=[0.985486,-0.00279116,0.163336,0.0461537]))
    pose_list.append(affine(t=[0.39173,-0.164432,0.541438],q=[0.965233,-0.112271,0.200508,-0.124569]))
    pose_list.append(affine(t=[0.35557,-0.127008,0.520651],q=[0.949973,0.180377,0.248776,-0.0559174]))
    pose_list.append(affine(t=[0.322704,0.0310063,0.579539],q=[0.972774,-0.0728059,0.213653,0.0525522]))
    pose_list.append(affine(t=[0.556391,-0.0110693,0.548328],q=[0.998624,0.0271286,-0.0325051,-0.0309418]))
     
    for index,move_pose in enumerate(pose_list):
        move(bestman, move_pose)
        rgb = camera_hand.get_rgb_image()
        # cv2.imshow("bgr",cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        # cv2.waitKey(0)
        cv2.imwrite(f"/home/liusong/temp/calibrate/color_{index}.png",rgb)
    bestman.release_robot()
    
    move(bestman, move_pose)
    
    
