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

def stirring(bestman, raw_pose, H=0.10, omega=1.5*math.pi, Rad=0.03, duration=10):

    ##########################################################################################
    ##############################    DEFINE Hyperparameters    ##############################
    ##########################################################################################

    frequency = bestman.frequency

    ##########################################################################################
    ######################    Associated Evaluations (DON'T Modify)   ########################
    ##########################################################################################

    center_position = raw_pose[0]
    stirring_orientation = raw_pose[1]
    period = 1.0 / frequency
    preparation_position = [center_position[0] + Rad * math.cos(0), 
                            center_position[1] + Rad * math.sin(0), 
                            center_position[2] + H]
    loop_counter = 0

    ##########################################################################################
    ####################################    START Main    ####################################
    ##########################################################################################

    # Move robot to preparation pose
    print('MOVING to preparation pose')
    preparation_pose = Pose(preparation_position, stirring_orientation)
    bestman.move_eef_to_goal_pose(preparation_pose)
    print('MOVED to preparation pose')
    time.sleep(0.2)

    # Move robot to startting pose
    print('MOVING to starting pose')
    start_position = preparation_position.copy()
    start_position[2] = center_position[2]         # descend by H
    start_pose = Pose(start_position, stirring_orientation)
    bestman.move_eef_to_goal_pose(start_pose, constrained_eef=True)
    print('MOVED to starting pose')
    time.sleep(0.2)

    # START main task
    print('START Stirring')
    start_time = time.time()
    stirring_orientation = [math.radians(k) for k in stirring_orientation]
    while time.time() - start_time < duration:

        time.sleep(period)

        target_position = center_position.copy()
        t = loop_counter * period
        angular_displace = t * omega
        target_position[0] = center_position[0] + Rad * math.cos(angular_displace)
        target_position[1] = center_position[1] + Rad * math.sin(angular_displace)
        bestman.robot.rm_movep_canfd(target_position+stirring_orientation, follow=True)

        if time.time() - start_time < 1.0:
            loop_counter += (time.time() - start_time) / 1.0
        elif duration - (time.time() - start_time) < 1.0:
            loop_counter += (duration - (time.time() - start_time)) / 1.0
        else:
            loop_counter += 1.0


if __name__ == "__main__":
    
    bestman = Bestman_Real_Realman(arm='right')

    position = [0.30, 0.40, 0.20]
    orientation = auto_orientation(position, arm=bestman.arm)
    move_pose = [position, orientation]

    stirring(bestman, move_pose, H=0.10, omega=2.0*math.pi, Rad=0.025, duration=10)
