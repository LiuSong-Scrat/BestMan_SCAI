#!/usr/bin/env python
import time
import math
import argparse
import numpy as np
import sys, os
# sys.path.insert(1, "../../..")
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Franka3, Pose


# 1. 实例化 Session 对象
bestman = Bestman_Real_Franka3()

# 初始化机器人（原代码逻辑）
if bestman.initialize_robot() is not True:
    exit(-1)

print(bestman.robot.current_cartesian_state)
print(bestman.robot.current_joint_state)
print("Gripper: ",bestman.gripper.width) #0.08-0.00


bestman.release_robot()