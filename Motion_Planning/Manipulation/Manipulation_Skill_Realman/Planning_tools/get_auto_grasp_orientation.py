#!/usr/bin/env python

import time
import math
import numpy as np
import sys, os
# sys.path.insert(1, "../../../..")
from scipy.spatial.transform import Rotation as R

def auto_orientation(position, arm, constrained_pitch = None):

    X, Y, Z = position
    if arm == 'left':
        X = X + 0.4
    elif arm == 'right':
        X = X - 0.4
    
    horizontal_dis = (X**2 + Y**2) ** 0.5

    roll = 0

    pitch_compensation1 = min(max(horizontal_dis-0.30, 0) / 0.20 * 45.0, 45.0)
    pitch_compensation2 = min(max(Z - 0.10, 0) / 0.30 * 45.0, 45.0)
    pitch = 180.0 - pitch_compensation1 - pitch_compensation2

    if arm == 'left':
        yaw = min(90 - math.degrees(math.atan(X / Y)*1.8), 120)
    elif arm == 'right':
        yaw = max(90 - math.degrees(math.atan(X / Y)*1.8), 60)
    # yaw = 90 - math.degrees(math.atan(X / Y))

    ori = [roll, pitch, yaw] if constrained_pitch is None else [roll, constrained_pitch, yaw]
    print("AUTO generated orientation, Euler: " + str(ori))
    return ori

if __name__ == "__main__":

    position = [0.40, 0.40, 0.25]
    # print(auto_orientation(position, arm='right'))
    print([math.radians(k) for k in auto_orientation(position, arm ='right')])