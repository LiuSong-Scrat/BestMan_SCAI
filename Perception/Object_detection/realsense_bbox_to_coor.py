
import numpy as np

def realsense_bbox_to_coor(bestman, bbox, obj_diameter, Z, image, level_height=0.000):

    Z = Z - level_height
    vcp = bestman.cfg.Manipulation.base_level_visual_calibration_parameters
    predicted_bbox_factor = vcp[0] * Z**3 + vcp[1] * Z**2 + vcp[2] * Z**1 + vcp[3]
    scale = 0.123 / predicted_bbox_factor


#    { 910.75244140625, 0.0, 648.3623046875,
#      0.0, 910.4097290039062, 382.8463134765625
#    , 0.0, 0.0, 1.0}
    # camera_pixel_center = np.array([640, 360])
    camera_pixel_center = np.array([648.3623046875, 382.8463134765625])
    CAMERA_CENTER_X = camera_pixel_center[0]
    CAMERA_CENTER_Y = camera_pixel_center[1]
    # cases to deal with visual distorsion (object tilted when far from the camera center)   0123->x1y1x2y2


    pixel_dia = obj_diameter / scale
    if bbox[3] < CAMERA_CENTER_Y and bbox[0] > CAMERA_CENTER_X:   
        print('CASE 1')#UP_RIGHT
        compensated_center = [bbox[0] + 0.5 * pixel_dia, bbox[3] - 0.5 * pixel_dia]
    elif bbox[3] < CAMERA_CENTER_Y and bbox[2] < CAMERA_CENTER_X:
        print('CASE 2')#UP_LEFT
        compensated_center = [bbox[2] - 0.5 * pixel_dia, bbox[3] - 0.5 * pixel_dia]
    elif bbox[1] > CAMERA_CENTER_Y and bbox[2] < CAMERA_CENTER_X:
        print('CASE 3')#DOWN_LEFT 
        compensated_center = [bbox[2] - 0.5 * pixel_dia, bbox[1] + 0.5 * pixel_dia]
    elif bbox[1] > CAMERA_CENTER_Y and bbox[0] > CAMERA_CENTER_X:
        print('CASE 4')#DOWN_RIGHT
        compensated_center = [bbox[0] + 0.5 * pixel_dia, bbox[1] + 0.5 * pixel_dia]
    elif bbox[3] < CAMERA_CENTER_Y:
        print('CASE 5')
        compensated_center = [(bbox[0]+bbox[2])/2, bbox[3] - 0.5 * pixel_dia]
    elif bbox[1] > CAMERA_CENTER_Y:
        print('CASE 6')
        compensated_center = [(bbox[0]+bbox[2])/2, bbox[1] + 0.5 * pixel_dia]
    elif bbox[0] > CAMERA_CENTER_X:
        print('CASE 7')
        compensated_center = [bbox[0] + 0.5 * pixel_dia, (bbox[1]+bbox[3])/2]
    elif bbox[2] < CAMERA_CENTER_X:
        print('CASE 8')
        compensated_center = [bbox[2] - 0.5 * pixel_dia, (bbox[1]+bbox[3])/2]
    else:
        print('CASE 0')
        compensated_center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    

 
    
    print("compensated_center = ",compensated_center)
    compensated_move_dis = [compensated_center[0] - CAMERA_CENTER_X, compensated_center[1] - CAMERA_CENTER_Y]
    compensated_world_dis = [compensated_move_dis[0]*scale, compensated_move_dis[1]*scale]





    return compensated_world_dis,compensated_center


def realsense_pt_to_coor(bestman, bbox, obj_diameter, Z, new_img, best_pt, best_pt_infer_pt, slope,level_height=0.000):

    Z = Z - level_height
    vcp = bestman.cfg.Manipulation.base_level_visual_calibration_parameters
    predicted_bbox_factor = vcp[0] * Z**3 + vcp[1] * Z**2 + vcp[2] * Z**1 + vcp[3]
    scale = 0.123 / predicted_bbox_factor


#    { 910.75244140625, 0.0, 648.3623046875,
#      0.0, 910.4097290039062, 382.8463134765625
#    , 0.0, 0.0, 1.0}
    # camera_pixel_center = np.array([640, 360])
    camera_pixel_center = np.array([648.3623046875, 382.8463134765625])
    CAMERA_CENTER_X = camera_pixel_center[0]
    CAMERA_CENTER_Y = camera_pixel_center[1]
    # cases to deal with visual distorsion (object tilted when far from the camera center)   0123->x1y1x2y2


    pixel_dia = obj_diameter / scale
    import math;import cv2
    theta_radian = math.atan(slope)
    half_pixel_dia = pixel_dia/2
    compensated_center = []
    if best_pt[0]-best_pt_infer_pt[0] > 0 :  #Can't Disappear == 0
        compensated_center = [best_pt[0]-half_pixel_dia*math.cos(theta_radian), best_pt[1]-half_pixel_dia*math.sin(theta_radian)]
    elif best_pt[0]-best_pt_infer_pt[0] < 0:
        compensated_center = [best_pt[0]+half_pixel_dia*math.cos(theta_radian), best_pt[1]+half_pixel_dia*math.sin(theta_radian)]
    
    if bbox[3] < CAMERA_CENTER_Y and bbox[0] > CAMERA_CENTER_X:   
        print('CASE 1')#UP_RIGHT
    elif bbox[3] < CAMERA_CENTER_Y and bbox[2] < CAMERA_CENTER_X:
        print('CASE 2')#UP_LEFT
    elif bbox[1] > CAMERA_CENTER_Y and bbox[2] < CAMERA_CENTER_X:
        print('CASE 3')#DOWN_LEFT 
    elif bbox[1] > CAMERA_CENTER_Y and bbox[0] > CAMERA_CENTER_X:
        print('CASE 4')#DOWN_RIGHT
    elif bbox[3] < CAMERA_CENTER_Y: 
        print('CASE 5')#UP
    elif bbox[1] > CAMERA_CENTER_Y:
        print('CASE 6')#DONW
    elif bbox[0] > CAMERA_CENTER_X:
        print('CASE 7')#RIGHT
    elif bbox[2] < CAMERA_CENTER_X:
        print('CASE 8')#LEFT
    else:
        print('CASE 0')
        compensated_center = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    
    
    cv2.circle(new_img,(int(compensated_center[0]),int(compensated_center[1])),10,(0,255,255),-1)
    cv2.circle(new_img,(int(best_pt[0]),int(best_pt[1])),10,(255,0,0),-1)
    cv2.imshow("new_img",new_img)
    cv2.waitKey(0)
 




    
    print("compensated_center = ",compensated_center)
    compensated_move_dis = [compensated_center[0] - CAMERA_CENTER_X, compensated_center[1] - CAMERA_CENTER_Y]
    compensated_world_dis = [compensated_move_dis[0]*scale, compensated_move_dis[1]*scale]





    return compensated_world_dis,compensated_center

