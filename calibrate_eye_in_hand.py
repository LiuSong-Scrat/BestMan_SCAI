import cv2
import numpy as np
import glob
from math import *
import pandas as pd
import os
import json
from scipy.spatial.transform import Rotation as R 
# K=np.array([[4283.95148301687,-0.687179973528103,2070.79900757240],
#             [0,4283.71915784510,1514.87274457919],
#             [0,0,1]],dtype=np.float64)#大恒相机内参
# K=np.matrix([[4283.95148301687,-0.687179973528103,2070.79900757240],
#             [0,4283.71915784510,1514.87274457919],
#             [0,0,1]],dtype=np.float64)#大恒相机内参

#

K=np.array( [[909.8029174804688, 0.0, 630.8021240234375],
             [0.0, 909.768310546875, 366.5152893066406],
             [0.0, 0.0, 1.0]],dtype=np.float64)#大恒相机内参

dist = np.array([0.0, 0.0, 0.0, 0.0])

#
# dist = np.array([[0.027380026193468753, -0.08801521475213679, 0.0011824388616034057,
#                   -0.0008063100320946023, 0.18819181491200773]],dtype=np.float64)

chess_board_x_num=11#棋盘格x方向格子数 -1
chess_board_y_num=8#棋盘格y方向格子数 -1
chess_board_len=20#单位棋盘格长度,mm


#用于根据欧拉角计算旋转矩阵
def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

#用于根据位姿计算变换矩阵
def pose_robot(x, y, z, Tx, Ty, Tz):

    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])  # 列合并
    RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
    # RT1=np.linalg.inv(RT1)
    return RT1

#用来从棋盘格图片得到相机外参
def get_RT_from_chessboard(img_path,chess_board_x_num,chess_board_y_num,K,chess_board_len):
    '''
    :param img_path: 读取图片路径
    :param chess_board_x_num: 棋盘格x方向格子数
    :param chess_board_y_num: 棋盘格y方向格子数
    :param K: 相机内参
    :param chess_board_len: 单位棋盘格长度,mm
    :return: 相机外参
    '''
    img=cv2.imread(img_path)
    ret, corners = cv2.findChessboardCorners(img, (chess_board_x_num, chess_board_y_num), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    cv2.drawChessboardCorners(img, (chess_board_x_num, chess_board_y_num), corners, ret)
    cv2.imshow("img",img)
    cv2.waitKey(0)

    corner_points=np.zeros((2,corners.shape[0]),dtype=np.float64)
    for i in range(corners.shape[0]):
        corner_points[:,i]=corners[i,0,:]
    # print(corner_points)
    object_points=np.zeros((3,chess_board_x_num*chess_board_y_num),dtype=np.float64)


    flag=0
    for i in range(chess_board_y_num):
        for j in range(chess_board_x_num):
            object_points[:2,flag]=np.array([(11-j-1)*chess_board_len,(8-i-1)*chess_board_len])
            flag+=1

    # retval,rvec,tvec  = cv2.solvePnP(object_points.T,corner_points.T, K, distCoeffs=None)


    # print("object_points.T:")
    # print(object_points.T)
    # print(type(object_points.T))
    # print("-" * 20)
    #
    # print("corner_points.T:")
    # print(corner_points.T)
    # print(type(corner_points.T))
    # print("-" * 20)

    # print("objp:")
    # print(object_points)
    # print(type(object_points))
    # print("*" * 20)
    #
    # print("imgp:")
    # print(corner_points)
    # print(type(corner_points))
    # print("*" * 20)

    # print("K:")
    # print(K)
    # print(type(K))
    # print("*" * 20)
    #
    # print("D_0:")
    # print(dist)
    # print(type(dist))
    # print("*" * 20)

    print("开始根据角点信息与对应该组信息的机械臂返回，通过2D点和3D点求解相机的位姿")
    print("\n")
    retval, rvec, tvec = cv2.solvePnP(object_points.T, corner_points.T, K, distCoeffs=dist)
    # print(rvec.reshape((1,3)))
    # RT=np.column_stack((rvec,tvec))
    RT=np.column_stack(((cv2.Rodrigues(rvec))[0],tvec))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
    # RT=pose(rvec[0,0],rvec[1,0],rvec[2,0],tvec[0,0],tvec[1,0],tvec[2,0])
    # print(RT)

    # print(retval, rvec, tvec)
    # print(RT)
    # print('')
    return RT

def set_setting_file(path):
    try:
        with open(path) as json_file:
            setting_dict_string = json_file.readline()[:-2]
        return json.loads(setting_dict_string)
    except:
        return {}

def read_xyz_Rxyz(good_picture, folder, ele_data_dict={}):
    """用于读取每个图片对应的json文件中，对应的"""
    with open(folder+'/info.json', 'r', encoding='utf-8') as f:
        ele_data_dict = json.load(f)

    return ele_data_dict





folder = r"/home/liusong/temp/calibrate"#棋盘格图片存放文件夹
# files = os.listdir(folder)
# file_num=len(files)
# RT_all=np.zeros((4,4,file_num))

# print(get_RT_from_chessboard('calib/2.bmp', chess_board_x_num, chess_board_y_num, K, chess_board_len))
'''
这个地方很奇怪的特点，有些棋盘格点检测得出来，有些检测不了，可以通过函数get_RT_from_chessboard的运行时间来判断
'''
# good_picture=[2,3,4,5,6,7,8,9]#存放可以检测出棋盘格角点的图片
# good_picture=[1,3,10,11,12]
good_picture=[0,1,2,3,4,5,6,7,8,9,10,11]#存放可以检测出棋盘格角点的图片

ele_data_dict = read_xyz_Rxyz(good_picture,folder)

# print(ele_data_dict)

file_num=len(good_picture)

#计算board to cam 变换矩阵
R_all_chess_to_cam_1=[]
T_all_chess_to_cam_1=[]
for i in good_picture:
    # print(i)
    image_path=folder+'/'+f"color_{str(i)}.png"
    print("开始解析"+ folder+'/'+str(i)+'.bmp'+ "的角点信息")
    RT=get_RT_from_chessboard(image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)

    # RT=np.linalg.inv(RT)

    R_all_chess_to_cam_1.append(RT[:3,:3])
    T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3,1)))
# print(T_all_chess_to_cam.shape)

#计算end to base变换矩阵
R_all_end_to_base_1=[]
T_all_end_to_base_1=[]
# print(sheet_1.iloc[0]['ax'])
for i in good_picture:
    print("开始分析第" + str(i) + "张图片对应的json文件")

    rotation = R.from_quat(ele_data_dict[str(i)]['q']).as_matrix()
    translation = ele_data_dict[str(i)]['t']
    translation = np.array(translation)*1000 #mm单位对齐
    R_all_end_to_base_1.append(rotation)
    T_all_end_to_base_1.append(translation.reshape((3, 1)))


print("基坐标系的旋转矩阵与平移向量求解完毕......")
print("进入手眼标定函数。。。。。。")
# print(R_all_end_to_base_1)
# print(T_all_end_to_base_1)

print("\n")
R,T=cv2.calibrateHandEye(R_all_end_to_base_1,T_all_end_to_base_1,R_all_chess_to_cam_1,T_all_chess_to_cam_1)#手眼标定

print("手眼矩阵分解得到的旋转矩阵")
print(R)
print("\n")


print("手眼矩阵分解得到的平移矩阵")
print(T)

RT=np.column_stack((R,T))
RT = np.row_stack((RT, np.array([0, 0, 0, 1])))#即为cam to end变换矩阵
print("\n")
print('相机相对于末端的变换矩阵为：')
print(RT)

#结果验证，原则上来说，每次结果相差较小
for i in range(len(good_picture)):

    # 得到机械手末端到基座的变换矩阵，通过机械手末端到基座的旋转矩阵与平移向量先按列合并，然后按行合并形成变换矩阵格式
    RT_end_to_base=np.column_stack((R_all_end_to_base_1[i],T_all_end_to_base_1[i]))
    RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
    # print(RT_end_to_base)

    # 标定版相对于相机的齐次矩阵
    RT_chess_to_cam=np.column_stack((R_all_chess_to_cam_1[i],T_all_chess_to_cam_1[i]))
    RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
    # print(RT_chess_to_cam)

    # 手眼标定变换矩阵
    RT_cam_to_end=np.column_stack((R,T))
    RT_cam_to_end=np.row_stack((RT_cam_to_end,np.array([0,0,0,1])))
    # print(RT_cam_to_end)

    # 即为固定的棋盘格相对于机器人基坐标系位姿
    RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam
    RT_chess_to_base=np.linalg.inv(RT_chess_to_base)
    print('第',i,'次')

    # 使用 array2string 进行临时格式化
    formatted_output = np.array2string(
        RT_chess_to_base[:3, :], 
        precision=2, 
        suppress_small=True, 
        formatter={'float': '{:.2f}'.format}
    )
    print(formatted_output)
    print('')


