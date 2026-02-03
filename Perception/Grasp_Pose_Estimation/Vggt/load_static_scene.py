import open3d as o3d
import cv2
import os
import numpy as np



path_dir = os.path.abspath(os.path.dirname(__file__))



# 读取PLY文件
ply_path = path_dir+"/scene_20250423_195410_238.ply"
pcd = o3d.io.read_point_cloud(ply_path)
# o3d.visualization.draw_geometries([pcd])
# voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=0.05)

color_grid = np.load(path_dir+"/color_grid.npy")
occupancy = np.load(path_dir+"/occupancy.npy")
origin = np.load(path_dir+"/origin.npy")
voxel_size = np.load(path_dir+"/voxel_size.npy")






# 获取点云数据
points = pcd.points
# 打印前三个点的坐标
for i in range(min(3, len(points))):
    print("Point", i+1, ":", points[i])
