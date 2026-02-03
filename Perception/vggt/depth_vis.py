import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import open3d as o3d

def depth_to_pointcloud(depth_image, K):
    """
    将深度图像转换为点云
    
    参数:
        depth_image: 深度图像 (H, W)
        K: 相机内参矩阵 [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    
    返回:
        points: 点云坐标 (N, 3)
    """
    # 提取相机内参
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]
    
    # 创建像素坐标网格
    height, width = depth_image.shape
    v, u = np.mgrid[0:height, 0:width]
    
    # 过滤无效深度值
    valid_mask = (depth_image > 0) & (depth_image < 10.0)  # 设置合理的深度范围，例如0-10米
    
    # 只处理有效深度值
    z = depth_image[valid_mask]
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    
    # 转换为相机坐标系
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy
    
    # 创建点云
    points = np.stack((x, y, z), axis=-1)
    
    return points

def detect_table_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    使用RANSAC检测桌面平面
    
    参数:
        pcd: Open3D点云对象
        distance_threshold: RANSAC距离阈值
        ransac_n: 用于估计平面的点数
        num_iterations: RANSAC迭代次数
    
    返回:
        plane_model: 平面模型参数 [a, b, c, d] 对应于 ax + by + cz + d = 0
        inliers: 平面内点索引
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return plane_model, inliers

def calculate_plane_distance(plane_model):
    """
    计算平面到相机原点的距离
    
    参数:
        plane_model: 平面模型参数 [a, b, c, d]
    
    返回:
        distance: 平面到相机原点的距离
    """
    a, b, c, d = plane_model
    # 平面到原点的距离公式: |d| / sqrt(a^2 + b^2 + c^2)
    distance = abs(d) / np.sqrt(a*a + b*b + c*c)
    return distance

def visualize_pointcloud_with_plane(points, plane_model, inliers):
    """
    可视化点云和检测到的平面
    
    参数:
        points: 点云坐标 (N, 3)
        plane_model: 平面模型参数 [a, b, c, d]
        inliers: 平面内点索引
    """
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 提取平面内点
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 红色表示平面点
    
    # 提取非平面点
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 0.5, 0.5])  # 青色表示非平面点
    
    # 创建坐标系可视化
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0])
    
    # 可视化
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, coordinate_frame])

def main():
    # 加载深度图像
    # 请替换为你的深度图像路径
    depth_image_path = "/home/yan/BestMan_Chemistry_Test_SONG/Perception/vggt/captured_images/depth_16bit_20250522_091536.png"
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    # 检查深度图像是否成功加载
    if depth_image is None:
        print("无法加载深度图像，请检查路径是否正确")
        return
    
    print(f"深度图像尺寸: {depth_image.shape}")
    print(f"深度图像类型: {depth_image.dtype}")
    print(f"深度值范围: {np.min(depth_image[depth_image > 0])} - {np.max(depth_image)}")
    
    # 根据深度图像格式进行适当的转换
    if depth_image.dtype == np.uint16:
        # 对于RealSense、Kinect等相机，通常是毫米单位
        depth_scale = 0.001  # 毫米转米
        depth_image = depth_image.astype(np.float32) * depth_scale
    elif depth_image.dtype == np.float32 or depth_image.dtype == np.float64:
        # 如果已经是浮点数，可能已经是米单位
        pass
    else:
        print(f"未知的深度图像格式: {depth_image.dtype}")
        return
    
    # 相机内参
    K = [910.75244140625, 0.0, 648.3623046875, 
         0.0, 910.4097290039062, 382.8463134765625, 
         0.0, 0.0, 1.0]
    
    # 将深度图像转换为点云
    points = depth_to_pointcloud(depth_image, K)
    
    print(f"有效点云数量: {points.shape[0]}")
    print(f"点云深度范围: {np.min(points[:, 2])} - {np.max(points[:, 2])} 米")
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 可选：应用统计离群值移除
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # 检测桌面平面
    print("正在检测桌面平面...")
    plane_model, inliers = detect_table_plane(pcd, distance_threshold=0.02)
    a, b, c, d = plane_model
    
    # 计算平面到相机的距离
    distance = calculate_plane_distance(plane_model)
    
    print(f"检测到的平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    print(f"桌面平面到相机的距离: {distance:.4f} 米")
    
    # 计算平面法向量
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)
    print(f"平面法向量: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
    
    # 计算平面内点的平均深度
    inlier_points = np.asarray(pcd.points)[inliers]
    avg_depth = np.mean(inlier_points[:, 2])
    print(f"平面内点的平均深度: {avg_depth:.4f} 米")
    
    # 可视化点云和检测到的平面
    visualize_pointcloud_with_plane(np.asarray(pcd.points), plane_model, inliers)

if __name__ == "__main__":
    main()
