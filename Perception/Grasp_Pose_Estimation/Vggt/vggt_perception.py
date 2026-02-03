import torch
import open3d as o3d
import numpy as np
import cv2
import copy
from depth_vis import depth_to_pointcloud

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import matplotlib.pyplot as plt
device = "cuda" if torch.cuda.is_available() else "cpu"

# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

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
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
def main():
    # model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Load and preprocess example images (replace with your own image paths)
    image_names = ["captured_images/color_20250519_143455.png"] 
    depth_image_path = "captured_images/depth_raw_20250519_143455.npy"
 
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)


    rgb_pcl = predictions["world_points"]
    points = rgb_pcl.squeeze(0).permute(1, 2, 3, 0).contiguous().cpu().numpy().reshape(-1, 3)


    pose_enc = predictions["pose_enc"]
    depth_map = predictions["depth"]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))

    print(extrinsic)

    # 获取颜色信息 - 从原始图像中提取
    original_image = images[0].permute(1, 2, 0).cpu().numpy()
    h, w = original_image.shape[:2]
    colors = original_image.reshape(-1, 3)  # 将图像展平为点云颜色

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 可选：应用统计离群值移除
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # 更新颜色数组以匹配过滤后的点
    colors = colors[ind]

    # 首先根据距离过滤点云，只保留近处的点（桌面）
    print("根据距离过滤点云，只保留近处的点...")
    points_array = np.asarray(pcd.points)
    distances = np.linalg.norm(points_array, axis=1)  # 计算每个点到相机的距离

    # 设置距离阈值，只保留近处的点
    # 这个阈值需要根据实际场景调整，确保桌面在阈值内，地板在阈值外
    distance_threshold = 0.8  # 单位：米，假设桌面距离相机不超过1.5米8
    close_indices = np.where(distances < distance_threshold)[0]
    close_pcd = pcd.select_by_index(close_indices)

    print(f"原始点云点数: {len(points_array)}")
    print(f"过滤后点云点数: {len(close_indices)}")

    # 在过滤后的点云中检测平面
    print("在过滤后的点云中检测桌面平面...")
    plane_model, inliers = close_pcd.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000
    )
    a, b, c, d = plane_model

    # 计算模型预测的平面到相机的距离
    predicted_distance = abs(d) / np.sqrt(a*a + b*b + c*c)
    print(f"模型预测的平面距离: {predicted_distance:.4f} 米")

    
    # 检查文件扩展名，决定如何加载深度图像
    if depth_image_path.endswith('.npy'):
        # 加载NPY格式的深度图像
        depth_image = np.load(depth_image_path)
        print("已加载NPY格式的深度图像")
    else:
        # 加载图像格式的深度图像
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

    # 检查深度图像是否成功加载


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


    # 相机内参
    # K = [924.034423828125, 0.0, 656.5450439453125, 
    #      0.0, 924.034423828125, 360.77032470703125, 
    #      0.0, 0.0, 1.0] #own camera
    K = [910.4097290039062, 0.0, 648.3623046875, 
        0.0, 910.4097290039062, 382.8463134765625, 
        0.0, 0.0, 1.0]
    # 将深度图像转换为点云
    gt_points = depth_to_pointcloud(depth_image, K)

    print(f"有效点云数量: {points.shape[0]}")
    print(f"点云深度范围: {np.min(points[:, 2])} - {np.max(points[:, 2])} 米")

    # 创建Open3D点云对象
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_points)

    gt_pcd, _ = gt_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 检测真值点云中的平面
    print("正在检测真值点云中的平面...")
    gt_plane_model, gt_inliers = gt_pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    a_gt, b_gt, c_gt, d_gt = gt_plane_model
    gt_normal = np.array([a_gt, b_gt, c_gt])
    gt_normal = gt_normal / np.linalg.norm(gt_normal)


    distance = calculate_plane_distance(gt_plane_model)

    print(f"检测到的平面方程: {a_gt:.4f}x + {b_gt:.4f}y + {c_gt:.4f}z + {d_gt:.4f} = 0")
    print(f"桌面平面到相机的距离: {distance:.4f} 米")
        # 计算缩放比例

    # 计算缩放比例
    scale_factor =  distance/ predicted_distance  # 实际距离/预测距离
    print(f"缩放比例: {scale_factor:.4f}")

    # 对点云进行缩放 - 只缩放近处的点（桌面部分）
    # 注意：这里我们只处理已经过滤后的近处点云
    close_points = np.asarray(close_pcd.points)
    scaled_close_points = close_points * scale_factor
    scaled_close_pcd = o3d.geometry.PointCloud()
    scaled_close_pcd.points = o3d.utility.Vector3dVector(scaled_close_points)

    # 获取对应的颜色
    close_colors = colors[close_indices]
    scaled_close_pcd.colors = o3d.utility.Vector3dVector(close_colors)

    # 在缩放后的近处点云中检测平面
    scaled_plane_model, scaled_inliers = scaled_close_pcd.segment_plane(
        distance_threshold=0.01 * scale_factor,  # 相应缩放阈值
        ransac_n=3,
        num_iterations=1000
    )
    a_scaled, b_scaled, c_scaled, d_scaled = scaled_plane_model

    # 计算缩放后平面到相机的距离
    scaled_distance = abs(d_scaled) / np.sqrt(a_scaled*a_scaled + b_scaled*b_scaled + c_scaled*c_scaled)

    # 计算平面法向量
    pred_normal = np.array([a_scaled, b_scaled, c_scaled])
    pred_normal = pred_normal / np.linalg.norm(pred_normal)

    # 计算平面内点的平均深度
    inlier_points = np.asarray(scaled_close_pcd.points)[scaled_inliers]
    avg_depth = np.mean(inlier_points[:, 2])

    print(f"缩放后的平面方程: {a_scaled:.4f}x + {b_scaled:.4f}y + {c_scaled:.4f}z + {d_scaled:.4f} = 0")
    print(f"缩放后平面到相机的距离: {scaled_distance:.4f} 米")
    print(f"平面法向量: [{pred_normal[0]:.4f}, {pred_normal[1]:.4f}, {pred_normal[2]:.4f}]")
    print(f"平面内点的平均深度: {avg_depth:.4f} 米")

    # 可视化点云和检测到的平面
    # 提取平面内点
    inlier_cloud = scaled_close_pcd.select_by_index(scaled_inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 红色表示平面点

    # 提取非平面点
    outlier_cloud = scaled_close_pcd.select_by_index(scaled_inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 0.5, 0.5])  # 青色表示非平面点

    # 创建坐标系可视化
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5 * scale_factor, origin=[0, 0, 0])

    # 可视化平面分割结果
    print("显示平面分割结果...")
    #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, coordinate_frame])

    # 可视化带颜色的完整点云
    print("显示带颜色的完整点云...")
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(scaled_close_points)
    colored_pcd.colors = o3d.utility.Vector3dVector(close_colors)
    #o3d.visualization.draw_geometries([colored_pcd, coordinate_frame])

    # # 加载真值深度图并重建点云
    # depth_image_path = "/home/yixuan/vggt/captured_images/depth_raw_20250519_143455.npy"

    # # 检查文件扩展名，决定如何加载深度图像
    # if depth_image_path.endswith('.npy'):
    #     # 加载NPY格式的深度图像
    #     depth_image = np.load(depth_image_path) 
    # else:
    #     # 加载图像格式的深度图像
    #     depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

    # # 检查深度图像是否成功加载
    # if depth_image is None:
    #     print("无法加载深度图像，请检查路径是否正确")
    # else:
    #     print(f"深度图像尺寸: {depth_image.shape}")
    #     print(f"深度图像类型: {depth_image.dtype}")
    #     print(f"深度值范围: {np.min(depth_image[depth_image > 0])} - {np.max(depth_image)}")
        
    #     # 根据深度图像格式进行适当的转换
    #     if depth_image.dtype == np.uint16:
    #         # 对于RealSense、Kinect等相机，通常是毫米单位
    #         depth_scale = 0.001  # 毫米转米
    #         depth_image = depth_image.astype(np.float32) * depth_scale
        
    #     # 相机内参 - 使用RealSense相机的内参
    #     # K = [924.034423828125, 0.0, 656.5450439453125, 
    #     #      0.0, 924.034423828125, 360.77032470703125, 
    #     #      0.0, 0.0, 1.0]
    #     K = [910.4097290039062, 0.0, 648.3623046875, 
    #         0.0, 910.4097290039062, 382.8463134765625, 
    #         0.0, 0.0, 1.0]
        
    #     # 将深度图像转换为点云
    #     gt_points = depth_to_pointcloud(depth_image, K)
        
    #     print(f"真值点云数量: {gt_points.shape[0]}")
        
    #     # 创建真值点云对象
    #     gt_pcd = o3d.geometry.PointCloud()
    #     gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
        
    #     # 可选：应用统计离群值移除
    #     gt_pcd, _ = gt_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
    #     # 检测真值点云中的平面
    #     print("正在检测真值点云中的平面...")
    #     gt_plane_model, gt_inliers = gt_pcd.segment_plane(
    #         distance_threshold=0.01,
    #         ransac_n=3,
    #         num_iterations=1000
    #     )
    #     a_gt, b_gt, c_gt, d_gt = gt_plane_model
        
    #     # 计算真值平面法向量
    #     gt_normal = np.array([a_gt, b_gt, c_gt])
    #     gt_normal = gt_normal / np.linalg.norm(gt_normal)
        
    #     print(f"真值平面方程: {a_gt:.4f}x + {b_gt:.4f}y + {c_gt:.4f}z + {d_gt:.4f} = 0")
    #     print(f"真值平面法向量: [{gt_normal[0]:.4f}, {gt_normal[1]:.4f}, {gt_normal[2]:.4f}]")
        
    # 使用平面对齐方法
    print("使用平面对齐方法...")

    # 计算旋转矩阵，将预测平面的法向量旋转到真值平面的法向量
    rotation_axis = np.cross(pred_normal, gt_normal)
    if np.linalg.norm(rotation_axis) < 1e-6:
        # 法向量几乎平行
        rotation_matrix = np.eye(3)
        print("平面法向量几乎平行，无需旋转")
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        cos_angle = np.dot(pred_normal, gt_normal)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        print(f"平面法向量夹角: {np.degrees(angle):.2f} 度")
        
        # 使用罗德里格斯公式计算旋转矩阵
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

        # 构建完整的变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        
        # 计算平移部分（使平面原点对齐）
        pred_d = d_scaled
        gt_d = d_gt
        
        # 计算平面上的点（原点到平面的投影点）
        pred_origin = -pred_d * pred_normal
        gt_origin = -gt_d * gt_normal
        
        # 旋转后的预测平面原点
        rotated_pred_origin = np.dot(rotation_matrix, pred_origin)
        translation = gt_origin - rotated_pred_origin
        
        transform[:3, 3] = translation
        
        print(f"变换矩阵:\n{transform}")
        
        # 应用变换 - 只对近处的点云（桌面部分）应用变换
        aligned_close_pcd = copy.deepcopy(scaled_close_pcd)
        aligned_close_pcd.transform(transform)
        
        # 可视化对齐前的结果
        print("显示对齐前的结果...")
        gt_pcd.paint_uniform_color([1.0, 0, 0])  # 红色表示真值点云
        scaled_close_pcd.paint_uniform_color([0, 0, 1.0])  # 蓝色表示重建点云
        #o3d.visualization.draw_geometries([scaled_close_pcd, gt_pcd, coordinate_frame])
        
        # 可视化对齐后的结果
        print("显示平面对齐后的结果...")
        aligned_close_pcd.paint_uniform_color([0, 0, 1.0])  # 蓝色表示对齐后的重建点云
        #o3d.visualization.draw_geometries([aligned_close_pcd, gt_pcd, coordinate_frame])
        
        # 可选：使用ICP进一步精细对齐
        print("使用ICP进行精细对齐...")
        # 首先过滤真值点云，只保留近处的点
        gt_points_array = np.asarray(gt_pcd.points)
        gt_distances = np.linalg.norm(gt_points_array, axis=1)
        gt_close_indices = np.where(gt_distances < distance_threshold)[0]
        gt_close_pcd = gt_pcd.select_by_index(gt_close_indices)

        print(f"真值点云点数: {len(gt_points_array)}")
        print(f"过滤后真值点云点数: {len(gt_close_indices)}")

        # 使用过滤后的点云进行ICP配准
        reg_p2p = o3d.pipelines.registration.registration_icp(
            aligned_close_pcd, gt_close_pcd, 0.01,  # 最大对应距离
            np.identity(4),  # 初始变换矩阵
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300)
        )

        print(f"ICP配准结果: {reg_p2p}")
        print(f"ICP变换矩阵:\n{reg_p2p.transformation}")

        # 应用ICP变换
        final_close_pcd = copy.deepcopy(aligned_close_pcd)
        final_close_pcd.transform(reg_p2p.transformation)
        
        # 可视化最终对齐结果
        print("显示最终对齐结果...")
        final_close_pcd.paint_uniform_color([0, 0, 1.0])  # 蓝色表示最终对齐的重建点云
        o3d.visualization.draw_geometries([final_close_pcd, gt_close_pcd, coordinate_frame])

        # 在过滤后的对齐点云中检测平面
        print("在过滤后的对齐点云中检测桌面平面...")
        final_plane_model, final_inliers = final_close_pcd.segment_plane(
            distance_threshold=0.01,
            ransac_n=3,
            num_iterations=1000
        )
        a_final_model, b_final_model, c_final_model, d_final_model = final_plane_model

        # 计算原始平面模型法向量的模长，用于后续d值的归一化
        norm_of_plane_model_abc = np.linalg.norm(np.array([a_final_model, b_final_model, c_final_model]))
        if norm_of_plane_model_abc < 1e-6: # 避免除以零
            norm_of_plane_model_abc = 1.0 # 对于有效平面，此情况不应发生

        # 计算桌面平面法向量 (这段逻辑保持不变，确保final_normal和d_final与您之前的打印一致)
        final_normal = np.array([a_final_model, b_final_model, c_final_model])
        if np.linalg.norm(final_normal) > 1e-6: # 避免在归一化时除以零
            final_normal = final_normal / np.linalg.norm(final_normal)
        else: # 简易处理退化情况
            final_normal = np.array([0.0,0.0,1.0]) # 默认法向量
        
        # 确保法向量朝上（假设y轴负方向朝上）
        # 如果y分量为正，则翻转法向量
        # 注意：这里的 a_final, b_final, c_final, d_final 是为了打印，它们是原始尺度翻转后的值
        a_final, b_final, c_final, d_final = a_final_model, b_final_model, c_final_model, d_final_model
        if final_normal[1] > 0:  
            final_normal = -final_normal
            # d_final 也需要根据法向量的翻转进行调整，以保持平面方程 ax+by+cz+d=0 的一致性
            # （如果使用原始尺度的a,b,c,d进行打印和后续计算）
            a_final, b_final, c_final, d_final = -a_final_model, -b_final_model, -c_final_model, -d_final_model
        
        print(f"对齐后桌面平面方程: {a_final:.4f}x + {b_final:.4f}y + {c_final:.4f}z + {d_final:.4f} = 0")
        print(f"对齐后桌面法向量: [{final_normal[0]:.4f}, {final_normal[1]:.4f}, {final_normal[2]:.4f}]")

        
        # 提取桌面平面点
        table_cloud = final_close_pcd.select_by_index(final_inliers)
        table_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色表示桌面点
        
        # 提取非桌面点（可能是桌面上的物体）
        non_table_cloud = final_close_pcd.select_by_index(final_inliers, invert=True)
        non_table_cloud.paint_uniform_color([1.0, 0.0, 1.0])  # 紫色表示非桌面点
        
        # 创建表示法向量的箭头
        # 计算桌面平面的中心点
        table_points = np.asarray(table_cloud.points)
        table_center = np.mean(table_points, axis=0)
        
        # 创建一个箭头来可视化法向量
        arrow_length = 0.2  # 箭头长度
        cylinder_radius = 0.01  # 箭头柱体半径
        cone_radius = 0.02  # 箭头锥体半径
        
        # 创建箭头几何体
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=cylinder_radius,
            cone_radius=cone_radius,
            cylinder_height=arrow_length * 0.7,
            cone_height=arrow_length * 0.3,
            resolution=20
        )
        
        # 计算旋转矩阵，使箭头指向法向量方向
        # 默认箭头指向(0,0,1)，需要旋转到法向量方向
        z_axis = np.array([0, 0, 1])
        rotation_axis = np.cross(z_axis, final_normal)
        
        if np.linalg.norm(rotation_axis) < 1e-6:
            # 如果法向量与z轴平行
            if final_normal[2] > 0:
                # 如果法向量与z轴同向，不需要旋转
                rotation_matrix = np.eye(3)
            else:
                # 如果法向量与z轴反向，旋转180度
                rotation_matrix = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ])
        else:
            # 计算旋转角度和旋转轴
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            cos_angle = np.dot(z_axis, final_normal)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            # 使用罗德里格斯公式计算旋转矩阵
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        # 构建完整的变换矩阵
        transform_arrow = np.eye(4)
        transform_arrow[:3, :3] = rotation_matrix
        transform_arrow[:3, 3] = table_center
        
        # 应用变换
        arrow.transform(transform_arrow)
        arrow.paint_uniform_color([1.0, 0.0, 0.0])  # 红色箭头
        
        # 可视化桌面平面、非桌面点和法向量箭头
        print("显示桌面平面和法向量...")
        #o3d.visualization.draw_geometries([table_cloud, non_table_cloud, arrow, coordinate_frame])

        # 提取桌面平面点
        table_cloud = final_close_pcd.select_by_index(final_inliers)
        table_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # 绿色表示桌面点
        
        # 提取非桌面点（可能是桌面上的物体）
        non_table_points_indices = list(set(range(len(np.asarray(final_close_pcd.points)))) - set(final_inliers))
        non_table_points = np.asarray(final_close_pcd.points)[non_table_points_indices]
        
        # 计算点到平面的有符号距离
        # 平面方程: ax + by + cz + d = 0
        # 点到平面的有符号距离: (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2)
        # 由于法向量已归一化，分母为1
        distances = np.abs(a_final * non_table_points[:, 0] + 
                        b_final * non_table_points[:, 1] + 
                        c_final * non_table_points[:, 2] + 
                        d_final)
        
        # 计算点在法向量方向上的投影
        # 如果点在法向量方向上（桌面上方），则投影为正；否则为负
        projections = (non_table_points[:, 0] * final_normal[0] + 
                    non_table_points[:, 1] * final_normal[1] + 
                    non_table_points[:, 2] * final_normal[2])
        
        # 设置高度阈值（单位：米）
        height_threshold = 0.01  # 1厘米，可以根据需要调整
        
        # 筛选出高于桌面阈值的点
        # 条件1：点到平面的距离大于阈值
        # 条件2：点在法向量方向上（考虑到Y轴负方向朝上，法向量已经调整为指向桌面上方）
        above_mask = (distances > height_threshold) & (np.sign(projections) == np.sign(-d_final))
        above_indices = np.where(above_mask)[0]
        
        # 创建桌面上物体的点云
        if len(above_indices) > 0:
            # 从非桌面点中选择高于桌面的点
            objects_indices = [non_table_points_indices[i] for i in above_indices]
            objects_cloud = final_close_pcd.select_by_index(objects_indices)
            objects_cloud.paint_uniform_color([1.0, 0.0, 1.0])  # 紫色表示桌面上的物体
            
            # 其他非桌面点（既不是桌面也不是桌面上物体的点）
            other_indices = [non_table_points_indices[i] for i in range(len(non_table_points)) if i not in above_indices]
            if other_indices:
                other_cloud = final_close_pcd.select_by_index(other_indices)
                other_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色表示其他点
            else:
                other_cloud = None
            
            print(f"检测到 {len(above_indices)} 个点高于桌面 {height_threshold} 米")
            
            # 可视化桌面平面、桌面上物体、其他点和法向量箭头
            print("显示桌面平面和桌面上的物体...")
            geometries_to_show = [table_cloud, objects_cloud, arrow, coordinate_frame]
            if other_cloud is not None:
                geometries_to_show.append(other_cloud)
            #o3d.visualization.draw_geometries(geometries_to_show)
        else:
            print("没有检测到高于桌面的物体点")
            # 如果没有检测到物体，仍然显示桌面和法向量
        # non_table_cloud = final_pcd.select_by_index(non_table_points_indices)
        # non_table_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色表示非桌面点
        # o3d.visualization.draw_geometries([table_cloud, non_table_cloud, arrow, coordinate_frame])

            
        # 对桌面上的物体进行聚类，找出各个独立物体
        print("对桌面上的物体进行聚类...")
        objects_points = np.asarray(objects_cloud.points)

        # 将点投影到桌面平面上（忽略高度信息）
        # 创建一个从3D空间到桌面平面的投影矩阵
        # 首先找到与桌面法向量垂直的两个基向量
        v1 = np.array([1.0, 0.0, 0.0])  # 尝试使用x轴作为第一个基向量
        if abs(np.dot(v1, final_normal)) > 0.9:  # 如果x轴与法向量接近平行
            v1 = np.array([0.0, 1.0, 0.0])  # 使用y轴
        v1 = v1 - np.dot(v1, final_normal) * final_normal  # 使v1垂直于法向量
        v1 = v1 / np.linalg.norm(v1)  # 归一化

        v2 = np.cross(final_normal, v1)  # 第二个基向量，垂直于法向量和v1
        v2 = v2 / np.linalg.norm(v2)  # 归一化

        # 投影点到平面上（仅保留平面内的坐标）
        projected_points = np.zeros((len(objects_points), 2))
        for i, point in enumerate(objects_points):
            # 计算点在v1和v2方向上的投影
            projected_points[i, 0] = np.dot(point, v1)
            projected_points[i, 1] = np.dot(point, v2)

        # 使用DBSCAN在2D投影平面上进行聚类
        eps = 0.007  # 聚类半径，可根据物体大小和密度调整
        min_points = 10  # 一个聚类的最小点数，可根据点云密度调整
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(projected_points)
        labels = clustering.labels_

        # 获取聚类数量（不包括噪声点，噪声点标签为-1）
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        print(f"检测到 {n_clusters} 个物体")

        # 为每个聚类创建一个点云，并拟合圆柱体
        cylinders = []
        cylinder_meshes = []
        cylinder_point_clouds = []  # 存储圆柱体点云

        # 设置高度阈值，过滤掉高度较低的圆柱体
        min_height_threshold = 0.04  # 2厘米，可以根据需要调整
        max_radius_threshold = 0.05  # 5厘米
        # 为每个聚类分配不同的颜色
        # 确保 n_clusters > 0 以避免 plt.get_cmap 报错
        if n_clusters > 0:
            colors_map = plt.get_cmap("tab20")(np.linspace(0, 1, n_clusters))
        else:
            colors_map = [] # 如果没有聚类，则为空
        
        color_index = 0
        center_list = []
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
            
            # 提取当前聚类的点
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < min_points : # 如果聚类点数过少，也跳过 (可选)
                continue
            cluster_points = objects_points[cluster_indices]
            
            # 创建当前聚类的点云
            cluster_cloud = o3d.geometry.PointCloud()
            cluster_cloud.points = o3d.utility.Vector3dVector(cluster_points)
            if color_index < len(colors_map):
                cluster_cloud.paint_uniform_color(colors_map[color_index][:3])
            else: # Fallback color if not enough colors in map (should not happen if n_clusters is correct)
                cluster_cloud.paint_uniform_color([0.5,0.5,0.5])

            axis = final_normal # 桌面法向量作为圆柱体轴向

            # 计算点在法向量方向上的投影（高度）
            projections_on_axis = np.dot(cluster_points, axis)
            min_coord_along_axis = np.min(projections_on_axis)
            max_coord_along_axis = np.max(projections_on_axis)
            height = max_coord_along_axis - min_coord_along_axis
            if height < 0.001: # 避免高度过小或为零的圆柱体
                height = 0.001
            

            # radius_idx = np.searchsorted(cumulative_weights, 0.95)
            # if radius_idx >= len(sorted_distances):
            #     radius_idx = len(sorted_distances) - 1
            # radius = sorted_distances[radius_idx]
                    # 找到95%权重对应的距离作为半径

            # 计算物体的高度分布，并根据高度给点分配权重
            # 高度归一化到[0,1]区间
            normalized_heights = (projections_on_axis - min_coord_along_axis) / height
            
            # 创建权重，使上部点的权重更高
            # 使用指数函数：weight = exp(alpha * normalized_height)
            # alpha控制权重增长的速率，值越大，上部点相对于下部点的权重越高
            alpha = 10 # 可以调整这个参数
            weights = np.exp(alpha * normalized_heights)
            
            # 使用加权平均计算物体的质心，更偏向上部点
            weighted_centroid = np.average(cluster_points, axis=0, weights=weights)
            
            # 计算点在平面上的投影，用于估计圆柱体半径
            # 使用加权方式计算投影中心
            projected_points_2d = np.array([[np.dot(p, v1), np.dot(p, v2)] for p in cluster_points])
            projected_centroid_2d = np.array([np.dot(weighted_centroid, v1), np.dot(weighted_centroid, v2)])
            
            # 计算投影点到投影中心的距离
            dist_to_centroid_proj = np.linalg.norm(projected_points_2d - projected_centroid_2d, axis=1)
            
            # 使用加权百分位数来估计半径，同样更偏向上部点
            # 由于numpy.percentile不直接支持权重，我们使用排序和累积权重的方法
            sorted_indices = np.argsort(dist_to_centroid_proj)
            sorted_distances = dist_to_centroid_proj[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cumulative_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)
            
            radius_idx = np.searchsorted(cumulative_weights, 0.95)
            if radius_idx >= len(sorted_distances):
                radius_idx = len(sorted_distances) - 1
            radius = sorted_distances[radius_idx]
            
            if radius < 0.001:  # 避免半径过小
                radius = 0.001
                
            # 跳过高度低于阈值的物体
            if height < min_height_threshold:
                print(f"跳过高度为 {height:.4f}m 的物体（低于阈值 {min_height_threshold}m）")
                continue
            if radius > max_radius_threshold:
                print(f"跳过半径为 {radius:.4f}m 的物体（大于阈值 {max_radius_threshold}m）")
                continue
            # 计算圆柱体底面中心在桌面平面上的投影点
            # 平面方程: dot(X, axis) + d_consistent = 0
            d_consistent_for_normalized_normal = d_final / norm_of_plane_model_abc
            
            # 使用加权质心计算到桌面的距离
            dist_centroid_to_table_plane = np.dot(weighted_centroid, axis) + d_consistent_for_normalized_normal
            cylinder_base_center = weighted_centroid - dist_centroid_to_table_plane * axis
            
            # 直接增加圆柱体高度，确保底面在桌面上
            # 简单地增加一个固定值，避免复杂计算可能导致的错误
            extra_height = 0.01  # 额外增加2厘米高度
            adjusted_height = height + extra_height
            
            # 计算圆柱体的中心位置（底面中心 + 高度/2 * 轴向）
            cylinder_center = cylinder_base_center + axis * (adjusted_height / 2.0)
            
            # 创建圆柱体网格
            cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(
                radius=radius,
                height=adjusted_height,  # 使用调整后的高度
                resolution=20,
                split=4,
                create_uv_map=False
            )
            
            # 旋转圆柱体，使其轴向与计算的轴向一致
            z_axis = np.array([0, 0, 1])
            temp_rotation_axis = np.cross(z_axis, axis)
            if np.linalg.norm(temp_rotation_axis) < 1e-6:
                if np.dot(z_axis, axis) > 0.999:
                    rotation_matrix = np.eye(3)
                else:
                    # 处理轴向与z轴反向的情况
                    angle_rot = np.pi
                    K_rot = np.array([[0,0,0],[0,0,-1],[0,1,0]])
                    rotation_matrix = np.eye(3) + np.sin(angle_rot)*K_rot + (1-np.cos(angle_rot)) * (K_rot @ K_rot)
            else:
                angle = np.arccos(np.clip(np.dot(z_axis, axis), -1.0, 1.0))
                temp_rotation_axis_normalized = temp_rotation_axis / np.linalg.norm(temp_rotation_axis)
                R_obj = o3d.geometry.get_rotation_matrix_from_axis_angle(temp_rotation_axis_normalized * angle)
                rotation_matrix = R_obj
                
            # 构建完整的变换矩阵
            transform_cylinder = np.eye(4)
            transform_cylinder[:3, :3] = rotation_matrix
            transform_cylinder[:3, 3] = cylinder_center
            
            # 应用变换
            cylinder_mesh.transform(transform_cylinder)
            
            # 为圆柱体设置颜色
            if color_index < len(colors_map):
                cylinder_color = colors_map[color_index][:3]
            else:
                cylinder_color = [0.5,0.5,0.5]
            cylinder_mesh.paint_uniform_color(cylinder_color)
            
            # 将圆柱体网格转换为点云
            # 首先对网格进行采样
            cylinder_point_cloud = cylinder_mesh.sample_points_uniformly(number_of_points=5000)
            cylinder_point_cloud.paint_uniform_color(cylinder_color)
            
            # 保存圆柱体参数、网格和点云
            cylinders.append({
                'center': cylinder_center,
                'axis': axis,
                'radius': radius,
                'height': adjusted_height,
                'bottom_center': cylinder_base_center
            })
            cylinder_meshes.append(cylinder_mesh)
            cylinder_point_clouds.append(cylinder_point_cloud)


            cylinder_3d_center = cylinder_base_center.copy()
            cylinder_3d_center[2] += adjusted_height / 2  # 在Z轴上加上高度的一半
            
            print(f"物体 {color_index+1}: 半径 = {radius:.4f}m, 高度 = {adjusted_height:.4f}m, 底面中心: {cylinder_base_center}, 3D中心: {cylinder_3d_center}")
            color_index += 1
            center_list.append(cylinder_center)
        # 可视化聚类结果
        # print("显示物体聚类结果...")
        # cluster_clouds = []
        # color_index = 0
        # for label in unique_labels:
        #     if label == -1:  # 噪声点
        #         noise_indices = np.where(labels == -1)[0]
        #         if len(noise_indices) > 0:
        #             noise_cloud = objects_cloud.select_by_index(noise_indices)
        #             noise_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色表示噪声点
        #             cluster_clouds.append(noise_cloud)
        #     else:
        #         cluster_indices = np.where(labels == label)[0]
        #         cluster_cloud = objects_cloud.select_by_index(cluster_indices)
        #         if color_index < len(colors_map):
        #             cluster_cloud.paint_uniform_color(colors_map[color_index][:3])
        #             color_index += 1
        #         else:
        #             cluster_cloud.paint_uniform_color([0.5, 0.5, 0.5])
        #         cluster_clouds.append(cluster_cloud)



        # 可视化拟合的圆柱体点云
        print("显示拟合的圆柱体点云...")
        o3d.visualization.draw_geometries(cylinder_point_clouds + [table_cloud, coordinate_frame])
        # 可视化原始点云和拟合的圆柱体点云
        print("显示原始点云和拟合的圆柱体点云...")
        o3d.visualization.draw_geometries([objects_cloud, table_cloud, coordinate_frame] + cylinder_point_clouds)
        
        print("显示真值点云和圆柱体点云...")
        o3d.visualization.draw_geometries([gt_pcd, table_cloud, coordinate_frame] + cylinder_point_clouds)

        print("显示拟合的圆柱体点云...")
        o3d.visualization.draw_geometries(cylinder_point_clouds + [table_cloud, coordinate_frame])
        
        # 创建圆柱体中心点的可视化
        centers_pcd = o3d.geometry.PointCloud()
        centers_pcd.points = o3d.utility.Vector3dVector(np.array(center_list))
        centers_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 红色表示中心点
        
        # 为每个中心点创建一个小球体，使其更容易看到
        center_spheres = []
        for center in center_list:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 1厘米半径的球体
            sphere.translate(center)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
            center_spheres.append(sphere)
        
        # 可视化原始点云和拟合的圆柱体点云，以及中心点
        print("显示原始点云、拟合的圆柱体点云和中心点...")
        o3d.visualization.draw_geometries([objects_cloud, table_cloud, coordinate_frame] + cylinder_point_clouds + center_spheres)
        
        print("显示真值点云、圆柱体点云和中心点...")
        o3d.visualization.draw_geometries([gt_pcd, table_cloud, coordinate_frame] + cylinder_point_clouds + center_spheres)
        #print(center_list)
        print(center_list)







        return center_list








if __name__ == "__main__":
    main()