## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.
## Modified to include Open3D PointCloud Visualization

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# ----------------------------------------------------
# 1. 初始化 RealSense 管道
# ----------------------------------------------------
pipeline = rs.pipeline()
config = rs.config()

ctx = rs.context()
devices = ctx.query_devices()
device_sno = None
for dev in devices:
    # get the name
    dev_name = dev.get_info(rs.camera_info.name)
    if "D435I" not in dev_name:
        device_sno = dev.get_info(rs.camera_info.serial_number)
config.enable_device(device_sno)

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)



device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

# 设置分辨率 (保持与你原代码一致)
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# 获取深度比例
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# 背景裁剪设置
clipping_distance_in_meters = 1
clipping_distance = clipping_distance_in_meters / depth_scale

# 对齐设置
align_to = rs.stream.color
align = rs.align(align_to)

# ----------------------------------------------------
# 2. 初始化 Open3D 可视化
# ----------------------------------------------------
# 创建点云对象
pcd = o3d.geometry.PointCloud()

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='RealSense PointCloud', width=640, height=480)
vis.add_geometry(pcd)

# 获取渲染选项以便调整背景色等
render_opt = vis.get_render_option()
render_opt.background_color = np.array([0.1, 0.1, 0.1]) # 深灰色背景
render_opt.point_size = 2.0  # 点的大小

# 创建用于计算点云的工具类
pc = rs.pointcloud()

print("Press 'Q' in the CV2 window or 'ESC' in the Open3D window to exit.")

try:
    while True:
        # --- A. 获取并对齐帧 ---
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # --- B. 数据处理 (NumPy) ---
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 背景移除 (保留你原有的逻辑)
        grey_color = 153
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        
        # 显示 2D 视图 (原代码逻辑)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        cv2.imshow('Align Example', images)

        # --- C. 生成点云 (核心修改部分) ---
        
        # 1. 将点云映射到彩色图像坐标系 (因为我们要用彩色图的颜色)
        # 注意：这里我们直接使用 aligned_frames，因为它们已经对齐过了
        pc.map_to(color_frame)
        
        # 2. 计算点云坐标 (x,y,z) 和 纹理坐标
        points = pc.calculate(aligned_depth_frame)
        
        # 3. 提取顶点 (N, 3) 和 纹理 (N, 2)
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        # 4. 过滤掉无效点 (深度为0的点会产生 [0,0,0] 或无效坐标)
        # 简单的过滤：去除全零向量
        valid_indices = np.any(vtx != 0, axis=1)
        vtx = vtx[valid_indices]
        tex = tex[valid_indices]

        if len(vtx) > 0:
            # 5. 根据纹理坐标从彩色图中提取颜色
            # 纹理坐标范围是 [0, 1]，需要映射到像素坐标 [0, width], [0, height]
            h, w, _ = color_image.shape
            tex_x = (tex[:, 0] * (w - 1)).astype(int)
            tex_y = (tex[:, 1] * (h - 1)).astype(int)
            
            # 边界检查，防止索引越界
            mask = (tex_x >= 0) & (tex_x < w) & (tex_y >= 0) & (tex_y < h)
            tex_x = tex_x[mask]
            tex_y = tex_y[mask]
            vtx = vtx[mask]

            # 提取颜色 (OpenCV 读取的是 BGR，Open3D 需要 RGB 且归一化到 0-1)
            colors_bgr = color_image[tex_y, tex_x]
            colors_rgb = colors_bgr[:, ::-1].astype(float) / 255.0

            # 6. 更新 Open3D 点云对象
            pcd.points = o3d.utility.Vector3dVector(vtx)
            pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            o3d.visualization.draw_geometries([pcd])
            # 7. 更新可视化窗口
            vis.update_geometry(pcd)
            
            # 处理窗口事件（旋转、缩放、关闭）
            if not vis.poll_events():
                break
            vis.update_renderer()

        # --- D. 退出检查 ---
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break

except KeyboardInterrupt:
    pass

finally:
    # 清理资源
    pipeline.stop()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print("Viewer closed.")