import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime

def setup_realsense():
    """设置并配置RealSense相机，分辨率为1280*720"""
    # 创建管道
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 获取设备信息
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    
    # 检查是否有彩色传感器
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("未找到RGB相机")
        exit(0)
    
    # 配置流 - 使用1280*720分辨率
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    # 启动管道
    profile = pipeline.start(config)
    
    # 获取深度传感器
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"深度比例因子: {depth_scale}")
    
    # 创建对齐对象（将深度帧对齐到彩色帧）
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # 获取相机内参
    color_profile = profile.get_stream(rs.stream.color)
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    depth_profile = profile.get_stream(rs.stream.depth)
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    
    print(f"彩色相机内参: fx={color_intrinsics.fx}, fy={color_intrinsics.fy}, "
          f"ppx={color_intrinsics.ppx}, ppy={color_intrinsics.ppy}")
    print(f"深度相机内参: fx={depth_intrinsics.fx}, fy={depth_intrinsics.fy}, "
          f"ppx={depth_intrinsics.ppx}, ppy={depth_intrinsics.ppy}")
    
    return pipeline, align, depth_scale, color_intrinsics

def capture_images(pipeline, align, depth_scale, output_dir="/home/yan/BestMan_Chemistry_Test_SONG/Perception/vggt/captured_images"):
    """捕获并保存深度图和RGB图"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 等待相机稳定
    print("等待相机稳定...")
    for i in range(30):
        pipeline.wait_for_frames()
    
    try:
        while True:
            # 等待一组连贯的帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到彩色帧
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # 转换为numpy数组
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # 创建可视化的深度图
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # 显示图像
            images = np.hstack((color_image, depth_colormap))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            
            # 按键控制
            key = cv2.waitKey(1)
            
            # 按ESC退出
            if key == 27:
                break
            # 按空格键保存图像
            elif key == 32:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                color_path = os.path.join(output_dir, f"color_{timestamp}.png")
                depth_path = os.path.join(output_dir, f"depth_{timestamp}.png")
                depth_raw_path = os.path.join(output_dir, f"depth_raw_{timestamp}.npy")
                
                # 保存彩色图像
                cv2.imwrite(color_path, color_image)
                
                # 保存深度图像（可视化版本）
                cv2.imwrite(depth_path, depth_colormap)
                
                # 保存原始深度数据（以米为单位）
                depth_meters = depth_image.astype(np.float32) * depth_scale
                np.save(depth_raw_path, depth_meters)
                
                # 同时保存16位PNG格式的深度图
                depth_mm = (depth_image).astype(np.uint16)  # 以毫米为单位
                depth_16bit_path = os.path.join(output_dir, f"depth_16bit_{timestamp}.png")
                cv2.imwrite(depth_16bit_path, depth_mm)
                
                print(f"已保存图像: {timestamp}")
                print(f"  彩色图像: {color_path}")
                print(f"  深度图像(可视化): {depth_path}")
                print(f"  深度图像(16位): {depth_16bit_path}")
                print(f"  原始深度数据: {depth_raw_path}")
                
                # 打印深度信息
                valid_depth = depth_meters[depth_meters > 0]
                if len(valid_depth) > 0:
                    print(f"  深度范围: {np.min(valid_depth):.3f}m - {np.max(valid_depth):.3f}m")
                    print(f"  平均深度: {np.mean(valid_depth):.3f}m")
                
    finally:
        # 停止流
        pipeline.stop()
        cv2.destroyAllWindows()

def save_intrinsics(color_intrinsics, output_dir="captured_images"):
    """保存相机内参到文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    intrinsics_path = os.path.join(output_dir, "camera_intrinsics.txt")
    
    with open(intrinsics_path, 'w') as f:
        f.write(f"fx: {color_intrinsics.fx}\n")
        f.write(f"fy: {color_intrinsics.fy}\n")
        f.write(f"ppx: {color_intrinsics.ppx}\n")
        f.write(f"ppy: {color_intrinsics.ppy}\n")
        f.write(f"width: {color_intrinsics.width}\n")
        f.write(f"height: {color_intrinsics.height}\n")
        f.write(f"model: {color_intrinsics.model}\n")
        f.write(f"coeffs: {color_intrinsics.coeffs}\n")
    
    print(f"相机内参已保存到: {intrinsics_path}")

def main():
    print("初始化RealSense相机...")
    pipeline, align, depth_scale, color_intrinsics = setup_realsense()
    
    # 保存相机内参
    save_intrinsics(color_intrinsics)
    
    print("\n按空格键捕获图像，按ESC退出")
    capture_images(pipeline, align, depth_scale)

if __name__ == "__main__":
    main()
