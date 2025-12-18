#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @FileName       : Camera_Realsense.py
# @Time           : 2025-01-03
# @Author         : Yan 
# @Email          : yding25@binghamton.edu
# @Description    : RealSense D435 Camera (Eye-in-Hand)
"""

import sys
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import matplotlib.pyplot as plt
import threading
import signal
import time

from datetime import datetime
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional  # 确保包含 Optional

from Robotics_API.Pose import Pose


class Camera_Realsense:
    """RealSense D435 Camera (Eye-in-Hand)."""

    def __init__(self, cfg):
        """
        初始化 RealSense D435。

        Args:
            cfg (object): 配置对象，包含相机分辨率、深度范围等信息。
        """
        # 从 cfg 中获取参数（若有多余的 fov、nearVal、farVal 等，可以保留以保持接口一致，也可忽略）
        self.width = cfg.width
        self.height = cfg.height
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

        # 初始化 RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, cfg.fps)
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, cfg.fps)

        # 开启相机流
        try:
            profile = self.pipeline.start(self.config)
        except Exception as e:
            print(f"Failed to start RealSense pipeline: {e}")
            sys.exit(1)

        # 获取深度缩放因子
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth Scale: {self.depth_scale}")
        
        # 对齐：将深度帧和彩色帧对齐到同一个坐标系
        self.align = rs.align(rs.stream.color)

        # 获取内参
        for _ in range(5):  # 连续读取几帧，确保相机已经稳定输出
            frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Could not retrieve frames from RealSense camera.")

        intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy

        # 将内参写回 cfg
        cfg.fx = self.fx
        cfg.fy = self.fy
        cfg.cx = self.cx
        cfg.cy = self.cy

        # # test
        # print('-'*30)
        # print(f'camera fx:{self.fx}; fy:{self.fy}; cx:{self.cx}; cy:{self.cy}')
        # print('-'*30)

        # 用于暂存获取的彩色图和深度图
        self.colors = None  # (H, W, 3)
        self.depths = None  # (H, W), 单位: 米

        # 默认先更新一次图像
        self.update()

        # -------------------------
        # 以下为“眼在手上”下典型的相机到机械臂末端(或基座)的外参矩阵
        # 具体数值需在真实项目中通过手眼标定或测量得到
        # 这里仅给出一个示例化的 4x4 矩阵
        self.camera_to_arm_base = np.eye(4)  # 根据实际标定结果来
        # 如果您已经知道相机坐标系相对机械臂基座(arm base)的平移/旋转，可填充到此矩阵
        # self.camera_to_arm_base[:3, :3] = R(3x3)
        # self.camera_to_arm_base[:3, 3] = t(3,)

        # 初始化 ArUco 字典和参数·
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_parameters = cv2.aruco.DetectorParameters()

        # 打印以验证初始化
        print("Initialized aruco_dict and aruco_parameters.")
        print(f"aruco_dict: {self.aruco_dict}")
        print(f"aruco_parameters: {self.aruco_parameters}")
    
        # 标志用于控制线程停止
        self._stop_event = threading.Event()

    def __enter__(self):
        """进入上下文管理器"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，确保资源被正确释放"""
        self.close()

    def close(self):
        """
        停止 RealSense pipeline 并释放资源。
        """
        if hasattr(self, 'pipeline') and self.pipeline:
            self.pipeline.stop()
            print("RealSense pipeline stopped.")
        cv2.destroyAllWindows()
        self._stop_event.set()

    # def __del__(self):
    #     """析构函数，确保资源被释放"""
    #     self.close()

    def get_focal_length(self):
        """
        获取相机焦距 (fx, fy)。
        """
        return self.fx, self.fy

    def get_camera_pose(self):
        """
        获取相机相对于机械臂末端或基座的位姿。
        在“眼在手上”环境中，通常是相机相对末端执行器(或基座)的 Pose。
        这里示例直接从 camera_to_arm_base 矩阵返回 Pose。
        """
        pose_mat = self.camera_to_arm_base.copy()
        translation = pose_mat[:3, 3]
        rotation = pose_mat[:3, :3]
        return Pose(translation, rotation)

    def update(self):
        """
        从 RealSense 管线中获取最新的 RGB 和深度图，并存储在成员变量。
        建议在需要的时候定期调用此方法，以更新图像。
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Warning: incomplete frame, skip update.")
                return

            # 取出图像数据
            color_image = np.asanyarray(color_frame.get_data())  # shape: (H, W, 3), BGR
            depth_image = np.asanyarray(depth_frame.get_data())  # shape: (H, W), uint16

            # 检查分辨率是否一致
            assert color_image.shape[:2] == depth_image.shape[:2], "Depth and color maps are not aligned!"

            # 使用动态获取的深度缩放因子
            depth_meters = depth_image.astype(np.float32) * self.depth_scale

            # 处理无效深度值
            valid_depth_mask = (depth_meters > self.min_depth) & (depth_meters < self.max_depth)
            depth_meters[~valid_depth_mask] = 0

            # 转换颜色图到 RGB 格式
            self.colors = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            self.depths = depth_meters
        except Exception as e:
            print(f"Error during update: {e}")
            self.close()


    def get_rgb_image(self, enable_show=False, enable_save=False, filename=None):
        """
        获取当前帧的 RGB 图像。

        Args:
            enable_show (bool): 是否显示图像。默认 False
            enable_save (bool): 是否保存图像。默认 False
            filename (str): 文件名（不带扩展名）

        Returns:
            np.ndarray: RGB 图像 (H, W, 3)
        """
        self.update()

        if self.colors is None:
            print("Warning: No color image available. Call update() first.")
            return None

        rgb_image = self.colors.copy()  # (H, W, 3), RGB

        if enable_show:
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.title("RealSense RGB Image")
            plt.show()

        if enable_save:
            if filename is None:
                filename = "rgb_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"./{filename}.png"
            # 需要存储时，OpenCV 缺省用 BGR，因此要再转回去
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr_image)

        return rgb_image

    def get_depth_image(self, enable_show=False, enable_save=False, filename=None):
        """
        获取当前帧的深度图 (单位: 米)。

        Args:
            enable_show (bool): 是否显示图像。默认 False
            enable_save (bool): 是否保存图像。默认 False
            filename (str): 文件名（不带扩展名）

        Returns:
            np.ndarray: Depth 图 (H, W)，单位: 米
        """
        if self.depths is None:
            print("Warning: No depth image available. Call update() first.")
            return None

        depth_img = self.depths.copy()

        if enable_show:
            # 自定义线性灰度 colormap
            cdict = {
                "red":   [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                "green": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                "blue":  [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            }
            custom_cmap = LinearSegmentedColormap("custom_cmap", cdict)
            plt.imshow(depth_img, cmap=custom_cmap)
            plt.colorbar()
            plt.title("RealSense Depth Image (m)")
            plt.show()

        if enable_save:
            if filename is None:
                filename = "depth_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"./{filename}.png"
            # 存储时，通常将深度从米转换回 mm，并以 uint16 形式保存
            depth_mm = (depth_img * 1000).astype(np.uint16)
            Image.fromarray(depth_mm).save(save_path)

        return depth_img

    # ----------------------------------------------------------------
    # 一些实用函数，与仿真中函数名对应，只是去掉了 "sim_" 前缀
    # ----------------------------------------------------------------

    def rotate_around_y(self, vector, angle):
        """
        绕 Y 轴旋转一个向量。

        Args:
            vector (np.ndarray): (3,) 向量
            angle (float): 旋转角度（弧度）

        Returns:
            np.ndarray: 旋转后的 (3,) 向量
        """
        rotation_matrix = np.array([
            [np.cos(angle),  0, np.sin(angle)],
            [0,              1, 0            ],
            [-np.sin(angle), 0, np.cos(angle)],
        ])
        return rotation_matrix @ vector

    def visualize_3d_points(self):
        """
        使用 Open3D 可视化当前帧转换成的 3D 点云。
        """
        if self.colors is None or self.depths is None:
            print("Warning: No image data. Call update() first.")
            return

        # 转为 open3d.Image
        color_o3d = o3d.geometry.Image(self.colors)
        depth_o3d = o3d.geometry.Image((self.depths * 1000).astype(np.uint16))

        # 构建 RGBD 图像
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            convert_rgb_to_intensity=False
        )

        # 相机内参
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )

        # 从 RGBD 构建点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

        # 如果需要将点云转到机械臂基座坐标系，也可在此处做变换
        # pcd.transform(self.camera_to_arm_base)

        # 可视化
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="RealSense 3D Points")
        vis.add_geometry(pcd)
        vis.add_geometry(coord_frame)
        vis.run()
        vis.destroy_window()

    def get_3d_points(self):
        """
        将当前帧的深度图转换为点云数据 (points + colors)。

        Returns:
            tuple: (points, colors)
                points: (N, 3) np.ndarray
                colors: (N, 3) np.ndarray, 范围 [0, 1]
        """
        self.update()
        if self.colors is None or self.depths is None:
            print("Warning: No image data. Call update() first.")
            return None, None

        h, w = self.depths.shape
        xmap, ymap = np.meshgrid(np.arange(w), np.arange(h))
        Z = self.depths

        # 剔除无效深度值
        mask = (Z > self.min_depth) & (Z < self.max_depth) & ~np.isnan(Z)
        Z[~mask] = 0
        # print(Z)

        X = (xmap - self.cx) / self.fx * Z
        Y = (ymap - self.cy) / self.fy * Z

        points = np.stack([X, Y, Z], axis=-1)[mask]
        # print(points)
        # print(points.shape)
        colors = (self.colors.astype(np.float32) / 255.0)[mask]

        # print(f"Generated {points.shape[0]} 3D points.")
        return points, colors
    def get_cam_3d_points_from_mouse(self) -> Optional[np.ndarray]:
        """
        通过鼠标在 RGB 图像上点击，获取对应的相机坐标系下的 3D 点。

        左键：添加点
        右键：撤销上一个点
        按 q 或 ESC：结束并返回点云

        Returns:
            np.ndarray (N, 3): 相机坐标系下的 3D 点（单位：米）
                            若未选点，返回 None
        """
        self.update()
        if self.colors is None or self.depths is None:
            print("No image available.")
            return None

        img = self.colors.copy()
        window_name = "Click RGB Image (L: add, R: undo, Q/ESC: quit)"

        points_2d = []   # [(x, y), ...]
        points_3d = []   # [(X, Y, Z), ...]

        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy

        def mouse_callback(event, x, y, flags, param):
            nonlocal img, points_2d, points_3d

            # 左键：添加点
            if event == cv2.EVENT_LBUTTONDOWN:
                if x < 0 or y < 0 or x >= self.width or y >= self.height:
                    return

                z = self.depths[y, x]
                if z <= 0:
                    print(f"Invalid depth at ({x}, {y})")
                    return

                X = (x - cx) / fx * z
                Y = (y - cy) / fy * z
                Z = z

                points_2d.append((x, y))
                points_3d.append((X, Y, Z))

                print(f"[ADD] Pixel ({x},{y}) -> 3D ({X:.3f}, {Y:.3f}, {Z:.3f})")

            # 右键：撤销上一个点
            elif event == cv2.EVENT_RBUTTONDOWN:
                if points_2d:
                    removed_2d = points_2d.pop()
                    removed_3d = points_3d.pop()
                    print(f"[UNDO] Pixel {removed_2d} -> 3D {removed_3d}")

            # 重新绘制
            img = self.colors.copy()
            for i, (px, py) in enumerate(points_2d):
                cv2.circle(img, (px, py), 5, (0, 255, 0), -1)
                cv2.putText(
                    img,
                    str(i + 1),
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q 或 ESC
                break

        cv2.destroyWindow(window_name)

        if len(points_3d) == 0:
            print("No points selected.")
            return None

        return np.array(points_3d, dtype=np.float32)


    def trans_to_arm_base(self, pose):
        """
        将相机坐标系下的 Pose 转到机械臂基座坐标系下。

        Args:
            pose (Pose): 在相机坐标系下的位姿

        Returns:
            Pose: 在机械臂基座坐标系下的位姿
        """
        # 先把 pose 转成 4x4 矩阵
        pose_mat = np.eye(4)
        pose_mat[:3, :3] = pose.get_orientation("rotation_matrix")
        pose_mat[:3,  3] = pose.get_position()

        # 乘以 camera_to_arm_base
        # camera_to_arm_base 是从 相机坐标系 -> 机械臂基座坐标系 的 4x4 变换
        final_pose_mat = self.camera_to_arm_base @ pose_mat

        # 示例：若需要对 Z 进行校正，可在此处进行
        # final_pose_mat[2, 3] -= 0.03

        # 提取最终位置、姿态
        trans = final_pose_mat[:3, 3]
        rot = final_pose_mat[:3, :3]
        return Pose(trans, rot)

    def check_realsense_connection(self) -> bool:
        """
        检查是否有 RealSense 设备连接。

        Returns:
            bool: 如果检测到设备，返回 True；否则返回 False。
        """
        context = rs.context()
        if len(context.devices) == 0:
            print("No RealSense devices connected.")
            return False
        else:
            print("RealSense device detected.")
            return True

    def get_marker_positions(self, debug: bool = False) -> Optional[np.ndarray]:
        """
        检测 ArUco 标记并返回其位置。

        Args:
            debug (bool): 如果为 True，显示带标记的帧进行调试。

        Returns:
            np.ndarray 或 None: 标记位置数组，格式为 [y, x, -z]，如果未检测到标记则返回 None。
        """
        try:
            while not self._stop_event.is_set():
                self.update()
                if self.colors is None:
                    continue

                # 转换为灰度图
                gray = cv2.cvtColor(self.colors, cv2.COLOR_RGB2GRAY)

                # 检测 ArUco 标记
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_parameters)

                marker_positions: Dict[int, np.ndarray] = {}

                # 如果检测到标记
                if ids is not None:
                    # 绘制检测到的标记
                    annotated_image = cv2.aruco.drawDetectedMarkers(self.colors.copy(), corners, ids)

                    for i in range(len(ids)):
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners[i], 0.035, 
                            np.array([[self.fx, 0, self.cx], 
                                      [0, self.fy, self.cy], 
                                      [0, 0, 1]], dtype=float), 
                            np.zeros((4, 1))  # 假设无镜头畸变
                        )
                        # 绘制每个标记的轴
                        if hasattr(cv2.aruco, 'drawAxis'):
                            cv2.aruco.drawAxis(annotated_image, 
                                               np.array([[self.fx, 0, self.cx], 
                                                         [0, self.fy, self.cy], 
                                                         [0, 0, 1]], dtype=float), 
                                               np.zeros((4, 1)), 
                                               rvec, tvec, 0.1)
                        # 存储标记位置
                        marker_positions[ids[i][0]] = tvec[0][0]

                    # 转换为 NumPy 数组
                    positions_array = np.array([[pos[1], pos[0], -pos[2]] for pos in marker_positions.values()])

                    # 调试模式：显示带标记的图像
                    if debug:
                        cv2.imshow('RealSense Debug - Marker Positions', annotated_image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.close()
                            break

                    return positions_array

                # 如果未检测到标记，仍在调试模式下显示图像
                if debug:
                    cv2.imshow('RealSense Debug - No Markers', self.colors)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.close()
                        break

                # 添加短暂的休眠以减少 CPU 使用率
                time.sleep(0.1)

            return None
        finally:
            if debug:
                cv2.destroyAllWindows()

    def display(self, option="rgb"):
        """
        显示来自 RealSense 相机的帧。支持 RGB、深度或两者。

        Parameters:
            option (str): 显示模式。
                        "rgb" - 仅显示 RGB 帧。
                        "d" - 仅显示深度帧。
                        "rgbd" - 同时显示 RGB 和深度帧。
        """
        stop_flag = [False]  # 可变标志以控制线程终止

        def signal_handler(sig, frame):
            """
            信号处理器，用于在检测到 Ctrl+C 时优雅地退出程序。

            Args:
                sig: 信号编号。
                frame: 当前栈帧。
            """
            stop_flag[0] = True
            print("Ctrl+C detected. Exiting...")

        # 绑定 SIGINT 信号（Ctrl+C）到信号处理器
        signal.signal(signal.SIGINT, signal_handler)

        def process_frames():
            """
            获取来自 RealSense 相机的帧并显示，直到 stop_flag 被设置为 True。
            """
            try:
                while not stop_flag[0] and not self._stop_event.is_set():
                    self.update()

                    # 根据选项获取帧
                    if option in ["rgb", "rgbd"]:
                        if self.colors is not None:
                            cv2.imshow('RGB Frame', cv2.cvtColor(self.colors, cv2.COLOR_RGB2BGR))
                    if option in ["d", "rgbd"]:
                        if self.depths is not None and np.max(self.depths) != 0:
                            depth_colormap = cv2.convertScaleAbs(self.depths, alpha=255.0 / np.max(self.depths))
                            cv2.imshow('Depth Frame', depth_colormap)
                        else:
                            # 如果深度图全为0，显示黑图
                            black_image = np.zeros((self.height, self.width), dtype=np.uint8)
                            cv2.imshow('Depth Frame', black_image)

                    # 检测 'q' 键以退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_flag[0] = True
                        self.close()
                        break
            finally:
                # 确保资源被正确释放
                self.close()

        # 启动帧处理的线程
        thread = threading.Thread(target=process_frames)
        thread.start()

        # 等待线程完成（主线程在此阻塞）
        thread.join()