# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
# @FileName       : Anygrasp.py
# @Time           : 2024-08-03 15:07:35
# @Author         : yan & yk
# @Email          : yding25@binghamton.edu
# @Description:   : AnyGrasp: Grasp pose estimation algorithm
"""

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from graspnetAPI import GraspGroup
from gsnet import AnyGrasp
from PIL import Image
from utils import *
from yacs.config import CfgNode as CN

# from Robotics_API import Pose
from Utils import *


class Anygrasp:
    """
    使用 AnyGrasp 模型进行抓取姿态估计的类。

    属性：
        anygrasp_cfg (CfgNode): AnyGrasp 模型的配置信息。
        camera_cfg (CfgNode): 相机参数的配置信息。
        grasping_model (AnyGrasp): AnyGrasp 模型的实例。
    """

    def __init__(self, anygrasp_cfg, camera_cfg):
        """
        初始化 Anygrasp 类，加载相关配置信息，并创建 AnyGrasp 模型实例。

        参数：
            anygrasp_cfg (CfgNode): AnyGrasp 模型的配置信息。
            camera_cfg (CfgNode): 相机参数的配置信息。
        """
        self.anygrasp_cfg = anygrasp_cfg
        self.camera_cfg = camera_cfg

        self.anygrasp_cfg.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.anygrasp_cfg.output_dir
        )
        self.anygrasp_cfg.checkpoint_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            self.anygrasp_cfg.checkpoint_path,
        )
        self.grasping_model = AnyGrasp(self.anygrasp_cfg)
        self.grasping_model.load_net()

    def Grasp_Pose_Estimation(
        self,
        points: np.ndarray,
        image: Image.Image,
        colors: np.ndarray,
        seg_mask: np.ndarray,
        bbox: Bbox,
        crop_flag: bool = False,
    ):
        """
        计算目标物体的最优抓取姿态。

        参数：
            points (np.ndarray): 场景的三维点云数据。
            image (PIL.Image): 与点云对应的 RGB 图像。
            colors (np.ndarray): 图像对应的颜色数组（已归一化）。
            seg_mask (np.ndarray): 物体的分割掩码。
            bbox (Bbox): 场景中物体的边界框，一般由 (x_min, y_min, x_max, y_max) 四个坐标定义。
            crop_flag (bool, 可选): 默认为 False。如果为 False，则会根据分割掩码裁剪点云，只关注目标物体区域；为 True 则不裁剪。

        返回：
            list 或 None: 表示最优抓取姿态的 [平移向量, 旋转矩阵]。若未检测到抓取姿态，则返回 None。
        """




        # 若采样率小于1，则对点云进行下采样
        if self.anygrasp_cfg.sampling_rate < 1:
            points, indices = sample_points(points, self.anygrasp_cfg.sampling_rate)
            colors = colors[indices]



        # 获取相机工作空间的边界
        xmin = self.camera_cfg.xmin
        xmax = self.camera_cfg.xmax
        ymin = self.camera_cfg.ymin
        ymax = self.camera_cfg.ymax
        zmin = self.camera_cfg.zmin
        zmax = self.camera_cfg.zmax
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        # 调用抓取模型进行抓取推理
        gg, cloud = self.grasping_model.get_grasp(
            points,
            colors,
            lims=lims,
            apply_object_mask=True,
            dense_grasp=False,
            collision_detection=True,
        )
        # 检查推理后点云的大小
        cloud_np = np.asarray(cloud.points)  # open3d -> numpy
        print(f"[DEBUG] After get_grasp, gg size: {len(gg)}, cloud points #: {cloud_np.shape[0]}")
        if cloud_np.size > 0:
            print(f"[DEBUG] cloud points min: {cloud_np.min(axis=0)}, max: {cloud_np.max(axis=0)}")

        trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        cloud.transform(trans_mat)

        if len(gg) == 0:
            print(
                "[AnyGrasp] \033[33mwarning\033[0m: No Grasp detected after collision detection!"
            )
            return None

        # 对所有抓取结果进行 nms 并按分数进行排序
        gg = gg.nms().sort_by_score()
        print(
            "[AnyGrasp] \033[34mInfo\033[0m: Grasp point number of all objects:",
            len(gg),
        )

        # 若调试模式开启，可视化
        if self.anygrasp_cfg.debug:
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            visualize_cloud_geometries(
                cloud,
                grippers,
                save_file=os.path.join(self.anygrasp_cfg.output_dir, "poses.png"),
            )

        # 根据分割掩码过滤抓取候选，并进行投影可视化
        # GraspGroup结构（https://graspnetapi.readthedocs.io/_/downloads/en/latest/pdf/）page 30： (score_1, width_1, height_1, depth_1, rotation_matrix_1(9), translation_1(3), object_id_1) 

        filter_gg = GraspGroup()
        # 参考方向向量，用于筛选抓取姿态时可能的角度偏好
        ref_vec = np.array([0, 0, 1])

        # 在图像上绘制目标的边界框
        image = copy.deepcopy(image)
        img_drw = draw_rectangle(image, bbox)

        
        for g in gg:
            # 获取抓取中心点并投影到图像坐标系
            grasp_center = g.translation
            ix = max(
                0,
                min(
                    self.camera_cfg.width - 1,
                    int(
                        ((grasp_center[0] * self.camera_cfg.fx) / grasp_center[2])
                        + self.camera_cfg.cx
                    ),
                ),
            )
            iy = max(
                0,
                min(
                    self.camera_cfg.height - 1,
                    int(
                        ((grasp_center[1] * self.camera_cfg.fy) / grasp_center[2])
                        + self.camera_cfg.cy
                    ),
                ),
            )

            # 如果不裁剪，则根据分割掩码判断该抓取是否在目标物体区域内
            if crop_flag:
                filter_gg.add(g)
            else:
                if seg_mask[
                    iy, ix
                ]:  # 以绿色圆点标记在目标物体上的抓取点
                    img_drw.ellipse([(ix - 2, iy - 2), (ix + 2, iy + 2)], fill="green")
                    filter_gg.add(g)
                else:
                    # 以红色圆点标记在物体外的抓取点
                    img_drw.ellipse([(ix - 2, iy - 2), (ix + 2, iy + 2)], fill="red")

        # 若过滤后没有可抓取姿态，则返回
        if len(filter_gg) == 0:
            print(
                "[AnyGrasp] \033[33mwarning\033[0m: No grasp poses detected for this object try to move the object a little and try again"
            )
            return None

        # 显示并保存最终投影可视化结果
        projections_file_name = os.path.join(
            self.anygrasp_cfg.output_dir, "grasp_projections.png"
        )
        if self.anygrasp_cfg.debug:
            plt.imshow(image)
            plt.title("visualize grasp projections")
            plt.axis("off")
            plt.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(projections_file_name, image)
        print(
            f"[AnyGrasp] \033[34mInfo\033[0m: Saved projections of grasps at {projections_file_name}"
        )

        # 对过滤后的抓取再次进行 nms 并排序
        filter_gg = filter_gg.nms().sort_by_score()
        print(
            "[AnyGrasp] \033[34mInfo\033[0m: Filter grasp point number of grasp object:",
            len(filter_gg),
        )

        self.print_filter_gg(filter_gg)
        print('\n\n')
        print(vars(filter_gg[0]))
        print(vars(filter_gg[1]))
        print(vars(filter_gg[2]))
        print(filter_gg[0].width)
        # print(filter_gg[0].translation)
        # print(filter_gg[0].rotation_matrix)
        print('\n\n')

        # 若调试模式开启，展示最好抓取姿态可视化
        if self.anygrasp_cfg.debug:
            filter_grippers = filter_gg.to_open3d_geometry_list()
            for gripper in filter_grippers:
                gripper.transform(trans_mat)
            visualize_cloud_geometries(
                cloud,
                [filter_grippers[0].paint_uniform_color([1.0, 0.0, 0.0])],
                save_file=os.path.join(self.anygrasp_cfg.output_dir, "best_pose.png"),
            )

        # 返回分数最高的抓取姿态
        best_pose = [filter_gg[0].translation, filter_gg[0].rotation_matrix]
        return best_pose

    def print_filter_gg(self, filter_gg):
        """print grasp pose and score, Descending.

        Args:
            filter_gg (): anygrasp grasp pose info
        """
        print(f"[AnyGrasp] \033[34mInfo\033[0m: AnyGrasp output pose about object:")
        for g in filter_gg:
            print(
                f"[AnyGrasp] \033[34mInfo\033[0m: translation: {g.translation}, z_vec: {g.rotation_matrix[:, 2]}, score: {g.score}"
            )


if __name__ == "__main__":

    # set work dir to AnyGrasp
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    input = Submodule()
    pkl_file = os.path.abspath("./data.pkl")
    input.deserialize(pkl_file)

    anygrasp_cfg = input.get("anygrasp_cfg", CN)
    camera_cfg = input.get("camera_cfg", CN)
    points = input.get("points", np.ndarray).astype(np.float32)
    image = input.get("image", Image.Image)
    colors = input.get("colors", np.ndarray).astype(np.float32)
    seg_mask = input.get("seg_mask", np.ndarray)
    bbox = input.get("bbox", np.ndarray)

    anygrasp = Anygrasp(anygrasp_cfg, camera_cfg)
    best_pose = anygrasp.Grasp_Pose_Estimation(points, image, colors, seg_mask, bbox)

    input.clear()
    input.add("best_pose", best_pose)
    input.serialize(pkl_file)
