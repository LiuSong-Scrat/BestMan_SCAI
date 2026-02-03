#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import sys
sys.path.append("/home/liusong/ProgramFiles/IssacSim/Tasks_Wokspace/VisionStacking/code/FastUmiDataProcessing")
from simulation_data_collection import *


try:
    # Third Party
    import isaacsim
except ImportError:
    pass


# Third Party
import torch

a = torch.zeros(
    4, device="cuda:0"
)  # this is necessary to allow isaac sim to use this torch instance
# Third Party
import numpy as np

np.set_printoptions(suppress=True)
# Standard Library

# Standard Library
import argparse

## import curobo:

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)
args = parser.parse_args()

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Standard Library
from typing import Optional

# Third Party
import carb
from helper import add_extensions
from omni.isaac.core import World
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.tasks import Stacking as BaseStacking
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.franka import Franka

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
    PoseCostMetric,
)


class CuroboController(BaseController):
    def __init__(
        self,
        my_world: World,
        my_task: BaseStacking,
        name: str = "curobo_controller",
        constrain_grasp_approach: bool = False,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._save_log = False
        self.my_world = my_world
        self.my_task = my_task
        self._step_idx = 0
        n_obstacle_cuboids = 20
        n_obstacle_mesh = 2
        # warmup curobo instance
        self.usd_help = UsdHelper()
        self.init_curobo = False
        self.world_file = "collision_table.yml"
        self.cmd_js_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.tensor_args = TensorDeviceType()
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.robot_cfg["kinematics"][
            "base_link"
        ] = "panda_link0"  # controls which frame the controller is controlling

        self.robot_cfg["kinematics"][
            "ee_link"
        ] = "panda_hand"  # controls which frame the controller is controlling
        # self.robot_cfg["kinematics"]["cspace"]["max_acceleration"] = 10.0 # controls how fast robot moves
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
        # @self.robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.0
        self.robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_collision_mesh.yml"

        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        self._world_cfg_table = world_cfg_table

        world_cfg1 = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        ).get_mesh_world()
        world_cfg1.mesh[0].pose[2] = -10.5

        self._world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self._world_cfg,
            self.tensor_args,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.01,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            store_ik_debug=self._save_log,
            store_trajopt_debug=self._save_log,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up...")
        self.motion_gen.warmup(parallel_finetune=True)
        pose_metric = None
        if constrain_grasp_approach:
            pose_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=0.1, tstep_fraction=0.8
            )

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            max_attempts=10,
            enable_graph_attempt=None,
            enable_finetune_trajopt=True,
            partial_ik_opt=False,
            parallel_finetune=True,
            pose_cost_metric=pose_metric,
            time_dilation_factor=0.75,
        )
        self.usd_help.load_stage(self.my_world.stage)
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.idx_list = None
        self.ee_orientation_goal = np.array([ 0,0.92388, 0, 0.38268 ])
        # self.ee_orientation_goal = np.array([0, 1, 0, 0])

    def attach_obj(
        self,
        sim_js: JointState,
        js_names: list,
    ) -> None:
        cube_name = self.my_task.get_cube_prim(self.my_task.target_cube)

        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )

        self.motion_gen.attach_objects_to_robot(
            cu_js,
            [cube_name],
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
        )

    def detach_obj(self) -> None:
        self.motion_gen.detach_object_from_robot()

    def plan(
        self,
        ee_translation_goal: np.array,
        ee_orientation_goal: np.array,
        sim_js: JointState,
        js_names: list,
    ) -> MotionGenResult:
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())
        if self._save_log:  # and not result.success.item(): # logging for debugging
            UsdHelper.write_motion_gen_log(
                result,
                {"robot_cfg": self.robot_cfg},
                self._world_cfg,
                cu_js,
                ik_goal,
                join_path("log/usd/", "cube") + "_debug",
                write_ik=False,
                write_trajopt=True,
                visualize_robot_spheres=True,
                link_spheres=self.motion_gen.kinematics.kinematics_config.link_spheres,
                grid_space=2,
                write_robot_usd_path="log/usd/assets",
            )
        return result

    def forward(
        self,
        sim_js: JointState,
        js_names: list,
    ) -> ArticulationAction:
        assert self.my_task.target_position is not None
        assert self.my_task.target_cube is not None

        if self.cmd_plan is None:
            self.cmd_idx = 0
            self._step_idx = 0
            # Set EE goals
            ee_translation_goal = self.my_task.target_position
            ee_orientation_goal = self.ee_orientation_goal
            # compute curobo solution:
            result = self.plan(ee_translation_goal, ee_orientation_goal, sim_js, js_names)
            succ = result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                self.idx_list = [i for i in range(len(self.cmd_js_names))]
                self.cmd_plan = cmd_plan.get_ordered_joint_state(self.cmd_js_names)
            else:
                carb.log_warn("Plan did not converge to a solution.")
                return None
        if self._step_idx % 3 == 0:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.cmd_idx += 1

            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy() * 0.0,
                joint_indices=self.idx_list,
            )
            if self.cmd_idx >= len(self.cmd_plan.position):
                self.cmd_idx = 0
                self.cmd_plan = None
        return art_action

    def reached_target(self, observations: dict) -> bool:
        curr_ee_position = observations["my_franka"]["end_effector_position"]
        if np.linalg.norm(
            self.my_task.target_position - curr_ee_position
        ) < 0.04 and (  # This is half gripper width, curobo succ threshold is 0.5 cm
            self.cmd_plan is None
        ):
            if self.my_task.cube_in_hand is None:
                print("reached picking target: ", self.my_task.target_cube)
            else:
                print("reached placing target: ", self.my_task.target_cube)
            return True
        else:
            return False

    def reset(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        # init
        self.update(ignore_substring, robot_prim_path)
        self.init_curobo = True
        self.cmd_plan = None
        self.cmd_idx = 0

    def update(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        # print("updating world...")
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=robot_prim_path
        ).get_collision_check_world()
        # add ground plane as it's not readable:
        obstacles.add_obstacle(self._world_cfg_table.cuboid[0])
        self.motion_gen.update_world(obstacles)
        self._world_cfg = obstacles


class MultiModalStacking(BaseStacking):
    def __init__(
        self,
        name: str = "multi_modal_stacking",
        offset: Optional[np.ndarray] = None,
    ) -> None:
        BaseStacking.__init__(
            self,
            name=name,
            cube_initial_positions=np.array(
                [
                    [0.55, 0.15, 0.5],
                    [0.55, -0.15, 0.5],
                ]
            )
            / get_stage_units(),
            cube_initial_orientations=None,
            stack_target_position=None,
            cube_size=np.array([0.1, 0.1, 0.1]),
            offset=offset,
        )
        self.cube_list = None
        self.target_position = None
        self.target_cube = None
        self.cube_in_hand = None

    def reset(self) -> None:
        self.cube_list = self.get_cube_names()
        self.target_position = None
        self.target_cube = None
        self.cube_in_hand = None

    def update_task(self) -> bool:
        # after detaching the cube in hand
        assert self.target_cube is not None
        assert self.cube_in_hand is not None
        self.cube_list.insert(0, self.cube_in_hand)
        self.target_cube = None
        self.target_position = None
        self.cube_in_hand = None
        if len(self.cube_list) <= 1:
            task_finished = True
        else:
            task_finished = False
        return task_finished

    def get_cube_prim(self, cube_name: str):
        for i in range(self._num_of_cubes):
            if cube_name == self._cubes[i].name:
                return self._cubes[i].prim_path

    def get_place_position(self, observations: dict) -> None:
        assert self.target_cube is not None
        self.cube_in_hand = self.target_cube
        self.target_cube = self.cube_list[0]
        ee_to_grasped_cube = (
            observations["my_franka"]["end_effector_position"][2]
            - observations[self.cube_in_hand]["position"][2]
        )
        self.target_position = observations[self.target_cube]["position"] + [
            0,
            0,
            self._cube_size[2] + ee_to_grasped_cube + 0.02,
        ]
        self.cube_list.remove(self.target_cube)

    def get_pick_position(self, observations: dict) -> None:
        assert self.cube_in_hand is None
        self.target_cube = self.cube_list[1]
        self.target_position = observations[self.target_cube]["position"] + [
            0,
            0,
            self._cube_size[2] / 2 + 0.092,
        ]
        # self.cube_list.remove(self.target_cube)

    def set_robot(self) -> Franka:
        franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return Franka(
            prim_path=franka_prim_path, name=franka_robot_name, end_effector_prim_name="panda_hand"
        )





robot_prim_path = "/World/Franka/panda_link0"
ignore_substring = ["Franka", "TargetCube", "material", "Plane","kitchen_scene_source"]
my_world = World(stage_units_in_meters=1.0)
stage = my_world.stage
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

my_task = MultiModalStacking()
my_world.add_task(my_task)
my_world.reset()
robot_name = my_task.get_params()["robot_name"]["value"]
my_franka = my_world.scene.get_object(robot_name)
my_controller = CuroboController(
    my_world=my_world, my_task=my_task, constrain_grasp_approach=args.constrain_grasp_approach
)
articulation_controller = my_franka.get_articulation_controller()
set_camera_view(eye=[2, 0, 1], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp")
wait_steps = 30

my_franka.set_solver_velocity_iteration_count(4)
my_franka.set_solver_position_iteration_count(124)
my_world._physics_context.set_solver_type("TGS")
initial_steps = 10
################################################################
print("Start simulation...")
robot = my_franka
print(
    my_world._physics_context.get_solver_type(),
    robot.get_solver_position_iteration_count(),
    robot.get_solver_velocity_iteration_count(),
)
print(my_world._physics_context.use_gpu_pipeline)
print(articulation_controller.get_gains())
print(articulation_controller.get_max_efforts())
robot = my_franka
print("**********************")
cube_init_heigth = my_world.get_observations()["cube"]["position"][2]
print("Updated gains:")
print(articulation_controller.get_gains())
print(articulation_controller.get_max_efforts())
# exit()

my_franka.gripper.open()
for _ in range(wait_steps):
    my_world.step(render=True)
my_task.reset()

#SONG
franka_init_joints_position = np.array([0.012, -0.57000005, 0.0, -2.15, 0.0, 1.57, 0.741, 0.04, 0.04])
my_franka.set_joint_positions(franka_init_joints_position, my_controller.idx_list)
for _ in range(wait_steps):
    my_world.step(render=True)


task_finished = False
observations = my_world.get_observations()
my_task.get_pick_position(observations)

i = 0

add_extensions(simulation_app, args.headless_mode)




#-----------------------------SONG-----------------------------------#
import sys
sys.path.append("/home/liusong/ProgramFiles/IssacSim/Tasks_Wokspace/VisionStacking/code/FastUmiDataProcessing")
from simulation_data_collection import SimulationDataCollection
sys.path.append("/home/liusong/ProgramFiles/Huggingface/lerobot/lerobot/")
from record_song import SmolVLA_ModelInference,ACT_ModelInference,DP_ModelInference
sys.path.append("/home/liusong/ProgramFiles/VA-VLA/DP3/3D-Diffusion-Policy/3D-Diffusion-Policy/")
from DP3_ModelInference import DP3_ModelInference

from scipy.spatial.transform import Rotation as R
import open3d as o3d
import omni
import cv2
import os 
import threading
import select
import pytorch3d.ops as torch3d_ops
import time
import copy
import omni.replicator.core as rep
import isaacsim.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import UsdGeom, UsdShade, Gf,Sdf
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import XFormPrim
import omni.isaac.core.utils.prims as prim_utils
sys.path.append("/home/liusong/ProgramFiles/REAP/StageGen")
from stagegen.geometry_utils import GeometryUtils
from stagegen.projection_utils import ProjectionUtils
from stagegen.visualization_utils import VisualizationUtils
from stagegen.stage2_editing import Stage2Editing

#隐藏默认灯光
prim = stage.GetPrimAtPath("/World/defaultGroundPlane/SphereLight")
prim.GetAttribute('visibility').Set('invisible')



direct_light = prim_utils.create_prim(
    "/World/DirectLight",
    "DistantLight",
    position=np.array([1.0, 1.0, 1.0]),
    orientation=np.array([0.84293, -0.41131, 0.20301,-0.28122]),
    attributes={
        "inputs:intensity": 1e3,
        "inputs:color": (1.0, 1.0, 1.0),
        'inputs:enableColorTemperature': True,
        'inputs:exposure':0.5 
        # 'inputs:temperature':0.5 
    }
    # attributes={'inputs:radius': 0.02, 'inputs:intensity': 5e3, 'inputs:exposure': 0.1, 'inputs:color': (1.0, 0.0, 1.0), 'inputs:enableColorTemperature': True}
)
demolight = prim_utils.create_prim(
    "/World/DemotLight",
    "DistantLight",
    position=np.array([1.0, 1.0, 1.0]),
    attributes={
        "inputs:intensity": 1.0,
        "inputs:color": (1.0, 1.0, 1.0),
        'inputs:enableColorTemperature': True,
        'inputs:exposure':9.0 ,
        'inputs:angle':-180,
        # 'inputs:temperature':0.5 
    }
    # attributes={'inputs:radius': 0.02, 'inputs:intensity': 5e3, 'inputs:exposure': 0.1, 'inputs:color': (1.0, 0.0, 1.0), 'inputs:enableColorTemperature': True}
)



################CAMERA
camera_width = 640
camera_height = 480
camera_fps=30
camera_focus_distance= 200
camera_f_stop = 0.5
camera_hand = rep.create.camera(
    parent="/World/Franka/panda_hand",
    position=(0.35,0.0,-0.2),
    rotation=(0, 45, 0),
    clipping_range=(0.1,2),
    # focus_distance=camera_focus_distance,
    # f_stop=camera_f_stop,
)
camera_overhead= rep.create.camera(
    parent="/World",
    position=(0.65,0.0,1.55),
    rotation=(180, -80, 0),
    clipping_range=(0.1,2),
        # focus_distance=camera_focus_distance,
    # f_stop=camera_f_stop,
)

def get_rgb_annators(cam):
    rp = rep.create.render_product(cam, (camera_width, camera_height))
    ldr= rep.AnnotatorRegistry.get_annotator("LdrColor")
    ldr.attach(rp)
    return ldr
def get_pointcloud_anotator(cam):
    rp = rep.create.render_product(cam, (camera_width, camera_height))
    pointcloud_anno = rep.annotators.get("pointcloud",init_params={"includeUnlabelled": True})
    pointcloud_anno.attach(rp)
    return pointcloud_anno

def get_rgb(cam):
    ldr = get_rgb_annators(cam)
    my_world.step(render=True)
    data = ldr.get_data()[:, :, :3]
    return data
def get_rgb_pointcloud(pointcloud_anno):
    pointcloud_dict = {"cloud":{},"rgb":{}}
    pc_data = pointcloud_anno.get_data()
    pointcloud_dict['cloud'] = pc_data["data"]
    pointcloud_dict['rgb'] = pc_data["info"]["pointRgb"].reshape(-1, 4)[:, :3]
    cloud_rgb = np.concatenate((pointcloud_dict['cloud'],pointcloud_dict['rgb']),axis=-1)
    return cloud_rgb

def get_double_rgb(cam1,cam2):
    ldr1 = get_rgb_annators(cam1)
    ldr2 = get_rgb_annators(cam2)
    # rep.orchestrator.step()->my_world.step(render=True)
    my_world.step(render=True)
    my_world.step(render=True)
    my_world.step(render=True)
    my_world.step(render=True)
    data1 = ldr1.get_data()[:, :, :3]
    data2 = ldr2.get_data()[:, :, :3]
    return data1,data2

def get_double_all(cam1,cam2):
    data_dict1 =  {"img_rgb":{},"cloud_rgb":{}}
    data_dict2 =  {"img_rgb":{},"cloud_rgb":{}}
    ldr1 = get_rgb_annators(cam1)
    ldr2 = get_rgb_annators(cam2)
    pointcloud_anno1=get_pointcloud_anotator(cam1)
    pointcloud_anno2=get_pointcloud_anotator(cam2)
    # rep.orchestrator.step()->my_world.step(render=True)
    my_world.step(render=True)
    my_world.step(render=True)
    my_world.step(render=True)
    my_world.step(render=True)
    cloud_rgb1 = get_rgb_pointcloud(pointcloud_anno1)
    cloud_rgb2 = get_rgb_pointcloud(pointcloud_anno2)
    img_rgb1 = ldr1.get_data()[:, :, :3]
    img_rgb2 = ldr2.get_data()[:, :, :3]
    data_dict1['img_rgb']=img_rgb1
    data_dict1['cloud_rgb']=cloud_rgb1
    data_dict2['img_rgb']=img_rgb2
    data_dict2['cloud_rgb']=cloud_rgb2
    # print(data.shape, data.dtype)   # ((512, 1024, 4), uint8)
    return data_dict1,data_dict2


def visualize_cloud_rgb(cloud_rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_rgb[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(cloud_rgb[:,3:]/255)
    geometries = []
    geometries.append(pcd)
    o3d.visualization.draw_geometries(geometries)

def predict_trajectory_visualize(model_inference,inference_action,my_controller,my_franka,overhead_cloud_rgb):
    pre_actions_list = []
    pre_actions_list.append(inference_action.cpu().numpy())
    for one_action in model_inference.policy._action_queue:
        pre_actions_list.append(one_action.cpu().numpy().squeeze())
    pre_actions_array = np.array(pre_actions_list)
    pre_actions_array[:-1]*=3.14/180  #convert to radin
    
    pre_js = JointState(
        position=my_controller.tensor_args.to_device(pre_actions_array),
        joint_names= my_franka.dof_names[:-1],
    )
    state = my_controller.motion_gen.compute_kinematics(pre_js)
    pre_translation = np.round(np.array(state.ee_pose.position.cpu().squeeze(0)),3)
    pre_translation[:,-1] -= 0.1  #gripper offeset
    pre_translation+=my_franka.get_world_pose()[0]
    
    # 提取位置数据用于轨迹显示
    positions = pre_translation  # x, y, z坐标
    # quaternions = poses[:, 3:7]  # 四元数
    positions_color = np.zeros(positions.shape)
    trajectory_clouds_rgb = np.hstack((positions,positions_color))
    clouds_rgb = overhead_cloud_rgb
    clouds_rgb_with_trajectory = np.vstack((clouds_rgb,trajectory_clouds_rgb))
    visualize_cloud_rgb(clouds_rgb_with_trajectory)

def get_sam2_masked_frame(frame,predictor):
    out_obj_ids, out_mask_logits = predictor.track(frame)
    (height, width) = frame.shape[:2]
    all_mask = np.zeros((height, width, 3), dtype=np.uint8)
    all_mask[..., 1] = 255
    # print(all_mask.shape)
    for i in range(0, len(out_obj_ids)):
        out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
            np.uint8
        ) * 255

        hue = (i + 3) / (len(out_obj_ids) + 3) * 255
        all_mask[out_mask[..., 0] == 255, 0] = hue
        all_mask[out_mask[..., 0] == 255, 2] = 255
    all_mask = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
    frame_overlay = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
    frame_overlay = cv2.cvtColor(frame_overlay, cv2.COLOR_BGR2RGB)
    # cv2.imshow("frame_overlay", frame_overlay)
    # cv2.waitKey(1)

    invalid_region_mask =  ~(np.any(all_mask > 0, axis=-1)) #all_mask Maybe [255,0,0]
    valid_region_frame = copy.deepcopy(frame) #RGB
    valid_region_frame[invalid_region_mask] = 0
    return valid_region_frame

def change_cube_color(cube_path, color):
    # 打开场景
    # 获取立方体的路径
    # 获取立方体的 Prim
    prim = get_prim_at_path(cube_path)
    while not prim.IsValid(): 
        my_world.step(render=True)  # necessary to visualize changes
    # 假设prim是您想要改变颜色的cube的USDA prim对象
    material = UsdShade.Material.Define(prim.GetStage(), prim.GetPath().AppendChild("Material"))
    shader = UsdShade.Shader.Define(prim.GetStage(), material.GetPath().AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(color))  # 设置颜色为红色
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def red_set_cube_position(cube_path, position,table_x_len,table_y_len,cube_distance,cube_x_bias):
    position = np.array(position)
    position[0]=table_x_len*(position[0]+1)+cube_x_bias
    position[1]=-(table_y_len*(position[1]+1)+cube_distance/2)
    position = tuple(position.tolist())
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cube_path)
    while  not prim.IsValid():
        my_world.step(render=True)  # necessary to visualize changes
    xformable = UsdGeom.Xformable(prim)
    # 获取所有的变换操作
    ops = xformable.GetOrderedXformOps()
    # 遍历查找 translate 操作
    found_translate = False
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            # 修改位置值
            op.Set(position)  # 设置为你想的新位置
            found_translate = True
            break

def blue_set_cube_position(cube_path, position,table_x_len,table_y_len,cube_distance,cube_x_bias):
    position = np.array(position)
    position[0]=table_x_len*(position[0]+1)+cube_x_bias
    position[1]=(table_y_len*(position[1]+1)+cube_distance/2)
    position = tuple(position.tolist())
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cube_path)
    while  not prim.IsValid():
        my_world.step(render=True)  # necessary to visualize changes
    xformable = UsdGeom.Xformable(prim)
    # 获取所有的变换操作
    ops = xformable.GetOrderedXformOps()
    # 遍历查找 translate 操作
    found_translate = False
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            # 修改位置值
            op.Set(position)  # 设置为你想的新位置
            found_translate = True
            break

def red_set_cube_position_middle(cube_path, position,table_x_len,table_y_len,cube_distance,cube_x_bias):
    position = np.array(position)
    position[0]=cube_x_bias-(table_x_len*(position[0]+1)+cube_distance/2)
    position[1]=(table_y_len*(position[1]+1))
    position = tuple(position.tolist())
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cube_path)
    while  not prim.IsValid():
        my_world.step(render=True)  # necessary to visualize changes
    xformable = UsdGeom.Xformable(prim)
    # 获取所有的变换操作
    ops = xformable.GetOrderedXformOps()
    # 遍历查找 translate 操作
    found_translate = False
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            # 修改位置值
            op.Set(position)  # 设置为你想的新位置
            found_translate = True
            break

def blue_set_cube_position_middle(cube_path, position,table_x_len,table_y_len,cube_distance,cube_x_bias):
    position = np.array(position)
    position[0]=cube_x_bias+(table_x_len*(position[0]+1)+cube_distance/2)
    position[1]=(table_y_len*(position[1]+1))
    position = tuple(position.tolist())
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cube_path)
    while  not prim.IsValid():
        my_world.step(render=True)  # necessary to visualize changes
    xformable = UsdGeom.Xformable(prim)
    # 获取所有的变换操作
    ops = xformable.GetOrderedXformOps()
    # 遍历查找 translate 操作
    found_translate = False
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            # 修改位置值
            op.Set(position)  # 设置为你想的新位置
            found_translate = True
            break
################ example
# yellow_mug_prim_path = "/World/kitchen_scene_source/SM_Mug_C1"
# mug_target_translation = (0.06523,-0.82723,0.27334) #LocalPosition
# mug_target_eular_degrees = (0,0,96.29427) #LocalPosition
# mug_world_position,mug_world_eular_degrees = get_usd_pose(yellow_mug_prim_path)
# set_usd_pose(yellow_mug_prim_path,mug_target_translation,mug_target_eular_degrees)

def get_usd_pose(prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    world_transform_matrix = omni.usd.get_world_transform_matrix(prim)
    world_position = world_transform_matrix[3][:3]
    world_euler_degrees = np.array([0,0,0])
    world_orientation_wxyz = np.array([1,0,0,0])
    prim = stage.GetPrimAtPath(prim_path)
    xform_translate = prim.GetAttribute('xformOp:translate')
    if xform_translate:
        world_position = xform_translate.Get()
    xform_orientation = prim.GetAttribute('xformOp:orient')
    if xform_orientation:
        world_orientation_wxyz_object = xform_orientation.Get()
        real_part = np.array([world_orientation_wxyz_object.real])
        imaginary = np.array(world_orientation_wxyz_object.GetImaginary()) 
        world_orientation_wxyz=np.concatenate((real_part,imaginary))
        world_orientation_xyzw = world_orientation_wxyz[[1,2,3,0]]
        r = R.from_quat(world_orientation_xyzw)
        # 转换为欧拉角 (XYZ顺序，单位为弧度)
        world_euler_degrees = r.as_euler('xyz', degrees=True)  # 如果需要角度单位为度，可以设置 degrees=True
    
    return np.array(world_position),np.array(world_euler_degrees),np.array(world_orientation_wxyz)


def set_usd_pose(prim_path,target_translation,target_eular_degrees):
    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    op_translate = None
    op_rotate = None
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op_translate = op
            continue
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op_rotate = op
            continue
    if op_translate == None:
        op_translate = xformable.AddTranslateOp()  # 如果已存在，不会重复添加，而是返回已有的
    if op_rotate == None:
        op_rotate = xformable.AddOrientOp()  # 如果已存在，不会重复添加，而是返回已有的
    op_translate.Set(target_translation)
    orientation = euler_xyz_degree_to_quaternion_wxyz(target_eular_degrees)
    orientation_gfvec4f = Gf.Quatf(orientation[0],orientation[1],orientation[2],orientation[3])
    op_rotate.Set(orientation_gfvec4f) 

def euler_xyz_degree_to_quaternion_wxyz(euler_xyz_deg):
    """
    将 XYZ 顺序的欧拉角（单位：度）转换为四元数 (w, x, y, z)
    参数:
        euler_xyz_deg: list 或 tuple，包含 [x, y, z] 三个角度（单位：度）
    返回:
        numpy.ndarray: 四元数 [w, x, y, z]
    """
    # 将度数转换为弧度
    angles_rad = np.radians(euler_xyz_deg)
    # 创建 Rotation 对象，指定旋转顺序为 'xyz'
    rotation = R.from_euler('xyz', angles_rad, degrees=False)
    # 转换为四元数 (scipy 返回的是 [x, y, z, w])
    quat_xyzw = rotation.as_quat()
    # 转换为 [w, x, y, z] 格式
    quat_wxyz = np.roll(quat_xyzw, 1)  # 将最后一个元素 w 移到最前面
    # 或者手动重组：
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return quat_wxyz


def get_additional_observation():
    yellow_mug_prim_path = "/World/kitchen_scene_source/SM_Mug_C1"
    mug_cur_world_position,mug_cur_world_eular_degrees,mug_cur_orientation_wxyz = get_usd_pose(yellow_mug_prim_path)
    observations = my_world.get_observations()
    observations['yellow_mug'] = {'position':mug_cur_world_position,'orientation':mug_cur_orientation_wxyz,'eular_xyz':mug_cur_world_eular_degrees}
    return observations


def yellow_mug_set_world_position(cube_path, position,table_x_len,table_y_len,cube_distance,cube_x_bias):
    position = np.array(position)
    position[0]=table_x_len*(position[0]+1)+cube_x_bias
    position[1]=-(table_y_len*(position[1]+1)+cube_distance/2)
    position = tuple(position.tolist())

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cube_path)
    parent_prim = prim.GetParent()
    parent_xformable = UsdGeom.Xformable(parent_prim)
    # 3. 计算父级的世界变换矩阵
    if parent_xformable:
        parent_world_transform = parent_xformable.ComputeLocalToWorldTransform(0)
        parent_world_matrix = Gf.Matrix4d(parent_world_transform)
    else:
        parent_world_matrix = Gf.Matrix4d()  # 根节点，无父级
    # 4. 将目标世界坐标 → 转换为局部坐标
    # 局部坐标 = parent_world_matrix 的逆 × 世界坐标
    inv_parent_world_matrix = parent_world_matrix.GetInverse()
    local_pos_target = inv_parent_world_matrix.Transform(Gf.Vec3d(position))
    while  not prim.IsValid():
        my_world.step(render=True)  # necessary to visualize changes
    xformable = UsdGeom.Xformable(prim)
    # 获取所有的变换操作
    ops = xformable.GetOrderedXformOps()
    # 遍历查找 translate 操作
    found_translate = False
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            # 修改位置值
            op.Set(local_pos_target)  # 设置为你想的新位置
            found_translate = True
            break


def yellow_mug_set_local_position(cube_path, position,table_x_len,table_y_len,cube_distance,cube_x_bias,eular_angle):
    position = np.array(position)
    position[0]=table_x_len*(position[0]+1)+cube_x_bias
    position[1]=-(table_y_len*(position[1]+1)+cube_distance/2)
    position[2]+=0.08
    position = tuple(position.tolist())
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cube_path)
    while  not prim.IsValid():
        my_world.step(render=True)  # necessary to visualize changes
    xformable = UsdGeom.Xformable(prim)
    # 获取所有的变换操作
    op_translate = None
    op_rotate = None
    ops = xformable.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op_translate = op
            continue
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            op_rotate = op
            continue
    if op_translate == None:
        op_translate = xformable.AddTranslateOp()  # 如果已存在，不会重复添加，而是返回已有的
    if op_rotate == None:
        op_rotate = xformable.AddOrientOp()  # 如果已存在，不会重复添加，而是返回已有的
    op_translate.Set(position)
    orientation = euler_xyz_degree_to_quaternion_wxyz(eular_angle)
    orientation_gfvec4f = Gf.Quatf(orientation[0],orientation[1],orientation[2],orientation[3])
    op_rotate.Set(orientation_gfvec4f) 




def task_scene_reset(my_world,my_task,red_cube_prim_path,blue_cube_prim_path,yellow_mug_prim_path,my_franka,franka_init_joints_position):
    my_world.reset()
    #SONG Franka Joint Reset
    my_franka.set_joint_positions(franka_init_joints_position)
    red_cube_repostion = (np.random.uniform(-1, 1),np.random.uniform(-1, 1),cube_init_heigth)
    red_set_cube_position(red_cube_prim_path,red_cube_repostion,table_x_len,table_y_len,cube_distance,cube_x_bias)
    blue_cube_repostion =  (np.random.uniform(-1, 1),np.random.uniform(-1, 1),cube_init_heigth)
    blue_set_cube_position(blue_cube_prim_path,blue_cube_repostion,table_x_len,table_y_len,cube_distance,cube_x_bias)
    yellow_mug_target_eular_degrees = (0,0,90+int(45*np.random.uniform(-1, 1))) #LocalPosition
    # yellow_mug_target_eular_degrees = (0,0,90) #LocalPosition
    yellow_mug_set_local_position(yellow_mug_prim_path,red_cube_repostion,table_x_len,table_y_len,cube_distance,cube_x_bias,yellow_mug_target_eular_degrees)
    for step in range(2*int(initial_steps)):
        my_world.step(render=True)  # necessary to visualize changes
    my_task.cube_list = ["cube","yellow_mug"] # ["cube","cube_1"]
    my_task.cube_in_hand=None

    

def point_cloud_filter(points):
    WORK_SPACE = [
        [0.35, 0.7],
        [-0.4, 0.4],
        [0, 0.8]
    ]
     # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]
    return points


#--------------------SONG----------------------
def song_min_max_normalize_points(points_src_data):
    min_array = np.expand_dims(points_src_data.min(axis=-2),-2)
    min_array = np.repeat(min_array,repeats=1024,axis=-2)
    max_array = np.expand_dims(points_src_data.max(axis=-2),-2)
    max_array = np.repeat(max_array,repeats=1024,axis=-2)
    points_normalized = (points_src_data - min_array) / (max_array - min_array)
    return torch.Tensor(points_normalized)


def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def collection_data_update(my_world,camera_overhead,camera_hand,out_overhead,out_hand,sim_data_collection,observations):
    camera_overhead_data_dict,camera_hand_data_dict = get_double_all(camera_overhead,camera_hand)
    img_overhead_rgb,img_hand_rgb = camera_overhead_data_dict['img_rgb'],camera_hand_data_dict['img_rgb']
    img_overhead = cv2.cvtColor(img_overhead_rgb,cv2.COLOR_RGB2BGR)
    img_hand = cv2.cvtColor(img_hand_rgb,cv2.COLOR_RGB2BGR)
    out_overhead.write(img_overhead)
    out_hand.write(img_hand)
    sim_data_collection.cur_pos = observations['my_franka']['joint_positions'][:7]

    gripper_width =(observations["my_franka"]["joint_positions"][-2]+observations["my_franka"]["joint_positions"][-1])
    #Producer
    cur_eff_pose_position = observations["my_franka"]["end_effector_position"]
    cur_eff_pose_orientation =  observations["my_franka"]["end_effector_orientation"] #wxyz
    # Create a Rotation object
    cur_eff_pose_orientation_xyzw = cur_eff_pose_orientation[[1,2,3,0]]
    rotation = R.from_quat(cur_eff_pose_orientation_xyzw)#( x, y, z, w)
    cur_eff_pose_orientation_eular_zyx = rotation.as_euler('zyx', degrees=False)  #zyx  机械臂以zyx为轴
    cur_eff_pose_orientation_eular_xyz = cur_eff_pose_orientation_eular_zyx[[2,1,0]]*180/3.14 #zyx->xyz   #为与IsaacsimGUI统一，转为XYZ,方便理解
    # 解码
    # cur_eff_pose_orientation_eular_zyx = cur_eff_pose_orientation_eular_zyx
    # rotation = R.from_euler('zyx', cur_eff_pose_orientation_eular_zyx, degrees=False)
    # quat_xyzw = rotation.as_quat()
    # quat_wxyz = quat_xyzw[[3,0,1,2]]
    sim_data_collection.pose_eular = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)#由于是zxy轴，不能用xyz
    sim_data_collection.overhead_cloud_rgb = camera_overhead_data_dict['cloud_rgb']
    sim_data_collection.hand_cloud_rgb = camera_hand_data_dict['cloud_rgb']
    sim_data_collection.overhead_frame = img_overhead_rgb #overhead_frame
    sim_data_collection.hand_frame = img_hand_rgb #hand_frame
    sim_data_collection.eff_angular = np.array([gripper_width]) #xyz/xyzw
    
####################仿射变换##################
def normalize_2d_points(pts):
    """归一化2D点坐标"""
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    scale = np.sqrt(2) / std
    T = np.array([
        [scale[0], 0, -scale[0]*mean[0]],
        [0, scale[1], -scale[1]*mean[1]],
        [0, 0, 1]
    ])
    pts_hom = np.column_stack([pts, np.ones(pts.shape[0])])
    pts_norm = (T @ pts_hom.T).T
    return pts_norm[:, :2], T

def normalize_3d_points(pts):
    """归一化3D点坐标"""
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    scale = np.sqrt(3) / std
    T = np.array([
        [scale[0], 0, 0, -scale[0]*mean[0]],
        [0, scale[1], 0, -scale[1]*mean[1]],
        [0, 0, scale[2], -scale[2]*mean[2]],
        [0, 0, 0, 1]
    ])
    pts_hom = np.column_stack([pts, np.ones(pts.shape[0])])
    pts_norm = (T @ pts_hom.T).T
    return pts_norm[:, :3], T

def compute_projection_matrix(pts3d, pts2d):
    """
    通过3D-2D点对计算投影矩阵P（3x4）
    pts3d: 世界坐标系下的3D点，形状为(n, 3)
    pts2d: 图像上的2D像素坐标，形状为(n, 2)
    返回: 3x4投影矩阵P
    """
    n = pts3d.shape[0]
    A = []
    
    for i in range(n):
        X, Y, Z = pts3d[i, :3]
        u, v = pts2d[i, :2]
        
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    
    A = np.array(A)
    # 使用SVD求解A的最小特征值对应的特征向量
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    
    return P

def compute_projection_matrix_improved(pts3d, pts2d):
    """改进的投影矩阵计算（包含归一化）"""
    # 归一化2D点
    pts2d_norm, T2d = normalize_2d_points(pts2d)
    
    # 归一化3D点
    pts3d_norm, T3d = normalize_3d_points(pts3d)
    
    n = pts3d.shape[0]
    A = []
    
    for i in range(n):
        X, Y, Z = pts3d_norm[i, :3]
        u, v = pts2d_norm[i, :2]
        
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])
    
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    P_norm = Vt[-1].reshape(3, 4)
    
    # 反归一化
    P = np.linalg.inv(T2d) @ P_norm @ T3d
    
    return P

def project_3d_to_2d(P, point3d):
    """
    使用投影矩阵P将3D点投影到2D像素坐标
    point3d: 3D坐标，形状为(3,)
    返回: 2D像素坐标(u, v)
    """
    # 转换为齐次坐标
    point3d_hom = np.append(point3d, 1)
    
    # 投影
    point2d_hom = P @ point3d_hom
    # 转换为非齐次坐标
    u = point2d_hom[0] / point2d_hom[2]
    v = point2d_hom[1] / point2d_hom[2]
    
    return int(u), int(v)



############1.ADD DESK USD
add_reference_to_stage(usd_path="/home/liusong/下载/download_usd/Collected_KitchenRoom/KitchenRoom.usd",
                        prim_path="/World/kitchen_scene_source")
my_world.scene.add(XFormPrim(prim_path="/World/kitchen_scene_source" ,
                                        name="fancy_robot",
                                        scale=np.array([1.0, 1.0, 1.0]),
                                        position=np.array([0, 0, 0.01])))
observations = get_additional_observation()
my_task.cube_list[1] = "yellow_mug"########LIUSONG


############2.CUBE RESET
table_x_len = 0.1
table_y_len=0.1
cube_distance = 0.2
cube_x_bias = 0.35  #y:(+/-)0.1~0.3 x:0.35~0.55
red_cube_prim_path = "/World/Cube_1"  # 替换为你的立方体路径
red_cube_prim_color = (1,0,0)
blue_cube_prim_path = "/World/Cube"  # 替换为你的立方体路径
blue_cube_prim_color = (0,0,1)
yellow_mug_prim_path = "/World/kitchen_scene_source/SM_Mug_C1"
blue_cube_repostion =  (0,np.random.uniform(-1, 1),cube_init_heigth)
red_cube_repostion = (0,np.random.uniform(-1, 1),cube_init_heigth) #Useless
change_cube_color(red_cube_prim_path,red_cube_prim_color)
change_cube_color(blue_cube_prim_path,blue_cube_prim_color)
red_set_cube_position(red_cube_prim_path,red_cube_repostion,table_x_len,table_y_len,cube_distance,cube_x_bias)
blue_set_cube_position(blue_cube_prim_path,blue_cube_repostion,table_x_len,table_y_len,cube_distance,cube_x_bias)
yellow_mug_target_eular_degrees = (0,0,90+int(45*np.random.uniform(-1, 1))) #LocalPosition
# yellow_mug_target_eular_degrees = (0,0,90) #LocalPosition
yellow_mug_set_local_position(yellow_mug_prim_path,red_cube_repostion,table_x_len,table_y_len,cube_distance,cube_x_bias,yellow_mug_target_eular_degrees)
# mug_world_position,mug_world_eular_degrees = get_usd_pose(yellow_mug_prim_path)
# set_usd_pose(yellow_mug_prim_path,mug_target_translation,mug_target_eular_degrees)




############3.PREPARE THE PARAMETERS
REAL_RANDOM = False 
IMAGE_STANDARD = False
############4.READY FOR SAM2
USE_CAMERA = "overhead"#hand/overhead/all
USE_SAM2 = True
USE_MARK = "cloud_rgb"#arrowmark/dotmark/overlaymarkn
############5.Model Mode
inference_model = "act"#dp3/smolvla/act/dp
action_mode = "absolute"#relative/absolute
action_type = "joint"#joint/pose/wait

if inference_model=="smolvla":
    model_inference = SmolVLA_ModelInference()
elif inference_model == "dp3":
    model_inference = DP3_ModelInference()
elif inference_model == "act":
    model_inference = ACT_ModelInference()
elif inference_model == "dp":
    model_inference = DP_ModelInference()
gripper_width_thresh = 0.035

if USE_MARK:
    #When Use to inference, mark point with the gripper center
    gripper_front_axis = 2
    gripper_front_bias = 0.1
    H_eff2mark = np.eye(4)
    H_eff2mark[gripper_front_axis,3] = gripper_front_bias   
    H_eff2mark_arrow = copy.deepcopy(H_eff2mark)
    H_eff2mark_arrow[gripper_front_axis,3] -= gripper_front_bias/2

    load_mark_points_np = np.load("/home/liusong/ProgramFiles/IssacSim/Tasks_Wokspace/dataset/test3/camera/mark_points.npy")
    load_mark_pixel_pts_np = np.load("/home/liusong/ProgramFiles/IssacSim/Tasks_Wokspace/dataset/test3/camera/mark_pixel_pts.npy")
    # 估计投影矩阵
    P_estimated = compute_projection_matrix_improved(load_mark_points_np, load_mark_pixel_pts_np)

if USE_SAM2:
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    import time
    import h5py
    sys.path.append("/home/liusong/ProgramFiles/SAM2/sam2")
    from sam2.build_sam import build_sam2_camera_predictor

    sam2_checkpoint = "/home/liusong/ProgramFiles/SAM2/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    frame = cv2.imread("/home/liusong/ProgramFiles/SAM2/sam2/notebooks/images/yellowmug-img_overhead.png")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    width, height = frame.shape[:2][::-1]
    # cv2.imshow("overhead_frame", frame)
    # cv2.waitKey(0)
    predictor.load_first_frame(frame)
    if_init = True
    ann_frame_idx = 0  # the frame index we interact with
    # First annotation
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    ##! add points, `1` means positive click and `0` means negative click
    points = np.array([[496, 266],[507,269],[485,298],[491,336]], dtype=np.float32)
    labels = np.array([1,1,1,1], dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    )
    ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
    points = np.array([[227, 266],[266,305]], dtype=np.float32)
    labels = np.array([1,1], dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    )
    # frame = cv2.imread("/home/liusong/ProgramFiles/SAM2/sam2/notebooks/images/yellowmug-img_hand.png")
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.imshow("hand_frame", frame)
    # cv2.waitKey(0)
    # out_obj_ids, out_mask_logits = predictor.track(frame)
    # predictor.add_conditioning_frame(frame)
    # ann_frame_idx = 1  # the frame index we interact with
    # # First annotation
    # ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    # ##! add points, `1` means positive click and `0` means negative click
    # points = np.array([[321, 206],[318,297]], dtype=np.float32)
    # labels = np.array([1,1], dtype=np.int32)
    # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
    #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    # )
    # ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
    # points = np.array([[215, 251]], dtype=np.float32)
    # labels = np.array([1], dtype=np.int32)
    # _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
    #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    # )



#SONG
my_world.reset()
franka_init_joints_position = np.array([0.3936, -1.4529, -0.1870, -2.7861, -0.4544,  2.0824,  1.15228, 0.04, 0.04])
# my_franka.set_joint_positions(franka_init_joints_position)
task_scene_reset(my_world,my_task,red_cube_prim_path,blue_cube_prim_path,yellow_mug_prim_path,my_franka,franka_init_joints_position)
for _ in range(wait_steps):
    my_world.step(render=True)

while simulation_app.is_running():

    my_world.step(render=True)  # necessary to visualize changes

    if not my_controller.init_curobo:
        my_controller.reset(ignore_substring, robot_prim_path)

    step_index = my_world.current_time_step_index
    observations = get_additional_observation()
    sim_js = my_franka.get_joints_state()


    camera_overhead_data_dict,camera_hand_data_dict = get_double_all(camera_overhead,camera_hand)
    img_overhead_rgb,img_hand_rgb = camera_overhead_data_dict['img_rgb'],camera_hand_data_dict['img_rgb']
    overhead_cloud_rgb,hand_cloud_rgb = camera_overhead_data_dict['cloud_rgb'],camera_hand_data_dict['cloud_rgb']
    
    if IMAGE_STANDARD:
        img_hand_rgb = cv2.resize(img_hand_rgb,(224,224),interpolation=cv2.INTER_AREA)
        img_overhead_rgb = cv2.resize(img_overhead_rgb,(224,224),interpolation=cv2.INTER_AREA)

    #################SAM2##############
    if USE_SAM2:
        img_overhead_rgb = get_sam2_masked_frame(img_overhead_rgb,predictor)
        img_hand_rgb = get_sam2_masked_frame(img_hand_rgb,predictor)
        if USE_MARK:
            cur_eff_pose_position = observations["my_franka"]["end_effector_position"]
            cur_eff_pose_orientation =  observations["my_franka"]["end_effector_orientation"] #wxyz
            rotation = R.from_quat(cur_eff_pose_orientation)#(w, x, y, z)
            euler_angles = rotation.as_euler('xyz', degrees=False)  # 'xyz' specifies the rotation order
            cur_eff_pose_eular_angles =  euler_angles #wxyz
            data = np.concatenate((cur_eff_pose_position,cur_eff_pose_eular_angles))

            H_world2eff = np.eye(4)
            data_tranlation=data[:3]
            H_world2eff[:3,3] = data_tranlation
            #OLD VERSION  data[3:] collected with wrong way
            data_eular_radin= data[3:]
            data_rotation = R.from_euler('xyz', data_eular_radin, degrees=False)
            data_orientation_wxyz = data_rotation.as_quat()
            data_orientation_xyzw = data_orientation_wxyz[[1,2,3,0]]
            correct_rotation = R.from_quat(data_orientation_xyzw).as_matrix()#(w, x, y, z)
            H_world2eff[:3,:3] = correct_rotation
            #Method1
            H_world2mark = np.dot(H_world2eff,H_eff2mark)
            H_world2mark_arrow = np.dot(H_world2eff,H_eff2mark_arrow)
            mark_point = H_world2mark[:3,3]
            mark_arrow_point = H_world2mark_arrow[:3,3]

            world_base = get_usd_pose("/World/Franka")[0]
            mark_point+=world_base
            mark_arrow_point+=world_base
            projected_2d = project_3d_to_2d(P_estimated, mark_point)
            mark_arrow_projected_2d = project_3d_to_2d(P_estimated, mark_arrow_point)
            
            img_overhead_bgr = cv2.cvtColor(img_overhead_rgb,cv2.COLOR_BGR2RGB)
            img_hand_bgr = cv2.cvtColor(img_hand_rgb,cv2.COLOR_BGR2RGB)
            if USE_MARK=="cloud_rgb":
                gripper_width = (observations["my_franka"]["joint_positions"][-2]+observations["my_franka"]["joint_positions"][-1])
                gripper_width = 1 if gripper_width>0.05 else 0
                    
                cur_eff_pose_position = observations["my_franka"]["end_effector_position"]
                cur_eff_pose_orientation =  observations["my_franka"]["end_effector_orientation"] #wxyz
                cur_eff_pose_orientation_xyzw = cur_eff_pose_orientation[[1,2,3,0]]
                rotation = R.from_quat(cur_eff_pose_orientation_xyzw)#( x, y, z, w)
                cur_eff_pose_orientation_eular_zyx = rotation.as_euler('zyx', degrees=False)  #zyx  机械臂以zyx为轴
                cur_eff_pose_orientation_eular_xyz = cur_eff_pose_orientation_eular_zyx[[2,1,0]]*180/3.14 #zyx->xyz   #为与IsaacsimGUI统一，转为XYZ,方便理解
                eff_pose_xyz_eular = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)#由于是zxy轴，不能用xyz
                # 添加夹爪点云
                gripper_mesh = VisualizationUtils.update_gripper(gripper_width, eff_pose_xyz_eular)
                gripper_pcd = gripper_mesh.sample_points_uniformly(number_of_points=2000)
                gripper_cloud_rgb = GeometryUtils.pcd_to_cloud_rgb(gripper_pcd)
                stage1_result = {
                    'subtask': None,
                    'obj_sets': None,
                    'trans_range': None,
                    'move_task_dict': None,
                    'subtask_cut_frame_with_end': None,
                    'obj_seg_dict': None,
                    'obj_initial_center': None,
                    'eff_angular': None,
                    'eff_trajectory': None,
                    'background_cloud_rgb': None,
                    'task_name': None
                }
                stage2editing = Stage2Editing(stage1_result)
                gripper_rgb = stage2editing.generate_cloud_rgb_to_image(gripper_cloud_rgb)
                frame_overlay_rgb = cv2.addWeighted(gripper_rgb, 1, img_overhead_rgb, 0.5, 0)
                frame_overlay_bgr = cv2.cvtColor(frame_overlay_rgb, cv2.COLOR_BGR2RGB)
                img_overhead_rgb = frame_overlay_rgb
                img_overhead_bgr = frame_overlay_bgr
            elif USE_MARK=="dotmark":
                cv2.circle(img_overhead_bgr,projected_2d,2,(0,255,0),5,-1)
                img_overhead_rgb = cv2.cvtColor(img_overhead_bgr,cv2.COLOR_BGR2RGB)
            elif USE_MARK=="arrowmark":
                cv2.arrowedLine(img_overhead_bgr,mark_arrow_projected_2d,projected_2d,(0,255,0),5,-1,tipLength=0.5)
                img_overhead_rgb = cv2.cvtColor(img_overhead_bgr,cv2.COLOR_BGR2RGB)
            elif USE_MARK=="overlaymark":
                cv2.arrowedLine(img_overhead_bgr,mark_arrow_projected_2d,projected_2d,(0,255,0),5,-1,tipLength=0.5)
                img_overhead_rgb = cv2.cvtColor(img_overhead_bgr,cv2.COLOR_BGR2RGB)
                img_overhead_bgr_src = cv2.cvtColor(camera_overhead_data_dict['img_rgb'],cv2.COLOR_RGB2BGR)
                img_overhead_bgr = cv2.addWeighted(img_overhead_bgr_src,0.5,img_overhead_bgr,0.5,0)
                img_overhead_rgb = cv2.cvtColor(img_overhead_bgr,cv2.COLOR_BGR2RGB)

                img_hand_bgr_src = cv2.cvtColor(camera_hand_data_dict['img_rgb'],cv2.COLOR_RGB2BGR)
                marked_input_image_bgr = copy.deepcopy(img_hand_bgr_src)
                height,width = marked_input_image_bgr.shape[:2]
                mark_arrow_end =( int(width/2),int(height/2)+50)
                mark_arrow_start =( mark_arrow_end[0] ,mark_arrow_end[1]+ int(height/7))
                cv2.arrowedLine(img_hand_bgr,mark_arrow_start,mark_arrow_end,(0,255,0),10,-1,tipLength=0.5)
                frame_overlay_bgr = cv2.addWeighted(img_hand_bgr, 1, img_hand_bgr_src, 0.5, 0)
                frame_overlay_rgb = cv2.cvtColor(frame_overlay_bgr, cv2.COLOR_BGR2RGB)
                img_hand_rgb = frame_overlay_rgb
                img_hand_bgr = frame_overlay_bgr

                # cv2.imshow("img_overhead_bgr",img_overhead_bgr)
                # cv2.imshow("img_hand_bgr",img_hand_bgr)
                # cv2.waitKey(0)ze
            # img_overhead_bgr_src = cv2.cvtColor(camera_overhead_data_dict['img_rgb'],cv2.COLOR_RGB2BGR)
            # mixed_image = cv2.addWeighted(img_overhead_bgr_src,0.5,img_overhead_bgr,0.5,0)
            # cv2.imshow("mixed_image",mixed_image)
            # cv2.waitKey(0)

    if USE_CAMERA=="hand":
        img_overhead_rgb[img_overhead_rgb>0] = 0
    elif USE_CAMERA=="overhead":
        img_hand_rgb[img_hand_rgb>0] = 0
    


    cur_model_observation = {'joint_1':None,'joint_2':None,'joint_3': None,'joint_4':None,'joint_5':None,'joint_6': None,'joint_7':None,'gripper_width':None,
                            'overhead':img_overhead_rgb,"hand":img_hand_rgb,"point_cloud":{},"pose_eular":{}}
    cur_observation_key_list = list(cur_model_observation.keys())
    for idx, value in enumerate(sim_js.positions[:7]):
        cur_model_observation[cur_observation_key_list[idx]] =180*value/3.14
    cur_model_observation['gripper_width'] = 0.5*(sim_js.positions[-1]+sim_js.positions[-2])*1000


    overhead_cloud_rgb_filter = point_cloud_filter(overhead_cloud_rgb)  ###################DOWNSAMPLE
    points_xyz = overhead_cloud_rgb_filter[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz)
    sample_indices = sample_indices.cpu()
    points_rgb = overhead_cloud_rgb_filter[sample_indices, 3:][0]

    # points_xyz *=0.1
    # points_xyz *=np.random.uniform(0.1,3)
    # points_xyz = song_min_max_normalize_points(points_xyz)

    points = np.hstack((points_xyz, points_rgb))
    cur_model_observation["point_cloud"] = points

    cur_eff_pose_position = observations["my_franka"]["end_effector_position"]
    # Create a Rotation object
    cur_eff_pose_orientation =  observations["my_franka"]["end_effector_orientation"] #wxyz
    cur_eff_pose_orientation_xyzw = cur_eff_pose_orientation[[1,2,3,0]]
    rotation = R.from_quat(cur_eff_pose_orientation_xyzw)#( x, y, z, w)
    euler_angles = rotation.as_euler('zyx', degrees=False)  #zyx  机械臂以zyx为轴
    cur_eff_pose_orientation_eular_zyx =  euler_angles #wxyz
    cur_model_observation['pose_eular'] = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)

    if inference_model=="dp":
        cur_observation_key_list = list(cur_model_observation.keys())
        for idx, value in enumerate(sim_js.positions[:6]):
            cur_model_observation[cur_observation_key_list[idx]] =cur_model_observation['pose_eular'][idx]
        gripper_width =cur_model_observation['gripper_width']/1000  # 2finger /mm——>middle position /m
        gripper_width = 1 if gripper_width>0.035 else 0
        cur_model_observation['gripper_width'] = gripper_width
    
    if inference_model=="smolvla" and action_type == "pose":
        cur_observation_key_list = list(cur_model_observation.keys())
        for idx, value in enumerate(sim_js.positions[:6]):
            cur_model_observation[cur_observation_key_list[idx]] =cur_model_observation['pose_eular'][idx]
        gripper_width =cur_model_observation['gripper_width']/1000  # 2finger /mm——>middle position /m
        gripper_width = 1 if gripper_width>0.035 else 0
        cur_model_observation['gripper_width'] = gripper_width


    if inference_model=="act" and action_type == "pose":
        cur_observation_key_list = list(cur_model_observation.keys())
        for idx, value in enumerate(sim_js.positions[:6]):
            cur_model_observation[cur_observation_key_list[idx]] =cur_model_observation['pose_eular'][idx]
        gripper_width =cur_model_observation['gripper_width']/1000  # 2finger /mm——>middle position /m
        gripper_width = 1 if gripper_width>0.035 else 0
        cur_model_observation['gripper_width'] = gripper_width

    cv2.imshow("img_overhead_rgb",img_overhead_bgr)
    cv2.waitKey(1)

    inference_action =  model_inference.single_inference(cur_model_observation)

    observations = my_world.get_observations()
    observations["my_franka"]["end_effector_position"]
    

    if inference_action is None:
        continue

    if action_type == "joint":
        if action_mode == "relative":
            if inference_model=="dp3":
                inference_7dof_positions =sim_js.positions[:7]+ np.array(3.14*inference_action[:7]/180)
                inference_gripper_width = 0.5*(sim_js.positions[-1]+sim_js.positions[-2])+inference_action[-1]/1000
            elif inference_model=="smolvla":  
                inference_7dof_positions =sim_js.positions[:7]+ np.array(3.14*inference_action[:7]/180)
                inference_gripper_width = 0.5*(sim_js.positions[-1]+sim_js.positions[-2])+inference_action[-1]/1000
            elif inference_model=="act":  
                inference_7dof_positions =sim_js.positions[:7]+ np.array(3.14*inference_action[:7]/180)
                inference_gripper_width = 0.5*(sim_js.positions[-1]+sim_js.positions[-2])+inference_action[-1]/1000
                if inference_action[-1]<-10:
                    print("yes")
        elif action_mode=="absolute":
            inference_7dof_positions = np.array(3.14*inference_action[:7]/180)
            inference_gripper_width = inference_action[-1]/1000

            
        my_controller.idx_list = [i for i in range(len(my_controller.cmd_js_names))] 
        # get full dof state
        art_action = ArticulationAction(
            inference_7dof_positions,
            inference_7dof_positions * 0.0,
            joint_indices=my_controller.idx_list,
        )
        articulation_controller.apply_action(art_action)

    elif action_type=="pose":
        if action_mode == "relative":
            if inference_model=="dp":
                target_pose_position =cur_eff_pose_position+ np.array(inference_action[:3])
            elif inference_model=="dp3":
                target_pose_position =cur_eff_pose_position+ np.array(inference_action[:3])
            elif inference_model=="dp3":
                target_pose_position =cur_eff_pose_position+ np.array(inference_action[:3])
            elif inference_model=="smolvla":
                target_pose_position =cur_eff_pose_position+ np.array(inference_action[:3])
            elif inference_model=="act":
                target_pose_position =cur_eff_pose_position+ np.array(inference_action[:3])

                # target_pose_orientation_eular_zyx = cur_eff_pose_orientation_eular_zyx + np.array(inference_action[3:6])
                # target_pose_orientation_xyzw = R.from_euler('zyx', target_pose_orientation_eular_zyx).as_quat() 
                # target_pose_orientation_wxyz = target_pose_orientation_xyzw[[3,0,1,2]]
                # my_controller.ee_orientation_goal = target_pose_orientation_wxyz


            if inference_action[-1] < -0.5:
                inference_gripper_width=0.022
            elif inference_action[-1] > 0.5:
                inference_gripper_width=0.040
            else:
                inference_gripper_width = 0.5*(sim_js.positions[-1]+sim_js.positions[-2])
        elif action_mode == "absolute":
            target_pose_position =np.array(inference_action[:3])

            if inference_action[-1] < 0.5:
                inference_gripper_width=0.022
            elif inference_action[-1] > 0.5:
                inference_gripper_width=0.040
            else:
                inference_gripper_width = 0.5*(sim_js.positions[-1]+sim_js.positions[-2])
        
            # target_pose_orientation_eular_zyx = cur_eff_pose_orientation_eular_zyx + np.array(inference_action[3:6])
            # target_pose_orientation_xyzw = R.from_euler('zyx', target_pose_orientation_eular_zyx).as_quat() 
            # target_pose_orientation_wxyz = target_pose_orientation_xyzw[[3,0,1,2]]
            # my_controller.ee_orientation_goal = target_pose_orientation_wxyz


        my_controller.my_task.target_position = target_pose_position
        
        ee_translation_goal = my_controller.my_task.target_position
        ee_orientation_goal = my_controller.ee_orientation_goal
        # compute curobo solution:
        result = my_controller.plan(ee_translation_goal, ee_orientation_goal, sim_js, my_franka.dof_names)
        succ = result.success.item()
        if succ:
            cmd_plan = result.get_interpolated_plan()
            my_controller.idx_list = [i for i in range(len(my_controller.cmd_js_names))]
            my_controller.cmd_plan = cmd_plan.get_ordered_joint_state(my_controller.cmd_js_names)
        else:
            carb.log_warn("Plan did not converge to a solution.")
            continue
        cmd_state = my_controller.cmd_plan[-1]
        # get full dof state
        art_action = ArticulationAction(
            cmd_state.position.cpu().numpy(),
            cmd_state.velocity.cpu().numpy() * 0.0,
            joint_indices=my_controller.idx_list,
        )
        # art_action = my_controller.forward(sim_js, my_franka.dof_names)
        if art_action is not None:
            articulation_controller.apply_action(art_action)
            my_world.step(render=True)  # necessary to visualize changes
            my_world.step(render=True)  # necessary to visualize changes
            my_world.step(render=True)  # necessary to visualize changes
    elif action_type == "wait":
        inference_gripper_width = 0.04
        my_world.step(render=True)  # necessary to visualize changes
        

    if inference_gripper_width<gripper_width_thresh: #0.04
        my_franka.gripper.close()
    else:
        my_franka.gripper.open()

    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        pressed_key = line.strip()
        if line:
            print(f"You pressed: {pressed_key}")
        if pressed_key == 'v':
            action_type="wait"
            predict_trajectory_visualize(model_inference,inference_action,my_controller,my_franka,overhead_cloud_rgb)
            model_inference.policy.reset() #Policy Reset

        if pressed_key == 'c':
            gripper_width_thresh=1
        if pressed_key == 'o':
            gripper_width_thresh=0
        if pressed_key == 'n':
            if inference_model == "dp3":
                model_inference.policy_reset()

            model_inference.policy.reset() #Policy Reset
            model_inference.cur_model_observation = None
            model_inference.inference_action = None
            gripper_width_thresh =0.035
            my_world.reset()
            my_franka.set_joint_positions(franka_init_joints_position)
            ############0.Random Seed!!!!!!!!!!!!!
            if REAL_RANDOM == True:
                np.random.seed(int(time.time()))
            red_cube_repostion = (np.random.uniform(-1, 1),np.random.uniform(-1, 1),cube_init_heigth)
            red_set_cube_position(red_cube_prim_path,red_cube_repostion,table_x_len,table_y_len,cube_distance,cube_x_bias)
            blue_cube_repostion =  (np.random.uniform(-1, 1),np.random.uniform(-1, 1),cube_init_heigth)
            blue_set_cube_position(blue_cube_prim_path,blue_cube_repostion,table_x_len,table_y_len,cube_distance,cube_x_bias)
            yellow_mug_target_eular_degrees = (0,0,90+int(45*np.random.uniform(-1, 1))) #LocalPosition
            # yellow_mug_target_eular_degrees = (0,0,90) #LocalPosition
            yellow_mug_set_local_position(yellow_mug_prim_path,red_cube_repostion,table_x_len,table_y_len,cube_distance,cube_x_bias,yellow_mug_target_eular_degrees)
            for i in range(10):
                my_world.step(render=True)  # necessary to visualize changes
        print("gripper_width_thresh = ",gripper_width_thresh)

simulation_app.close()
