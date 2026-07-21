import sys
import torch
import numpy as np 
import cv2
import copy

sys.path.append("/home/liusong/ProgramFiles/REAP/StageGen")
from stagegen.geometry_utils import GeometryUtils
from stagegen.projection_utils import ProjectionUtils
from stagegen.visualization_utils import VisualizationUtils
from stagegen.stage2_editing import Stage2Editing
from stagegen.stage1_segmentation import Stage1Segmentation


sys.path.append("/home/liusong/ProgramFiles/REAP/SAVG/savg/")
sys.path.append("/home/liusong/ProgramFiles/REAP/SAVG/savg/models")
from models.pose_act_cvae import PoseACTCVAE
from utils.rot6d import pose9_to_homo
from preprocess_hdf5 import load_cloud_from_group,maybe_bytes_to_str,remove_outliers_fast
from eval import single_data_inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append("/home/liusong/ProgramFiles/Huggingface/lerobot/src/lerobot/scripts/")
from smolvla_model_inference import SmolVLA_ModelInference, pose9_to_traj6



import time
import math
import numpy as np
import cv2
import sys, os
import threading
import select
from collections import deque
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Franka3, Pose
from Motion_Planning.Manipulation.Skill_Franka3 import skill_database
from Sensor.Camera_Realsense import Camera_Realsense
from Dataset.scripts.data_collection import RealDataCollection
from franky import Affine
import pytorch3d.ops as torch3d_ops



CONST_POINTS_NUM=640*480
POLICY_EXEC_ACTION_STEPS = 20
POLICY_MAX_LINEAR_VEL = 0.35
POLICY_MAX_ANGULAR_VEL = math.radians(90)

POLICY_REPLAN_INITIAL_WINDOW = 8
POLICY_REPLAN_MIN_WINDOW = 1
POLICY_REPLAN_MAX_FAILURES = 2
POLICY_REPLAN_POSITION_TOL = 0.001
POLICY_REPLAN_ROTATION_TOL = math.radians(1)
POLICY_RECOVER_SETTLE_TIME = 0.18

POLICY_ACTION_DT = 0.08
POLICY_USE_TARGET_VELOCITIES = False
POLICY_REPLAN_PATH_TOL = 0.0
POLICY_VIRTUAL_GRIPPER_LEN = 0.02
POLICY_VIRTUAL_GRIPPER_LOCAL_OFFSET = np.array([0.0, 0.0, -POLICY_VIRTUAL_GRIPPER_LEN], dtype=np.float32)
VIRTUAL_GRIPPER_BAIS = 0.06

GRIPPER_CLOSE_THRESHOLD = 0.04 #0.03
GRIPPER_OPEN_THRESHOLD = 0.075

TERMINAL_INPUT_LOCK = threading.Lock()
POLICY_MANUAL_GRIPPER_KEYS = {"c", "o"}
POLICY_ROLLBACK_KEY = "r"
POLICY_ROLLBACK_CHUNKS = 2
POLICY_MANUAL_CONTROL_KEYS = POLICY_MANUAL_GRIPPER_KEYS | {POLICY_ROLLBACK_KEY}
POLICY_MANUAL_CONTROL_KEY_QUEUE = deque()
POLICY_MANUAL_CONTROL_KEY_LOCK = threading.Lock()


def poll_terminal_key(timeout=0.0):
    if not TERMINAL_INPUT_LOCK.acquire(blocking=False):
        return None
    try:
        if sys.stdin in select.select([sys.stdin], [], [], timeout)[0]:
            line = sys.stdin.readline()
            pressed_key = line.strip()
            if line:
                print(f"You pressed: {pressed_key}")
            return pressed_key if pressed_key else None
        return None
    finally:
        TERMINAL_INPUT_LOCK.release()


def queue_policy_control_key(pressed_key):
    if pressed_key not in POLICY_MANUAL_CONTROL_KEYS:
        return False
    with POLICY_MANUAL_CONTROL_KEY_LOCK:
        POLICY_MANUAL_CONTROL_KEY_QUEUE.append(pressed_key)
    return True


def pop_policy_control_key():
    with POLICY_MANUAL_CONTROL_KEY_LOCK:
        if POLICY_ROLLBACK_KEY in POLICY_MANUAL_CONTROL_KEY_QUEUE:
            POLICY_MANUAL_CONTROL_KEY_QUEUE.remove(POLICY_ROLLBACK_KEY)
            return POLICY_ROLLBACK_KEY
        if POLICY_MANUAL_CONTROL_KEY_QUEUE:
            return POLICY_MANUAL_CONTROL_KEY_QUEUE.popleft()
    return None


def pop_policy_rollback_key():
    with POLICY_MANUAL_CONTROL_KEY_LOCK:
        if POLICY_ROLLBACK_KEY in POLICY_MANUAL_CONTROL_KEY_QUEUE:
            POLICY_MANUAL_CONTROL_KEY_QUEUE.remove(POLICY_ROLLBACK_KEY)
            return POLICY_ROLLBACK_KEY
    return None


def clear_policy_gripper_control_keys():
    with POLICY_MANUAL_CONTROL_KEY_LOCK:
        queued_rollback_keys = [
            key for key in POLICY_MANUAL_CONTROL_KEY_QUEUE
            if key == POLICY_ROLLBACK_KEY
        ]
        POLICY_MANUAL_CONTROL_KEY_QUEUE.clear()
        POLICY_MANUAL_CONTROL_KEY_QUEUE.extend(queued_rollback_keys)


def reset_policy_for_new_observation(model_va):
    model_va.policy_reset()
    if hasattr(model_va, "predict_action_queue") and model_va.predict_action_queue is not None:
        model_va.predict_action_queue.clear()
    return None



def point_cloud_filter(points):
    WORK_SPACE = [
        [0.0, 1.0],
        [-0.5, 0.5],
        [0, 0.8]
    ]
     # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
                                (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
                                (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1]))]
    return points


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



def uniform_random_sample_points(xyzrgb: np.ndarray, M: int):
    N = xyzrgb.shape[0]
    if N == 0:
        return np.zeros((M, 6))
    if N >= M:
        idx = np.linspace(0, N - 1, M).astype(np.int64)
        return xyzrgb[idx]
    else:
        extra = np.random.choice(N, M - N, replace=True)
        return np.concatenate([xyzrgb, xyzrgb[extra]], axis=0)


def update_cam_extrinsics(bestman,camera_name):
    # #camera_hand2eff
    # camera_hand2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
    # camera_hand2ee_aff = Affine([-0.05,-0.03,-0.04],camera_hand2ee_quat) #x y z
    H_camera_hand2eff = np.array([[-3.79969778e-02,-9.98877888e-01,-2.82700204e-02,-0.0457490352],
                                    [9.99277851e-01,-3.79838244e-02,-1.00233611e-03,-0.0278605145],
                                    [-7.25921195e-05,-2.82876910e-02,9.99599821e-01,-0.0525787876],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00, 1.00000000e+00]])
    cur_eff_pose = bestman.get_current_eef_pose()
    H_eff2base = np.eye(4)
    H_eff2base[:3,:3] = R.from_quat(cur_eff_pose.orientation).as_matrix()
    H_eff2base[:3,3] = np.array(cur_eff_pose.position)
    H_camera_hand2base = H_eff2base@H_camera_hand2eff

    
    if camera_name == "D435I":
        H_camera_extrics = H_camera_hand2base
    elif camera_name == "L515":

        # # [ 1280x720  p[643.178 357.433]  f[898.481 899.104]  Brown Conrady [0.144588 -0.485378 0.0004702 -4.61056e-05 0.44427] ]
        # H_camera_overhead2camera_hand = np.array([[-0.994305,-0.046932,0.095686,0.006064],
        #                                             [0.093147,-0.818933,0.566278,-0.538343],
        #                                             [0.051785,0.571966,0.818641,0.017486],
        #                                             [0.000000,0.000000,0.000000,1.000000]]
        #                                                                                                     )
        # H_camera_hand2base=np.array([[-3.81971794e-02, -9.98831631e-01, -2.96031838e-02,
        #                                 3.04561658e-01],
        #                             [-9.99270213e-01,  3.81767610e-02,  1.25483779e-03,
        #                                 2.79467731e-02],
        #                             [-1.23218001e-04,  2.96295110e-02, -9.99560942e-01,
        #                                 7.02004808e-01],
        #                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        #                                 1.00000000e+00]])
        # H_camera_overhead2base = H_camera_hand2base@H_camera_overhead2camera_hand
        # H_camera_extrics = H_camera_overhead2base

        H_camera_extrics = np.array([[-0.05659152,  0.80283684, -0.59350569,  0.84141181],
                                [ 0.9972004 ,  0.01635126, -0.07297025, -0.00164086],
                                [-0.04887985, -0.59597368, -0.80151482,  0.66857453],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]])


    return H_camera_extrics

def policy_action_to_traj6_gripper(action):
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] >= 10:
        traj6 = pose9_to_traj6(action[:9])
        return np.concatenate([traj6, action[-1:]], axis=0)
    if action.shape[0] >= 7:
        return action[:7]
    raise ValueError(f"Expected policy action with 7 or 10 values, got shape {action.shape}")

def collect_policy_action_chunk(model_va, first_action, chunk_size):
    action_chunk = [policy_action_to_traj6_gripper(first_action)]
    while len(action_chunk) < chunk_size and len(model_va.predict_action_queue) > 0:
        action_chunk.append(policy_action_to_traj6_gripper(model_va.predict_action_queue.popleft()))
    return np.stack(action_chunk, axis=0)

def offset_pose_eular_local(pose_eular, local_offset):
    pose_eular = np.asarray(pose_eular, dtype=float).copy()
    pose_rotation = R.from_euler('zyx', pose_eular[3:6])
    pose_eular[:3] += pose_rotation.apply(np.asarray(local_offset, dtype=float))
    return pose_eular

def robot_pose_eular_to_policy_pose_eular(robot_pose_eular):
    return offset_pose_eular_local(robot_pose_eular, POLICY_VIRTUAL_GRIPPER_LOCAL_OFFSET)

def policy_pose_eular_to_robot_pose_eular(policy_pose_eular):
    return offset_pose_eular_local(policy_pose_eular, -POLICY_VIRTUAL_GRIPPER_LOCAL_OFFSET)

def pose_eular_to_pose(pose_eular):
    pose_eular = np.asarray(pose_eular, dtype=float)
    pose_orientation_xyzw = R.from_euler('zyx', pose_eular[3:6]).as_quat()
    return Pose(pose_eular[:3], pose_orientation_xyzw)

def policy_action_to_pose(action):
    target_policy_pose_eular = np.asarray(action[:6], dtype=float)
    target_robot_pose_eular = policy_pose_eular_to_robot_pose_eular(target_policy_pose_eular)
    return pose_eular_to_pose(target_robot_pose_eular)

def pose_position(pose):
    return np.asarray(pose.get_position(), dtype=float)

def pose_rotation(pose):
    return R.from_quat(np.asarray(pose.get_orientation(type="quaternion"), dtype=float))

def pose_error(current_pose, target_pose):
    position_error = np.linalg.norm(pose_position(current_pose) - pose_position(target_pose))
    rotation_error = np.linalg.norm((pose_rotation(target_pose) * pose_rotation(current_pose).inv()).as_rotvec())
    return position_error, rotation_error

def copy_pose(pose):
    return Pose(
        np.asarray(pose.get_position(), dtype=float).copy(),
        np.asarray(pose.get_orientation(type="quaternion"), dtype=float).copy(),
    )

def rollback_policy_chunks(bestman, chunk_start_pose_history):
    available_chunks = len(chunk_start_pose_history)
    if available_chunks == 0:
        print("Rollback requested, but no executed chunk start pose is available; reset policy from current pose.")
        return False

    rollback_chunks = min(POLICY_ROLLBACK_CHUNKS, available_chunks)
    rollback_pose = copy_pose(chunk_start_pose_history[-rollback_chunks])
    print(f"Rollback requested: move back before previous {rollback_chunks} policy chunk(s).")
    if force_move(
        bestman,
        rollback_pose,
        maxLinearVel=POLICY_MAX_LINEAR_VEL,
        maxAngularVel=POLICY_MAX_ANGULAR_VEL,
    ):
        for _ in range(rollback_chunks):
            chunk_start_pose_history.pop()
        print(f"Rollback complete; {len(chunk_start_pose_history)} earlier chunk start pose(s) remain.")
        return True
    return False

def estimate_resume_index(current_pose, target_poses, start_idx):
    if start_idx >= len(target_poses):
        return start_idx

    reached_idx = None
    for idx in range(start_idx, len(target_poses)):
        position_error, rotation_error = pose_error(current_pose, target_poses[idx])
        if position_error < POLICY_REPLAN_POSITION_TOL and rotation_error < POLICY_REPLAN_ROTATION_TOL:
            reached_idx = idx
    if reached_idx is not None:
        return reached_idx + 1

    return start_idx

def first_gripper_event(action_chunk, allow_gripper_open_flag, manual_gripper_key=None):
    for idx, action in enumerate(action_chunk):
        inference_gripper_width = action[-1] 
        if idx == 0 and manual_gripper_key == "c":
            inference_gripper_width = GRIPPER_CLOSE_THRESHOLD - 1e-6
        if idx == 0 and manual_gripper_key == "o":
            inference_gripper_width = GRIPPER_OPEN_THRESHOLD + 1e-6
        if allow_gripper_open_flag == 0 and inference_gripper_width < GRIPPER_CLOSE_THRESHOLD:
            return idx, "close", 1
        if allow_gripper_open_flag == 1 and inference_gripper_width > GRIPPER_OPEN_THRESHOLD:
            return idx, "open", 0
    return None, None, allow_gripper_open_flag

def execute_pose_segment(bestman, pose_segment):
    if len(pose_segment) == 1:
        bestman.move_eef_to_goal_pose(
            pose_segment[0],
            maxLinearVel=POLICY_MAX_LINEAR_VEL,
            maxAngularVel=POLICY_MAX_ANGULAR_VEL,
            asynchronous=False,
        )
        return

    if POLICY_USE_TARGET_VELOCITIES:
        try:
            bestman.move_eef_through_goal_poses(
                pose_segment,
                maxLinearVel=POLICY_MAX_LINEAR_VEL,
                maxAngularVel=POLICY_MAX_ANGULAR_VEL,
                waypoint_dt=POLICY_ACTION_DT,
                asynchronous=False,
                use_target_velocities=True,
            )
            print(f"SUCCESS velocity trajectory points={len(pose_segment)}------------------")
            return
        except Exception as e:
            bestman.robot.recover_from_errors()
            print(f"Velocity waypoint trajectory failed, fallback to plain waypoints: {e}")

    bestman.move_eef_through_goal_poses(
        pose_segment,
        maxLinearVel=POLICY_MAX_LINEAR_VEL,
        maxAngularVel=POLICY_MAX_ANGULAR_VEL,
        waypoint_dt=POLICY_ACTION_DT,
        asynchronous=False,
        use_target_velocities=False,
    )

def execute_policy_pose_chunk(bestman, move_towards_poses):
    if len(move_towards_poses) == 0:
        return 0

    next_idx = 0
    window_size = min(POLICY_REPLAN_INITIAL_WINDOW, len(move_towards_poses))
    failure_count = 0
    last_successful_idx = -1

    while next_idx < len(move_towards_poses):
        segment_end = min(len(move_towards_poses), next_idx + window_size)
        pose_segment = move_towards_poses[next_idx:segment_end]

        try:
            execute_pose_segment(bestman, pose_segment)
            # print(f"SUCCESS replanned segment {next_idx}:{segment_end}------------------")
            last_successful_idx = max(last_successful_idx, segment_end - 1)
            next_idx = segment_end
            failure_count = 0
            if window_size < POLICY_REPLAN_INITIAL_WINDOW:
                window_size += 1
        except Exception as e:
            failure_count += 1
            bestman.robot.recover_from_errors()
            time.sleep(POLICY_RECOVER_SETTLE_TIME)

            current_pose = bestman.get_current_eef_pose()
            resumed_idx = estimate_resume_index(current_pose, move_towards_poses, next_idx)
            if resumed_idx > next_idx:
                print(
                    f"Trajectory interrupted after progress, replan remaining "
                    f"{resumed_idx}/{len(move_towards_poses)}: {e}"
                )
                last_successful_idx = max(last_successful_idx, resumed_idx - 1)
                next_idx = resumed_idx
                failure_count = 0
                window_size = max(POLICY_REPLAN_MIN_WINDOW, min(window_size, POLICY_REPLAN_INITIAL_WINDOW))
                continue

            if window_size > POLICY_REPLAN_MIN_WINDOW:
                window_size = max(POLICY_REPLAN_MIN_WINDOW, window_size // 2)
                print(
                    f"Trajectory segment {next_idx}:{segment_end} failed, "
                    f"shrink window to {window_size} and replan from current pose: {e}"
                )
                continue

            print(f"Single waypoint {next_idx} failed while replanning from current pose: {e}")
            if failure_count >= POLICY_REPLAN_MAX_FAILURES:
                print(
                    f"Skip waypoint {next_idx}/{len(move_towards_poses)} after "
                    f"{failure_count} failed replanning attempts; try following waypoints."
                )
                next_idx += 1
                failure_count = 0
                window_size = POLICY_REPLAN_MIN_WINDOW

    return last_successful_idx + 1

def interaction_policy_inference(camera_hand,camera_overhead, bestman,stage1segmentation,predictor,model_va,allow_gripper_open_flag,visualize,task="None"):
    reset_policy_for_new_observation(model_va)
    model_va.policy.n_action_steps=POLICY_EXEC_ACTION_STEPS #26
    executed_chunk_start_pose_history = deque()
    while True:
        if pop_policy_rollback_key() == POLICY_ROLLBACK_KEY:
            rollback_policy_chunks(bestman, executed_chunk_start_pose_history)
            clear_policy_gripper_control_keys()
            reset_policy_for_new_observation(model_va)
            continue

        if  len(model_va.predict_action_queue)<model_va.policy.horizon-model_va.policy.n_action_steps+2:

            reset_policy_for_new_observation(model_va)
            cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
            
            ###################CLOUD_RGB EXTRACT#############
            
            # Scene CloudRgb 
            masked_scene_cloud_rgb = []
            
            # 添加夹爪点云
            eff_pose_zyx_eular = cur_model_observation['pose_eular']
            eff_gripper_width = cur_model_observation['gripper_width']
            normalize_eff_angular = 0 if eff_gripper_width<0.04 else 1 
            gripper_mesh = VisualizationUtils.update_gripper(
                normalize_eff_angular,
                eff_pose_zyx_eular,
                gripper_len=VIRTUAL_GRIPPER_BAIS,
            )
            gripper_pcd = gripper_mesh.sample_points_uniformly(number_of_points=500)
            gripper_cloud_rgb = GeometryUtils.pcd_to_cloud_rgb(gripper_pcd)
            # ADD CloudRgb to Scene
            # masked_scene_cloud_rgb.append(gripper_cloud_rgb)


            # WORKSPACE DOWNSAMPLE  
            overhead_cloud_rgb = cur_model_observation['point_cloud']
            overhead_cloud_rgb_workspace = point_cloud_filter(overhead_cloud_rgb)  
            # # Objects Segmentation 
            # obj_sets = ["yellow_mug","blue_cube"]  
            # img_overhead_rgb = cur_model_observation['overhead']
            # obj_masks = get_sam2_obj_masks_fast(predictor,img_overhead_rgb,obj_sets)
            # # Objects CloudRgb Extracted 
            # scene_pcd = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb_workspace)
            camera_intrics, H_world2image = stage1segmentation.setup_camera_transforms()
            # for obj_str, obj_mask in obj_masks.items():
            #     obj_seg_pcd = ProjectionUtils.get_seg_pcd(scene_pcd, H_world2image, camera_intrics, obj_mask)
            #     obj_seg_cloud_rgb = np.hstack((np.array(obj_seg_pcd.points), 
            #                                     np.array(255 * np.array(obj_seg_pcd.colors)).astype(np.uint8)))
            #     # obj_seg_cloud_rgb = remove_outliers_fast(obj_seg_cloud_rgb, nb_neighbors=50, std_ratio=1)
                
            #     # ADD CloudRgb to Scene
            #     masked_scene_cloud_rgb.append(obj_seg_cloud_rgb)
            # # Convert Scene CloudRgb to Numpy
            # masked_scene_cloud_rgb = np.vstack(masked_scene_cloud_rgb)

            # Align with Collection
            overhead_cloud_rgb_filter = GeometryUtils.random_repeat_sample_points(overhead_cloud_rgb_workspace,49500)
            # points_xyz = overhead_cloud_rgb_filter[..., :3]
            # points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points=49500)
            # sample_indices = sample_indices.cpu()
            # points_rgb = overhead_cloud_rgb_filter[sample_indices, 3:][0]
            points_xyz = overhead_cloud_rgb_filter[..., :3]
            points_rgb = overhead_cloud_rgb_filter[..., 3:]
            points = np.hstack((points_xyz, points_rgb))
            points = np.vstack((gripper_cloud_rgb,points))

            cur_model_observation["point_cloud"] = points


            ###################STATE ALIGNED WITH DP3#############
            cur_observation_key_list = list(cur_model_observation.keys())
            for idx in range(len(cur_model_observation['pose_eular'])):
                cur_model_observation[cur_observation_key_list[idx]] =cur_model_observation['pose_eular'][idx]
            cur_model_observation['joint_7'] = eff_gripper_width



        inference_action =  model_va.single_inference(cur_model_observation,visualize=visualize,task=task)
        if inference_action is None:
            continue
        
        # Image Visualize
        # overhead_pcd_filter = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb_filter)
        # depth, rgb = ProjectionUtils.project_pcd_to_image_depth(overhead_pcd_filter, H_world2image, camera_intrics, (480, 640))
        # cv2.imwrite("/home/liusong/temp/temp.png",rgb)

        ###################POSE CONTROAL#############
        action_chunk = collect_policy_action_chunk(
            model_va,
            inference_action,
            model_va.policy.n_action_steps,
        )
        policy_control_key = pop_policy_control_key()
        if policy_control_key == POLICY_ROLLBACK_KEY:
            rollback_policy_chunks(bestman, executed_chunk_start_pose_history)
            clear_policy_gripper_control_keys()
            reset_policy_for_new_observation(model_va)
            continue

        manual_gripper_key = (
            policy_control_key
            if policy_control_key in POLICY_MANUAL_GRIPPER_KEYS
            else None
        )
        if manual_gripper_key == "c":
            print("Manual policy gripper close key queued.")
        if manual_gripper_key == "o":
            print("Manual policy gripper open key queued.")


        gripper_event_idx, gripper_command, next_gripper_flag = first_gripper_event(
            action_chunk,
            allow_gripper_open_flag,
            manual_gripper_key=manual_gripper_key,
        )
        if gripper_event_idx is not None:
            action_chunk_to_execute = action_chunk[:gripper_event_idx + 1]
        else:
            action_chunk_to_execute = action_chunk
        move_towards_poses = [policy_action_to_pose(action) for action in action_chunk_to_execute]
        chunk_start_pose = copy_pose(bestman.get_current_eef_pose())
        
        try:
            executed_points = execute_policy_pose_chunk(bestman, move_towards_poses)
        except Exception as e:
            bestman.robot.recover_from_errors() 
            reset_policy_for_new_observation(model_va)
            print(f"ERROR all trajectory execution fallbacks failed: {e}------------------")
            continue
        executed_chunk_start_pose_history.append(chunk_start_pose)

        # ###################Gripper CONTROAL#############
        # inference_gripper_width=inference_action[-1]*2 #recover the normal scale
        # bestman.open_gripper_width(inference_gripper_width)

        gripper_event_reached = gripper_event_idx is not None and gripper_event_idx < executed_points
        if gripper_event_reached and gripper_command == "close":
            allow_gripper_open_flag = next_gripper_flag
            bestman.close_gripper() 
            return allow_gripper_open_flag
        if gripper_event_reached and gripper_command == "open":
            bestman.open_gripper()
            allow_gripper_open_flag = next_gripper_flag
            return allow_gripper_open_flag
def force_move(
    bestman,
    pose,
    maxLinearVel=0.22,
    maxAngularVel=POLICY_MAX_ANGULAR_VEL,
    max_attempts=5,
    retry_sleep=0.2,
):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            bestman.move_eef_to_goal_pose(pose, maxLinearVel=maxLinearVel, maxAngularVel=maxAngularVel)
            print("Force move SUCCESS------------------")
            return True
        except Exception as e:
            last_error = e
            try:
                bestman.robot.recover_from_errors()
            except Exception as recover_error:
                print(f"Recover from errors failed: {recover_error}")
            print(
                f"Force move ERROR attempt {attempt}/{max_attempts}: "
                f"{type(e).__name__}: {e}"
            )
            time.sleep(retry_sleep)

    print(
        f"Force move GIVE UP after {max_attempts} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )
    return False


def mission_execution(task_name,camera_hand,camera_overhead, bestman,predictor,model_va,visualize=False):
    #4.load stage1 segmentation for camera parameters
    stagegen_config_file = f"/home/liusong/ProgramFiles/REAP/StageGen/config/{task_name}.yaml"
    stagegen_src_hdf5_path = f"/home/liusong/ProgramFiles/REAP/StageGen/source/{task_name}.hdf5"
    stage1segmentation = Stage1Segmentation(stagegen_config_file, stagegen_src_hdf5_path)
    # import pickle
    # with open(f"/home/liusong/ProgramFiles/REAP/StageGen/out/{task_name}/{task_name}_stage1_result.pkl", 'rb') as file:
    #     stage1_result = pickle.load(file)
    # stage2editing = Stage2Editing(stage1_result)

    object_marker_list = object_marker_dict[task_name]
    predictor = predictor_initialize(predictor,task_name,object_marker_list)

    # 6.LOAD SAVG MODEL
    SAVG_PRETRAINED_CKPT_PATH = f"/home/liusong/ProgramFiles/REAP/SAVG/out/checkpoints/{task_name}.pt"
    # model
    model_savg = PoseACTCVAE(
        pc_in_dim=6,
        pc_dim=256,
        pc_grid_size=0.005,
        pc_tokens=256,
        geo_k=256,
        model_dim=256,
        latent_dim=32,
        n_enc_layers=4,
        n_dec_layers=4,
        heads=4,
        ff_dim=1024,
        dropout=0.1,
        pre_norm=True,
    ).to(DEVICE)
    ckpt = torch.load(SAVG_PRETRAINED_CKPT_PATH)
    model_savg.load_state_dict(ckpt["model"])
    model_savg.eval()
    assert os.path.isfile(SAVG_PRETRAINED_CKPT_PATH), f"ckpt not found: {SAVG_PRETRAINED_CKPT_PATH}"



    
    
    ##############Subtask##########
    print("###########Subtask##########")
    subtasks = stage1segmentation.task_config['subtask']
    [print(subtask) for subtask in subtasks]

    allow_gripper_open_flag = 0
    for subtask in subtasks:
        subtask_label = subtask.split(',')
        action = subtask_label[0] 
        target_obj = subtask_label[1].strip() 
        cond_obj = subtask_label[3].strip()
        if action == "move_towards":
            continue
            # if subtask == "move_towards, mug, eff_open, None":
            #     continue
            cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
            target_pose_position,target_pose_orientation_xyzw = savg_inference(model_savg,cur_model_observation,predictor,stage1segmentation,target_obj,cond_obj,visualize=visualize)
            target_policy_pose_eular = np.concatenate(
                (
                    target_pose_position,
                    R.from_quat(target_pose_orientation_xyzw).as_euler('zyx'),
                ),
                axis=0,
            )
            move_towards_pose = pose_eular_to_pose(policy_pose_eular_to_robot_pose_eular(target_policy_pose_eular))
            force_move(bestman,move_towards_pose, maxLinearVel=0.3, maxAngularVel=POLICY_MAX_ANGULAR_VEL)
        else:
            allow_gripper_open_flag = interaction_policy_inference(camera_hand,camera_overhead, bestman,stage1segmentation,predictor,model_va,allow_gripper_open_flag,visualize=visualize,task = "Place the Yellow Mug on the Red Pole")
            if action=="place":
                # Lift Up 5cm
                cur_eff_pose = bestman.get_current_eef_pose()
                move_towards_pose = Pose(cur_eff_pose.position+np.array([0,0,0.08]), cur_eff_pose.orientation)
                force_move(bestman,move_towards_pose, maxLinearVel=0.3, maxAngularVel=POLICY_MAX_ANGULAR_VEL)
            # if action=="pick":
            #     # Lift Up 5cm
            #     cur_eff_pose = bestman.get_current_eef_pose()
            #     move_towards_pose = Pose(cur_eff_pose.position+np.array([0,0,0.05]), cur_eff_pose.orientation)
            #     force_move(bestman,move_towards_pose, maxLinearVel=0.3, maxAngularVel=POLICY_MAX_ANGULAR_VEL)
    # Go Back Home
    bestman.open_gripper()
    bestman.go_home(home_js)




#Visualize With Predict Trajectory
{
            # cur_model_observation = get_cur_model_observation(camera_hand,camera_overhead, bestman)
            # ###################CLOUD_RGB EXTRACT#############
            # # 添加夹爪点云
            # eff_pose_zyx_eular = cur_model_observation['pose_eular']
            # eff_gripper_width = cur_model_observation['gripper_width']
            # normalize_eff_angular = 0 if eff_gripper_width<0.03 else 1
            # gripper_mesh = VisualizationUtils.updaqte_gripper(normalize_eff_angular, eff_pose_zyx_eular, gripper_len = 0.06)
            # gripper_pcd = gripper_mesh.sample_points_uniformly(number_of_points=500)
            # gripper_cloud_rgb = GeometryUtils.pcd_to_cloud_rgb(gripper_pcd)

            # # Objects CloudRgb Extracted 
            # masked_scene_cloud_rgb = []
            # masked_scene_cloud_rgb.append(gripper_cloud_rgb)

            # # Objects Segmentation 
            # obj_sets = ["yellow_mug","blue_cube"]  
            # img_overhead_rgb = cur_model_observation['overhead']
            # obj_masks = get_sam2_obj_masks_fast(predictor,img_overhead_rgb,obj_sets)

            # # Objects CloudRgb Extracted 
            # obj_seg_dict={}
            # overhead_cloud_rgb = cur_model_observation['point_cloud']

            # #WORKSPACE DOWNSAMPLE
            # overhead_cloud_rgb_workspace = point_cloud_filter(overhead_cloud_rgb)  
            # scene_pcd = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb_workspace)
            # camera_intrics, H_world2image = stage1segmentation.setup_camera_transforms()
            # for obj_str, obj_mask in obj_masks.items():
            #     obj_seg_pcd = ProjectionUtils.get_seg_pcd(scene_pcd, H_world2image, camera_intrics, obj_mask)
            #     obj_seg_cloud_rgb = np.hstack((np.array(obj_seg_pcd.points), 
            #                                     np.array(255 * np.array(obj_seg_pcd.colors)).astype(np.uint8)))
            #     masked_scene_cloud_rgb.append(obj_seg_cloud_rgb)
            # masked_scene_cloud_rgb = np.vstack(masked_scene_cloud_rgb)

            # # Align with Collection
            # overhead_cloud_rgb_filter = GeometryUtils.random_repeat_sample_points(masked_scene_cloud_rgb,48*64)
            # points_xyz = overhead_cloud_rgb_filter[..., :3]
            # points_xyz, sample_indices = farthest_point_sampling(points_xyz)
            # sample_indices = sample_indices.cpu()
            # points_rgb = overhead_cloud_rgb_filter[sample_indices, 3:][0]
            # points = np.hstack((points_xyz, points_rgb))
            # cur_model_observation["point_cloud"] = points

            # ###################STATE ALIGNED WITH DP3#############
            # cur_observation_key_list = list(cur_model_observation.keys())
            # for idx in range(len(cur_model_observation['pose_eular'])):
            #     cur_model_observation[cur_observation_key_list[idx]] =cur_model_observation['pose_eular'][idx]
            # cur_model_observation['joint_7'] = cur_model_observation["gripper_width"]
            # inference_action =  model_va.single_inference(cur_model_observation)

            # np_list = [t.detach().cpu().numpy() for t in model_va.predict_action_queue]
            # np_array = np.stack(np_list)  # 可选：拼成一个数组
            # pre_traj_points = np_array[:,:3]
            # pre_traj_colors = np.zeros(pre_traj_points.shape)
            # pre_traj_cloud_rgb = np.concatenate((pre_traj_points,pre_traj_colors),axis=1)
            # pre_traj_pcd = GeometryUtils.cloud_rgb_to_pcd(pre_traj_cloud_rgb)
            # overhead_pcd_filter = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb_filter)
            # import open3d as o3d
            # o3d.visualization.draw_geometries([pre_traj_pcd,overhead_pcd_filter])
            # from collections import deque
            # model_va.predict_action_queue = deque()
        }




def get_cur_model_observation(camera_hand,camera_overhead,bestman):
    gripper_width = bestman.gripper.width

    cur_pos = bestman.get_current_joint_values()
    
    cur_eff_pose = bestman.get_current_eef_pose()
    cur_eff_pose_position = cur_eff_pose.position
    cur_eff_rotation =  R.from_quat(cur_eff_pose.orientation)
    cur_eff_pose_orientation_eular_zyx = cur_eff_rotation.as_euler('zyx')
    robot_pose_eular = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)
    policy_pose_eular = robot_pose_eular_to_policy_pose_eular(robot_pose_eular)
    

    img_hand_rgb = camera_hand.get_rgb_image()
    points, colors = camera_hand.get_3d_points()
    # points, colors = np.zeros((10,3)),np.zeros((10,3))#-----------temp_use
    H_camera_hand_extrinsic = update_cam_extrinsics(bestman,camera_hand.dev_name)
    world_points = ((H_camera_hand_extrinsic[:3,:3]@points.T).T+H_camera_hand_extrinsic[:3,3].T).astype(np.float32)
    hand_cloud_rgb_var_len = np.hstack((world_points,((colors*255).astype(np.uint8)))) 
    hand_cloud_rgb = uniform_random_sample_points(hand_cloud_rgb_var_len,CONST_POINTS_NUM)
    # camera_hand.visualize_3d_points()


    img_overhead_rgb = camera_overhead.get_rgb_image()
    img_overhead_rgb = cv2.resize(img_overhead_rgb,(640,480),interpolation=cv2.INTER_LINEAR)
    points, colors = camera_overhead.get_3d_points()
    # points, colors = np.zeros((10,3)),np.zeros((10,3))#-----------temp_use
    H_camera_overhead_extrinsic = update_cam_extrinsics(bestman,camera_overhead.dev_name)
    world_points = ((H_camera_overhead_extrinsic[:3,:3]@points.T).T+H_camera_overhead_extrinsic[:3,3].T).astype(np.float32)
    overhead_cloud_rgb_var_len = np.hstack((world_points,((colors*255).astype(np.uint8)))) 
    overhead_cloud_rgb = uniform_random_sample_points(overhead_cloud_rgb_var_len,CONST_POINTS_NUM)
    # camera_overhead.visualize_3d_points()


    # Lerobot-VA
    # cur_model_observation = {'joint_1':None,'joint_2':None,'joint_3': None,'joint_4':None,'joint_5':None,'joint_6': None,'joint_7':None,'gripper_width':None,
    #                         'overhead':img_overhead_rgb,"hand":img_hand_rgb,"point_cloud":{},"pose_eular":{}}
    # cur_observation_key_list = list(cur_model_observation.keys())
    # for idx, value in enumerate(cur_pos):
    #     cur_model_observation[cur_observation_key_list[idx]] =180*value/3.14
    # cur_model_observation['gripper_width'] = gripper_width*1000*0.5
    # cur_model_observation['pose_eular'] = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)
    # cur_model_observation['point_cloud'] = overhead_cloud_rgb



    #DP3
    cur_model_observation = {'joint_1':None,'joint_2':None,'joint_3': None,'joint_4':None,'joint_5':None,'joint_6': None,'joint_7':None,'gripper_width':None,
                            'overhead':img_overhead_rgb,"hand":img_hand_rgb,"point_cloud":{},"pose_eular":{}}
    cur_model_observation['pose_eular'] = policy_pose_eular
    cur_model_observation['gripper_width'] = gripper_width
    cur_model_observation['point_cloud'] = overhead_cloud_rgb

    return cur_model_observation


def get_sam2_obj_masks(predictor,frame,obj_sets):
    out_obj_ids, out_mask_logits = predictor.track(frame)
    all_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    all_mask[..., 1] = 255
    obj_masks = {}
    for obj_index in range(len(out_obj_ids)):
        if obj_index >= len(obj_sets):
            continue
        
        obj_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        obj_mask[..., 1] = 255
        
        out_mask = (out_mask_logits[obj_index] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        hue = (obj_index + 3) / (len(out_obj_ids) + 3) * 255
        
        obj_mask[out_mask[..., 0] == 255, 0] = hue
        obj_mask[out_mask[..., 0] == 255, 2] = 255
        obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_HSV2RGB)
        mask = np.any(obj_mask > 0, axis=-1)
        obj_mask[mask] = [255, 255, 255]

        obj_mask_gray = cv2.cvtColor(obj_mask, cv2.COLOR_BGR2GRAY)
        obj_masks[obj_sets[obj_index]] = obj_mask_gray
    return obj_masks

def get_sam2_obj_masks_fast(predictor, frame, obj_sets):
    out_obj_ids, out_mask_logits = predictor.track(frame)

    H, W = frame.shape[:2]
    num_obj = min(len(out_obj_ids), len(obj_sets))

    # 🔴 1. 一次性 GPU → CPU（最重要）
    # out_mask_logits: [N, 1, H, W] or [N, H, W]
    masks = (out_mask_logits[:num_obj] > 0.0) \
        .squeeze(1) \
        .cpu() \
        .numpy() \
        .astype(np.uint8) * 255  # [N, H, W]

    obj_masks = {}

    # 🔴 2. 只生成 grayscale mask
    for i in range(num_obj):
        # masks[i] already H×W uint8 {0,255}
        obj_masks[obj_sets[i]] = masks[i]

    return obj_masks


def from_trajectory_to_H(trajectory):
    """从轨迹数据转换为齐次矩阵"""
    position = trajectory[:3]
    euler_zyx = trajectory[3:]
    rotation_matrix = R.from_euler('zyx', euler_zyx, degrees=False).as_matrix()
    H = np.eye(4)
    H[:3, :3] = rotation_matrix
    H[:3, 3] = position
    return H
def from_H_to_trajectory(H):
    """从齐次矩阵转换为轨迹数据"""
    position = H[:3, 3]
    rotation_matrix = H[:3, :3]
    euler_zyx = R.from_matrix(rotation_matrix).as_euler('zyx', degrees=False)
    trajectory = np.hstack((position, euler_zyx))
    return trajectory


def get_base_points_from_cam_points(bestman,mouse_get_cam_3d_points,camera_name):
    base_3d_points = []
    for cam_obj_translation in mouse_get_cam_3d_points:
        H_obj2cam = np.eye(4)
        H_obj2cam[:3,3] = np.array(cam_obj_translation)
        
        
        # #camera_hand2eff
        # camera_hand2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
        # camera_hand2ee_aff = Affine([-0.05,-0.03,-0.04],camera_hand2ee_quat) #x y z
        H_camera_hand2eff = np.array([[-3.79969778e-02,-9.98877888e-01,-2.82700204e-02,-0.0457490352],
                                        [9.99277851e-01,-3.79838244e-02,-1.00233611e-03,-0.0278605145],
                                        [-7.25921195e-05,-2.82876910e-02,9.99599821e-01,-0.0525787876],
                                        [0.00000000e+00,0.00000000e+00,0.00000000e+00, 1.00000000e+00]])
        cur_eff_pose = bestman.get_current_eef_pose()
        H_eff2base = np.eye(4)
        H_eff2base[:3,:3] = R.from_quat(cur_eff_pose.orientation).as_matrix()
        H_eff2base[:3,3] = np.array(cur_eff_pose.position)
        H_camera_hand2base = H_eff2base@H_camera_hand2eff

        if camera_name == "hand":
            H_camera_extrics = H_camera_hand2base
        elif camera_name == "overhead":

            # # [ 1280x720  p[643.178 357.433]  f[898.481 899.104]  Brown Conrady [0.144588 -0.485378 0.0004702 -4.61056e-05 0.44427] ]
            # H_camera_overhead2camera_hand = np.array([[-0.994305,-0.046932,0.095686,0.006064],
            #                                             [0.093147,-0.818933,0.566278,-0.538343],
            #                                             [0.051785,0.571966,0.818641,0.017486],
            #                                             [0.000000,0.000000,0.000000,1.000000]]
            #                                                                                                     )
            # H_camera_hand2base=np.array([[-3.81971794e-02, -9.98831631e-01, -2.96031838e-02,
            #                                 3.04561658e-01],
            #                             [-9.99270213e-01,  3.81767610e-02,  1.25483779e-03,
            #                                 2.79467731e-02],
            #                             [-1.23218001e-04,  2.96295110e-02, -9.99560942e-01,
            #                                 7.02004808e-01],
            #                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            #                                 1.00000000e+00]])
            # H_camera_overhead2base = H_camera_hand2base@H_camera_overhead2camera_hand
            # H_camera_extrics = H_camera_overhead2base

            H_camera_extrics = np.array([[-0.05659152,  0.80283684, -0.59350569,  0.84141181],
                                    [ 0.9972004 ,  0.01635126, -0.07297025, -0.00164086],
                                    [-0.04887985, -0.59597368, -0.80151482,  0.66857453],
                                    [ 0.        ,  0.        ,  0.        ,  1.        ]])


        H_obj2base = H_camera_extrics@H_obj2cam
        base_3d_points.append(H_obj2base[:3,3])


    return base_3d_points



def savg_inference(model_savg,cur_model_observation,predictor,stage1segmentation,target_obj,cond_obj,visualize):
    obj_sets = stage1segmentation.task_config['obj_sets']
    
    cond_name = cond_obj
    if cond_name=="None":
        has_cond=False
    else:
        has_cond=True

    # Objects Segmentation 
    img_overhead_rgb = cur_model_observation['overhead']
    obj_masks = get_sam2_obj_masks_fast(predictor,img_overhead_rgb,obj_sets)

    # Objects CloudRgb Extracted 
    obj_seg_dict={}
    overhead_cloud_rgb = cur_model_observation['point_cloud']
    scene_pcd = GeometryUtils.cloud_rgb_to_pcd(overhead_cloud_rgb)
    camera_intrics, H_world2image = stage1segmentation.setup_camera_transforms()
    for obj_str, obj_mask in obj_masks.items():
        obj_seg_pcd = ProjectionUtils.get_seg_pcd(scene_pcd, H_world2image, camera_intrics, obj_mask)
        obj_seg_cloud_rgb = np.hstack((np.array(obj_seg_pcd.points), 
                                        np.array(255 * np.array(obj_seg_pcd.colors)).astype(np.uint8)))
        obj_seg_dict[obj_str] = {"cloud_rgb": obj_seg_cloud_rgb}

    # ApproachMode Switch
    if has_cond:
        cond_raw= obj_seg_dict[cond_obj]['cloud_rgb']
        target_raw= obj_seg_dict[target_obj]['cloud_rgb']
    else:
        cond_raw = np.zeros((0, 6), dtype=np.float32)
        target_raw= obj_seg_dict[target_obj]['cloud_rgb']

    # SAVG Inference
    cur_eff_trajectory = cur_model_observation['pose_eular']
    with torch.no_grad():
        #MakeSure torch.float32 //Avoid SAM2 Collision
        with torch.cuda.amp.autocast(False):
            approaching_raw = cur_eff_trajectory # NO USE
            if has_cond:
                H_pr = np.eye(4)
                for i in range(3):
                    H_pr_slice,target_raw,cond_raw,approaching_raw,cond_name = single_data_inference(model_savg,target_raw,cond_raw,approaching_raw,cond_name,visualize=visualize)
                    H_pr = H_pr_slice@H_pr
            if has_cond != True:
                H_pr_slice,target_raw,cond_raw,approaching_raw,cond_name = single_data_inference(model_savg,target_raw,cond_raw,approaching_raw,cond_name,visualize=visualize)
                H_pr = H_pr_slice
        
    if has_cond:
        H_cur_eff_trajectory = from_trajectory_to_H(cur_eff_trajectory)
        H_next_eff_trajectory = H_pr@H_cur_eff_trajectory
    else:
        H_next_eff_trajectory = H_pr
    #Move Towards Action Execution 
    target_pose_position = H_next_eff_trajectory[:3, 3]
    target_pose_orientation_xyzw = R.from_matrix(H_next_eff_trajectory[:3, :3]).as_quat()
    return target_pose_position,target_pose_orientation_xyzw

def sam2_initialize():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    import time
    import h5py
    sys.path.append("/home/liusong/ProgramFiles/SAM2/sam2")
    from sam2.build_sam import build_sam2_camera_predictor

    # sam2_checkpoint = "/home/liusong/ProgramFiles/SAM2/sam2/checkpoints/sam2.1_hiera_small.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    # Better -->Base_Plus SAM2-----------------
    # Need Be Careful of the order of the object! SAM-Object Order Not Always Correct 

    sam2_checkpoint = "/home/liusong/ProgramFiles/SAM2/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    return predictor

def predictor_initialize(predictor,task_name,object_marker_list):

    
    # img_rgb = cv2.resize(camera_overhead.get_rgb_image(),(640,480),cv2.INTER_LINEAR)
    # img_bgr = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB)
    # cv2.imshow("img",img_bgr)
    # cv2.waitKey(0)
    # cv2.imwrite("/home/liusong/ProgramFiles/BestMan/Dataset/Images/Desk_TrashSweep.png",img_bgr)


    frame = cv2.imread(f"/home/liusong/ProgramFiles/BestMan/Dataset/Images/{task_name}.png")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    width, height = frame.shape[:2][::-1]
    predictor.load_first_frame(frame)
    if_init = True

    ann_frame_idx = 0  # the frame index we interact with
    # First annotation
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    ##! add points, `1` means positive click and `0` means negative click
    points = np.array(object_marker_list[0], dtype=np.float32)
    labels = np.array([1,1,1], dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    )
    
    ann_obj_id = 2  # give a unique id to each object we inter act with (it can be any integers)
    points = np.array(object_marker_list[1], dtype=np.float32)
    labels = np.array([1,1,1], dtype=np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels
    )

    return predictor


# 1.初始化机器人（原代码逻辑）
bestman = Bestman_Real_Franka3()
if bestman.initialize_robot() is not True:
    exit(-1)
bestman.install_exit_handlers()
bestman.open_gripper()
#Twist 90 degree
bestman.go_home()
skill_franka3_database = skill_database.SkillFranka3Database()

#2.Camera Initialize
camera_devices = list( bestman.cfg.Camera.keys())
camera_hand,camera_overhead = None,None
for cam_dev in camera_devices:
    if cam_dev == "D435I":
        # camera_hand = Camera_Realsense(bestman.cfg.Camera[cam_dev])
        camera_hand=camera_overhead
    if cam_dev == "L515":
        camera_overhead = Camera_Realsense(bestman.cfg.Camera[cam_dev])
if camera_overhead == None:
    camera_overhead = camera_hand
time.sleep(bestman.cfg.Camera.init_delay)




task_name = "Desk_CubeStacking"
object_marker_dict = {"PutStationeryBox":[[[477,293],[485,320],[494,363]],[[342,306],[340,333],[341,362]]],
                      "CubeStacking":[[[327, 310],[337,323],[332,334]],[[441, 308],[454,321],[447,335]]],
                      "TrashSweep":[[[273, 205],[280,263],[284,286]],[[422, 339],[432,400],[437,457]]],
                      "MugRack":[[[459, 287],[468,344],[434,315]],[[329, 318],[303,337],[321,330]]],
                      "Desk_MugRack":[[[459, 287],[468,344],[434,315]],[[329, 318],[303,337],[321,330]]],
                      "Desk_CubeStacking":[[[308,169],[308,180],[307,186]],[[391,327],[393,333],[390,352]]],
                      "Desk_TrashSweep":[[[490,132],[472,181],[471,203]],[[224,306],[208,358],[193,431]]],
                      }
predictor = sam2_initialize()

x_home_js = np.array([-0.07188314616233507, -0.5007457342122718, 0.07313486429670638, -2.7816527503720883, 0.05476125807473123, 2.2630911769337083, -0.7468963222873954])
y_home_js = np.array([-0.0684262,-0.512061,0.0723129,-2.794,0.0469305,2.28233,0.749217])
home_js = y_home_js

# DP3_PRETRAINED_CKPT_PATH = f"/home/liusong/scp_receive/dp3/franka_real_simple_pose9/checkpoints/{task_name}.ckpt"
# sys.path.append("/home/liusong/ProgramFiles/VA-VLA/DP3/3D-Diffusion-Policy/3D-Diffusion-Policy/")
# from DP3_ModelInference import DP3_ModelInference
# model_va = DP3_ModelInference(DP3_PRETRAINED_CKPT_PATH)

model_va = SmolVLA_ModelInference(
    policy_path="/home/liusong/ProgramFiles/Huggingface/lerobot/benchmarks/song_real_libero/outputs/real_setting/train/ep_vla/checkpoints/last/pretrained_model",
    policy_repo_id="/home/liusong/scp_receive/smolvla",
    device=DEVICE,
)



mission_execution_flag= False
visualize=False
mission_execution_thread = None

def close_cameras(*cameras):
    closed_camera_ids = set()
    for camera in cameras:
        if camera is None or id(camera) in closed_camera_ids:
            continue
        closed_camera_ids.add(id(camera))
        try:
            camera.close()
        except Exception as e:
            print(f"Close camera failed: {e}")

exit_code = 0
try:
    while True:

        window_name = "Overhead RGB"
        cv2.namedWindow(window_name)
        img_hand_rgb = camera_hand.get_rgb_image()
        img_overhead_rgb = camera_overhead.get_rgb_image()
        cv2.imshow(window_name, cv2.resize(cv2.cvtColor(img_overhead_rgb,cv2.COLOR_RGB2BGR),(640,360)))
        cv2.waitKey(1)

        if mission_execution_flag == True:
            
            #Window Overview
            mission_execution_thread = threading.Thread(
                target=mission_execution,
                args=(task_name,camera_hand,camera_overhead, bestman,predictor,model_va,visualize))
            mission_execution_thread.daemon = True
            mission_execution_thread.start()
            mission_execution_flag = False


        pressed_key = poll_terminal_key(timeout=0.01)
        if pressed_key is not None:
            mission_is_running = (
                mission_execution_thread is not None
                and mission_execution_thread.is_alive()
            )
            if mission_is_running and queue_policy_control_key(pressed_key):
                print(f"Queued policy control: {pressed_key}")
                continue
            if pressed_key == 'n':
                mission_execution_flag = True
            if pressed_key == 'q':
                print("Program Over!")
                break
            if pressed_key in object_marker_dict.keys():
                task_name = pressed_key
                if task_name == "Desk_CubeStacking":
                    home_js = y_home_js
                else:
                    home_js = x_home_js
                home_js = y_home_js
                bestman.go_home(home_js)
                mission_execution_flag = True
except KeyboardInterrupt:
    signal_number = getattr(bestman, "_shutdown_signal", None)
    exit_code = 128 + signal_number if signal_number is not None else 130
    print("Shutdown requested. Cleaning up resources.")
finally:
    close_cameras(camera_hand, camera_overhead)
    bestman.release_robot()
    cv2.destroyAllWindows()

sys.exit(exit_code)
