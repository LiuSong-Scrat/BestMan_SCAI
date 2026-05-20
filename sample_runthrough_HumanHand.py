import time
import math
import numpy as np
import cv2
import sys, os
import json
import threading
# sys.path.insert(1, "../../..")
from scipy.spatial.transform import Rotation as R
from Robotics_API import Bestman_Real_Franka3, Pose
from Motion_Planning.Manipulation.Skill_Franka3 import skill_database
from Sensor.Camera_Realsense import Camera_Realsense
from Dataset.scripts.data_collection import RealDataCollection
from franky import Affine
import argparse
import select
parser = argparse.ArgumentParser()


CONST_POINTS_NUM=640*480


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

def collection_data_update(camera_hand,camera_overhead,bestman,sim_data_collection):
    while True:
        # gripper_width = bestman.gripper.width
        # cur_pos = bestman.get_current_joint_values()
        # cur_eff_pose = bestman.get_current_eef_pose()
        # H_eff2base = np.eye(4)
        # H_eff2base[:3,:3] = R.from_quat(cur_eff_pose.orientation).as_matrix()
        # H_eff2base[:3,3] = np.array(cur_eff_pose.position)
        # cur_eff_pose_position = cur_eff_pose.position
        # cur_eff_rotation =  R.from_quat(cur_eff_pose.orientation)
        # cur_eff_pose_orientation_eular_zyx = cur_eff_rotation.as_euler('zyx')
        
        
        cur_pos = np.zeros(7)
        H_eff2base = np.eye(4)
        rgbd_frame = camera_overhead.get_rgbd_image()
        results = bestman.process_frame(rgbd_frame)
        if len(results) == 0 or results[0].gripper is None:
            continue
        gripper_width = results[0].gripper.opening_width_m

        H_hand2camera = np.eye(4)
        H_hand2camera[:3,:3] = R.from_quat(results[0].gripper.metadata['quaternion_xyzw']).as_matrix()
        H_hand2camera[:3,3] = np.array(results[0].gripper.position_m)
        H_camera_overhead_extrinsic = update_cam_extrinsics(H_eff2base,camera_overhead.dev_name)
        H_eff2base = H_camera_overhead_extrinsic@H_hand2camera
        
        cur_eff_pose_position = H_eff2base[:3,3]
        cur_eff_rotation =  R.from_matrix(H_eff2base[:3,:3])
        cur_eff_pose_orientation_eular_zyx = cur_eff_rotation.as_euler('zyx')



        img_hand_rgb = camera_hand.get_rgb_image()
        points, colors = camera_hand.get_3d_points()
        H_camera_hand_extrinsic = update_cam_extrinsics(H_eff2base,camera_hand.dev_name)
        world_points = ((H_camera_hand_extrinsic[:3,:3]@points.T).T+H_camera_hand_extrinsic[:3,3].T)
        hand_cloud_rgb_var_len = np.hstack((world_points,((colors*255).astype(np.uint8)))) 
        hand_cloud_rgb = uniform_random_sample_points(hand_cloud_rgb_var_len,CONST_POINTS_NUM)
        # camera_hand.visualize_3d_points()


        img_overhead_rgb = camera_overhead.get_rgb_image()
        img_overhead_rgb = cv2.resize(img_overhead_rgb,(640,480),interpolation=cv2.INTER_LINEAR)
        points, colors = camera_overhead.get_3d_points()
        H_camera_overhead_extrinsic = update_cam_extrinsics(H_eff2base,camera_overhead.dev_name)
        world_points = ((H_camera_overhead_extrinsic[:3,:3]@points.T).T+H_camera_overhead_extrinsic[:3,3].T)
        overhead_cloud_rgb_var_len = np.hstack((world_points,((colors*255).astype(np.uint8)))) 
        overhead_cloud_rgb = uniform_random_sample_points(overhead_cloud_rgb_var_len,CONST_POINTS_NUM)
        # camera_overhead.visualize_3d_points()






        sim_data_collection.cur_pos = cur_pos
        sim_data_collection.pose_eular = np.concatenate((cur_eff_pose_position, cur_eff_pose_orientation_eular_zyx), axis=0)#由于是zxy轴，不能用xyz
        sim_data_collection.hand_cloud_rgb = hand_cloud_rgb
        sim_data_collection.overhead_cloud_rgb = overhead_cloud_rgb #-----------temp_use
        sim_data_collection.hand_frame = img_hand_rgb 
        sim_data_collection.overhead_frame = img_overhead_rgb #-----------temp_use
        sim_data_collection.eff_angular = np.array([gripper_width]) #xyz/xyzw
        time.sleep(0.02)

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

def update_cam_extrinsics(H_eff2base,camera_name):
    # #camera_hand2eff
    # camera_hand2ee_quat = R.from_matrix(np.array([[0,1,0],[-1,0,0],[0,0,1]])).as_quat()
    # camera_hand2ee_aff = Affine([-0.05,-0.03,-0.04],camera_hand2ee_quat) #x y z
    H_camera_hand2eff = np.array([[-3.79969778e-02,-9.98877888e-01,-2.82700204e-02,-0.0457490352],
                                    [9.99277851e-01,-3.79838244e-02,-1.00233611e-03,-0.0278605145],
                                    [-7.25921195e-05,-2.82876910e-02,9.99599821e-01,-0.0525787876],
                                    [0.00000000e+00,0.00000000e+00,0.00000000e+00, 1.00000000e+00]])

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


def _draw_gripper_pose(cv2, image, result, intrinsics) -> None:
    pts = np.asarray(result.prediction.keypoints_2d, dtype=np.float64)
    if np.all(np.isfinite(pts[[4, 8]])):
        thumb_tip_px = tuple(np.round(pts[4]).astype(int))
        index_tip_px = tuple(np.round(pts[8]).astype(int))
        cv2.line(image, thumb_tip_px, index_tip_px, (255, 0, 255), 3)

    origin = np.asarray(result.gripper.position_m, dtype=np.float64)
    rotation = np.asarray(result.gripper.rotation_camera_gripper, dtype=np.float64)
    if origin.shape != (3,) or rotation.shape != (3, 3):
        return
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(rotation)):
        return

    origin_px = _project_one(origin, intrinsics)
    if origin_px is None:
        return

    wrist = result.gripper.metadata.get("wrist_position_m")
    if wrist is not None:
        wrist_px = _project_one(np.asarray(wrist, dtype=np.float64), intrinsics)
        if wrist_px is not None:
            cv2.line(image, wrist_px, origin_px, (200, 200, 200), 1)
            cv2.circle(image, wrist_px, 4, (200, 200, 200), -1)

    opening = float(result.gripper.opening_width_m)
    axis_length_m = float(np.clip(max(opening * 1.8, 0.055), 0.055, 0.12))
    axes = [
        ("X", rotation[:, 0], (0, 0, 255)),
        ("Y", rotation[:, 1], (0, 255, 0)),
        ("Z", rotation[:, 2], (255, 0, 0)),
    ]
    cv2.circle(image, origin_px, 6, (255, 255, 255), -1)
    cv2.circle(image, origin_px, 8, (20, 20, 20), 2)
    for label, axis, color in axes:
        endpoint = origin + axis * axis_length_m
        _draw_projected_arrow(cv2, image, intrinsics, origin, endpoint, color, label)


def _draw_projected_arrow(cv2, image, intrinsics, start_xyz, end_xyz, color, label) -> None:
    start_px = _project_one(start_xyz, intrinsics)
    end_px = _project_one(end_xyz, intrinsics)
    if start_px is None or end_px is None:
        return

    cv2.arrowedLine(image, start_px, end_px, color, 3, tipLength=0.05)
    text_anchor = (end_px[0] + 4, end_px[1] - 4)
    cv2.putText(
        image,
        label,
        text_anchor,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )

def project_points(points_xyz_m: np.ndarray, intrinsics) -> np.ndarray:
    points = np.asarray(points_xyz_m, dtype=np.float64)
    z = points[..., 2]
    u = points[..., 0] / z * intrinsics.fx + intrinsics.ppx
    v = points[..., 1] / z * intrinsics.fy + intrinsics.ppy
    return np.stack([u, v], axis=-1)

def _project_one(point_xyz, intrinsics) -> tuple[int, int] | None:
    point = np.asarray(point_xyz, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)) or point[2] <= 0.0:
        return None
    pixel = project_points(point[None, :], intrinsics)[0]
    if not np.all(np.isfinite(pixel)):
        return None
    return tuple(np.round(pixel).astype(int))



def _draw_preview(cv2, image, results, intrinsics) -> None:
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ]
    for result in results:
        pts = result.prediction.keypoints_2d
        for a, b in edges:
            if np.all(np.isfinite(pts[[a, b]])):
                cv2.line(
                    image,
                    tuple(np.round(pts[a]).astype(int)),
                    tuple(np.round(pts[b]).astype(int)),
                    (0, 220, 180),
                    2,
                )
        for idx, point in enumerate(pts):
            if np.all(np.isfinite(point)):
                color = (40, 220, 40) if result.fusion.valid_depth_mask[idx] else (60, 60, 255)
                cv2.circle(image, tuple(np.round(point).astype(int)), 3, color, -1)

        if result.gripper is not None:
            _draw_gripper_pose(cv2, image, result, intrinsics)


def _serialize_array(value) -> list:
    return np.asarray(value, dtype=float).tolist()


def _serialize_result(result) -> dict:
    gripper = None
    if result.gripper is not None:
        quat = rotation_matrix_to_quaternion_xyzw(result.gripper.rotation_camera_gripper)
        gripper = {
            "position_m": _serialize_array(result.gripper.position_m),
            "rotation_camera_gripper": _serialize_array(result.gripper.rotation_camera_gripper),
            "quaternion_xyzw": _serialize_array(quat),
            "opening_width_m": float(result.gripper.opening_width_m),
            "confidence": float(result.gripper.confidence),
        }

    return {
        "handedness": result.prediction.handedness,
        "score": float(result.prediction.score),
        "bbox_xyxy": (
            None
            if result.prediction.bbox_xyxy is None
            else _serialize_array(result.prediction.bbox_xyxy)
        ),
        "keypoints_2d_px": _serialize_array(result.prediction.keypoints_2d),
        "keypoints_3d_m": _serialize_array(result.fusion.keypoints_3d_m),
        "keypoint_depth_m": _serialize_array(result.fusion.keypoint_depth_m),
        "valid_depth_mask": result.fusion.valid_depth_mask.astype(bool).tolist(),
        "model_filled_mask": result.fusion.model_filled_mask.astype(bool).tolist(),
        "fusion_confidence": float(result.fusion.confidence),
        "fusion_mode": result.fusion.transform.get("mode"),
        "gripper": gripper,
    }

def _fusion_mode(value: str) -> str:
    return "model_depth_aligned" if value == "model-depth" else "keypoint_depth"

# 1.初始化机器人（原代码逻辑）
bestman = Bestman_Real_Franka3()
# if bestman.initialize_robot() is not True:
#     exit(-1)
# bestman.open_gripper()
# home_js = np.array([-0.11582,-0.476437,0.0715459,-1.69814,0.0351751,1.22258,0.745147])
# bestman.go_home()
# skill_franka3_database = skill_database.SkillFranka3Database()

#2.Camera Initialize
camera_devices = list( bestman.cfg.Camera.keys())
camera_hand,camera_overhead = None,None
for cam_dev in camera_devices:
    if cam_dev == "D435I":
        camera_hand = Camera_Realsense(bestman.cfg.Camera[cam_dev])
    if cam_dev == "L515":
        camera_overhead = Camera_Realsense(bestman.cfg.Camera[cam_dev])
if camera_overhead == None:
    camera_overhead = camera_hand

time.sleep(bestman.cfg.Camera.init_delay)



sys.path.append("/home/liusong/ProgramFiles/HandPoseExtraction")
from handpose_extraction import DepthFusionConfig, HandPosePipeline
from handpose_extraction.adapters import WiLoREstimator
from handpose_extraction.geometry import rotation_matrix_to_quaternion_xyzw
from handpose_extraction.realsense import RealSenseConfig, RealSenseD435i
from handpose_extraction.smoothing import ExponentialPointSmoother

parser.add_argument("--wilor-repo", required=False, help="Path to the official WiLoR repository.",default="/home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR")
parser.add_argument("--checkpoint", default="pretrained_models/wilor_final.ckpt")
parser.add_argument("--model-cfg", default="pretrained_models/model_config.yaml")
parser.add_argument("--detector", default="pretrained_models/detector.pt")
parser.add_argument("--bag", default=None, help="Optional RealSense Viewer .bag recording.")
parser.add_argument("--jsonl", default=None, help="Optional output JSONL path.")
parser.add_argument("--show", action="store_true", help="Show OpenCV preview.")
parser.add_argument("--fast", action="store_true", help="Use WiLoR fast FP16 mode when available.")
parser.add_argument("--color-width", type=int, default=640)
parser.add_argument("--color-height", type=int, default=480)
parser.add_argument("--depth-width", type=int, default=640)
parser.add_argument("--depth-height", type=int, default=480)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--disable-realsense-filters", action="store_true")
parser.add_argument("--depth-window", type=int, default=7)
parser.add_argument("--min-depth", type=float, default=0.15)
parser.add_argument("--max-depth", type=float, default=3.0)
parser.add_argument("--max-keypoint-depth-deviation", type=float, default=0.35)
parser.add_argument(
    "--fusion-mode",
    choices=("model-depth", "keypoint-depth"),
    default="model-depth",
    help="model-depth uses WiLoR/MANO 3D structure with global depth alignment; keypoint-depth uses legacy per-keypoint depth.",
)
parser.add_argument("--disable-model-depth-correction", action="store_true")
parser.add_argument("--model-depth-max-residual-m", type=float, default=0.18)
parser.add_argument("--gripper-x-offset-cm", type=float, default=0.0) ##########Gripper X Offset
parser.add_argument("--gripper-z-offset-cm", type=float, default=3.5) ##########Gripper Len Offset
parser.add_argument("--disable-gripper-stabilizer", action="store_true")
parser.add_argument("--gripper-position-alpha", type=float, default=0.55)
parser.add_argument("--gripper-rotation-alpha", type=float, default=0.45)
parser.add_argument("--gripper-max-position-jump-m", type=float, default=0.12)
parser.add_argument("--gripper-max-rotation-jump-deg", type=float, default=70.0)
parser.add_argument("--gripper-max-consecutive-holds", type=int, default=12)
parser.add_argument("--disable-adaptive-gripper-smoothing", action="store_true")
parser.add_argument("--gripper-fast-alpha", type=float, default=0.85)
parser.add_argument("--gripper-position-fast-threshold-m", type=float, default=0.04)
parser.add_argument("--gripper-rotation-fast-threshold-deg", type=float, default=25.0)
parser.add_argument("--track-match-distance-m", type=float, default=0.22)
parser.add_argument("--max-track-missing-frames", type=int, default=30)
parser.add_argument("--force-handedness", choices=("left", "right"), default=None)
parser.add_argument("--smooth-alpha", type=float, default=0.9)
parser.add_argument("--stats-window", type=int, default=30)
args = parser.parse_args()



estimator = WiLoREstimator(
    repo_root=args.wilor_repo,
    checkpoint_path=args.checkpoint,
    cfg_path=args.model_cfg,
    detector_path=args.detector,
    fast=args.fast,
)
pipeline = HandPosePipeline(
    estimator=estimator,
    fusion_config=DepthFusionConfig(
        fusion_mode=_fusion_mode(args.fusion_mode),
        sample_window=args.depth_window,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
        max_keypoint_depth_deviation_m=args.max_keypoint_depth_deviation,
        model_depth_correction=not args.disable_model_depth_correction,
        model_depth_max_residual_m=args.model_depth_max_residual_m,
    ),
    smoother=ExponentialPointSmoother(alpha=args.smooth_alpha),
    gripper_x_offset_m=args.gripper_x_offset_cm / 100.0,
    gripper_z_offset_m=args.gripper_z_offset_cm / 100.0,
    gripper_stabilizer=None,
    track_match_distance_m=args.track_match_distance_m,
    max_track_missing_frames=args.max_track_missing_frames,
    forced_handedness=args.force_handedness,
)


############4.ADD SIMDATACOLLETION
# Parse command line arguments
parser.add_argument('--task', type=str, default="test3")  # open_lid, open_fridge, open_drawer, pick_place_pot
parser.add_argument('--config_file', type=str, default='Config/real_data_collection_config.json')
sim_data_collection = RealDataCollection(parser) 


#Consumer
send_data_thread = threading.Thread(
    target=collection_data_update,
    args=(camera_hand,camera_overhead, pipeline, sim_data_collection))
send_data_thread.start()


videowriter = cv2.VideoWriter('hand_camera_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1280,720))
click_camera_name = "overhead"
print("Collecting Data...")

while True:
    collect_data_thread = threading.Thread(target=sim_data_collection.data_collection)
    collect_data_thread.start()

    rgbd_frame = camera_overhead.get_rgbd_image()
    results = pipeline.process_frame(rgbd_frame)
    preview = rgbd_frame.color_bgr.copy()
    _draw_preview(cv2, preview, results, rgbd_frame.intrinsics)
    videowriter.write(preview)
    if cv2.waitKey(33) & 0xFF in (27, ord("q")):
        break
    cv2.imshow("WiLoR RealSense hand pose", preview)


    if sys.stdin in select.select([sys.stdin], [], [],  0.01)[0]:
        line = sys.stdin.readline()
        pressed_key = line.strip()
        if line:
            print(f"You pressed: {pressed_key}")
        if pressed_key == 'n':
            # #2.Consumer Save & Restart
            sim_data_collection.episode_finish=True 
            sim_data_collection.data_save_hdf5()
            videowriter.release()
            videowriter = cv2.VideoWriter('hand_camera_output1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640,480))
        if pressed_key == 'q':
            print("Program Over!")
            break




# bestman.release_robot()
# exit(-1)


# #Overhead Extrics Label with CloudCompare
# import open3d as o3d
# camera_hand_points = camera_hand.get_3d_points()
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(camera_hand_points[0])
# pcd.colors = o3d.utility.Vector3dVector(camera_hand_points[1])
# # 保存为ASCII格式的PLY文件
# o3d.io.write_point_cloud("/home/liusong/桌面/temp.ply", pcd, write_ascii=True)

# import open3d as o3d
# camera_overhead_points = camera_overhead.get_3d_points()
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(camera_overhead_points[0])
# pcd.colors = o3d.utility.Vector3dVector(camera_overhead_points[1])
# # 保存为ASCII格式的PLY文件
# o3d.io.write_point_cloud("/home/liusong/桌面/temp1.ply", pcd, write_ascii=True)

