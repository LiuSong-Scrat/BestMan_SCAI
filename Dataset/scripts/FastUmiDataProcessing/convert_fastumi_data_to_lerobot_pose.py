"""
Script to convert Fastumi HDF5 data to the LeRobot dataset v2.0 format.

This version supports:
1. **Recursive batch import** – all `*.hdf5` files inside `--raw-dir` (including
   sub‑folders) are automatically detected and treated as episodes.
2. **Custom task name** via `--task`.
3. **Custom FPS** via `--fps` (frame‑rate stored in the LeRobot dataset).
4. **Custom output directory** via `--output-dir`; the finished dataset folder
   will be copied there instead of the default `HF_LEROBOT_HOME`.

Example
-------
```bash
uv run convert_fastumi_data_to_lerobot.py \
  --raw-dir /root/dataset/unzip/sweep_trash/merged_sweep_trash_deltatcp \
  --repo-id myteam/sweep_trash \
  --task  "grab the small broom and sweep the trash into the dustpan" \
  --fps 20 \
  --output-dir /root/dataset/unzip/sweep_trash/merged_sweep_trash_deltatcp_rgb224 \
  --mode image        
```
"""
from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path
from typing import Literal, List, Dict, Tuple

import h5py
import numpy as np
import torch
import tqdm
import tyro
import time
import threading


from lerobot.common.datasets.lerobot_dataset import (
    HF_LEROBOT_HOME,
    LeRobotDataset,
)
from lerobot.common.datasets.push_dataset_to_hub._download_raw import (
    download_raw,
)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1e-4
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


# ---------------------------------------------------------------------------
# Dataset creation utilities
# ---------------------------------------------------------------------------

def create_empty_dataset(
    repo_id: str,
    fps: int,
    robot_type: str = "fastumi",
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Initialise an empty LeRobot dataset ready to receive frames."""

    # # FastUMI state definition: (x, y, z, qw, qx, qy, qz, gripper_width)
    # state_names = [
    #     "joint_1",
    #     "joint_2",
    #     "joint_3",
    #     "joint_4",
    #     "joint_5",
    #     "joint_6",
    #     "joint_7",
    #     "gripper_width",
    # ]

    # cameras = ["overhead","hand"]
    # features: Dict[str, Dict[str, object]] = {
    #     "observation.state": {
    #         "dtype": "float32",
    #         "shape": (len(state_names),),
    #         "names": state_names,
    #     },
    #     "action": {
    #         "dtype": "float32",
    #         "shape": (8,),
    #         "names": [
    #             "joint_1",
    #             "joint_2",
    #             "joint_3",
    #             "joint_4",
    #             "joint_5",
    #             "joint_6",
    #             "joint_7",
    #             "gripper_width",
    #         ],
    #     },
    # }

    # FastUMI state definition: (x, y, z, qw, qx, qy, qz, gripper_width)
    state_names = [
        "joint_1",#X
        "joint_2",#Y
        "joint_3",#Z
        "joint_4",#R
        "joint_5",#P
        "joint_6",#Y
        "gripper_width",
    ]

    cameras = ["overhead","hand"]
    features: Dict[str, Dict[str, object]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "gripper_width",
            ],
        },
    }


    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(state_names),),
            "names": state_names,
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (480, 640, 3),  # (H, W, C) 224p RGB
            "names": ["height", "width", "channels"],
        }

    # Clear previous dataset folder if it already exists
    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    print(dataset_config.image_writer_processes, dataset_config.image_writer_threads)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_images(ep: h5py.File, cameras: List[str]) -> Dict[str, np.ndarray]:
    """加载并处理视频帧：解码（如有需要）、裁剪左右各13%、缩放。"""
    import cv2  # 本地导入避免额外依赖
    imgs: Dict[str, np.ndarray] = {}

    for cam in cameras:
        imgs[cam] = ep[f"/observations/images/{cam}"][:].astype(np.float32)/255
        
        # ds = ep[f"/observations/images/{cam}"]
        # # 解码
        # if ds.ndim == 4:
        #     arr = ds[:]                  # shape (T, H, W, 3) BGR
        #     arr = arr[..., ::-1]         # convert BGR→RGB 
        # else:
        #     # JPEG bytes，需要逐帧解码
        #     decoded: List[np.ndarray] = []
        #     for buf in ds:
        #         bgr = cv2.imdecode(buf, 1)
        #         rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        #         decoded.append(rgb)
        #     arr = np.stack(decoded, axis=0)  # (T, H, W, 3)
        # cv2.imwrite("/home/liusong/ProgramFiles/IssacSim/Tasks_Wokspace/dataset/test3/camera/cam.png", arr[0])

        # # 计算裁剪范围：左右各13%
        # T, H, W, C = arr.shape
        # left = int(0.0 * W)
        # right = W - left
        # cropped = arr[:, :, left:right, :]  # (T, H, W*0.74, 3)

        # # 缩放到 256×256
        # resized_frames = []
        # for frame in cropped:
        #     # 注意：cv2.resize 的尺寸参数是 (宽, 高)
        #     frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        #     # cv2.imwrite("/home/chenpengan/hny/vla/lerobot/lerobot/scripts/cam.jpg", frame_resized)

        #     resized_frames.append(frame_resized)
        # imgs[cam] = np.stack(resized_frames, axis=0)  # (T, 224, 224, 3)

    return imgs

def _load_episode(ep_path: Path) -> Tuple[
    Dict[str, np.ndarray],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    cameras = ["overhead","hand"]
    with h5py.File(ep_path, "r") as ep:
        # state = torch.from_numpy(ep["/observations/qpos"][:])  # (T, 8)
        # action = torch.from_numpy(ep["/action"][:])           # (T, 7)
        state = torch.from_numpy(ep["/observations/qpos"][:].astype(np.float32))  # (T, 8)
        action = torch.from_numpy(ep["/action"][:].astype(np.float32))            # (T, 7)
        velocity = None  # FastUMI data do not contain these by default
        effort = None
        # imgs = _load_images(ep, cameras)
        imgs: Dict[str, np.ndarray] = {}
        for cam in cameras:
            imgs[cam] = np.round(ep[f"/observations/images/{cam}"][:].astype(np.float32)/255,4)

        # imgs: Dict[str, np.ndarray] = {}
        # for cam in cameras:
        #     ds = ep[f"/observations/images/{cam}"]
        #     arr = ds[:]/255                  # shape (T, H, W, 3) BGR
        #     imgs[cam] = arr


    return imgs, state, action, velocity, effort



def _add_episode(
    dataset: LeRobotDataset,
    ep_path: Path,
    task: str,
):
    imgs, state, action, velocity, effort = _load_episode(ep_path)

    for i in range(state.shape[0]):
        frame = {
            "observation.state": state[i],
            "action": action[i],
            "task": task,
        }
        for cam, arr in imgs.items():
            frame[f"observation.images.{cam}"] = arr[i]
        if velocity is not None:
            frame["observation.velocity"] = velocity[i]
        if effort is not None:
            frame["observation.effort"] = effort[i]
        dataset.add_frame(frame)

    dataset.save_episode()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def port_fastumi(
    *,
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    fps: int = 30,
    output_dir: Path | None = None,
    raw_repo_id: str | None = None,
    episodes: List[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Convert FastUMI episodes to LeRobot dataset format."""

    # ---------------------------------------------------------------------
    # 1. Gather raw episodes ------------------------------------------------
    # ---------------------------------------------------------------------
    if not raw_dir.exists():
        if raw_repo_id is None:
            raise FileNotFoundError(
                f"{raw_dir} does not exist and --raw-repo-id was not provided."
            )
        download_raw(raw_dir, repo_id=raw_repo_id)

    hdf5_files: List[Path] = sorted(raw_dir.rglob("*.hdf5"))
    if not hdf5_files:
        raise RuntimeError(f"No .hdf5 files found under {raw_dir}")

    if episodes is None:
        ep_indices = range(len(hdf5_files))
    else:
        ep_indices = episodes

    # ---------------------------------------------------------------------
    # 2. Create empty dataset container -----------------------------------
    # ---------------------------------------------------------------------
    dataset = create_empty_dataset(
        repo_id=repo_id,
        fps=fps,
        robot_type="fastumi",
        mode=mode,
        has_velocity=False,
        has_effort=False,
        dataset_config=dataset_config,
    )

    # time.sleep(10)  # 确保目录创建完成
    # ---------------------------------------------------------------------
    # 3. Populate dataset ---------------------------------------------------
    # ---------------------------------------------------------------------
    for idx in tqdm.tqdm(ep_indices, desc="Converting episodes"):
        # if idx>52:
        #     break
        print("i = ",idx)
        _add_episode(dataset, hdf5_files[idx], task)
        
    # dataset.consolidate()

    # ---------------------------------------------------------------------
    # 4. Persist to custom output location if requested --------------------
    # ---------------------------------------------------------------------
    if output_dir is not None:
        target = output_dir / repo_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(HF_LEROBOT_HOME / repo_id, target)
        print(f"[Info] Dataset copied to {target}")
    # ---------------------------------------------------------------------
    # 5. Push to Hub (optional) -------------------------------------------
    # ---------------------------------------------------------------------
    if push_to_hub:
        dataset.push_to_hub()


def convert_pick_lid_data():
    """专门用于转换pick_lid数据的函数"""
    task_name = "simple_isaacsim_pose_absolute"
    port_fastumi(
        raw_dir=Path(f"dataset/test3/src_hdf5_to_lerobot/src_hdf5_to_processed_hdf5_{task_name}"),
        # raw_dir=Path("/home/chenpengan/hny/hdf5s_song/pick_lid_processed"),
        repo_id="LiuSong-Scrat/smolvla",
        task="Put the red cube on top of the blue cube.",
        fps=10,
        output_dir=Path(f"/home/liusong/ProgramFiles/IssacSim/Tasks_Wokspace/dataset/test3/src_hdf5_to_lerobot/lerobot_datasets/processed_hdf5_to_lerobot_{task_name}"),
        mode="image",
    )


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 直接调用转换函数
    convert_pick_lid_data()