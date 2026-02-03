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
import json

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

    # FastUMI state definition: (x, y, z, qw, qx, qy, qz, gripper_width)
    state_names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
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
            "shape": (8,),
            "names": [
                "joint_1",
                "joint_2",
                "joint_3",
                "joint_4",
                "joint_5",
                "joint_6",
                "joint_7",
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


import threading
import time

# 定义一个函数，用于线程执行
def thread_function(args):
    dataset,hdf5_files_sliced,task = args

    for hdf5_file in tqdm.tqdm(hdf5_files_sliced):
        _add_episode(dataset, hdf5_file, task)

    
    # dataset.consolidate()
    print(f"线程 加载")



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

def merge_datasets(thread_num,repo_id,chunk_name: str = "chunk-000",delete_source=True):
    """
    Merge multiple LeRobot dataset folders into a single dataset folder.

    - dataset_dirs: list of sub dataset folders (e.g. smolvla0, smolvla1...)
    - merged_dir: output merged dataset folder
    - chunk_name: data chunk subfolder name (default "chunk-000")
    """

    merged_dir = HF_LEROBOT_HOME / repo_id
    dataset_dirs = []
    for i in range(thread_num):
        sub_dataset_path = HF_LEROBOT_HOME / (repo_id+str(i))
        dataset_dirs.append(sub_dataset_path)



    merged_dir = Path(merged_dir)
    merged_data_dir = merged_dir / "data" / chunk_name
    merged_meta_dir = merged_dir / "meta"
    merged_data_dir.mkdir(parents=True, exist_ok=True)
    merged_meta_dir.mkdir(parents=True, exist_ok=True)

    # accumulators
    global_episode_counter = 0
    merged_episodes: List[Dict[str, Any]] = []
    merged_episode_stats: List[Dict[str, Any]] = []
    merged_tasks: List[Dict[str, Any]] = []
    merged_info: Dict[str, Any] | None = None

    # We need per-dataset mapping:
    # for each dataset we'll build:
    #  old_task_index -> new_task_index
    #  old_episode_index -> new_episode_index
    for dset in dataset_dirs:
        dset = Path(dset)
        data_dir = dset / "data" / chunk_name
        meta_dir = dset / "meta"

        # load tasks from this dataset and allocate new task indices
        task_map: Dict[int, int] = {}
        task_file = meta_dir / "tasks.jsonl"
        if task_file.exists():
            with open(task_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    task_obj = json.loads(line)
                    old_tidx = task_obj.get("task_index")
                    # assign new index sequentially
                    new_tidx = len(merged_tasks)
                    # update the task object to new index
                    task_obj["task_index"] = new_tidx
                    merged_tasks.append(task_obj)
                    if old_tidx is None:
                        # if no explicit old index, we can't map; but episodes may reference
                        # by old index; assume they refer by order - we won't attempt to guess.
                        pass
                    else:
                        task_map[int(old_tidx)] = new_tidx
        else:
            # no tasks file: leave task_map empty
            task_map = {}

        # load episodes metadata from this dataset (list in order)
        src_episodes = []
        episodes_file = meta_dir / "episodes.jsonl"
        if episodes_file.exists():
            with open(episodes_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ep = json.loads(line)
                    src_episodes.append(ep)

        # Build mapping for episodes: old episode_index -> new index
        episode_map: Dict[int, int] = {}
        # For robustness, also map by filename if present
        # list files in data dir sorted (matching written order)
        src_parquets = sorted((data_dir).glob("episode_*.parquet"))
        # We prefer to iterate src_episodes (meta) to preserve original order and to update fields.
        for i, ep in enumerate(src_episodes):
            # determine the old index:
            old_idx = ep.get("episode_index", i)
            new_idx = global_episode_counter
            # copy corresponding parquet file:
            # prefer to locate parquet by episode_path in ep if available, else use list order
            src_ep_path = None
            if "episode_path" in ep:
                # episode_path might be like "data/chunk-000/episode_000001.parquet"
                candidate = (dset / ep["episode_path"]).resolve()
                if candidate.exists():
                    src_ep_path = candidate
            if src_ep_path is None:
                # fallback: use file at same order
                if i < len(src_parquets):
                    src_ep_path = src_parquets[i]
                elif len(src_parquets) == 1:
                    src_ep_path = src_parquets[0]
                else:
                    raise FileNotFoundError(f"Cannot find parquet for episode {ep} in {dset}")

            # copy and rename
            new_name = f"episode_{new_idx:06d}.parquet"
            shutil.copy(src_ep_path, merged_data_dir / new_name)

            # update episode metadata fields
            ep["episode_index"] = new_idx
            ep["episode_path"] = f"data/{chunk_name}/{new_name}"

            # update task_index in episode meta if present (map old->new)
            if "task_index" in ep:
                old_tidx = ep["task_index"]
                if old_tidx is None:
                    ep["task_index"] = None
                else:
                    # if mapping exists use it, else assign None (safe fallback)
                    ep["task_index"] = task_map.get(int(old_tidx), None)

            merged_episodes.append(ep)
            episode_map[int(old_idx)] = new_idx
            global_episode_counter += 1

        # If there are parquet files but no episodes.jsonl (edge case), try to import them by filename
        if not src_episodes and src_parquets:
            for i, src_parquet in enumerate(src_parquets):
                new_idx = global_episode_counter
                new_name = f"episode_{new_idx:06d}.parquet"
                shutil.copy(src_parquet, merged_data_dir / new_name)
                # create a minimal episode entry
                ep = {
                    "episode_index": new_idx,
                    "episode_path": f"data/{chunk_name}/{new_name}",
                }
                # no task info available
                merged_episodes.append(ep)
                episode_map[i] = new_idx
                global_episode_counter += 1

        # Merge episodes_stats.jsonl: need to remap episode_index (and task_index if present)
        stats_file = meta_dir / "episodes_stats.jsonl"
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    st = json.loads(line)
                    # remap episode_index if present
                    if "episode_index" in st:
                        old_eidx = st["episode_index"]
                        if old_eidx is None:
                            st["episode_index"] = None
                        else:
                            # map using episode_map, fallback to None if missing
                            st["episode_index"] = episode_map.get(int(old_eidx), None)
                    # remap task_index in stats if present
                    if "task_index" in st:
                        old_tidx = st["task_index"]
                        st["task_index"] = task_map.get(int(old_tidx), None) if old_tidx is not None else None

                    merged_episode_stats.append(st)

        # Merge info.json (consistency check)
        info_file = meta_dir / "info.json"
        if info_file.exists():
            with open(info_file, "r", encoding="utf-8") as f:
                info = json.load(f)
            if merged_info is None:
                merged_info = info
            else:
                # check a few key fields for consistency (fps, robot_type, mode)
                for key in ("fps", "robot_type", "mode"):
                    if key in info and key in merged_info:
                        if info[key] != merged_info[key]:
                            raise ValueError(f"Info.json mismatch for key '{key}': {info[key]} != {merged_info[key]} (in {dset})")
                # Optionally we could merge some fields; for now we keep merged_info from first dataset.

    # ---------- Save merged meta ----------
    # episodes.jsonl
    with open(merged_meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in merged_episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # episodes_stats.jsonl
    with open(merged_meta_dir / "episodes_stats.jsonl", "w", encoding="utf-8") as f:
        for st in merged_episode_stats:
            f.write(json.dumps(st, ensure_ascii=False) + "\n")

    # tasks.jsonl (we kept duplicates and assigned new indices)
    with open(merged_meta_dir / "tasks.jsonl", "w", encoding="utf-8") as f:
        for t in merged_tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # info.json
    if merged_info is None:
        merged_info = {"note": "no info.json found in inputs"}
    with open(merged_meta_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(merged_info, f, ensure_ascii=False, indent=2)

    print(f"Merged {len(dataset_dirs)} datasets into {merged_dir}")
    print(f"Total episodes: {len(merged_episodes)}")
    print(f"Total tasks: {len(merged_tasks)}")

    # ===============================================================
    # 删除源数据集目录（若启用）
    # ===============================================================
    if delete_source:
        print("\n[INFO] Deleting source dataset folders...")
        for d in dataset_dirs:
            try:
                shutil.rmtree(d)
                print(f"  ✔ deleted {d}")
            except Exception as e:
                print(f"  ✘ failed to delete {d}: {e}")
        print("[INFO] All source datasets deleted.")

    print("\n[DONE]")

def run_create_datasets(thread_num,hdf5_files,repo_id,fps,mode,dataset_config,task,ep_indices):
    ###########串行############
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
        # if idx>5:
        #     break
        print("i = ",idx)
        _add_episode(dataset, hdf5_files[idx], task)
    # dataset.consolidate()

def run_threads_create_datasets(thread_num,hdf5_files,repo_id,fps,mode,dataset_config,task):
    args_list = []
    for i in range(thread_num):
        dataset = create_empty_dataset(
                        repo_id=repo_id+str(i),
                        fps=fps,
                        robot_type="fastumi",
                        mode=mode,
                        has_velocity=False,
                        has_effort=False,
                        dataset_config=dataset_config,
                    )
        all_file_len = len(hdf5_files)
        slice_len = int(all_file_len/thread_num)
        start_idx = i*slice_len
        end_idx = (i+1)*slice_len if i != thread_num-1 else all_file_len
        hdf5_files_sliced =  hdf5_files[start_idx:end_idx]
        args_list.append((dataset,hdf5_files_sliced,task))
    # 创建线程
    threads = []
    for i in range(thread_num):
        thread = threading.Thread(target=thread_function, args=(args_list[i],))
        threads.append(thread)
        thread.start()
    # 在每个线程上调用join方法
    for thread in threads:
        thread.join()
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
    # 3. Populate dataset ---------------------------------------------------
    # ---------------------------------------------------------------------
    # ############串行############
    # run_create_datasets(thread_num,hdf5_files,repo_id,fps,mode,dataset_config,task,ep_indices)
    ############并行############
    thread_num = 4
    run_threads_create_datasets(thread_num,hdf5_files,repo_id,fps,mode,dataset_config,task)
    merge_datasets(thread_num,repo_id)


    # ---------------------------------------------------------------------
    # 4. Persist to custom output location if requested --------------------
    # ---------------------------------------------------------------------
    if output_dir is not None:
        target = output_dir / repo_id
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(HF_LEROBOT_HOME / repo_id, target)
        print(f"[Info] Dataset copied to {target}")


    # # ---------------------------------------------------------------------
    # # 5. Push to Hub (optional) -------------------------------------------
    # # ---------------------------------------------------------------------
    # if push_to_hub:
    #     dataset.push_to_hub()


def convert_pick_lid_data():
    """专门用于转换pick_lid数据的函数"""
    task_name = "real_franka3_place_cube_joint_absolute"
    port_fastumi(
        raw_dir=Path(f"Dataset/dataset/test3/src_hdf5_to_lerobot/src_hdf5_to_processed_hdf5_{task_name}"),
        # raw_dir=Path("/home/chenpengan/hny/hdf5s_song/pick_lid_processed"),
        repo_id="LiuSong-Scrat/smolvla",
        task="Put the red cube on top of the blue cube.",
        fps=10,
        output_dir=Path(f"Dataset/dataset/test3/src_hdf5_to_lerobot/lerobot_datasets/processed_hdf5_to_lerobot_{task_name}"),
        mode="image",
    )




# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 直接调用转换函数
    convert_pick_lid_data()