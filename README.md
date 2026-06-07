# HumanHand Offline RGB-D To HDF5

This project-side workflow uses `record_bestman_rgbd.py` for offline RGB-D capture,
HandPoseExtraction WiLoR for offline gripper inference, then
`build_humanhand_hdf5_dataset.py` for visual slicing and BestMan-style HDF5 export.

## 1. Record RGB-D

```bash
python record_bestman_rgbd.py \
  --camera L515 \
  --output outputs/rgbd_records/humanhand_demo \
  --num-frames 300 \
  --show
```

By default this uses compact storage: `color_jpg/*.jpg` and `depth_png/*.png`.
Depth PNG is `uint16` millimeters, so it is much smaller than float32 NPY with about 1 mm quantization.

For maximum RGB compression, use MP4 color video plus depth PNG:

```bash
python record_bestman_rgbd.py \
  --camera L515 \
  --output outputs/rgbd_records/humanhand_demo_video \
  --num-frames 300 \
  --storage video \
  --video-fps 15
```

To mark training episodes during capture, start in pause mode and use Space to start/pause recording:

```bash
python record_bestman_rgbd.py \
  --camera L515 \
  --output outputs/rgbd_records/humanhand_demo_video \
  --storage video \
  --video-fps 15 \
  --space-toggle-recording
```

This writes:

- `segments.txt`: directly usable as `build_humanhand_hdf5_dataset.py --segments`
- `segments.json`: detailed start/pause events, including saved record indices and camera frame numbers

For the old lossless static-frame format, use:

```bash
python record_bestman_rgbd.py \
  --camera L515 \
  --output outputs/rgbd_records/humanhand_demo_legacy \
  --num-frames 300 \
  --storage legacy
```

## 2. Run Inference And Slice Interactively

```bash
python build_humanhand_hdf5_dataset.py \
  --input outputs/rgbd_records/humanhand_demo \
  --output-dir Dataset/dataset/humanhand_offline \
  --run-inference \
  --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
  --fast \
  --force-handedness right \
  --fusion-mode model-depth \
  --gripper-x-offset-cm 0.0 \
  --gripper-z-offset-cm 3.5 \
  --camera-names overhead,hand
```

Controls:

- `Right/D`: next frame
- `Left/A`: previous frame
- `Up/W`: set current frame as segment start
- `Down/S`: set current frame as segment end and save one `episode_*.hdf5`
- `R`: clear current start
- `U`: delete the last saved HDF5 in this session
- `Q/Esc`: quit

## 3. Reuse Existing Inference

```bash
python build_humanhand_hdf5_dataset.py \
  --input outputs/rgbd_records/humanhand_demo \
  --jsonl outputs/rgbd_records/humanhand_demo/handpose_wilor.jsonl \
  --output-dir Dataset/dataset/humanhand_offline
```

## 4. Batch Export Known Segments

```bash
python build_humanhand_hdf5_dataset.py \
  --input outputs/rgbd_records/humanhand_demo \
  --jsonl outputs/rgbd_records/humanhand_demo/handpose_wilor.jsonl \
  --output-dir Dataset/dataset/humanhand_offline \
  --gripper-x-offset-cm 0.0 \
  --gripper-z-offset-cm 3.5 \
  --no-interactive \
  --segments 0:120,150:260
```

If the segments were marked with `--space-toggle-recording`, reuse them directly:

```bash
python build_humanhand_hdf5_dataset.py \
  --input outputs/rgbd_records/humanhand_demo_video \
  --jsonl outputs/rgbd_records/humanhand_demo_video/handpose_wilor.jsonl \
  --output-dir Dataset/dataset/humanhand_offline \
  --no-interactive \
  --segments "$(cat outputs/rgbd_records/humanhand_demo_video/segments.txt)" \
  --transform-to-world \
  --camera-names overhead \
  --max-points 4096 \
  --segment-workers 4
```

`build_humanhand_hdf5_dataset.py` 默认保存每帧 4096 个点。需要旧的全分辨率点云时显式加
`--max-points 307200`，但 HDF5 会明显变大、切片也会更慢。非交互式 `--segments`
导出可以用 `--segment-workers N` 按 segment 并行；点云较大时建议先用 `2` 到 `4`。

The saved HDF5 fields match `Dataset/scripts/data_collection.py`:

- `observations/images/{camera_name}`
- `observations/cloud_rgb/{camera_name}`
- `observations/qpos`
- `observations/pose_eular`
- `observations/eff_angular`
- `action`

Extra debug/training fields are also saved:

- `observations/gripper_quat_xyzw`
- `observations/keypoints_3d_m`
- `source_record_index`
- `timestamp_ms`

By default pose and cloud data are in the camera frame. To export using the fixed L515
camera-to-base transform from `sample_runthrough_HumanHand.py`, add:

```bash
--pose-frame base --camera-to-base-preset humanhand_l515
```





###########Record##########如果使用了--show则是持续录制#######--space-toggle-recording 是键盘控制启停#####
推荐采集：
python record_bestman_rgbd.py \
  --camera L515 \
  --output outputs/rgbd_records/humanhand_demo \
  --num-frames 300 \
  --storage compressed \
  --jpeg-quality 92 \
  --space-toggle-recording 

更省 RGB 空间：
python record_bestman_rgbd.py \
  --camera L515 \
  --output outputs/rgbd_records/humanhand_demo_video \
  --num-frames 0 \
  --storage video \
  --video-fps 30 \
  --space-toggle-recording 



###############################
#inference 可视化  --run-inference or --show-inference
#切片可视化 --show 
#如果要一键离线推理再交互切片：
python build_humanhand_hdf5_dataset.py \
  --input outputs/rgbd_records/humanhand_demo_video \
  --output-dir Dataset/dataset/humanhand_offline \
  --run-inference \
  --wilor-repo /home/liusong/ProgramFiles/HandPoseExtraction/external/WiLoR \
  --fast \
  --force-handedness right \
  --fusion-mode model-depth \
  --camera-names overhead,hand
  --transform-to-world 
  --show


#直接用推理后的结果交互切片
python build_humanhand_hdf5_dataset.py \
  --input outputs/rgbd_records/humanhand_demo_video \
  --output-dir Dataset/dataset/humanhand_offline \
  --transform-to-world 

#离线推理后切分点切片
python build_humanhand_hdf5_dataset.py \
  --run-inference \
  --input outputs/rgbd_records/humanhand_demo_video \
  --jsonl outputs/rgbd_records/humanhand_demo_video/handpose_wilor.jsonl \
  --output-dir Dataset/dataset/humanhand_offline \
  --no-interactive \
  --transform-to-world \
  --segments "$(cat outputs/rgbd_records/humanhand_demo_video/segments.txt)" \
  --max-points 50000 \
  --segment-workers 4
  
