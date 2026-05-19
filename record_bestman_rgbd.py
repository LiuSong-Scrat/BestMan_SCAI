#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import cv2
import numpy as np


DEFAULT_BESTMAN_ROOT = Path("/home/liusong/ProgramFiles/BestMan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record aligned RGB-D frames from BestMan RealSense cameras.")
    parser.add_argument("--bestman-root", default=str(DEFAULT_BESTMAN_ROOT))
    parser.add_argument(
        "--config",
        default=str(DEFAULT_BESTMAN_ROOT / "Config/default_franka3.yaml"),
        help="BestMan YAML config containing Camera.L515/D435I.",
    )
    parser.add_argument(
        "--camera",
        default="overhead",
        choices=("overhead", "hand", "L515", "D435I"),
        help="overhead prefers L515 and falls back to D435I; hand prefers D435I.",
    )
    parser.add_argument("--output", default=None, help="Output sequence directory.")
    parser.add_argument("--num-frames", type=int, default=0, help="0 means record until Ctrl-C.")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 means no duration limit.")
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save-depth-png", action="store_true", help="Also save uint16 millimeter depth PNG.")
    args = parser.parse_args()

    bestman_root = Path(args.bestman_root).resolve()
    if str(bestman_root) not in sys.path:
        sys.path.insert(0, str(bestman_root))

    from Sensor.Camera_Realsense import Camera_Realsense

    cfg = _load_yaml_namespace(Path(args.config))
    output_dir = Path(args.output) if args.output else _default_output_dir(args.camera)
    output_dir.mkdir(parents=True, exist_ok=True)
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth_m"
    depth_png_dir = output_dir / "depth_png"
    color_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    if args.save_depth_png:
        depth_png_dir.mkdir(exist_ok=True)



    camera = None
    frames_file = None
    try:
        camera_name, camera_cfg, camera = _open_camera(Camera_Realsense, cfg.Camera, args.camera)
        init_delay = float(getattr(cfg.Camera, "init_delay", 0.0))
        if init_delay > 0.0:
            time.sleep(init_delay)

        for _ in range(max(0, int(args.warmup_frames))):
            camera.get_rgbd_image()

        metadata = {
            "format": "handpose_rgbd_sequence_v1",
            "camera_request": args.camera,
            "camera_config_name": camera_name,
            "camera_dev_name": getattr(camera_cfg, "dev_name", None),
            "created_unix_s": time.time(),
            "intrinsics": _intrinsics_to_dict(camera.get_rgbd_image().intrinsics),
            "depth_unit": "meter",
            "color_format": "bgr_png",
            "depth_format": "float32_npy_meter",
            "bestman_root": str(bestman_root),
            "config": str(Path(args.config).resolve()),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        frames_file = open(output_dir / "frames.jsonl", "w", encoding="utf-8")

        start_s = time.time()
        index = 0
        while True:
            if args.num_frames > 0 and index >= args.num_frames:
                break
            if args.duration_s > 0.0 and time.time() - start_s >= args.duration_s:
                break

            frame = camera.get_rgbd_image()
            color_rel = Path("color") / f"{index:06d}.png"
            depth_rel = Path("depth_m") / f"{index:06d}.npy"
            _save_color(output_dir / color_rel, frame.color_bgr)
            np.save(output_dir / depth_rel, np.asarray(frame.depth_m, dtype=np.float32))

            record = {
                "index": index,
                "timestamp_ms": frame.timestamp_ms,
                "frame_number": frame.frame_number,
                "color_path": color_rel.as_posix(),
                "depth_m_path": depth_rel.as_posix(),
                "intrinsics": _intrinsics_to_dict(frame.intrinsics),
            }
            if args.save_depth_png:
                depth_png_rel = Path("depth_png") / f"{index:06d}.png"
                depth_mm = np.clip(frame.depth_m * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
                cv2.imwrite(str(output_dir / depth_png_rel), depth_mm)
                record["depth_png_path"] = depth_png_rel.as_posix()

            frames_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            frames_file.flush()

            preview = frame.color_bgr.copy()
            cv2.putText(
                preview,
                f"{args.camera} frame {index}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("record RGB-D", preview)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            index += 1
    except KeyboardInterrupt:
        pass
    finally:
        if frames_file is not None:
            frames_file.close()
        if camera is not None:
            camera.close()
        if cv2 is not None:
            cv2.destroyAllWindows()

    print(f"Saved RGB-D sequence to {output_dir}")


def _load_yaml_namespace(path: Path) -> SimpleNamespace:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load BestMan YAML config") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _to_namespace(data)


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{str(k): _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _camera_candidates(camera_cfg, camera: str) -> list[tuple[str, object]]:
    if camera == "overhead":
        candidates = []
        for name in ("L515", "D435I"):
            if hasattr(camera_cfg, name):
                candidates.append((name, getattr(camera_cfg, name)))
        return candidates
    if camera == "hand":
        candidates = []
        for name in ("D435I", "L515"):
            if hasattr(camera_cfg, name):
                candidates.append((name, getattr(camera_cfg, name)))
        return candidates
    if hasattr(camera_cfg, camera):
        return [(camera, getattr(camera_cfg, camera))]
    raise KeyError(f"Camera.{camera} not found in config")


def _open_camera(camera_cls, camera_cfg, camera: str):
    last_error: BaseException | None = None
    for name, cfg in _camera_candidates(camera_cfg, camera):
        try:
            return name, cfg, camera_cls(cfg)
        except SystemExit as exc:
            last_error = exc
            print(f"Failed to open {name}, trying next candidate if available.")
        except Exception as exc:
            last_error = exc
            print(f"Failed to open {name}: {exc}")
    raise RuntimeError(f"Could not open RealSense camera for request {camera}") from last_error


def _default_output_dir(camera: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("outputs/rgbd_records") / f"{camera}_{timestamp}"


def _intrinsics_to_dict(intrinsics) -> dict:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "coeffs": [float(v) for v in getattr(intrinsics, "coeffs", ())],
        "model": getattr(intrinsics, "model", None),
    }


def _save_color(path: Path, color_bgr: np.ndarray) -> None:
    import cv2

    ok = cv2.imwrite(str(path), color_bgr)
    if not ok:
        raise RuntimeError(f"Failed to save {path}")


if __name__ == "__main__":
    main()
