#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Dobot E6 raw collection folders into a LeRobot v2.1-style dataset.

Install dependencies:
    pip install pandas pyarrow opencv-python numpy

This script is intentionally standalone:
- It does not modify robot_server.py.
- It does not modify the raw Dobot collection folders.
- It only creates/recreates OUT_ROOT.
"""

from pathlib import Path
import json
import shutil

import cv2
import numpy as np
import pandas as pd


RAW_ROOT = Path("/media/billye6/새 볼륨/Dobot/SmolVLA")
OUT_ROOT = Path("/media/billye6/새 볼륨/Dobot/SmolVLA_lerobot_v21")
FPS = 20
OVERWRITE = True
TEST_LEROBOT_LOAD = True
CHUNK_NAME = "chunk-000"

VIDEO_SIZE = (512, 512)
RAW_SIZE = (640, 480)
CAMERA_MAPPING = {
    "OBS_IMAGE_1": "HIK_top",
    "OBS_IMAGE_2": "ZED_side",
}
VIDEO_KEYS = {
    "hik": "observation.images.OBS_IMAGE_1",
    "zed": "observation.images.OBS_IMAGE_2",
}
CROP_XYXY = {
    "hik": [94, 0, 574, 480],
    "zed": [80, 0, 560, 480],
}

STATE_COLUMNS = [
    "j1", "j2", "j3", "j4", "j5", "j6",
    "x", "y", "z", "rx", "ry", "rz",
    "gripper_tooldo1",
]


def load_episode_meta(ep_dir):
    path = ep_dir / "episode_meta.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_instruction(meta):
    instruction = meta.get("instruction")
    if instruction:
        return str(instruction)
    zone_id = meta.get("zone_id")
    if zone_id:
        return f"move to zone {zone_id}"
    return "move to target zone"


def load_frame_paths(ep_dir, camera_name):
    image_dir = ep_dir / "images" / camera_name
    if not image_dir.exists():
        return []
    paths = sorted(image_dir.glob("frame_*.jpg"))
    if not paths:
        paths = sorted(image_dir.glob("*.jpg"))
    return paths


def crop_resize_for_lerobot(img, camera_name):
    if img is None:
        raise RuntimeError(f"{camera_name}: image is None")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    if (w, h) != RAW_SIZE:
        img = cv2.resize(img, RAW_SIZE, interpolation=cv2.INTER_AREA)

    if camera_name not in CROP_XYXY:
        raise RuntimeError(f"Unknown camera_name: {camera_name}")

    x1, y1, x2, y2 = CROP_XYXY[camera_name]
    crop = img[y1:y2, x1:x2]
    if crop.shape[:2] != (480, 480):
        raise RuntimeError(
            f"{camera_name}: crop shape must be 480x480, got {crop.shape}"
        )

    return cv2.resize(crop, VIDEO_SIZE, interpolation=cv2.INTER_AREA)


def write_mp4_from_frames(frame_paths, out_path, camera_name, fps):
    if not frame_paths:
        raise RuntimeError(f"{camera_name}: no frames to write")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), VIDEO_SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_path}")

    try:
        for frame_path in frame_paths:
            img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
            frame = crop_resize_for_lerobot(img, camera_name)
            if frame.shape != (512, 512, 3):
                raise RuntimeError(
                    f"{camera_name}: frame shape must be 512x512x3, got {frame.shape}"
                )
            writer.write(frame)
    finally:
        writer.release()

    cap = cv2.VideoCapture(str(out_path))
    ok, first = cap.read()
    cap.release()
    if not ok or first is None:
        raise RuntimeError(f"Failed to read first mp4 frame: {out_path}")
    if first.shape != (512, 512, 3):
        raise RuntimeError(f"MP4 first frame shape must be 512x512x3, got {first.shape}")

    return len(frame_paths)


def _numeric_columns(df, columns):
    try:
        out = df[columns].apply(pd.to_numeric, errors="raise")
    except Exception as exc:
        raise RuntimeError(f"Failed to convert numeric columns {columns}: {exc}") from exc
    if out.isna().any().any():
        missing = out.columns[out.isna().any()].tolist()
        raise RuntimeError(f"Missing numeric values in columns: {missing}")
    return out


def make_state_array(df):
    states = _numeric_columns(df, STATE_COLUMNS).to_numpy(dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 13:
        raise RuntimeError(f"observation.state must have shape [N, 13], got {states.shape}")
    return states


def make_action_array(states):
    if states.ndim != 2 or states.shape[1] != 13:
        raise RuntimeError(f"states must have shape [N, 13], got {states.shape}")
    deltas = states[1:, :12] - states[:-1, :12]
    gripper_next = states[1:, 12:13]
    actions = np.concatenate([deltas, gripper_next], axis=1).astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 13:
        raise RuntimeError(f"action must have shape [N-1, 13], got {actions.shape}")
    return actions


def compute_stats(all_states, all_actions):
    states = np.concatenate(all_states, axis=0).astype(np.float32)
    actions = np.concatenate(all_actions, axis=0).astype(np.float32)

    def one(arr):
        std = np.std(arr, axis=0)
        std = np.clip(std, 1e-6, None)
        return {
            "mean": np.mean(arr, axis=0).astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "min": np.min(arr, axis=0).astype(float).tolist(),
            "max": np.max(arr, axis=0).astype(float).tolist(),
        }

    return {
        "observation.state": one(states),
        "action": one(actions),
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_video_frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for counting: {path}")
    count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    return count


def validate_frame_id_alignment(df, hik_frames, zed_frames):
    frame_ids = pd.to_numeric(df["frame_id"], errors="raise").astype(int).tolist()
    for frame_id, hik_path, zed_path in zip(frame_ids, hik_frames, zed_frames):
        expected = f"frame_{frame_id:06d}.jpg"
        if hik_path.name != expected:
            raise RuntimeError(f"HIK frame mismatch: expected {expected}, got {hik_path.name}")
        if zed_path.name != expected:
            raise RuntimeError(f"ZED frame mismatch: expected {expected}, got {zed_path.name}")


def validate_episode(out_root, episode_index):
    parquet_path = out_root / "data" / CHUNK_NAME / f"episode_{episode_index:06d}.parquet"
    hik_video = (
        out_root / "videos" / CHUNK_NAME / VIDEO_KEYS["hik"] / f"episode_{episode_index:06d}.mp4"
    )
    zed_video = (
        out_root / "videos" / CHUNK_NAME / VIDEO_KEYS["zed"] / f"episode_{episode_index:06d}.mp4"
    )

    rows = len(pd.read_parquet(parquet_path))
    hik_frames = count_video_frames(hik_video)
    zed_frames = count_video_frames(zed_video)
    if not (rows == hik_frames == zed_frames):
        raise RuntimeError(
            f"episode_{episode_index:06d} length mismatch: "
            f"parquet={rows}, hik={hik_frames}, zed={zed_frames}"
        )
    print(f"[Validate] episode_{episode_index:06d} OK")


def validate_all_episodes(out_root, total_episodes):
    ok_count = 0
    for episode_index in range(total_episodes):
        try:
            validate_episode(out_root, episode_index)
            ok_count += 1
        except Exception as exc:
            raise RuntimeError(f"[Validate All] FAILED at episode_{episode_index:06d}: {exc}") from exc
    print(f"[Validate All] {ok_count}/{total_episodes} episodes OK")
    print("[Validate All] parquet rows and video frames are aligned.")


def test_lerobot_dataset_load(out_root):
    if not TEST_LEROBOT_LOAD:
        print("[LeRobot Load Test] SKIP: TEST_LEROBOT_LOAD=False")
        return
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        print(f"[LeRobot Load Test] SKIP: import failed: {exc}")
        return

    try:
        dataset = LeRobotDataset(str(out_root))
        print(f"[LeRobot Load Test] dataset loaded: len={len(dataset)}")
        sample = dataset[0]
        required_keys = [
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "task_index",
        ]
        for key in required_keys:
            if key not in sample:
                raise RuntimeError(f"missing key: {key}")
        print("[LeRobot Load Test] first sample keys OK")
    except Exception as exc:
        raise RuntimeError(f"[LeRobot Load Test] FAILED: {exc}") from exc


def _prepare_output_dirs():
    if OUT_ROOT.exists():
        if OVERWRITE:
            shutil.rmtree(OUT_ROOT)
        else:
            raise RuntimeError(f"OUT_ROOT already exists and OVERWRITE=False: {OUT_ROOT}")

    data_dir = OUT_ROOT / "data" / CHUNK_NAME
    video_hik_dir = OUT_ROOT / "videos" / CHUNK_NAME / VIDEO_KEYS["hik"]
    video_zed_dir = OUT_ROOT / "videos" / CHUNK_NAME / VIDEO_KEYS["zed"]
    meta_dir = OUT_ROOT / "meta"
    for path in (data_dir, video_hik_dir, video_zed_dir, meta_dir):
        path.mkdir(parents=True, exist_ok=True)
    return data_dir, video_hik_dir, video_zed_dir, meta_dir


def _episode_dirs():
    if not RAW_ROOT.exists():
        raise RuntimeError(f"RAW_ROOT does not exist: {RAW_ROOT}")
    return sorted(
        [p for p in RAW_ROOT.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )


def _read_timestamps(df):
    if "timestamp" not in df.columns:
        raise RuntimeError("robot_data.csv missing timestamp column")
    try:
        ts = pd.to_numeric(df["timestamp"], errors="raise")
    except Exception as exc:
        raise RuntimeError(f"timestamp conversion failed: {exc}") from exc
    if ts.isna().any():
        raise RuntimeError("timestamp has missing values")
    ts = ts.to_numpy(dtype=np.float32)
    return ts[:-1] - ts[0]


def _build_info(total_episodes, total_frames):
    return {
        "codebase_version": "v2.1",
        "robot_type": "dobot_e6",
        "fps": FPS,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "chunks": 1,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [13],
                "names": [
                    "j1", "j2", "j3", "j4", "j5", "j6",
                    "x", "y", "z", "rx", "ry", "rz",
                    "gripper",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": [13],
                "names": [
                    "delta_j1", "delta_j2", "delta_j3", "delta_j4", "delta_j5", "delta_j6",
                    "delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz",
                    "gripper_next",
                ],
            },
            "observation.images.OBS_IMAGE_1": {
                "dtype": "video",
                "shape": [512, 512, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "camera": "HIK_top",
                    "original_size": [640, 480],
                    "crop_xyxy": [94, 0, 574, 480],
                    "resize": [512, 512],
                },
            },
            "observation.images.OBS_IMAGE_2": {
                "dtype": "video",
                "shape": [512, 512, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "camera": "ZED_side",
                    "original_size": [640, 480],
                    "crop_xyxy": [80, 0, 560, 480],
                    "resize": [512, 512],
                },
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
            },
        },
    }


def convert():
    data_dir, video_hik_dir, video_zed_dir, meta_dir = _prepare_output_dirs()
    episode_dirs = _episode_dirs()

    tasks = {}
    tasks_rows = []
    episodes_rows = []
    all_states = []
    all_actions = []
    global_index = 0
    out_episode_index = 0

    for ep_dir in episode_dirs:
        csv_path = ep_dir / "robot_data.csv"
        hik_dir = ep_dir / "images" / "hik"
        zed_dir = ep_dir / "images" / "zed"

        if not csv_path.exists():
            print(f"[Skip] source folder {ep_dir.name}: robot_data.csv missing")
            continue
        if not hik_dir.exists():
            print(f"[Skip] source folder {ep_dir.name}: images/hik missing")
            continue
        if not zed_dir.exists():
            print(f"[Skip] source folder {ep_dir.name}: images/zed missing")
            continue

        hik_frames = load_frame_paths(ep_dir, "hik")
        zed_frames = load_frame_paths(ep_dir, "zed")
        if len(hik_frames) < 2:
            print(f"[Skip] source folder {ep_dir.name}: HIK frame count < 2")
            continue
        if len(zed_frames) < 2:
            print(f"[Skip] source folder {ep_dir.name}: ZED frame count < 2")
            continue

        df = pd.read_csv(csv_path)
        if len(df) < 2:
            print(f"[Skip] source folder {ep_dir.name}: CSV row count < 2")
            continue
        if len(hik_frames) != len(df):
            raise RuntimeError(
                f"source folder {ep_dir.name}: HIK frame count {len(hik_frames)} "
                f"!= CSV rows {len(df)}"
            )
        if len(zed_frames) != len(df):
            raise RuntimeError(
                f"source folder {ep_dir.name}: ZED frame count {len(zed_frames)} "
                f"!= CSV rows {len(df)}"
            )
        validate_frame_id_alignment(df, hik_frames, zed_frames)

        meta = load_episode_meta(ep_dir)
        instruction = get_instruction(meta)
        if instruction not in tasks:
            task_index = len(tasks)
            tasks[instruction] = task_index
            tasks_rows.append({"task_index": task_index, "task": instruction})
        task_index = tasks[instruction]

        states_full = make_state_array(df)
        actions = make_action_array(states_full)
        states = states_full[:-1]
        timestamps = _read_timestamps(df)
        n_rows = len(states)
        if n_rows != len(actions) or n_rows != len(timestamps):
            raise RuntimeError(
                f"source folder {ep_dir.name}: row mismatch "
                f"states={n_rows}, actions={len(actions)}, timestamps={len(timestamps)}"
            )

        print(f"[Convert] Episode {out_episode_index:06d} from source folder {ep_dir.name}")

        hik_out = video_hik_dir / f"episode_{out_episode_index:06d}.mp4"
        zed_out = video_zed_dir / f"episode_{out_episode_index:06d}.mp4"
        write_mp4_from_frames(hik_frames[:-1], hik_out, "hik", FPS)
        print(f"[Video] OBS_IMAGE_1 saved: {hik_out}")
        write_mp4_from_frames(zed_frames[:-1], zed_out, "zed", FPS)
        print(f"[Video] OBS_IMAGE_2 saved: {zed_out}")

        rows = []
        for i in range(n_rows):
            state = states[i].astype(np.float32).tolist()
            action = actions[i].astype(np.float32).tolist()
            assert len(state) == 13
            assert len(action) == 13
            rows.append({
                "observation.state": state,
                "action": action,
                "timestamp": float(timestamps[i]),
                "frame_index": int(i),
                "episode_index": int(out_episode_index),
                "task_index": int(task_index),
                "index": int(global_index),
            })
            global_index += 1

        parquet_path = data_dir / f"episode_{out_episode_index:06d}.parquet"
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        print(f"[Parquet] saved: {parquet_path}")

        record_rate_hz = meta.get("record_rate_hz", FPS)
        episodes_rows.append({
            "episode_index": out_episode_index,
            "length": n_rows,
            "task_index": task_index,
            "tasks": [instruction],
            "source_folder": ep_dir.name,
            "record_rate_hz": float(record_rate_hz),
            "camera_mapping": dict(CAMERA_MAPPING),
        })

        all_states.append(states)
        all_actions.append(actions)
        validate_episode(OUT_ROOT, out_episode_index)
        out_episode_index += 1

    if out_episode_index == 0:
        raise RuntimeError(f"No valid episodes converted from RAW_ROOT: {RAW_ROOT}")

    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)
    print("[Meta] tasks.jsonl saved")

    write_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
    print("[Meta] episodes.jsonl saved")

    stats = compute_stats(all_states, all_actions)
    stats_path = meta_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("[Meta] stats.json saved")

    info = _build_info(total_episodes=out_episode_index, total_frames=global_index)
    info_path = meta_dir / "info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[Meta] info.json saved")

    validate_all_episodes(OUT_ROOT, out_episode_index)
    test_lerobot_dataset_load(OUT_ROOT)

    print(f"[Done] Output: {OUT_ROOT}")


if __name__ == "__main__":
    convert()
