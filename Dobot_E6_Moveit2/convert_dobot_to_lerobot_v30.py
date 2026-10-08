#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dobot E6 수집 데이터 → LeRobot **v3.0** 데이터셋 (SmolVLA 학습용).

## 왜 새로 썼는가 (v21 변환기를 대체)

설치된 lerobot 0.4.4 가 `CODEBASE_VERSION = "v3.0"` 이다
(`lerobot/datasets/lerobot_dataset.py:83`). 그런데 옛 변환기는 `"codebase_version": "v2.1"`
을 쓴다. `check_version_compatibility()` 가 **major 가 낮으면
`BackwardCompatibilityError` 를 던지므로**(2 < 3) 옛 출력은 **로드조차 되지 않는다.**

⚠️ 옛 `convert_dobot_to_lerobot_v21.py` 는 지우지 않았다 — 원본 해상도로 저장된
   과거 에피소드를 다시 볼 일이 있을 수 있다.

## 스키마를 손으로 만들지 않는다

v3.0 은 `meta/episodes/chunk-000/file-000.parquet` 안에
`data/chunk_index`, `dataset_from_index`, `videos/<key>/from_timestamp`,
flatten 된 per-episode stats … 같은 필드를 요구한다. 손으로 맞추면 한 칸만 틀려도
학습 직전에 터진다. 그래서 **lerobot 자신의 `LeRobotDataset.create/add_frame/
save_episode/finalize`** 를 쓴다. 청킹·통계·비디오 인코딩·메타 전부 lerobot 이 만든다.

## 계약 (SmolVLA_base_dobot7d_local 실측에서 확정)

    observation.state   7D  [j1..j6 (deg), gripper 0/1]
    action              7D  [j1..j6 (deg) **절대값** = 다음 프레임, gripper 0/1]
    이미지 키            observation.images.OBS_IMAGE_1 (HIK) / OBS_IMAGE_2 (ZED)
    해상도              512×512
    fps                 16

🔴 **action 이 절대값인 것은 의도된 변경이다.** base 체크포인트와 cond_C 는 delta 였다
   (실측: action mean 이 6축 모두 0, std 0.04~0.71). 절대값으로 바꾼 근거 셋:

   1. j5 저분산. delta 에서 j5 std **0.04** 로 다른 축의 1/17 이다. 정규화가
      MEAN_STD 라 이 축이 17배 증폭된다 — 이 저장소에 "E6 의 j5 같은 저분산 축이
      학습 그래디언트 30% 를 먹었던 전례"가 기록돼 있다. 절대값이면 std 0.68 이다.
   2. 개루프 16스텝 실행에 안전하다. delta 는 적분해야 해서 오차가 누적된다.
      절대 목표값은 각 스텝이 독립이고 ServoJ(목표 도달 시 정지)와 궁합이 맞는다.
   3. LeRobot 공식 SmolVLA 데이터셋(`lerobot/svla_so100_pickplace`)이 절대값이다
      — 실측: action mean 이 state mean 과 0.0~1.6 차이. 사전학습 규약에 맞는다.

   ⚠️ 그래서 이 데이터로 학습한 것은 cond_C/v4 체크포인트와 **규약이 다르다.**
      섞어 쓰면 안 된다.

🔴 **이미지 키를 top/side 로 바꾸지 말 것.** 이름 자체는 lerobot 관점에서 자유지만
   (`image_features` 는 VISUAL 타입 전부), `policies/factory.py` 가
   `if not cfg.input_features:` 라 **사전학습 config 의 키를 데이터셋 키로 덮어쓰지
   않는다.** base 가 `OBS_IMAGE_1/2` 로 선언해 뒀으므로 이름을 바꾸면 배치에서 못 찾아
   `modeling_smolvla.py:414` 의 "All image features are missing from the batch" 로 터진다.
   정규화 통계 키에도 같은 이름이 박혀 있다.

## 타임스탬프

`add_frame()` 에 `timestamp` 를 **넘기지 않는다** → lerobot 이 `frame_index / fps` 로
균일하게 만든다. 실제 수집 간격은 중앙 63.1ms(이상값 62.5ms), p95 75.4ms 로 지터가
있는데, `tolerance_s` 기본값이 **1e-4 초(0.1ms)** 라 실측값을 그대로 넣으면 검증에
걸린다. 원본 시각은 수집본 `robot_data.csv` 에 그대로 남아 있다.

## 사용법

    source ~/SmolVLA/.venv_SmolVLA310/bin/activate
    export LD_LIBRARY_PATH=~/SmolVLA/.venv_SmolVLA310/lib/python3.10/site-packages/nvidia/cu12/lib:$LD_LIBRARY_PATH
    python3 convert_dobot_to_lerobot_v30.py --src /mnt/robotdata/SmolVLA/SmolVLA_dataset \
                                            --out /mnt/robotdata/SmolVLA/lerobot_v30
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

FPS = 16
ROBOT_TYPE = "dobot_e6"
IMAGE_SIZE = (512, 512)          # (h, w)
STATE_COLUMNS = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper_tooldo1"]
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
CAMERA_TO_KEY = {                # 수집 폴더명 → LeRobot feature 키
    "hik": "observation.images.OBS_IMAGE_1",
    "zed": "observation.images.OBS_IMAGE_2",
}
# 이 태스크만 변환한다. 옛 zone_pick/zone_move 에피소드는 카메라 크롭도 태스크도
# 달라서 섞이면 조용히 오염된다.
ALLOWED_TASK_NAMES = {"pick_place_across_line"}


def build_features() -> dict:
    feats = {
        "observation.state": {
            "dtype": "float32", "shape": (len(JOINT_NAMES),), "names": list(JOINT_NAMES),
        },
        "action": {
            "dtype": "float32", "shape": (len(JOINT_NAMES),), "names": list(JOINT_NAMES),
        },
    }
    for key in CAMERA_TO_KEY.values():
        feats[key] = {
            "dtype": "video",
            "shape": (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            "names": ["height", "width", "channel"],
        }
    return feats


def episode_dirs(src: Path) -> list[Path]:
    return sorted((d for d in src.iterdir() if d.is_dir() and d.name.isdigit()),
                  key=lambda d: int(d.name))


def load_meta(ep: Path) -> dict:
    p = ep / "episode_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def crop_signature(meta: dict):
    """이 에피소드가 어떤 크롭으로 찍혔는지. 서로 다른 크롭을 한 데이터셋에 섞지 않으려고 쓴다."""
    per = (meta.get("image_save") or {}).get("per_camera") or {}
    return json.dumps({k: (per.get(k) or {}).get("crop_norm_xyxy") for k in ("hik", "zed")},
                      sort_keys=True)


def read_states(ep: Path) -> np.ndarray:
    with (ep / "robot_data.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [c for c in STATE_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise RuntimeError(f"{ep.name}: robot_data.csv 에 없는 컬럼 {missing}")
    return np.array([[float(r[c]) for c in STATE_COLUMNS] for r in rows], dtype=np.float32)


def read_image(path: Path) -> np.ndarray:
    """디스크의 jpg(BGR) → RGB uint8 (h, w, 3). LeRobot 은 RGB 를 기대한다."""
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"이미지를 읽지 못했다: {path}")
    if img.shape[:2] != IMAGE_SIZE:
        raise RuntimeError(f"{path}: 해상도 {img.shape[:2]} != {IMAGE_SIZE} — "
                           "수집 시점 크롭 설정이 다른 에피소드가 섞였다")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def frame_paths(ep: Path, cam: str) -> list[Path]:
    d = ep / "images" / cam
    return sorted(d.glob("frame_*.jpg")) if d.is_dir() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="수집 루트 (에피소드 폴더들의 부모)")
    ap.add_argument("--out", required=True, type=Path, help="출력 LeRobot 데이터셋 루트")
    ap.add_argument("--repo-id", default="local/dobot_pickplace_line")
    ap.add_argument("--vcodec", default="libsvtav1",
                    help="libsvtav1(기본, lerobot 표준) | h264_nvenc(Jetson HW, 빠름) | h264 | hevc")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N개만 (0=전부)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if args.out.exists():
        if not args.overwrite:
            print(f"[중단] 출력 경로가 이미 있다: {args.out}  (--overwrite 로 덮어쓰기)")
            return 1
        shutil.rmtree(args.out)

    eps = episode_dirs(args.src)
    if not eps:
        print(f"[중단] 에피소드를 찾지 못했다: {args.src}")
        return 1

    # ── 1단계: 무엇을 넣고 무엇을 뺄지 먼저 정하고 **이유를 출력**한다 ──────────
    selected, skipped, signatures = [], [], {}
    trailing_surplus = []
    for ep in eps:
        meta = load_meta(ep)
        task = meta.get("task_name")
        if task not in ALLOWED_TASK_NAMES:
            skipped.append((ep.name, f"task_name={task!r} (허용: {sorted(ALLOWED_TASK_NAMES)})"))
            continue
        if not meta.get("success", False):
            skipped.append((ep.name, "success=False"))
            continue
        n = {cam: len(frame_paths(ep, cam)) for cam in CAMERA_TO_KEY}
        n_rows = len(read_states(ep))
        # 🔴 이미지가 CSV 행보다 **맨 끝에 1~2장 더 많은** 경우가 있다 (180개 중 7개, 3.9%).
        #    에피소드 종료 시점 경합이다 — _stop_and_save 가 recording=False 로 바꾸고
        #    0.07초 기다리는 사이에 ROS2 save_worker 큐에 남아 있던 프레임이 디스크에
        #    한 장 더 쓰인다. 그 프레임에 대응하는 CSV 행(로봇 상태)은 없다.
        #
        #    이걸 "불일치"로 보고 버리면 멀쩡한 에피소드 7개를 통째로 잃는다. 실제로는
        #    아래 기록 루프가 `range(n_rows - 1)` 로 인덱스 0..n_rows-2 만 쓰므로
        #    꼬리의 여분 이미지는 **애초에 건드리지 않는다.** CSV 가 정본이다.
        #
        #    ⚠️ 반대 방향(이미지가 CSV 보다 **적은**)은 거른다 — 그건 중간에 프레임이
        #       빠진 것이라 image[i] 와 state[i] 의 짝이 어긋난다.
        if len(set(n.values())) != 1:
            skipped.append((ep.name, f"hik/zed 장수 불일치 {n}"))
            continue
        n_img = next(iter(n.values()))
        if n_img < n_rows:
            skipped.append((ep.name, f"이미지 부족 csv={n_rows} img={n_img}"))
            continue
        if n_img > n_rows:
            trailing_surplus.append((ep.name, n_img - n_rows))
        if n_rows < 2:
            skipped.append((ep.name, f"프레임이 너무 적다 ({n_rows})"))
            continue
        signatures.setdefault(crop_signature(meta), []).append(ep.name)
        selected.append((ep, meta, n_rows))

    print(f"[선별] 변환 {len(selected)}개 / 제외 {len(skipped)}개")
    if trailing_surplus:
        tot = sum(k for _, k in trailing_surplus)
        print(f"[꼬리여분] {len(trailing_surplus)}개 에피소드에 CSV 행 없는 끝 이미지 {tot}장 "
              f"— 사용하지 않는다: {[f'{n}(+{k})' for n, k in trailing_surplus]}")
    for name, why in skipped:
        print(f"   제외 {name}: {why}")
    if not selected:
        print("[중단] 변환할 에피소드가 없다")
        return 1

    # 🔴 크롭이 섞이면 같은 키의 이미지가 서로 다른 화각이 된다. 에러 없이 학습만
    #    망가지는 종류라 여기서 막는다.
    if len(signatures) > 1:
        print("[중단] 카메라 크롭이 서로 다른 에피소드가 섞여 있다:")
        for sig, names in signatures.items():
            print(f"   {sig}  ← {names}")
        return 1
    print(f"[크롭] 단일 설정 확인: {next(iter(signatures))}")

    if args.limit:
        selected = selected[: args.limit]

    # ── 2단계: lerobot 공식 API 로 기록 ────────────────────────────────────────
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, features=build_features(),
        root=args.out, robot_type=ROBOT_TYPE, use_videos=True, vcodec=args.vcodec,
    )

    total = 0
    for ep, meta, n_rows in selected:
        states = read_states(ep)
        paths = {cam: frame_paths(ep, cam) for cam in CAMERA_TO_KEY}
        task = meta.get("instruction") or meta.get("prompt") or "pick up the orange box"
        # 마지막 프레임은 action(= 다음 상태)이 없으므로 제외한다 → N-1 프레임
        for i in range(n_rows - 1):
            frame = {
                "observation.state": states[i],
                "action": states[i + 1],          # 🔴 절대값
                "task": task,
            }
            for cam, key in CAMERA_TO_KEY.items():
                frame[key] = read_image(paths[cam][i])
            ds.add_frame(frame)
        ds.save_episode()
        total += n_rows - 1
        print(f"   ep{ep.name}: {n_rows - 1} 프레임  task={task!r}")

    ds.finalize()
    print(f"[완료] 에피소드 {len(selected)}개 / 프레임 {total}개 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
