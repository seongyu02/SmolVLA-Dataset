#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI web server for Dobot E6 Pick-Place data collection.
Dual camera: HIKRobot (wrist) + ZED (scene, LEFT view only)
Dashboard: J1-J6, TCP pose, Robot Mode live display

Usage:
    cd /home/billye6/Dobot-Arm-DataCollect/Dobot_E6_Moveit2/src
    python3 robot_server.py

Open: http://<jetson-ip>:8000
"""

import sys
import os
import json
import time
import threading
import asyncio
import shutil
import random
from datetime import datetime
from typing import Optional, Set

# ═══════════════════════════════════════════════════════════════════════════
# PyQt5 Mock — PickPlaceStepWorker(QThread) → threading.Thread 교체
# (pick_place_gui_new import 전 반드시 먼저 선언)
# ═══════════════════════════════════════════════════════════════════════════
import types as _types

class _BoundSignal:
    def __init__(self):
        self._cbs = []
    def connect(self, cb):
        self._cbs.append(cb)
    def emit(self, *args):
        for cb in self._cbs:
            try:
                cb(*args)
            except Exception:
                pass
    def disconnect(self, cb=None):
        self._cbs = [] if cb is None else [c for c in self._cbs if c != cb]

class _SignalDescriptor:
    def __init__(self, *_):
        self._attr = None
    def __set_name__(self, owner, name):
        self._attr = f'_sig_{name}'
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        attr = self._attr or '_sig_unknown'
        if not hasattr(obj, attr):
            object.__setattr__(obj, attr, _BoundSignal())
        return object.__getattribute__(obj, attr)

def _pyqtSignal(*a, **kw):
    return _SignalDescriptor()

class _QThread(threading.Thread):
    def __init__(self, parent=None):
        super().__init__(daemon=True)
    def start(self):
        super().start()
    def isRunning(self):
        return self.is_alive()
    def wait(self, msecs=None):
        self.join(timeout=(msecs / 1000.0) if msecs else None)

class _MockQt:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return _MockQt()
    def __getattr__(self, name): return _MockQt()

_qt5     = _types.ModuleType('PyQt5')
_qw      = _types.ModuleType('PyQt5.QtWidgets')
_qc      = _types.ModuleType('PyQt5.QtCore')
_qg      = _types.ModuleType('PyQt5.QtGui')

for _n in ['QApplication','QMainWindow','QWidget','QVBoxLayout','QHBoxLayout',
           'QGroupBox','QGridLayout','QLabel','QLineEdit','QPushButton',
           'QTextEdit','QDoubleSpinBox','QMessageBox','QCheckBox']:
    setattr(_qw, _n, _MockQt)
_qc.QThread    = _QThread
_qc.pyqtSignal = _pyqtSignal
_qc.QTimer     = _MockQt
_qc.Qt         = _MockQt()
for _n in ['QFont', 'QImage', 'QPixmap']:
    setattr(_qg, _n, _MockQt)

sys.modules['PyQt5']             = _qt5
sys.modules['PyQt5.QtWidgets']   = _qw
sys.modules['PyQt5.QtCore']      = _qc
sys.modules['PyQt5.QtGui']       = _qg

# ═══════════════════════════════════════════════════════════════════════════
# 모듈 import
# ═══════════════════════════════════════════════════════════════════════════
import numpy as np
import cv2

_current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _current_dir)

if not os.environ.get('MVCAM_COMMON_RUNENV'):
    os.environ['MVCAM_COMMON_RUNENV'] = '/opt/MVS/lib'

import pick_place_gui_new as base
from pick_place_gui_random_pose import (
    RandomPosePickPlaceStepWorker,
    generate_random_initial_pose,
    INIT_SAFE_RX, INIT_SAFE_RY, INIT_SAFE_RZ,
)

# 데이터 저장 경로 — 외장 드라이브 마운트 확인 필요
# 외장 HDD 는 2026-07-28 부터 UUID 고정으로 /mnt/robotdata 에 마운트된다.
# 옛 경로 "/media/billye6/새 볼륨/..." 은 마운트가 아니라 **내장 디스크의 빈 잔재
# 디렉터리**여서, 그대로 두면 HDD 가 아니라 내장(여유 6.4G)에 조용히 쌓인다.
DATA_SAVE_DIR   = "/mnt/robotdata/SmolVLA/SmolVLA_dataset"
# 🔴 마운트 판정 기준. 옛 값 "/media/billye6/새 볼륨" 은 **내장 디스크에 빈
#    디렉터리로 실재**해서 os.path.isdir() 이 HDD 유무와 무관하게 항상 True 였다
#    — 안전장치가 무력화된 상태였다. 실제 마운트포인트를 가리키게 바꾼다.
DATA_DRIVE_ROOT = "/mnt/robotdata"   # 마운트 여부 판단 기준
from dobot_e6_controller import DobotE6Controller
from suction_gripper import SuctionGripper

_hik_available = False
try:
    from camera_viewer import HikRobotCamera
    # 카메라 설정(serial/AOI/gamma/출력크기)은 dobot_camera.py 가 단일 소유한다.
    # HIK 은 AOI·gamma 를 펌웨어에 영구 저장하므로 열 때 명시하지 않으면 같은 카메라를
    # 쓰는 다른 프로그램(xarm camera_publisher_node)이 남긴 값을 물려받는다.
    from dobot_camera import make_hik_camera
    _hik_available = True
except Exception as e:
    print(f"[Server] HIK camera unavailable: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# ZED 카메라 래퍼 (LEFT 뷰 단일, 640×480 리사이즈)
# ═══════════════════════════════════════════════════════════════════════════
_zed_available = False
try:
    import pyzed.sl as _sl
    _zed_available = True
except Exception as e:
    print(f"[Server] ZED SDK unavailable: {e}")

class ZedCamera:
    """ZED 2i / ZED X — LEFT 뷰 전용 래퍼."""
    def __init__(self):
        if not _zed_available:
            raise RuntimeError("pyzed not installed")
        self.cam   = _sl.Camera()
        self._mat  = _sl.Mat()
        self._rt   = _sl.RuntimeParameters()
        self.initialized = False

    def init_camera(self) -> bool:
        params = _sl.InitParameters()
        params.camera_resolution = _sl.RESOLUTION.HD1080
        params.camera_fps        = 30
        params.depth_mode        = _sl.DEPTH_MODE.NONE   # 깊이 불필요
        err = self.cam.open(params)
        if err != _sl.ERROR_CODE.SUCCESS:
            print(f"[ZED] Open failed: {err}")
            return False
        self.initialized = True
        print("[ZED] Camera initialized (HD720, LEFT view)")
        return True

    def get_frame(self):
        """(ok, RGB ndarray 640×480) 반환."""
        if not self.initialized:
            return False, None
        err = self.cam.grab(self._rt)
        if err != _sl.ERROR_CODE.SUCCESS:
            return False, None
        self.cam.retrieve_image(self._mat, _sl.VIEW.LEFT)
        data = self._mat.get_data()          # (H, W, 4) BGRA
        bgr  = data[:, :, :3]               # BGR
        # 리사이즈하지 않는다. HD1080(1920×1080, 16:9)을 640×480(4:3)으로 눌러
        # 담으면 종횡비가 깨진 채 저장된다. 512 변환은 학습 변환기가 맡는다.
        bgr  = bgr.copy()   # get_data() 는 SDK 내부 버퍼 뷰라 복사해서 들고 나간다
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return True, rgb

    def cleanup(self):
        if self.initialized:
            self.cam.close()
            self.initialized = False

# ═══════════════════════════════════════════════════════════════════════════
# FastAPI
# ═══════════════════════════════════════════════════════════════════════════
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

# ═══════════════════════════════════════════════════════════════════════════
# ROS2 레코더 (rclpy 미설치 시 graceful 비활성화)
# ═══════════════════════════════════════════════════════════════════════════
import ros2_recorder as _ros2
_ros2_ok: bool = False          # start() 성공 후 True 로 설정

# ═══════════════════════════════════════════════════════════════════════════
# 서버 상태
# ═══════════════════════════════════════════════════════════════════════════
_state = {
    "robot":          None,
    "gripper":        None,
    "camera_hik":     None,
    "camera_zed":     None,
    "worker":         None,
    "recording":      False,
    "recorded_data":  [],
    "record_save_dir":None,
    "record_frame_count": 0,
    "vacuum_pick":    0.0,
    "vacuum_place":   0.0,
    "episode_meta":   {},
    "auto_target":    0,
    "auto_done":      0,
    "auto_zone_order": [],
    "auto_episodes_per_zone": 0,
    "pick_section":   "A",
    "last_place_x":   None,
    "last_place_y":   None,
    "task_mode":      "zone_move",
    # pick_place_across_line 전용 상태. zone 태스크의 auto_* 와 분리해 둔다 —
    # 한 쪽을 돌리다 멈춰도 다른 쪽 카운터가 오염되지 않게.
    "ppl_box_xy":      None,   # 지금 박스가 있는 (x, y). None 이면 시작점에서 출발
    "ppl_box_section": None,   # "A" | "B"
    "ppl_auto_target": 0,
    "ppl_auto_done":   0,
    "ppl_episode_index": 0,
    "zone_episode_idx": 0,
    "zone_stats":     {},
    "current_zone_id": None,
    "current_zone_episode": 0,
    "last_auto_error": "",
    "last_auto_done": 0,
    "last_auto_target": 0,
}

_state_lock  = threading.Lock()
_ws_clients: Set[WebSocket] = set()
_log_queue: asyncio.Queue   = None
_main_loop: asyncio.AbstractEventLoop = None

FIXED_INIT = (89.3715, -378.5400, 250.0000, -179.5275, -2.4369, 2.3663)

# Zone move dataset parameters — 실기에서 실행 전 반드시 확인/튜닝할 값.
# - ZONE_INIT_POSE: 초기자세. 현재 실측 안전 대기 위치를 사용.
# - Zone move task keeps the base PickPlaceStepWorker travel/descent flow.
# - ZONE_TARGET_Z: 이번 task의 최종 하강 Z.
# - ZONE_XY_OFFSET_MM: 목표 zone 내부 XY 미세 랜덤 offset.
# Home 버튼 목표. 2026-09-15 실기에서 로봇을 그 자세에 놓고 읽은 실측값이라
# 도달 가능성이 구조적으로 보장된다. rpy 는 INIT_SAFE_* 와 0.001° 안에서 일치.
HOME_POSE = (89.3719, -378.5381, 249.9971, 176.4640, -1.7731, 8.1310)

ZONE_INIT_POSE = (89.3715, -378.5400, 250.0000, INIT_SAFE_RX, INIT_SAFE_RY, INIT_SAFE_RZ)
ZONE_TARGET_Z = 120.0
ZONE_X_OFFSET_MM = 5.0
ZONE_Y_OFFSET_MM = 3.0
ZONE_TRAVEL_Z_RANGE = (150.0, 230.0)
ZONE_EPISODES_PER_ZONE = 10
ZONE_ORDER = ["2", "5", "6", "8", "9"]
ZONE_TASK_TOTAL = len(ZONE_ORDER) * ZONE_EPISODES_PER_ZONE

def _new_zone_stats():
    return {
        zid: {"success": 0, "fail": 0, "status": "pending"}
        for zid in ZONE_ORDER
    }

def _reset_zone_progress():
    _state["zone_stats"] = _new_zone_stats()
    _state["current_zone_id"] = None
    _state["current_zone_episode"] = 0
    _state["auto_zone_order"] = []
    _state["auto_episodes_per_zone"] = 0
    _state["last_auto_error"] = ""
    _state["last_auto_done"] = 0
    _state["last_auto_target"] = 0

_reset_zone_progress()

ZONE_POSES = {
    "2": base.POS_2,
    "5": base.POS_5,
    "6": base.POS_6,
    "8": base.POS_8,
    "9": base.POS_9,
}

CAMERA_MAPPING = {
    "OBS_IMAGE_1": "HIK_top",
    "OBS_IMAGE_2": "ZED_side",
}

# 2026-09-16: 수집 시점에 512×512 로 확정해 저장한다. 이 블록은 **실제로 한 일**을
# 적는 곳이지 권고가 아니다 — 이전 값(640×480 / crop_applied False)은 실제 저장과
# 달랐고, 그대로 두면 변환기가 meta 를 믿고 엉뚱한 전처리를 한다.
IMAGE_SAVE_META = {
    "saved_size": [512, 512],
    "saved_format": "jpg",
    "crop_applied": True,            # hik·zed 둘 다 크롭됨 (아래 per_camera 참조)
    "resize_to_512_applied": True,
    "resize_interpolation": "INTER_AREA",
    "per_camera": {
        "hik": {
            "source_size": [2592, 1944],
            "crop_norm_xyxy": [324 / 2592, 0 / 1944, 2268 / 2592, 1944 / 1944],
            "note": "1944x1944 정사각 중앙 크롭 (2026-09-16). 왜곡 없음. "
                    "정사각을 만들며 가로 좌우 각 324px 을 버린다.",
        },
        "zed": {
            "source_size": [1920, 1080],
            "crop_norm_xyxy": [780 / 1920, 0 / 1080, 1590 / 1920, 810 / 1080],
            "note": "810x810 정사각.",
        },
    },
    "description": "Frames are cropped+resized to 512x512 at collection time "
                   "using the same CROP_NORM as convert_dobot_to_lerobot_v21.py.",
}

# 변환기가 해야 할 남은 전처리 — 이제 **없다**. 수집이 최종 크기로 저장한다.
FUTURE_LEROBOT_PREPROCESS_META = {
    "hik": {
        "raw_size": [512, 512],
        "crop_required": False,
        "resize_size": [512, 512],
        "description": "Already cropped to 1944x1944 and resized at collection time. "
                       "Do NOT crop again.",
    },
    "zed": {
        "raw_size": [512, 512],
        "crop_required": False,
        "resize_size": [512, 512],
        "description": "Already cropped to 810x810 and resized at collection time. "
                       "Cropping again would cut the field of view twice.",
    },
}

ROBOT_MODE_LABELS = {
    1:"INIT", 2:"BRAKE_OPEN", 4:"DISABLED", 5:"ENABLE",
    6:"BACKDRIVE", 7:"RUNNING", 8:"RECORDING", 9:"ERROR",
    10:"PAUSE", 11:"JOG"
}

# ─── 프레임 버퍼 (MJPEG + 레코딩 공용) ────────────────────────────────────
_buf_hik_jpg: Optional[bytes]       = None   # MJPEG용 JPEG 바이트
_buf_zed_jpg: Optional[bytes]       = None
# 크롭 전 원본 프레임. 크롭 영역을 정할 때 전체 화각을 봐야 해서 들고 있는다.
# 참조만 보관하므로(복사 없음) 프레임당 비용은 사실상 0 이다.
_buf_hik_raw: Optional[np.ndarray]  = None
_buf_zed_raw: Optional[np.ndarray]  = None
_buf_hik_np:  Optional[np.ndarray]  = None   # 레코딩용 BGR numpy
_buf_zed_np:  Optional[np.ndarray]  = None
_buf_lock     = threading.Lock()

_cam_hik_thread: Optional[threading.Thread] = None
_cam_zed_thread: Optional[threading.Thread] = None
_cam_hik_running = False
_cam_zed_running = False

_robot_pub_running = False   # _robot_pub_loop 제어 플래그

# ═══════════════════════════════════════════════════════════════════════════
# 헬퍼
# ═══════════════════════════════════════════════════════════════════════════

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if _main_loop and _log_queue:
        try:
            _main_loop.call_soon_threadsafe(_log_queue.put_nowait, line)
        except Exception:
            pass

_SAFE_INIT_FALLBACK_XYZ = (89.3715, -378.5400, 250.0)  # 실측 검증된 안전 대기 위치

def _set_random_init_pose(robot):
    rx, ry, rz = INIT_SAFE_RX, INIT_SAFE_RY, INIT_SAFE_RZ
    cx, cy, cz = _SAFE_INIT_FALLBACK_XYZ
    ok = False
    for _ in range(30):
        tx, ty, tz, *_ = generate_random_initial_pose()
        if robot and robot.connected:
            ok, _ = robot.check_ik_solution(tx, ty, tz, rx, ry, rz)
        else:
            ok = True
        if ok:
            cx, cy, cz = tx, ty, tz
            break
    base.INIT_X = cx;  base.INIT_Y = cy;  base.INIT_Z = cz
    base.INIT_RX = rx; base.INIT_RY = ry; base.INIT_RZ = rz
    _log(f"[RandomPose] INIT X={cx:.1f} Y={cy:.1f} Z={cz:.1f} (IK={ok})")

def _get_next_folder(base_dir: str) -> int:
    if not os.path.exists(base_dir):
        return 1
    nums = [int(d) for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    return max(nums, default=0) + 1

def ensure_640x480_bgr(img):
    """
    img: BGR numpy image
    return: BGR image with shape (480, 640, 3)
    """
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    if (w, h) != (640, 480):
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    return img


# ═══════════════════════════════════════════════════════════════════════════
# 512×512 저장 변환 — 변환기(convert_dobot_to_lerobot_v21.py)와 같은 규약
# ═══════════════════════════════════════════════════════════════════════════
#
# 🔴 이 값은 convert_dobot_to_lerobot_v21.py 의 CROP_NORM **사본**이다.
#    두 파일이 갈라지면 수집 때 만든 그림과 학습에 들어가는 그림이 달라지는데
#    **에러가 안 난다** — 조용히 다른 데이터가 된다. 한쪽을 고치면 반드시 같이 고칠 것.
#
#    hik  (324,0)-(2268,1944) @2592×1944 = 1944×1944 정사각.
#
#    🔴 2026-09-16: 전체 화각(2592×1944)을 512×512 로 **눌러 담던 것**을 정사각
#       크롭으로 바꿨다. 센서가 4:3 이라 정사각을 만들려면 높이 1944 에 맞춰야
#       하고, 가로 648px(좌우 각 324px)은 **버릴 수밖에 없다** — 이것이 정사각의
#       수학적 최소 손실이다.
#       버리는 것: 왼쪽 324px ≈ 빈 벽 / 오른쪽 324px ≈ 로봇 팔꿈치·베이스 일부.
#       체커보드·박스·그리퍼는 전부 남고 더 크게 들어온다.
#       얻는 것: 왜곡 0 (ZED 와 성격이 같아진다), 가로 축소비 5.06→3.80 배로
#       완화되어 남는 영역의 픽셀 밀도는 오히려 높아진다.
#
#    ⚠️ 홈 자세 프레임 한 장으로 고른 값이다. 팔이 오른쪽 place 존으로 갈 때
#       잘린 오른쪽 324px 때문에 프레임을 벗어나는지는 **아직 검증 안 됐다**.
#       /camera/hik/raw.jpg 로 실제 동작 중 프레임을 보고 확인할 것.
#    zed  (780,0)-(1590,810) @1920×1080 = 810×810 정사각.
#
#    🔴 2026-09-16: ZED 재배치 후 빈 책상이 화면 아래 ⅓ 을 먹어서 창을 **위로
#       최대치(270px)** 이동했다. y[270:1080] → y[0:810]. 크기는 810×810 그대로다.
#       프레임 높이 1080 · 크롭 810 이므로 270px 이 물리적 상한이다.
#
#    ⚠️ 들어온 것이 **창문**이다(원본 y0~270 이 창). 빈 책상을 창문과 맞바꾼 셈이라
#       이득이 자명하지 않다. 그리고 창은 밝은 광원이라 auto exposure 가 장면
#       전체를 어둡게 눌러버릴 수 있다 — 이 프로젝트에서 이미 겪은 실패 유형이다
#       (화각에서 광원 제거가 1순위였던 적 있음).
#       실제 프레임 밝기·포화를 재보고 되돌릴지 판단할 것. 되돌리려면 y 를
#       270/1080 ~ 1080/1080 로.
#
# 왜 수집 때 미리 줄이는가 (2026-09-16 실측):
#    _record_tick 의 imwrite 가 HIK 2592×1944 에서 84.67ms, ZED 1920×1080 에서
#    32.77ms 였다 — 프레임당 117ms 라 legacy tick 이 **5.5Hz** 로 밀렸다.
#    512 로 줄이면 3.56ms×2 = 7.1ms 다.
#    ⚠️ 리사이즈 자체는 새 비용이 아니다. ROS2 발행용으로 이미 640×480 축소를
#       매 프레임 하고 있었고(HIK 11.62ms), 512 로 바꿔도 10.97ms 로 같다.
VIDEO_SIZE_512 = (512, 512)
CROP_NORM_512 = {
    "hik": (324 / 2592, 0 / 1944, 2268 / 2592, 1944 / 1944),
    "zed": (780 / 1920, 0 / 1080, 1590 / 1920, 810 / 1080),
}


def to_512(img, camera_name):
    """BGR 원본 → 변환기와 동일한 크롭 → 512×512 BGR."""
    if img is None:
        return np.zeros((512, 512, 3), dtype=np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # 이미 512×512 면 그대로 둔다. 다시 크롭하면 화각이 두 번 깎인다.
    if img.shape[:2] == (512, 512):
        return img

    fx1, fy1, fx2, fy2 = CROP_NORM_512[camera_name]
    h, w = img.shape[:2]
    x1, x2 = int(round(fx1 * w)), int(round(fx2 * w))
    y1, y2 = int(round(fy1 * h)), int(round(fy2 * h))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    return cv2.resize(crop, VIDEO_SIZE_512, interpolation=cv2.INTER_AREA)


# ═══════════════════════════════════════════════════════════════════════════
# 카메라 그랩 루프 (MJPEG 버퍼 + 레코딩 numpy 버퍼 동시 갱신)
# ═══════════════════════════════════════════════════════════════════════════

# HIK MJPEG(화면) 인코딩 주기.
#
# 프레임이 센서 원본(2592×1944)이면 cv2.imencode 가 실측 69.1ms 로 16Hz 주기
# (62.5ms)를 혼자 넘긴다. 화면용이라 매 프레임 인코딩할 이유가 없어 주기를 따로 둔다.
# 화면이 끊기거나 부하가 높으면 이 값을 낮출 것.
_HIK_MJPEG_HZ = 8.0

def _hik_grab_loop():
    global _buf_hik_jpg, _buf_hik_np, _cam_hik_running, _buf_hik_raw
    cam = _state["camera_hik"]
    last_enc = 0.0
    enc_period = 1.0 / max(1.0, _HIK_MJPEG_HZ)
    while _cam_hik_running and cam and cam.initialized:
        ret, frame = cam.get_frame()   # RGB — dobot_camera.HIK_OUTPUT_SIZE=None 이면 원본
        if ret and frame is not None:
            # 여기서 한 번만 512×512 로 만든다. 화면·녹화·ROS2 가 **같은 그림**을 쓴다.
            # (이전에는 녹화=원본 / ROS2=640×480 이라 두 저장 경로의 화각이 달랐다)
            _raw = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bgr = to_512(_raw, "hik")

            enc_bytes = None
            now = time.time()
            if now - last_enc >= enc_period:
                _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                enc_bytes = enc.tobytes()
                last_enc = now

            with _buf_lock:
                if enc_bytes is not None:
                    _buf_hik_jpg = enc_bytes
                _buf_hik_np = bgr
                _buf_hik_raw = _raw

            # ROS2 에도 같은 512 버퍼를 그대로 보낸다.
            #    512×512×3 = 786KB/frame → 16Hz 에서 약 12MB/s.
            #    2026-09-14 에 프로세스를 멈춰 세웠던 원본 발행(14.4MB/frame ≈ 230MB/s)
            #    대비 1/19 이라 그 경로를 다시 밟지 않는다.
            if _ros2_ok:
                _ros2.publish_hik(bgr)
        else:
            time.sleep(0.02)

def _zed_grab_loop():
    global _buf_zed_jpg, _buf_zed_np, _cam_zed_running, _buf_zed_raw
    cam = _state["camera_zed"]
    while _cam_zed_running and cam and cam.initialized:
        ret, frame = cam.get_frame()   # RGB
        if ret and frame is not None:
            # HIK 과 같이 여기서 한 번만 512×512 로 만든다.
            # ZED 는 16:9 라 눌러 담으면 왜곡되므로 변환기와 같은 810×810 정사각을
            # 잘라 쓴다 — 로봇팔·박스가 화면을 채우고 벽·창문·책장이 빠진다.
            _raw = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bgr = to_512(_raw, "zed")
            _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _buf_lock:
                _buf_zed_jpg = enc.tobytes()
                _buf_zed_np  = bgr
                _buf_zed_raw = _raw
            if _ros2_ok:
                _ros2.publish_zed(bgr)   # 캡처 직후 타임스탬프로 퍼블리시
        else:
            time.sleep(0.02)

def _robot_pub_loop():
    """로봇 상태를 ~50Hz 로 ROS2 에 퍼블리시. startup 에서 데몬 스레드로 시작."""
    global _robot_pub_running
    while _robot_pub_running:
        if _ros2_ok:
            robot   = _state["robot"]
            gripper = _state["gripper"]
            if robot and robot.connected:
                try:
                    feed = robot.feed.feedBackData()
                    if feed is not None and len(feed) > 0:
                        joints     = feed['QActual'][0].tolist()
                        tcp_pose   = feed['ToolVectorActual'][0].tolist()
                        robot_mode = int(feed['RobotMode'][0]) if 'RobotMode' in feed.dtype.names else 0
                        gripper_on = 1 if (gripper and gripper.is_gripping) else 0
                        _ros2.publish_robot(joints, tcp_pose, gripper_on, robot_mode)
                except Exception:
                    pass
        time.sleep(0.02)   # ~50 Hz


def _mjpeg_gen(buf_getter):
    """공통 MJPEG 제너레이터."""
    placeholder = None
    while True:
        with _buf_lock:
            frame = buf_getter()
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else:
            if placeholder is None:
                blank = np.full((240, 320, 3), 60, dtype=np.uint8)
                cv2.putText(blank, "No Camera", (55, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2)
                _, enc = cv2.imencode('.jpg', blank)
                placeholder = enc.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
        time.sleep(0.04)

# ═══════════════════════════════════════════════════════════════════════════
# 20Hz 레코딩 (threading 기반, QTimer 대체)
# ═══════════════════════════════════════════════════════════════════════════

def _start_recording():
    if _state["recording"]:
        return

    # ── 외장 드라이브 마운트 확인 ──────────────────────────────────────────
    if not os.path.isdir(DATA_DRIVE_ROOT):
        _log(f"[ERROR] External drive not mounted: {DATA_DRIVE_ROOT}")
        _log("[ERROR] Data collection aborted — please connect the drive and retry")
        # 진행 중인 worker 도 중단
        w = _state.get("worker")
        if w:
            w._stop_requested = True
        # 녹화는 시작하지 않는다. 호출부가 실패로 처리한다.
        return
    # ────────────────────────────────────────────────────────────────────────

    n = _get_next_folder(DATA_SAVE_DIR)
    save_dir = os.path.join(DATA_SAVE_DIR, str(n))
    has_zed = bool(_state["camera_zed"] and _state["camera_zed"].initialized)
    try:
        os.makedirs(os.path.join(save_dir, "images", "hik"), exist_ok=True)
        if has_zed:
            os.makedirs(os.path.join(save_dir, "images", "zed"), exist_ok=True)
    except OSError as e:
        _log(f"[ERROR] Cannot create save directory: {e}")
        _log("[ERROR] Data collection aborted — check drive permissions")
        w = _state.get("worker")
        if w:
            w._stop_requested = True
        # 녹화는 시작하지 않는다. 호출부가 실패로 처리한다.
        return

    _state.update(recording=True, recorded_data=[], record_save_dir=save_dir,
                  record_frame_count=0)
    mode_str = "ROS2+sync" if (_ros2_ok and has_zed) else "legacy"
    _log(f"Recording started → {save_dir} (ZED={'ON' if has_zed else 'OFF'}, mode={mode_str})")
    if _ros2_ok:
        _ros2.start_recording(save_dir)
    threading.Thread(target=_record_loop, daemon=True).start()

def _record_loop():
    while _state["recording"]:
        # 🔴 항상 legacy tick 으로 저장한다 (2026-09-14).
        #
        #    이전에는 ros2 + ZED 가 둘 다 살아 있으면 ROS2 sync 콜백이 저장을 맡았다.
        #    그런데 그 경로(ros2_recorder._save_worker)는
        #    crop_center_480_resize_512() 를 거치며 **중앙 480×480 을 잘라 512×512 로**
        #    만든다. crop_size 가 480 고정이라 프레임이 원본(2592×1944)이면 폭의
        #    18.5% 만 남는다 — 원본 저장이 목적이므로 이 경로를 쓸 수 없다.
        #
        #    ⚠️ 대가: HIK/ZED 의 ApproximateTimeSynchronizer(35ms) 정합을 잃는다.
        #       여기서는 50ms 폴링으로 두 버퍼의 최신값을 같이 집는다. 엄밀한
        #       프레임 정합이 다시 필요해지면 _save_worker 의 크롭을 걷어낸 뒤
        #       이 분기를 되살릴 것.
        _record_tick()
        time.sleep(0.05)

def _record_tick():
    robot    = _state["robot"]
    gripper  = _state["gripper"]
    save_dir = _state["record_save_dir"]
    if not robot or not robot.connected or not save_dir:
        return
    try:
        feed = robot.feed.feedBackData()
        if feed is None or len(feed) == 0:
            return
        joints     = feed['QActual'][0].tolist()
        tcp_pose   = feed['ToolVectorActual'][0].tolist()
        robot_mode = int(feed['RobotMode'][0]) if 'RobotMode' in feed.dtype.names else 0
        gripper_on = 1 if (gripper and gripper.is_gripping) else 0
        fc = _state["record_frame_count"]
        fname = f"frame_{fc:06d}.jpg"

        # HIK 이미지 저장
        with _buf_lock:
            hik_np = _buf_hik_np.copy() if _buf_hik_np is not None else None
            zed_np = _buf_zed_np.copy() if _buf_zed_np is not None else None

        # 파이프라인 (2026-09-16):
        #   카메라  전체 화각(AOI 0,0,2592,1944) · 원본 해상도로 받는다
        #     ↓
        #   그랩루프  to_512() 로 512×512 (변환기와 같은 CROP_NORM)
        #     ↓
        #   저장    512×512 그대로. 변환기는 다시 크롭하지 않는다.
        #
        # 🔴 이미지는 **ROS2 sync 경로가 단독으로** 쓴다 (2026-09-16).
        #
        #    이전에는 여기(_record_tick)와 ros2_recorder._save_worker 가 **둘 다**
        #    같은 `frame_%06d.jpg` 에 썼다. 그런데 두 writer 가 **각자 0번부터**
        #    번호를 매기고 속도가 달라서, 같은 번호가 서로 다른 순간을 가리켰다.
        #
        #    실측 (2026-09-16, 241프레임 에피소드):
        #      ROS2 15.85Hz / tick 14.20Hz → index 212 까지는 tick 이 나중에 써서
        #      이겼고, CSV 는 ROS2 것이라 **이미지가 로봇 상태보다 최대 1.54초
        #      앞선 순간**이었다. 그 전(원본 해상도 저장) 에는 5.5Hz vs 14Hz 라
        #      **10.30초** 까지 벌어졌다.
        #
        #    ⇒ writer 를 하나로 만들면 이미지와 상태가 같은 콜백(_on_sync)에서
        #      나오므로 어긋남이 **구조적으로 0** 이 된다. 16fps 도 ROS2 쪽이
        #      카메라(16.19Hz)를 그대로 따라가므로 이쪽을 남긴다.
        #      (π0 비교 설정: 수집 16fps · action chunk 16스텝 ≈ 1.0초)
        #
        #    ⚠️ ROS2 가 없을 때만 여기서 쓴다. 이 폴백을 지우면 ros2 가 죽은 날
        #       **CSV 는 멀쩡한데 이미지 폴더만 비는** 에피소드가 조용히 쌓인다.
        if not _ros2_ok:
            hik_path = os.path.join(save_dir, "images", "hik", fname)
            if hik_np is not None:
                cv2.imwrite(hik_path, hik_np)
            else:
                # 검은 더미를 쓰면 문제가 조용히 묻힌다. 파일을 만들지 않고 남긴다.
                print(f"[record_tick] HIK 프레임 없음 → {fname} 저장 생략")

            if _state["camera_zed"] and _state["camera_zed"].initialized:
                zed_path = os.path.join(save_dir, "images", "zed", fname)
                if zed_np is not None:
                    cv2.imwrite(zed_path, zed_np)
                else:
                    print(f"[record_tick] ZED 프레임 없음 → {fname} 저장 생략")

        has_zed = bool(_state["camera_zed"] and _state["camera_zed"].initialized)

        record = {
            'frame_id':       fc,
            'timestamp':      time.time(),
            'image_path_hik': f"hik/{fname}",
            'image_path_zed': f"zed/{fname}" if has_zed else "",
            'joint_angles':   joints,
            'tcp_pose':       tcp_pose,
            'gripper_tooldo1':gripper_on,
            'gripper_tooldo2':0,
            'robot_mode':     robot_mode,
        }
        _state["recorded_data"].append(record)
        _state["record_frame_count"] += 1
    except Exception as e:
        print(f"[record_tick] {e}")

def _stop_and_save(success: bool):
    _state["recording"] = False
    time.sleep(0.07)
    save_dir = _state["record_save_dir"]

    # ros2 sync 데이터 우선 사용, 없으면 legacy fallback
    if _ros2_ok:
        ros2_data = _ros2.stop_recording()
        data = ros2_data if ros2_data else _state["recorded_data"]
    else:
        data = _state["recorded_data"]

    if not success:
        _log("Episode failed → not saved")
        if save_dir and os.path.isdir(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)
        _state.update(recorded_data=[], record_save_dir=None)
        return

    if not save_dir or not data:
        _log(f"No data recorded (dir={save_dir}, n={len(data) if data else 0})")
        return
    try:
        has_zed = bool(data[0].get('image_path_zed'))
        # CSV
        with open(os.path.join(save_dir, "robot_data.csv"), 'w', newline='') as f:
            f.write("frame_id,timestamp,image_path_hik")
            if has_zed:
                f.write(",image_path_zed")
            f.write(",j1,j2,j3,j4,j5,j6,x,y,z,rx,ry,rz"
                    ",gripper_tooldo1,gripper_tooldo2,robot_mode\n")
            for r in data:
                f.write(f"{r['frame_id']},{r['timestamp']},{r['image_path_hik']}")
                if has_zed:
                    f.write(f",{r['image_path_zed']}")
                f.write(',' + ','.join(map(str, r['joint_angles'])))
                f.write(',' + ','.join(map(str, r['tcp_pose'])))
                f.write(f",{r['gripper_tooldo1']},{r['gripper_tooldo2']},{r['robot_mode']}\n")
        # NPY
        np.save(os.path.join(save_dir, "dataset.npy"), data)
        # episode_meta.json
        import json as _json
        folder_num = os.path.basename(save_dir)
        n_frames = len(data)
        if n_frames >= 2:
            actual_fps = round((n_frames - 1) / (data[-1]['timestamp'] - data[0]['timestamp']), 3)
        else:
            actual_fps = 15.0
        ep_meta = dict(_state.get("episode_meta") or {})
        ep_meta.update({
            "folder": folder_num,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames": n_frames,
            "record_rate_hz": actual_fps,
            "cameras": "HIK+ZED" if has_zed else "HIK",
            "success": bool(success),
            "vacuum_pick_duration_s": round(_state['vacuum_pick'], 3),
            "vacuum_place_duration_s": round(_state['vacuum_place'], 3),
        })
        events_raw = ep_meta.pop("events", [])
        with open(os.path.join(save_dir, "episode_meta.json"), 'w', encoding='utf-8') as f:
            _json.dump(ep_meta, f, ensure_ascii=False, indent=2)
        # episode_events.csv
        if events_raw and data:
            ts_list = [(r['frame_id'], r['timestamp']) for r in data]
            with open(os.path.join(save_dir, "episode_events.csv"), 'w', newline='') as f:
                f.write("event,frame_id,timestamp\n")
                for ev_name, ev_ts in events_raw:
                    closest_fid = min(ts_list, key=lambda t: abs(t[1] - ev_ts))[0]
                    f.write(f"{ev_name},{closest_fid},{ev_ts:.6f}\n")
        # metadata.txt (호환성 유지)
        with open(os.path.join(save_dir, "metadata.txt"), 'w') as f:
            f.write("VLA Dataset - Pick-Place Step\n" + "="*50 + "\n\n")
            f.write(f"Folder: {folder_num}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {len(data)}\n")
            f.write(f"Record Rate: {actual_fps}Hz\n")
            f.write(f"Cameras: HIK{'+ ZED (LEFT)' if has_zed else ' only'}\n")
            f.write(f"Step Success: {success}\n")
            f.write(f"VacuumCommandPickDuration_s: {_state['vacuum_pick']:.3f}\n")
            f.write(f"VacuumCommandPlaceDuration_s: {_state['vacuum_place']:.3f}\n")
        _state["episode_meta"] = {}
        _log(f"Saved {len(data)} frames → {save_dir}")
    except Exception as e:
        _log(f"Save error: {e}")
    finally:
        _state.update(recorded_data=[], record_save_dir=None)

# ═══════════════════════════════════════════════════════════════════════════
# Zone move worker (흡착 없이 target zone까지 이동하는 궤적 수집)
# ═══════════════════════════════════════════════════════════════════════════

# ⚠️ 2026-09-15: 이 worker 를 직접 돌리는 경로(▶ Zone Step 버튼 · /zone-dataset/*)는
#    제거했다. **흡착을 하지 않아** 수집 목적에 맞지 않기 때문이다.
#    클래스 자체는 남긴다 — 아래 ZonePickWorker 가 이걸 상속해 _move / IK 샘플링 /
#    저장·복귀 로직을 그대로 쓴다. 지우면 그쪽이 깨진다.
class ZoneMoveWithoutSuctionWorker(threading.Thread):
    """초기자세 -> target zone 하강 완료까지만 recording하고, 복귀는 녹화하지 않는다."""
    def __init__(
        self,
        robot,
        zone_id: str,
        episode_in_zone: int,
        x_offset_mm: float = ZONE_X_OFFSET_MM,
        y_offset_mm: float = ZONE_Y_OFFSET_MM,
    ):
        super().__init__(daemon=True)
        self.robot = robot
        self.zone_id = str(zone_id)
        self.episode_in_zone = int(episode_in_zone)
        self.x_offset_mm = float(x_offset_mm)
        self.y_offset_mm = float(y_offset_mm)
        self._stop_requested = False
        self._recording_saved_before_finished = False
        self.log_signal = _BoundSignal()
        self.finished = _BoundSignal()
        self.recording_begin_at_initial = _BoundSignal()
        self.episode_meta_ready = _BoundSignal()
        self.episode_vacuum_durations = _BoundSignal()
        self._events = []

    def isRunning(self):
        return self.is_alive()

    def _log(self, msg: str):
        self.log_signal.emit(msg)

    def _move(self, x, y, z, rx, ry, rz, velocity=24.0):
        if self._stop_requested:
            return False
        ok = self.robot.move_j(
            x, y, z, rx, ry, rz,
            coordinate_mode=0,
            velocity=velocity,
            use_waypoint=False,
        )
        if ok:
            self.robot.wait_for_motion_complete()
        return ok

    def _return_to_initial_unrecorded(self):
        try:
            ix, iy, iz, irx, iry, irz = ZONE_INIT_POSE
            self._log("Return to initial pose (recording already stopped)...")
            self._move(ix, iy, iz, irx, iry, irz, velocity=35.0)
        except Exception as e:
            self._log(f"Return to initial failed: {e}")

    def _save_or_discard_recording(self, success: bool):
        if _state.get("recording"):
            self._recording_saved_before_finished = True
            _stop_and_save(success)

    def _build_episode_meta(self, target_pose, travel_z):
        tx, ty, tz, trx, try_, trz = target_pose
        return {
            "task": "move_to_zone_without_suction",
            "task_name": "move_to_zone_without_suction",
            "zone_id": self.zone_id,
            "zone_name": f"zone_{self.zone_id}",
            "instruction": f"move to zone {self.zone_id}",
            "suction": False,
            "return_to_home_recorded": False,
            "camera_mapping": dict(CAMERA_MAPPING),
            "image_save": dict(IMAGE_SAVE_META),
            "future_lerobot_preprocess": dict(FUTURE_LEROBOT_PREPROCESS_META),
            "episode_in_zone": self.episode_in_zone,
            "target_pose": {
                "x": round(tx, 3), "y": round(ty, 3), "z": round(tz, 3),
                "rx": round(trx, 3), "ry": round(try_, 3), "rz": round(trz, 3),
            },
            "travel_z": round(float(travel_z), 3),
            "events": list(self._events),
        }

    def _sample_target_pose_with_ik(self):
        base_pose = ZONE_POSES[self.zone_id]
        bx, by, *_ = [float(v) for v in base_pose[:6]]
        for attempt in range(60):
            x = bx + random.uniform(-self.x_offset_mm, self.x_offset_mm)
            y = by + random.uniform(-self.y_offset_mm, self.y_offset_mm)
            z = ZONE_TARGET_Z
            rx, ry, rz = base.get_descent_rpy(x, y)
            ok, msg = self.robot.check_ik_solution(x, y, z, rx, ry, rz)
            if ok:
                if attempt > 0:
                    self._log(f"IK target accepted after {attempt + 1} tries")
                return (x, y, z, rx, ry, rz)
            self._log(f"IK reject zone_{self.zone_id} try {attempt + 1}: {msg}")
        return None

    # 🔴 2026-09-16: 이 클래스의 zone_move / zone_pick **태스크는 폐기됐다**.
    #    엔드포인트·러너·UI 버튼을 전부 제거했고 run() 도 지웠다.
    #    클래스를 남긴 이유는 PickPlaceLineWorker 가 아래 헬퍼들을 상속해 쓰기 때문이다:
    #      _move · _goto_zone_and_descend · _lift_to_mid ·
    #      _save_or_discard_recording · _return_to_initial_unrecorded · _build_episode_meta
    #    즉 지금은 **공용 헬퍼 베이스**이지 실행 가능한 태스크가 아니다.

# ═══════════════════════════════════════════════════════════════════════════
# Worker 콜백
# ═══════════════════════════════════════════════════════════════════════════

def _on_log(msg):    _log(msg)
def _on_vacuum(ph, pl): _state["vacuum_pick"] = ph; _state["vacuum_place"] = pl
def _on_rec_begin():  _start_recording()
def _on_episode_meta(meta): _state["episode_meta"] = meta

# ═══════════════════════════════════════════════════════════════════════════
# Zone pick worker — 집는 구간만 녹화한다
# ═══════════════════════════════════════════════════════════════════════════
#
#   [녹화]     초기자세 → pick_zone 위 → 하강 → 흡착 ON → 상승 → 초기자세 → 저장
#   [녹화 없음] → place_zone 위 → 하강 → 흡착 OFF(놓기) → 상승 → 초기자세
#
# 다음 에피소드는 place_zone 에서 집는다(상자가 거기로 옮겨졌으므로).
# 상자를 옮기는 뒷구간은 리셋 동작이라 데이터에 넣지 않는다.
#
# ⚠️ ZoneMoveWithoutSuctionWorker 는 그대로 둔다. 그 worker 로 모은 기존 데이터와
#    계약이 어긋나면 안 되므로 동작을 바꾸지 않고 새 클래스를 만든다.

# 🔴 자연어 명령. 변환기(convert_dobot_to_lerobot_v21.py:448-453)가 이 문자열을
#    LeRobot 의 task 라벨로 그대로 쓴다. 존 번호를 넣으면 존마다 다른 task 가
#    생겨, 추론 때 "pick up the orange box" 같은 단일 문장과 맞지 않게 된다.
#    ⇒ 모든 존에서 **같은 문장**을 쓴다.
ZONE_PICK_INSTRUCTION = "pick up the orange box"


def _next_zone_id(zone_id: str) -> str:
    """ZONE_ORDER 에서 다음 존. 끝이면 처음으로 돌아온다."""
    try:
        i = ZONE_ORDER.index(str(zone_id))
    except ValueError:
        return ZONE_ORDER[0]
    return ZONE_ORDER[(i + 1) % len(ZONE_ORDER)]


class ZonePickWorker(ZoneMoveWithoutSuctionWorker):
    """pick_zone 에서 집는 구간만 녹화하고, place_zone 으로 옮겨 놓는 건 녹화하지 않는다."""

    def __init__(self, robot, gripper, pick_zone_id, place_zone_id,
                 episode_in_zone=0, x_offset_mm=ZONE_X_OFFSET_MM,
                 y_offset_mm=ZONE_Y_OFFSET_MM):
        super().__init__(robot, zone_id=pick_zone_id, episode_in_zone=episode_in_zone,
                         x_offset_mm=x_offset_mm, y_offset_mm=y_offset_mm)
        self.gripper = gripper
        self.place_zone_id = str(place_zone_id)

    def _sample_for(self, zone_id):
        """임의 존에 대해 IK 통과하는 목표 자세를 뽑는다(부모 구현을 존만 바꿔 재사용)."""
        saved, self.zone_id = self.zone_id, str(zone_id)
        try:
            return self._sample_target_pose_with_ik()
        finally:
            self.zone_id = saved

    def _goto_zone_and_descend(self, pose, label):
        """존 위로 이동 → MID Z → 목표 Z. 실패하면 False."""
        tx, ty, tz, trx, try_, trz = pose
        travel_z = random.uniform(*ZONE_TRAVEL_Z_RANGE)
        self._log(f"{label}: 존 위로 이동 X={tx:.1f} Y={ty:.1f} Z={travel_z:.1f}")
        if not self._move(tx, ty, travel_z, base.TRAVEL_RX, base.TRAVEL_RY, base.TRAVEL_RZ, velocity=34.0):
            return False
        if not self._move(tx, ty, base.DESCENT_MID_Z, base.TRAVEL_RX, base.TRAVEL_RY, base.TRAVEL_RZ, velocity=28.0):
            return False
        if not self._move(tx, ty, tz, trx, try_, trz, velocity=base.DESCENT_VELOCITY):
            return False
        return travel_z

    def _lift_to_mid(self, pose):
        tx, ty = pose[0], pose[1]
        return self._move(tx, ty, base.DESCENT_MID_Z,
                          base.TRAVEL_RX, base.TRAVEL_RY, base.TRAVEL_RZ, velocity=28.0)

    def _build_episode_meta(self, target_pose, travel_z):
        m = super()._build_episode_meta(target_pose, travel_z)
        m.update({
            "task": "zone_pick",
            "task_name": "zone_pick",
            "instruction": ZONE_PICK_INSTRUCTION,
            "suction": True,
            "return_to_home_recorded": True,   # 흡착 ON 후 초기자세 복귀까지 녹화
            "pick_zone_id": self.zone_id,
            "place_zone_id": self.place_zone_id,   # 녹화되지 않는 리셋 구간의 목적지
            "place_recorded": False,
        })
        return m

# ═══════════════════════════════════════════════════════════════════════════
# Pick-and-place across the center line — A ↔ B 핑퐁 (2026-09-16 신설)
# ═══════════════════════════════════════════════════════════════════════════
#
#   [녹화]  초기자세 → pick 좌표 하강 → 흡착 ON → 상승
#           → 반대 구역 무작위 좌표 하강 (가운데 선 통과) → 흡착 OFF
#           → +1.0초 → 저장·녹화 종료
#   [녹화 X] 상승 → 초기자세 복귀
#
# 놓은 좌표가 **다음 에피소드의 집을 좌표**가 되어 사람이 박스를 옮길 필요가 없고,
# 매 에피소드 방향이 A↔B 로 교대되므로 항상 가운데 선을 넘는다.
#
# 🔴 릴리즈 후 1.0초를 더 녹화하는 이유 (PICKPLACE_LINE_POST_RELEASE_SEC):
#    흡착 OFF 명령 즉시 끊으면 릴리즈가 **완전한 16스텝 청크 1개에만** 들어간다.
#    _stop_and_save 가 0.07초(≈1프레임)만 기다리므로 `gripper 1→0` 전이가 0프레임인
#    에피소드도 생긴다 — 실제로 옛 zone_pick 데이터가 그 상태였다(전이가 0→1 하나뿐).
#    그러면 정책은 "놓아라"를 배울 수가 없다.
#    1.0초 = 16프레임 = 청크 하나 길이라, 릴리즈를 포함한 완전한 청크가 16개 생긴다
#    (π0 와 같은 커버리지). 복귀 이동은 여전히 녹화 밖이다.
#
# ⚠️ 놓기가 어긋나면(박스가 튕기거나 구르면) 다음 에피소드가 빈 자리를 집으러 간다.
#    좌표를 기억해서 쓰는 구조의 대가이므로, 실패가 보이면 바로 멈출 것.

PICKPLACE_LINE_INSTRUCTION = "pick up the orange box"
PICKPLACE_LINE_POST_RELEASE_SEC = 1.0     # 릴리즈 후 더 녹화할 시간 (16fps → 16프레임)
PICKPLACE_LINE_START_XY = (float(base.POS_4[0]), float(base.POS_4[1]))
PICKPLACE_LINE_START_SECTION = "A"
SECTION_POINTS = {"A": base.A_SECTION_POINTS, "B": base.B_SECTION_POINTS}
SECTION_OPPOSITE = {"A": "B", "B": "A"}
# 가운데 선 = A·B 가 공유하는 변 (POS_6 ↔ POS_7). 기록용으로만 쓴다 —
# A 에서 집어 B 에 놓으면 경로상 반드시 지나므로 따로 강제할 것이 없다.
CENTER_LINE = ((float(base.POS_6[0]), float(base.POS_6[1])),
               (float(base.POS_7[0]), float(base.POS_7[1])))


class PickPlaceLineWorker(ZonePickWorker):
    """A↔B 핑퐁 pick-and-place. 구역 폴리곤 안에서만 좌표를 뽑는다."""

    def __init__(self, robot, gripper, pick_xy, pick_section, episode_index=0):
        super().__init__(robot, gripper,
                         pick_zone_id=str(pick_section),
                         place_zone_id=str(SECTION_OPPOSITE[str(pick_section)]),
                         episode_in_zone=episode_index)
        self.pick_xy = (float(pick_xy[0]), float(pick_xy[1]))
        self.pick_section = str(pick_section)
        self.place_section = SECTION_OPPOSITE[self.pick_section]
        self.place_xy = None
        self.episode_index = int(episode_index)

    # ── 좌표 만들기 ───────────────────────────────────────────────────────
    def _pose_at_xy(self, x, y, label):
        """주어진 xy 의 하강 자세. IK 가 거부하면 None."""
        z = ZONE_TARGET_Z
        rx, ry, rz = base.get_descent_rpy(x, y)
        ok, msg = self.robot.check_ik_solution(x, y, z, rx, ry, rz)
        if ok:
            return (float(x), float(y), z, rx, ry, rz)
        self._log(f"IK reject {label} ({x:.1f}, {y:.1f}): {msg}")
        return None

    def _sample_place_pose(self):
        """반대 구역 폴리곤 **내부**에서 IK 통과하는 좌표를 뽑는다.

        generate_random_point_in_section() 이 point_in_polygon 을 통과할 때까지
        다시 뽑으므로 체커보드(구역) 밖 좌표는 애초에 나오지 않는다.
        """
        pts = SECTION_POINTS[self.place_section]
        for attempt in range(60):
            x, y = base.generate_random_point_in_section(pts)
            pose = self._pose_at_xy(x, y, f"place {self.place_section} try{attempt + 1}")
            if pose is not None:
                if attempt > 0:
                    self._log(f"place 좌표 확정 (시도 {attempt + 1}회)")
                return pose
        return None

    def _build_episode_meta(self, target_pose, travel_z):
        m = super()._build_episode_meta(target_pose, travel_z)
        px, py = self.pick_xy
        qx, qy = (self.place_xy or (None, None))
        m.update({
            "task": "pick_place_across_line",
            "task_name": "pick_place_across_line",
            "instruction": PICKPLACE_LINE_INSTRUCTION,
            "suction": True,
            "return_to_home_recorded": False,    # 복귀는 녹화 밖
            "place_recorded": True,
            "post_release_recorded_sec": PICKPLACE_LINE_POST_RELEASE_SEC,
            "pick_section": self.pick_section,
            "place_section": self.place_section,
            "pick_x": round(px, 3), "pick_y": round(py, 3),
            "place_x": None if qx is None else round(qx, 3),
            "place_y": None if qy is None else round(qy, 3),
            "pick_xy_origin": ("fixed_start_pt4" if self.episode_index == 0
                               else "previous_episode_place"),
            "transport_direction": (None if qx is None
                                    else base._transport_direction(qx, qy)),
            "crossed_center_line": True,
            "center_line": {"p1": list(CENTER_LINE[0]), "p2": list(CENTER_LINE[1])},
            "section_polygons": {k: [list(p) for p in v] for k, v in SECTION_POINTS.items()},
            "episode_index": self.episode_index,
        })
        for k in ("zone_id", "zone_name", "pick_zone_id", "place_zone_id", "episode_in_zone"):
            m.pop(k, None)
        return m

    # ── 본 흐름 ───────────────────────────────────────────────────────────
    def run(self):
        try:
            if not self.robot or not self.robot.connected:
                self._log("Robot not connected"); self.finished.emit(False); return
            if self.gripper is None:
                self._log("Gripper 없음 — pick-place 불가"); self.finished.emit(False); return

            # 🔴 두 좌표를 **녹화 시작 전에** 확정한다. 녹화 중에 IK 를 60번 돌리면
            #    그 시간만큼 로봇이 멈춘 프레임이 데이터에 들어간다.
            pick_pose = self._pose_at_xy(*self.pick_xy, label=f"pick {self.pick_section}")
            if pick_pose is None:
                self._log("Failed: pick 좌표 IK 거부"); self.finished.emit(False); return
            place_pose = self._sample_place_pose()
            if place_pose is None:
                self._log(f"Failed: {self.place_section} 구역에서 IK 통과 좌표를 못 찾음")
                self.finished.emit(False); return
            self.place_xy = (place_pose[0], place_pose[1])
            self._log(f"pick {self.pick_section}({self.pick_xy[0]:.1f}, {self.pick_xy[1]:.1f}) "
                      f"→ place {self.place_section}({self.place_xy[0]:.1f}, {self.place_xy[1]:.1f})")

            ix, iy, iz, irx, iry, irz = ZONE_INIT_POSE
            self._log("1) 초기자세로 이동...")
            if not self._move(ix, iy, iz, irx, iry, irz, velocity=35.0):
                self._log("Failed: initial pose"); self.finished.emit(False); return
            try:
                self.gripper.release()      # 집으러 가기 전 흡착이 꺼져 있는지 확실히
            except Exception:
                pass

            self._log("2) 초기자세에서 1초 대기 후 녹화 시작...")
            for _ in range(10):
                if self._stop_requested:
                    self.finished.emit(False); return
                time.sleep(0.1)
            self._events.append(("recording_start_at_initial", time.time()))
            self.recording_begin_at_initial.emit()
            if self._stop_requested or not _state.get("recording"):
                self._log("Recording did not start; aborting")
                self.finished.emit(False); return

            travel_z = self._goto_zone_and_descend(pick_pose, f"3) pick {self.pick_section}")
            if travel_z is False:
                self._log("Failed: 집으러 이동/하강")
                self._save_or_discard_recording(False)
                self._return_to_initial_unrecorded()
                self.finished.emit(False); return

            self._log(f"4) 흡착 ON — {self.pick_section} 구역에서 집는다")
            self.gripper.grip()
            self._events.append(("suction_on_at_pick", time.time()))

            self._log("5) 상승")
            if not self._lift_to_mid(pick_pose):
                self._log("Failed: 상승")
                self._save_or_discard_recording(False)
                self._return_to_initial_unrecorded()
                self.finished.emit(False); return

            self._log(f"6) 가운데 선을 넘어 {self.place_section} 구역으로 이송·하강")
            if self._goto_zone_and_descend(place_pose, f"6) place {self.place_section}") is False:
                self._log("Failed: 이송/하강 — 박스를 든 채 복귀한다")
                self._save_or_discard_recording(False)
                self._return_to_initial_unrecorded()
                self.finished.emit(False); return

            self._log("7) 흡착 OFF — 놓는다")
            self.gripper.release()
            self._events.append(("suction_off_at_place", time.time()))

            self._log(f"8) 릴리즈 후 {PICKPLACE_LINE_POST_RELEASE_SEC:.1f}초 더 녹화")
            time.sleep(PICKPLACE_LINE_POST_RELEASE_SEC)
            self._events.append(("recording_end_after_release", time.time()))

            self.episode_vacuum_durations.emit(0.0, 0.0)
            self.episode_meta_ready.emit(self._build_episode_meta(place_pose, travel_z))
            self._log("9) 녹화 종료·저장")
            self._save_or_discard_recording(True)

            self._log("10) 상승·초기자세 복귀 (녹화 없음)")
            self._lift_to_mid(place_pose)
            self._return_to_initial_unrecorded()
            self.finished.emit(True)
        except Exception as e:
            self._log(f"PickPlaceLine worker error: {e}")
            self._save_or_discard_recording(False)
            self._return_to_initial_unrecorded()
            self.finished.emit(False)


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(title="Dobot E6 Server")

@app.on_event("startup")
async def _startup():
    global _log_queue, _main_loop, _ros2_ok, _robot_pub_running
    _log_queue = asyncio.Queue()
    _main_loop = asyncio.get_event_loop()
    asyncio.create_task(_broadcast())

    # ROS2 레코더 초기화 (rclpy 미설치 시 False 반환 → fallback 모드)
    _ros2_ok = _ros2.start()
    if _ros2_ok:
        _robot_pub_running = True
        threading.Thread(target=_robot_pub_loop, daemon=True).start()
        _log("ROS2 recorder ready (sync mode)")
    else:
        _log("ROS2 unavailable — legacy recording mode")

    _log("Server ready")

async def _broadcast():
    while True:
        msg = await _log_queue.get()
        dead = set()
        for ws in list(_ws_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        _ws_clients.difference_update(dead)

# ─── 연결 ────────────────────────────────────────────────────────────────

@app.post("/connect")
def connect(ip: str = "192.168.5.1"):
    if _state["robot"] and _state["robot"].connected:
        return {"ok": True, "msg": "Already connected"}
    try:
        robot = DobotE6Controller(ip=ip)
        if not robot.connect():
            return JSONResponse({"ok": False, "msg": "Connect failed — robot unreachable"}, status_code=500)
        _state["robot"]   = robot
        _state["gripper"] = SuctionGripper(robot, do_index=1)
        _log(f"Robot connected @ {ip}")
        return {"ok": True, "msg": f"Connected @ {ip}"}
    except Exception as e:
        _log(f"Connect error: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/enable")
def enable_robot():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        robot.dashboard.EnableRobot()
        _log("Robot enabled")
        return {"ok": True, "msg": "Robot enabled"}
    except Exception as e:
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/disable")
def disable_robot():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        robot.dashboard.DisableRobot()
        _log("Robot disabled")
        return {"ok": True, "msg": "Robot disabled"}
    except Exception as e:
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/clear-alarm")
def clear_alarm():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        result = robot.dashboard.ClearError()
        _log(f"ClearError → {result}")
        return {"ok": True, "msg": f"Alarm cleared ({result})"}
    except Exception as e:
        _log(f"ClearError failed: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/resume")
def resume_robot():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        robot.resume_robot()
        robot.clear_error()
        robot.enable_robot(sleep_after=0.1)
        _log("Resume → ClearError + EnableRobot 완료")
        return {"ok": True}
    except Exception as e:
        _log(f"Resume failed: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/disconnect")
def disconnect():
    if _state["robot"]:
        try:
            _state["robot"].disconnect()
        except Exception:
            pass
        _state["robot"] = _state["gripper"] = None
        _log("Robot disconnected")
    return {"ok": True}

@app.get("/status")
def status():
    robot     = _state["robot"]
    connected = bool(robot and robot.connected)
    pose = joints = None
    robot_mode = 0
    if connected:
        try:
            feed = robot.feed.feedBackData()
            if feed is not None and len(feed) > 0:
                joints     = [round(float(v), 3) for v in feed['QActual'][0]]
                pose       = [round(float(v), 3) for v in feed['ToolVectorActual'][0]]
                robot_mode = int(feed['RobotMode'][0]) if 'RobotMode' in feed.dtype.names else 0
        except Exception:
            pass
    return {
        "ppl":            _ppl_progress(),
        "connected":      connected,
        "pose":           pose,
        "joints":         joints,
        "robot_mode":     robot_mode,
        "robot_mode_str": ROBOT_MODE_LABELS.get(robot_mode, str(robot_mode)),
        "cam_hik":        bool(_state["camera_hik"] and _state["camera_hik"].initialized),
        "cam_zed":        bool(_state["camera_zed"] and _state["camera_zed"].initialized),
        "recording":      _state["recording"],
        "frames":         _state["record_frame_count"],
        "auto_target":    _state["auto_target"],
        "auto_done":      _state["auto_done"],
        "auto_zone_order": _state["auto_zone_order"],
        "auto_episodes_per_zone": _state["auto_episodes_per_zone"],
        "zone_order":     ZONE_ORDER,
        "zone_episodes_per_zone": ZONE_EPISODES_PER_ZONE,
        "zone_stats":     _state["zone_stats"],
        "current_zone_id": _state["current_zone_id"],
        "current_zone_episode": _state["current_zone_episode"],
        "last_auto_error": _state["last_auto_error"],
        "last_auto_done": _state["last_auto_done"],
        "last_auto_target": _state["last_auto_target"],
        "worker_running": bool(_state["worker"] and _state["worker"].isRunning()),
    }

# ─── 로봇 제어 ────────────────────────────────────────────────────────────

@app.post("/home")
def go_home():
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    x, y, z, rx, ry, rz = HOME_POSE
    # 이전 좌표 (300,0,400,180,0,0) 은 xArm 리그 값이라 이 Dobot 에서는 IK 해가 없다.
    # 컨트롤러가 알람 17(역연산 무해)을 내고 ERROR(mode 9)로 래치되는데, 이 엔드포인트가
    # 무조건 ok:True 를 돌려줘 UI 에는 성공으로 보였다 — 원인 찾는 데 오래 걸린 이유.
    reachable, why = robot.check_ik_solution(x, y, z, rx, ry, rz)
    if not reachable:
        _log(f"Home 취소 — 도달 불가: {why}")
        return JSONResponse({"ok": False, "msg": f"Home 취소 — {why}"}, status_code=400)

    def _do():
        ok = robot.move_j(x, y, z, rx, ry, rz, coordinate_mode=0, use_waypoint=False)
        if ok: robot.wait_for_motion_complete()
    threading.Thread(target=_do, daemon=True).start()
    return {"ok": True}

@app.post("/move")
def move(x: float, y: float, z: float,
         rx: float = 180.0, ry: float = 0.0, rz: float = 0.0,
         velocity: float = 30.0):
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    def _do():
        ok = robot.move_j(x, y, z, rx, ry, rz, coordinate_mode=0,
                          velocity=velocity, use_waypoint=False)
        if ok: robot.wait_for_motion_complete()
    threading.Thread(target=_do, daemon=True).start()
    return {"ok": True}

_jog_lock = threading.Lock()
_jog_axis_active: list = [None]   # [0] = currently jogging axis or None
_jog_stop_time: list  = [0.0]     # [0] = last stop timestamp

@app.post("/jog/start")
def jog_start(axis: str, speed: int = 20):
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    speed = max(1, min(100, speed))
    def _do():
        with _jog_lock:
            # cooldown: wait until 200ms after last stop
            elapsed = time.time() - _jog_stop_time[0]
            if elapsed < 0.20:
                time.sleep(0.20 - elapsed)
            try:
                robot.dashboard.EnableRobot()
            except Exception:
                pass
            try:
                robot.dashboard.SpeedFactor(speed)
            except Exception:
                pass
            for attempt in range(2):
                try:
                    if axis.startswith('J'):
                        result = robot.dashboard.MoveJog(axis)
                    else:
                        result = robot.dashboard.MoveJog(axis, coordtype=1, user=0, tool=0)
                    result_str = str(result).strip() if result else ""
                    first = result_str.split(',')[0].strip() if result_str else ""
                    if first and first != "0":
                        if attempt == 0:
                            time.sleep(0.15)
                            continue  # retry once
                        _log(f"[jog] {axis} error: {result_str}")
                    else:
                        _jog_axis_active[0] = axis
                        _log(f"[jog] {axis} start (speed={speed}%)")
                    break
                except Exception as e:
                    _log(f"[jog] {axis} exception: {e}")
                    break
    threading.Thread(target=_do, daemon=True).start()
    return {"ok": True}

@app.post("/jog/stop")
def jog_stop():
    robot = _state["robot"]
    if robot and robot.connected:
        def _stop():
            with _jog_lock:
                try:
                    robot.dashboard.MoveJog("")
                except Exception as e:
                    _log(f"[jog] stop error: {e}")
                try:
                    robot.dashboard.SpeedFactor(100)
                except Exception:
                    pass
                was = _jog_axis_active[0]
                _jog_axis_active[0] = None
                _jog_stop_time[0] = time.time()
                if was:
                    _log(f"[jog] {was} stopped")
        threading.Thread(target=_stop, daemon=True).start()
    return {"ok": True}

@app.get("/pose")
def get_pose():
    """현재 TCP 좌표 반환 (조그 후 위치 확인용)."""
    robot = _state["robot"]
    if not robot or not robot.connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    try:
        pose = robot.get_current_pose_from_feedback()
        if pose and len(pose) >= 6:
            return {"ok": True, "x": round(pose[0],3), "y": round(pose[1],3), "z": round(pose[2],3),
                    "rx": round(pose[3],3), "ry": round(pose[4],3), "rz": round(pose[5],3)}
        return JSONResponse({"ok": False, "msg": "No pose data"}, status_code=503)
    except Exception as e:
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/gripper/grip")
def grip():
    g = _state["gripper"]
    if not g: return JSONResponse({"ok": False, "msg": "No gripper"}, status_code=400)
    threading.Thread(target=g.grip, daemon=True).start()
    return {"ok": True}

@app.post("/gripper/release")
def release():
    g = _state["gripper"]
    if not g: return JSONResponse({"ok": False, "msg": "No gripper"}, status_code=400)
    threading.Thread(target=g.release, daemon=True).start()
    return {"ok": True}

@app.post("/estop")
def estop():
    w = _state["worker"]
    if w: w._stop_requested = True
    if _state["gripper"]:
        try: _state["gripper"].emergency_release()
        except Exception: pass
    if _state["robot"]: _state["robot"].disable_robot()
    _log("E-STOP triggered")
    return {"ok": True}

# ─── Zone Move Dataset ────────────────────────────────────────────────────

# 수집 목표. 진행률은 **디스크에서 센다** — 인메모리 카운터는 서버를 재시작하면 0 이
# 되어서, 며칠에 걸쳐 모으는 동안 몇 개를 모았는지 알 수 없게 된다.
PICKPLACE_LINE_TARGET_TOTAL = 180


def _ppl_scan_disk() -> dict:
    """저장된 pick_place_across_line 에피소드를 방향별로 센다.

    ⚠️ 폴더 번호가 아니라 **meta 의 task_name** 으로 거른다. 폴더 번호는 옛 태스크
       에피소드를 지우면 비는 자리가 생겨 개수와 어긋난다.
    """
    out = {"total": 0, "A_to_B": 0, "B_to_A": 0, "frames": 0, "unreadable": 0, "error": ""}
    try:
        names = sorted((d for d in os.listdir(DATA_SAVE_DIR) if d.isdigit()), key=int)
    except OSError as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    for n in names:
        try:
            with open(os.path.join(DATA_SAVE_DIR, n, "episode_meta.json"), encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            # meta 가 없거나 깨진 폴더(녹화 중이거나 중단된 것). 세어서 드러낸다.
            out["unreadable"] += 1
            continue
        if m.get("task_name") != "pick_place_across_line" or not m.get("success"):
            continue
        out["total"] += 1
        out["frames"] += int(m.get("total_frames") or 0)
        key = f"{m.get('pick_section')}_to_{m.get('place_section')}"
        if key in out:
            out[key] += 1
    return out


def _ppl_progress() -> dict:
    d = _ppl_scan_disk()
    xy = _state.get("ppl_box_xy")
    sec = _state.get("ppl_box_section")
    d.update({
        "target": PICKPLACE_LINE_TARGET_TOTAL,
        "remaining": max(0, PICKPLACE_LINE_TARGET_TOTAL - d["total"]),
        "box_section": sec,
        "box_x": None if not xy else round(float(xy[0]), 1),
        "box_y": None if not xy else round(float(xy[1]), 1),
        "next_direction": None if not sec else f"{sec}->{SECTION_OPPOSITE[sec]}",
        "auto_target": int(_state.get("ppl_auto_target", 0)),
        "auto_done": int(_state.get("ppl_auto_done", 0)),
    })
    return d


def _ppl_reset_box():
    _state["ppl_box_xy"] = tuple(PICKPLACE_LINE_START_XY)
    _state["ppl_box_section"] = PICKPLACE_LINE_START_SECTION
    _state["ppl_episode_index"] = 0


def _run_pickplace_line_step():
    """A↔B 핑퐁 한 에피소드. 놓은 좌표를 기억해 다음 에피소드의 집을 좌표로 쓴다."""
    robot   = _state.get("robot")
    gripper = _state.get("gripper")
    if _state.get("ppl_box_xy") is None:
        _ppl_reset_box()
        _log(f"박스 위치 초기화 → {PICKPLACE_LINE_START_SECTION} "
             f"({PICKPLACE_LINE_START_XY[0]:.1f}, {PICKPLACE_LINE_START_XY[1]:.1f}) [4번 고정 시작점]")

    worker = PickPlaceLineWorker(
        robot, gripper,
        pick_xy=_state["ppl_box_xy"],
        pick_section=_state["ppl_box_section"],
        episode_index=int(_state.get("ppl_episode_index", 0)),
    )
    worker.log_signal.connect(_on_log)
    worker.episode_vacuum_durations.connect(_on_vacuum)
    worker.episode_meta_ready.connect(_on_episode_meta)
    worker.recording_begin_at_initial.connect(_on_rec_begin)

    def _done(ok: bool):
        # 워커가 이미 저장했으면 다시 저장하지 않는다(중복 저장 방지).
        if not getattr(worker, "_recording_saved_before_finished", False):
            _stop_and_save(ok)
        if ok and worker.place_xy is not None:
            # 🔴 박스는 이제 놓은 자리에 있다. 이 좌표가 다음 에피소드의 pick 이 된다.
            _state["ppl_box_xy"] = tuple(worker.place_xy)
            _state["ppl_box_section"] = worker.place_section
            _state["ppl_episode_index"] = int(_state.get("ppl_episode_index", 0)) + 1
            _log(f"박스 위치 갱신 → {worker.place_section} "
                 f"({worker.place_xy[0]:.1f}, {worker.place_xy[1]:.1f})")
        if _state.get("ppl_auto_target", 0) > 0:
            if not ok:
                _state["ppl_auto_target"] = 0
                _log("⚠ 자동 수집 중단 (실패/STOP) — 박스 위치가 어긋났을 수 있으니 확인할 것")
                return
            _state["ppl_auto_done"] = int(_state.get("ppl_auto_done", 0)) + 1
            _log(f"Auto {_state['ppl_auto_done']}/{_state['ppl_auto_target']}")
            if _state["ppl_auto_done"] >= _state["ppl_auto_target"]:
                _state["ppl_auto_target"] = 0
                _log("Auto collect complete")
                return
            threading.Thread(target=_run_pickplace_line_step, daemon=True).start()

    worker.finished.connect(_done)
    _state["worker"] = worker
    worker.start()


@app.post("/pick-place-line/step")
def pickplace_line_step():
    if _state.get("worker") and _state["worker"].isRunning():
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    if not _state.get("robot") or not _state["robot"].connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    if not _state.get("gripper"):
        return JSONResponse({"ok": False, "msg": "No gripper"}, status_code=400)
    _state["ppl_auto_target"] = 0
    threading.Thread(target=_run_pickplace_line_step, daemon=True).start()
    sec = _state.get("ppl_box_section") or PICKPLACE_LINE_START_SECTION
    return {"ok": True, "msg": f"{sec} 에서 집어 {SECTION_OPPOSITE[sec]} 로 (가운데 선 통과)"}


@app.post("/pick-place-line/auto")
def pickplace_line_auto(n: int = 100):
    if _state.get("worker") and _state["worker"].isRunning():
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    if not _state.get("robot") or not _state["robot"].connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    if not _state.get("gripper"):
        return JSONResponse({"ok": False, "msg": "No gripper"}, status_code=400)
    _state.update(ppl_auto_target=int(n), ppl_auto_done=0)
    _log(f"pick_place_across_line 자동 수집 시작: {n} 에피소드")
    threading.Thread(target=_run_pickplace_line_step, daemon=True).start()
    return {"ok": True}


@app.post("/pick-place-line/set-box")
def pickplace_line_set_box(x: float = None, y: float = None, section: str = None):
    """박스를 지금 어디에 놓아뒀는지 알려준다.

    인자를 다 비우면 시작점(4번 고정, A 구역)으로 초기화한다.
    ⚠️ 이 값이 실제와 다르면 로봇이 빈 자리를 집으러 간다.
    """
    if x is None and y is None and section is None:
        _ppl_reset_box()
        return {"ok": True, "msg": f"시작점으로 초기화: A {PICKPLACE_LINE_START_XY}"}
    if x is None or y is None or section is None:
        return JSONResponse({"ok": False, "msg": "x, y, section 을 모두 주거나 모두 비울 것"},
                            status_code=400)
    section = str(section).upper()
    if section not in SECTION_POINTS:
        return JSONResponse({"ok": False, "msg": f"Unknown section: {section}"}, status_code=400)
    if not base.point_in_polygon(float(x), float(y), SECTION_POINTS[section]):
        return JSONResponse(
            {"ok": False, "msg": f"({x}, {y}) 는 {section} 구역 밖이다 — 체커보드 안 좌표를 줄 것"},
            status_code=400)
    _state["ppl_box_xy"] = (float(x), float(y))
    _state["ppl_box_section"] = section
    _log(f"박스 위치 수동 지정 → {section} ({x:.1f}, {y:.1f})")
    return {"ok": True, "msg": f"box = {section} ({x:.1f}, {y:.1f})"}


@app.post("/pick-place/stop")
def stop_collect():
    if _state["worker"]: _state["worker"]._stop_requested = True
    _state["auto_target"] = 0
    _log("Stop requested")
    return {"ok": True}

# ─── 카메라 ──────────────────────────────────────────────────────────────

@app.post("/camera/hik/start")
def hik_start():
    global _cam_hik_thread, _cam_hik_running
    if not _hik_available:
        return JSONResponse({"ok": False, "msg": "HIK SDK not available"}, status_code=400)
    if _state["camera_hik"] and _state["camera_hik"].initialized:
        return {"ok": True, "msg": "Already running"}
    try:
        cam = make_hik_camera()
        if not cam.init_camera():
            return JSONResponse({"ok": False, "msg": "Exterior Cam 1 (HIK) init failed — check USB connection"}, status_code=500)
        _state["camera_hik"] = cam
        _cam_hik_running = True
        _cam_hik_thread  = threading.Thread(target=_hik_grab_loop, daemon=True)
        _cam_hik_thread.start()
        _log("Exterior Cam 1 (HIKRobot) started")
        return {"ok": True}
    except Exception as e:
        _log(f"HIK start error: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/camera/hik/stop")
def hik_stop():
    global _cam_hik_running
    _cam_hik_running = False
    if _state["camera_hik"]:
        try: _state["camera_hik"].cleanup()
        except Exception: pass
        _state["camera_hik"] = None
    _log("Exterior Cam 1 (HIKRobot) stopped")
    return {"ok": True}

@app.post("/camera/zed/start")
def zed_start():
    global _cam_zed_thread, _cam_zed_running
    if not _zed_available:
        return JSONResponse({"ok": False, "msg": "ZED SDK not available"}, status_code=400)
    if _state["camera_zed"] and _state["camera_zed"].initialized:
        return {"ok": True, "msg": "Already running"}
    try:
        cam = ZedCamera()
        if not cam.init_camera():
            return JSONResponse({"ok": False, "msg": "Exterior Cam 2 (ZED) init failed — check USB3 connection"}, status_code=500)
        _state["camera_zed"] = cam
        _cam_zed_running = True
        _cam_zed_thread  = threading.Thread(target=_zed_grab_loop, daemon=True)
        _cam_zed_thread.start()
        _log("Exterior Cam 2 (ZED) started — LEFT view only")
        return {"ok": True}
    except Exception as e:
        _log(f"ZED start error: {e}")
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=500)

@app.post("/camera/zed/stop")
def zed_stop():
    global _cam_zed_running
    _cam_zed_running = False
    if _state["camera_zed"]:
        try: _state["camera_zed"].cleanup()
        except Exception: pass
        _state["camera_zed"] = None
    _log("Exterior Cam 2 (ZED) stopped")
    return {"ok": True}

@app.get("/camera/hik/stream")
async def hik_stream():
    return StreamingResponse(
        _mjpeg_gen(lambda: _buf_hik_jpg),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/{cam}/raw.jpg")
def camera_raw(cam: str):
    """크롭 전 원본 프레임 1장 (HIK 2592×1944 / ZED 1920×1080).

    크롭 영역(CROP_NORM_512)을 정하려면 전체 화각을 봐야 하는데, 스트림은 이미
    512 로 잘린 뒤라 그걸로는 못 정한다. 이 엔드포인트가 없으면 크롭을 바꿀 때마다
    서버를 재시작해서 눈으로 확인하는 수밖에 없다.

    ⚠️ 진단용이다. 녹화 경로는 이 버퍼를 쓰지 않는다.
    """
    from fastapi.responses import Response
    with _buf_lock:
        raw = _buf_hik_raw if cam == "hik" else _buf_zed_raw if cam == "zed" else None
        raw = None if raw is None else raw.copy()
    if raw is None:
        return Response(content=f"no frame for {cam}", status_code=404,
                        media_type="text/plain")
    ok, enc = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return Response(content="encode failed", status_code=500,
                        media_type="text/plain")
    return Response(content=enc.tobytes(), media_type="image/jpeg",
                    headers={"X-Frame-Shape": f"{raw.shape[1]}x{raw.shape[0]}"})

@app.get("/camera/zed/stream")
async def zed_stream():
    return StreamingResponse(
        _mjpeg_gen(lambda: _buf_zed_jpg),
        media_type="multipart/x-mixed-replace; boundary=frame")

# ─── WebSocket ────────────────────────────────────────────────────────────

@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)

# ─── Web UI ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_HTML)

_HTML = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dobot E6 Server</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#12121f;color:#dde;font-size:13px;min-width:900px}
h1{text-align:center;padding:9px;background:#0e0e1d;color:#00c8e8;font-size:1.05rem;letter-spacing:1px;border-bottom:1px solid #1e2a3a}

/* 3-column layout */
.layout{display:grid;grid-template-columns:300px 1fr 340px;gap:8px;padding:8px;align-items:start}
.col{display:flex;flex-direction:column;gap:8px;min-width:0}

/* card */
.card{background:#1a1a30;border-radius:7px;padding:11px;overflow:hidden}
h3{color:#00c8e8;font-size:.7rem;text-transform:uppercase;letter-spacing:.6px;margin-bottom:9px;border-bottom:1px solid #1e2a3a;padding-bottom:5px}

/* row, inputs, buttons */
.row{display:flex;gap:5px;margin-bottom:6px;align-items:center;flex-wrap:wrap}
input[type=text],input[type=number]{background:#0d1a2e;color:#dde;border:1px solid #2a4a6a;
  padding:4px 7px;border-radius:4px;flex:1;min-width:0;font-size:.8rem}
button{background:#0d1a2e;color:#aac8e0;border:1px solid #2a4a6a;padding:5px 10px;
  border-radius:4px;cursor:pointer;font-size:.78rem;white-space:nowrap;transition:background .12s}
button:hover{background:#00c8e8;color:#0d1a2e;border-color:#00c8e8}
button:active{filter:brightness(1.3)}
.btn-g{border-color:#2dc653;color:#2dc653}.btn-g:hover{background:#2dc653;color:#0d1a2e}
.btn-r{border-color:#e63946;color:#e63946}.btn-r:hover{background:#e63946;color:#fff}
.btn-y{border-color:#f4a261;color:#f4a261}.btn-y:hover{background:#f4a261;color:#0d1a2e}

/* status bars */
.sbar{background:#0d1a2e;padding:5px 9px;border-radius:4px;font-size:.72rem;margin-bottom:5px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:4px;vertical-align:middle}
.on{background:#2dc653}.off{background:#e63946}.rec{background:#e63946;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
#zone-table{width:100%;border-collapse:collapse;margin:6px 0 8px;background:#0d1a2e;border-radius:4px;overflow:hidden;font-size:.68rem}
#zone-table th,#zone-table td{padding:4px 6px;border-bottom:1px solid #1e2a3a;text-align:left}
#zone-table th{color:#00c8e8;font-weight:600;background:#0a0f1e}
#zone-table tr:last-child td{border-bottom:none}
.zstat.done{color:#2dc653}.zstat.running{color:#f4a261}.zstat.failed{color:#e63946}.zstat.pending{color:#889}

/* dashboard grid */
.dash-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px}
.dash-cell{background:#0d1a2e;border-radius:4px;padding:5px 4px;text-align:center}
.dash-label{font-size:.58rem;color:#557;text-transform:uppercase}
.dash-val{font-size:.88rem;font-weight:700;color:#00c8e8;font-family:monospace}

/* cameras */
.cam-row{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.cam-box{background:#0d1a2e;border-radius:5px;overflow:hidden}
.cam-label{font-size:.63rem;color:#446;padding:3px 6px;background:#0a0f1e}
img.stream{width:100%;height:480px;object-fit:contain;display:block;background:#000}

/* log */
#log{background:#060612;font-family:monospace;font-size:.67rem;height:140px;overflow-y:auto;
  padding:6px;border-radius:4px;color:#6fdf8f;word-break:break-all}

/* separator */
.sep{border-top:1px solid #1e2a3a;margin:8px 0}
.sub{font-size:.63rem;color:#557;margin-bottom:5px;margin-top:2px}
.zone-select{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.zone-select label{font-size:.68rem;color:#aac8e0;display:flex;align-items:center;gap:3px}

/* ── JOG ── */
/* D-pad: 3×3 grid */
.dpad{display:grid;grid-template-columns:repeat(3,52px);grid-template-rows:repeat(3,40px);gap:4px}
.dpad .jb{font-size:.9rem;font-weight:700;padding:0;display:flex;align-items:center;justify-content:center;
  border-radius:5px;user-select:none;-webkit-user-select:none;touch-action:none}
.dpad .jb.center{background:#1e2a3a;color:#446;font-size:.6rem;cursor:default;border:1px solid #1e2a3a}
.dpad .jb.center:hover{background:#1e2a3a;color:#446;border-color:#1e2a3a}

/* Z col */
.zcol{display:flex;flex-direction:column;gap:4px;margin-left:8px}
.zcol .jb{width:48px;height:40px;font-size:.85rem;font-weight:700;display:flex;align-items:center;
  justify-content:center;border-radius:5px;user-select:none;-webkit-user-select:none;touch-action:none}

/* rotation row */
.rot-row{display:grid;grid-template-columns:repeat(6,1fr);gap:4px}
.rot-row .jb{padding:5px 2px;font-size:.72rem;text-align:center;font-weight:600;
  border-radius:4px;user-select:none;-webkit-user-select:none;touch-action:none}

/* joint row */
.joint-table{display:grid;grid-template-columns:repeat(6,1fr);gap:4px}
.joint-table .jb{padding:6px 2px;font-size:.72rem;text-align:center;
  border-radius:4px;user-select:none;-webkit-user-select:none;touch-action:none}

/* active jog highlight */
.jb.jogging{background:#00c8e8 !important;color:#0d1a2e !important;border-color:#00c8e8 !important}

/* speed slider */
input[type=range]{width:100%;accent-color:#00c8e8}

/* pose capture */
#pose-display{font-size:.68rem;color:#9cf;margin-top:4px;font-family:monospace;word-break:break-all;min-height:16px}
</style>
</head>
<body>
<h1>⬡ Dobot E6 — Robot Control &amp; Data Collection</h1>
<div class="layout">

<!-- ══════════════ LEFT COL ══════════════ -->
<div class="col">

  <!-- Connection -->
  <div class="card">
    <h3>Connection</h3>
    <div class="row">
      <input id="ip" type="text" value="192.168.5.1" style="max-width:115px">
      <button class="btn-g" onclick="api('POST','/connect',{ip:$('ip').value})">Connect</button>
      <button onclick="api('POST','/disconnect')">Disconnect</button>
    </div>
    <div id="conn-bar" class="sbar"><span class="dot off"></span>Disconnected</div>
    <div id="mode-bar" class="sbar" style="margin-bottom:7px">Mode: —</div>
    <div class="row" style="margin-bottom:0;gap:4px">
      <button class="btn-g" onclick="api('POST','/enable')">Enable</button>
      <button onclick="api('POST','/disable')">Disable</button>
      <button class="btn-y" onclick="clearAlarm()">Clear Alarm</button>
      <button class="btn-y" onclick="api('POST','/resume').then(()=>addLog('▶ Resume sent'))">Resume</button>
      <button onclick="api('POST','/home')">Home</button>
    </div>
  </div>

  <!-- Dashboard -->
  <div class="card">
    <h3>Robot Dashboard</h3>
    <div class="sub">TCP Pose (mm / deg)</div>
    <div class="dash-grid">
      <div class="dash-cell"><div class="dash-label">X</div><div class="dash-val" id="dX">—</div></div>
      <div class="dash-cell"><div class="dash-label">Y</div><div class="dash-val" id="dY">—</div></div>
      <div class="dash-cell"><div class="dash-label">Z</div><div class="dash-val" id="dZ">—</div></div>
      <div class="dash-cell"><div class="dash-label">RX</div><div class="dash-val" id="dRX">—</div></div>
      <div class="dash-cell"><div class="dash-label">RY</div><div class="dash-val" id="dRY">—</div></div>
      <div class="dash-cell"><div class="dash-label">RZ</div><div class="dash-val" id="dRZ">—</div></div>
    </div>
    <div class="sub" style="margin-top:8px">Joint Angles (deg)</div>
    <div class="dash-grid">
      <div class="dash-cell"><div class="dash-label">J1</div><div class="dash-val" id="dJ1">—</div></div>
      <div class="dash-cell"><div class="dash-label">J2</div><div class="dash-val" id="dJ2">—</div></div>
      <div class="dash-cell"><div class="dash-label">J3</div><div class="dash-val" id="dJ3">—</div></div>
      <div class="dash-cell"><div class="dash-label">J4</div><div class="dash-val" id="dJ4">—</div></div>
      <div class="dash-cell"><div class="dash-label">J5</div><div class="dash-val" id="dJ5">—</div></div>
      <div class="dash-cell"><div class="dash-label">J6</div><div class="dash-val" id="dJ6">—</div></div>
    </div>
  </div>

  <!-- Data Collection -->
  <div class="card">
    <h3>Zone Move Dataset</h3>
    <div id="auto-bar" class="sbar">Ready</div>
    <div class="sub">Zone Selection</div>
    <div class="zone-select">
      <label><input type="checkbox" id="zone-all" checked>ALL</label>
      <label><input type="checkbox" class="zone-opt" value="2" checked>2</label>
      <label><input type="checkbox" class="zone-opt" value="5" checked>5</label>
      <label><input type="checkbox" class="zone-opt" value="6" checked>6</label>
      <label><input type="checkbox" class="zone-opt" value="8" checked>8</label>
      <label><input type="checkbox" class="zone-opt" value="9" checked>9</label>
    </div>
    <div class="sub">Collection Progress</div>
    <table id="zone-table">
      <tbody id="ppl-table-body"></tbody>
    </table>
    <div class="sub">Pick &amp; Place across center line (A &#8596; B)</div>
    <div class="row">
      <button class="btn-g" onclick="api('POST','/pick-place-line/step')">&#9654; 1 Episode</button>
      <button class="btn-g" onclick="pplAuto()">&#9654;&#9654; Auto</button>
    </div>
    <div class="row">
      <button class="btn-y" onclick="api('POST','/pick-place-line/set-box')">&#8635; Box &#8594; start (pt4, A)</button>
    </div>
    <div class="row">
      <button class="btn-y" onclick="api('POST','/pick-place/stop')">&#9632; Stop</button>
    </div>
    <button class="btn-r" style="width:100%;padding:8px;font-size:.82rem;font-weight:700" onclick="doEstop()">⚠ E-STOP</button>
  </div>

</div><!-- end left col -->

<!-- ══════════════ CENTER COL ══════════════ -->
<div class="col">

  <!-- Camera controls -->
  <div class="card">
    <h3>Exterior Cameras</h3>
    <div class="row" style="margin-bottom:4px">
      <span style="font-size:.72rem;color:#00c8e8;min-width:105px">Cam 1 — HIKRobot</span>
      <button class="btn-g" onclick="api('POST','/camera/hik/start')">Start</button>
      <button onclick="api('POST','/camera/hik/stop')">Stop</button>
      <span id="hik-stat" style="font-size:.72rem;color:#668;margin-left:6px">OFF</span>
    </div>
    <div class="row" style="margin-bottom:0">
      <span style="font-size:.72rem;color:#00c8e8;min-width:105px">Cam 2 — ZED</span>
      <button class="btn-g" onclick="api('POST','/camera/zed/start')">Start</button>
      <button onclick="api('POST','/camera/zed/stop')">Stop</button>
      <span id="zed-stat" style="font-size:.72rem;color:#668;margin-left:6px">OFF</span>
    </div>
  </div>

  <!-- Camera streams -->
  <div class="card" style="padding:8px">
    <div class="cam-row">
      <div class="cam-box">
        <div class="cam-label">Cam 1 — HIKRobot</div>
        <img class="stream" src="/camera/hik/stream" alt="HIK">
      </div>
      <div class="cam-box">
        <div class="cam-label">Cam 2 — ZED (LEFT)</div>
        <img class="stream" src="/camera/zed/stream" alt="ZED">
      </div>
    </div>
  </div>

  <!-- Log -->
  <div class="card">
    <h3>Log</h3>
    <div id="log"></div>
  </div>

</div><!-- end center col -->

<!-- ══════════════ RIGHT COL: JOG ══════════════ -->
<div class="col">
  <div class="card">
    <h3>Jog Control</h3>

    <!-- Gripper -->
    <div class="row" style="margin-bottom:7px">
      <button class="btn-g" style="flex:1;padding:7px" onclick="api('POST','/gripper/grip')">Grip ON [Q]</button>
      <button style="flex:1;padding:7px" onclick="api('POST','/gripper/release')">Grip OFF [W]</button>
    </div>

    <!-- Speed -->
    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
        <span style="font-size:.73rem;color:#aac8e0;font-weight:600">Jog Speed</span>
        <span style="font-size:.82rem;font-weight:700;color:#00c8e8"><span id="speed-val">5</span>%</span>
      </div>
      <input type="range" id="jog-speed" min="1" max="50" value="5"
             oninput="$('speed-val').textContent=this.value">
    </div>

    <!-- Capture pose -->
    <div style="margin-bottom:8px">
      <button onclick="capturePose()" style="width:100%;padding:6px;font-size:.75rem;background:#263445;border-color:#3a5a7a;color:#9cf">
        📍 현재 좌표 캡처
      </button>
      <div id="pose-display"></div>
    </div>

    <div class="sep"></div>

    <!-- TCP XY D-pad + Z -->
    <div class="sub">TCP XY / Z &nbsp;·&nbsp; 키보드: ←→↑↓ / Z=Z+ X=Z-</div>
    <div style="display:flex;align-items:center;margin-bottom:8px">
      <!-- D-pad 3×3 -->
      <div class="dpad">
        <div></div>
        <button class="jb btn-g" id="jb-Y+" data-axis="Y+">↑<br><small style="font-size:.55rem">Y+</small></button>
        <div></div>
        <button class="jb btn-g" id="jb-X+" data-axis="X+">←<br><small style="font-size:.55rem">X+</small></button>
        <div class="jb center">XY</div>
        <button class="jb btn-g" id="jb-X-" data-axis="X-">→<br><small style="font-size:.55rem">X-</small></button>
        <div></div>
        <button class="jb btn-g" id="jb-Y-" data-axis="Y-">↓<br><small style="font-size:.55rem">Y-</small></button>
        <div></div>
      </div>
      <!-- Z col -->
      <div class="zcol">
        <button class="jb btn-g" id="jb-Z+" data-axis="Z+">Z+<br><small style="font-size:.55rem">▲</small></button>
        <button class="jb btn-g" id="jb-Z-" data-axis="Z-">Z-<br><small style="font-size:.55rem">▼</small></button>
      </div>
    </div>

    <!-- Rotation -->
    <div class="sub">Rotation (Rx / Ry / Rz)</div>
    <div class="rot-row" style="margin-bottom:8px">
      <button class="jb" id="jb-Rx+" data-axis="Rx+">Rx+</button>
      <button class="jb" id="jb-Rx-" data-axis="Rx-">Rx-</button>
      <button class="jb" id="jb-Ry+" data-axis="Ry+">Ry+</button>
      <button class="jb" id="jb-Ry-" data-axis="Ry-">Ry-</button>
      <button class="jb" id="jb-Rz+" data-axis="Rz+">Rz+</button>
      <button class="jb" id="jb-Rz-" data-axis="Rz-">Rz-</button>
    </div>

    <div class="sep"></div>

    <!-- Joint jog -->
    <div class="sub">Joint Jog</div>
    <div class="joint-table">
      <button class="jb" id="jb-J1+" data-axis="J1+">J1+</button>
      <button class="jb" id="jb-J1-" data-axis="J1-">J1-</button>
      <button class="jb" id="jb-J2+" data-axis="J2+">J2+</button>
      <button class="jb" id="jb-J2-" data-axis="J2-">J2-</button>
      <button class="jb" id="jb-J3+" data-axis="J3+">J3+</button>
      <button class="jb" id="jb-J3-" data-axis="J3-">J3-</button>
      <button class="jb" id="jb-J4+" data-axis="J4+">J4+</button>
      <button class="jb" id="jb-J4-" data-axis="J4-">J4-</button>
      <button class="jb" id="jb-J5+" data-axis="J5+">J5+</button>
      <button class="jb" id="jb-J5-" data-axis="J5-">J5-</button>
      <button class="jb" id="jb-J6+" data-axis="J6+">J6+</button>
      <button class="jb" id="jb-J6-" data-axis="J6-">J6-</button>
    </div>

  </div>
</div><!-- end right col -->

</div><!-- end layout -->

<script>
function pplAuto(){
  // 기본값 = 180 까지 남은 개수 (디스크 기준)
  const rem = (window._pplRemaining == null) ? 100 : window._pplRemaining;
  const n = prompt('\uc218\uc9d1\ud560 \uc5d0\ud53c\uc18c\ub4dc \uac1c\uc218', String(rem));
  if(!n) return;
  api('POST','/pick-place-line/auto?n=' + encodeURIComponent(n));
}
const $ = id => document.getElementById(id);
const api = async (m, p, q={}) => {
  const url = p + (Object.keys(q).length ? '?' + new URLSearchParams(q) : '');
  try {
    const res = await fetch(url, {method:m});
    const data = await res.json();
    if (data && data.msg) addLog((data.ok ? '✓ ' : '✗ ') + data.msg);
    if (!res.ok && !data.msg) addLog(`✗ ${m} ${p} → HTTP ${res.status}`);
    return data;
  } catch(e) { addLog('✗ Network: ' + e); }
};
const addLog = msg => {
  const b=$('log'); b.innerHTML += msg+'<br>'; b.scrollTop=b.scrollHeight;
};
const zoneOpts = () => Array.from(document.querySelectorAll('.zone-opt'));
const selectedZones = () => zoneOpts().filter(el => el.checked).map(el => el.value);
const syncZoneAll = () => {
  const all = $('zone-all');
  const opts = zoneOpts();
  if(!all || !opts.length) return;
  all.checked = opts.every(el => el.checked);
};
const initZoneSelector = () => {
  const all = $('zone-all');
  const opts = zoneOpts();
  if(all){
    all.addEventListener('change', () => {
      opts.forEach(el => { el.checked = all.checked; });
    });
  }
  opts.forEach(el => el.addEventListener('change', syncZoneAll));
  syncZoneAll();
};

// WebSocket
const ws = new WebSocket(`ws://${location.host}/ws/logs`);
ws.onmessage = e => addLog(e.data);
ws.onopen = () => addLog('[WS] connected');
setInterval(() => { if(ws.readyState===1) ws.send('ping'); }, 10000);

// Status polling
setInterval(async () => {
  const s = await api('GET','/status');
  if(!s) return;
  $('conn-bar').innerHTML = `<span class="dot ${s.connected?'on':'off'}"></span>`
    + (s.connected ? 'Connected' : 'Disconnected')
    + (s.recording ? ' &nbsp;<span class="dot rec"></span><b> REC</b>' : '');

  // 로봇 MODE 표시 — ERROR(9)면 빨간 경고
  const isError = s.robot_mode === 9;
  $('mode-bar').textContent = (isError ? '⚠ ROBOT ERROR — ' : 'Mode: ')
    + (s.robot_mode_str||'—')
    + (s.recording ? ` | REC ${s.frames??''}f` : '');
  $('mode-bar').style.background = isError ? '#3a0a0a' : '#0d1a2e';
  $('mode-bar').style.color      = isError ? '#ff4444' : '#dde';
  $('mode-bar').style.fontWeight = isError ? '700' : 'normal';

  // ERROR 상태면 자동 수집 서버 측 중단 알림
  if (isError && s.auto_target > 0) {
    addLog('⚠ [ERROR] 로봇 충돌/알람 감지 — 자동 수집 중단됨. Clear Alarm 후 재시작하세요.');
    api('POST','/pick-place/stop');
  }
  if(s.joints) ['J1','J2','J3','J4','J5','J6'].forEach((k,i)=>{
    const el=$('d'+k); if(el) el.textContent=s.joints[i]?.toFixed(1)??'—';
  });
  if(s.pose) ['X','Y','Z','RX','RY','RZ'].forEach((k,i)=>{
    const el=$('d'+k); if(el) el.textContent=s.pose[i]?.toFixed(1)??'—';
  });
  $('hik-stat').textContent = s.cam_hik ? 'ON ●' : 'OFF';
  $('hik-stat').style.color  = s.cam_hik ? '#2dc653' : '#668';
  $('zed-stat').textContent  = s.cam_zed ? 'ON ●' : 'OFF';
  $('zed-stat').style.color  = s.cam_zed ? '#2dc653' : '#668';
  const activeZones = (s.auto_zone_order && s.auto_zone_order.length) ? s.auto_zone_order : (s.zone_order || []);
  const perZone = s.auto_episodes_per_zone || s.zone_episodes_per_zone;
  if(s.auto_target > 0) {
    $('auto-bar').textContent =
      `Auto: ${s.auto_done}/${s.auto_target} | Zones: ${activeZones.join(',')} | Current Zone: ${s.current_zone_id||'—'} | Episode: ${s.current_zone_episode||0}/${perZone}`;
  } else if((s.last_auto_target||0) > 0) {
    $('auto-bar').textContent = `Last Auto: ${s.last_auto_done}/${s.last_auto_target}`
      + (s.last_auto_error ? ` | Failed: ${s.last_auto_error}` : ' | Complete');
  } else if(s.worker_running) {
    $('auto-bar').textContent = 'Running…';
  } else {
    $('auto-bar').textContent = 'Ready';
  }
  const pplBody = $('ppl-table-body');
  if(pplBody) {
    const p = s.ppl || {};
    window._pplRemaining = p.remaining;
    const box = (p.box_section == null)
        ? '<span style="color:#e8a33a">\uBBF8\uC124\uC815</span>'
        : `${p.box_section} (${p.box_x}, ${p.box_y})`;
    const err = p.error ? `<tr><td colspan="2" style="color:#e8503a">${p.error}</td></tr>` : '';
    const unread = p.unreadable ? `<tr><td>\uC77D\uD790 \uC218 \uC5C6\uC74C</td><td style="color:#e8a33a">${p.unreadable}</td></tr>` : '';
    const auto = p.auto_target ? `<tr><td>Auto</td><td>${p.auto_done}/${p.auto_target}</td></tr>` : '';
    pplBody.innerHTML = `
      <tr><td>\uD569\uACC4</td><td><b>${p.total||0}</b> / ${p.target||0}  (\uB0A8\uC74C ${p.remaining||0})</td></tr>
      <tr><td>A &#8594; B</td><td>${p.A_to_B||0}</td></tr>
      <tr><td>B &#8594; A</td><td>${p.B_to_A||0}</td></tr>
      <tr><td>\uD504\uB808\uC784</td><td>${(p.frames||0).toLocaleString()}</td></tr>
      <tr><td>\uBC15\uC2A4 \uC704\uCE58</td><td>${box}</td></tr>
      <tr><td>\uB2E4\uC74C \uBC29\uD5A5</td><td>${p.next_direction||'-'}</td></tr>
      ${auto}${unread}${err}`;
  }
}, 800);

// ── Jog logic ──────────────────────────────
const getSpeed = () => parseInt($('jog-speed').value) || 5;
let _jogAxis = null;  // currently held axis (prevents duplicate stop calls)

const jogStart = axis => {
  if (_jogAxis === axis) return;  // already jogging this axis
  _jogAxis = axis;
  const el = document.getElementById('jb-' + axis);
  if (el) el.classList.add('jogging');
  api('POST','/jog/start',{axis, speed:getSpeed()});
};
const jogStop = () => {
  if (!_jogAxis) return;  // nothing to stop
  const el = document.getElementById('jb-' + _jogAxis);
  if (el) el.classList.remove('jogging');
  _jogAxis = null;
  api('POST','/jog/stop');
};

// Attach events to all .jb buttons with data-axis
document.querySelectorAll('.jb[data-axis]').forEach(btn => {
  const ax = btn.dataset.axis;
  btn.addEventListener('mousedown',  e => { e.preventDefault(); jogStart(ax); });
  btn.addEventListener('touchstart', e => { e.preventDefault(); jogStart(ax); });
  btn.addEventListener('mouseup',    () => jogStop());
  btn.addEventListener('touchend',   () => jogStop());
  btn.addEventListener('mouseleave', () => { if(_jogAxis===ax) jogStop(); });
});

// Keyboard jog
const KEY_MAP = {
  'ArrowLeft':'X+', 'ArrowRight':'X-',
  'ArrowUp':'Y+',   'ArrowDown':'Y-',
  'z':'Z+', 'Z':'Z+',
  'x':'Z-', 'X':'Z-',
};
document.addEventListener('keydown', e => {
  if (e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA') return;
  if (e.repeat) return;
  const k = e.key;
  if (k==='q'||k==='Q') { api('POST','/gripper/grip'); return; }
  if (k==='w'||k==='W') { api('POST','/gripper/release'); return; }
  const ax = KEY_MAP[k];
  if (ax) { jogStart(ax); e.preventDefault(); }
});
document.addEventListener('keyup', e => {
  const ax = KEY_MAP[e.key];
  if (ax && _jogAxis===ax) { jogStop(); e.preventDefault(); }
});
window.addEventListener('blur', () => jogStop());

// Capture pose
const capturePose = async () => {
  const d = await api('GET','/pose');
  if(d && d.ok)
    $('pose-display').textContent = `X:${d.x}  Y:${d.y}  Z:${d.z} | Rx:${d.rx}  Ry:${d.ry}  Rz:${d.rz}`;
  else
    $('pose-display').textContent = 'fetch failed';
};

// Misc handlers
const doEstop = () => { if(confirm('E-STOP?')) api('POST','/estop'); };
const clearAlarm = async () => {
  const d = await api('POST','/clear-alarm');
  if(d&&d.ok) addLog('✓ Alarm cleared');
};
initZoneSelector();
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn, socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close()
    except Exception:
        ip = "localhost"
    print(f"\n  Dobot E6 Robot Server")
    print(f"  Open: http://{ip}:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
