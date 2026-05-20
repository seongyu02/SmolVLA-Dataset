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
DATA_SAVE_DIR   = "/media/billye6/새 볼륨/Dobot/SmolVLA"
DATA_DRIVE_ROOT = "/media/billye6/새 볼륨"   # 마운트 여부 판단 기준
from dobot_e6_controller import DobotE6Controller
from suction_gripper import SuctionGripper

_hik_available = False
try:
    from camera_viewer import HikRobotCamera
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
        bgr  = cv2.resize(bgr, (640, 480))
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
    "pick_section":   "A",
    "last_place_x":   None,
    "last_place_y":   None,
    "task_mode":      "zone_move",
    "zone_episode_idx": 0,
    "single_zone_target": 0,
    "single_zone_done": 0,
    "single_zone_id": None,
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
ZONE_INIT_POSE = (89.3715, -378.5400, 250.0000, INIT_SAFE_RX, INIT_SAFE_RY, INIT_SAFE_RZ)
ZONE_TARGET_Z = 120.0
ZONE_XY_OFFSET_MM = 8.0
ZONE4_XY_OFFSET_MM = 5.0
ZONE_EPISODES_PER_ZONE = 10
ZONE_ORDER = ["1", "5", "8", "9", "10"]
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
    _state["single_zone_target"] = 0
    _state["single_zone_done"] = 0
    _state["single_zone_id"] = None
    _state["last_auto_error"] = ""
    _state["last_auto_done"] = 0
    _state["last_auto_target"] = 0

_reset_zone_progress()

def _mid_pose(a, b):
    return tuple((float(x) + float(y)) / 2.0 for x, y in zip(a, b))

ZONE_POSES = {
    "1": (base.POS_5[0] - 20.0, base.POS_1[1], base.POS_1[2], base.POS_1[3], base.POS_1[4], base.POS_1[5]),
    "4": base.POS_4,
    "5": (base.POS_5[0] - 20.0, base.POS_5[1], base.POS_5[2], base.POS_5[3], base.POS_5[4], base.POS_5[5]),
    "8": base.POS_8,
    "9": base.POS_9,
    "10": _mid_pose(base.POS_6, base.POS_7),
}

CAMERA_MAPPING = {
    "OBS_IMAGE_1": "HIK_top",
    "OBS_IMAGE_2": "ZED_side",
}

IMAGE_SAVE_META = {
    "saved_size": [640, 480],
    "saved_format": "jpg",
    "crop_applied": False,
    "resize_to_512_applied": False,
    "description": "Raw camera frames are saved as 640x480 during data collection.",
}

FUTURE_LEROBOT_PREPROCESS_META = {
    "hik": {
        "raw_size": [640, 480],
        "recommended_crop_480x480_xyxy": [94, 0, 574, 480],
        "resize_size": [512, 512],
        "description": "Recommended HIK crop/resize for later LeRobot conversion.",
    },
    "zed": {
        "raw_size": [640, 480],
        "recommended_crop_480x480_xyxy": [80, 0, 560, 480],
        "resize_size": [512, 512],
        "description": "Recommended ZED crop/resize for later LeRobot conversion.",
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
# 카메라 그랩 루프 (MJPEG 버퍼 + 레코딩 numpy 버퍼 동시 갱신)
# ═══════════════════════════════════════════════════════════════════════════

def _hik_grab_loop():
    global _buf_hik_jpg, _buf_hik_np, _cam_hik_running
    cam = _state["camera_hik"]
    while _cam_hik_running and cam and cam.initialized:
        ret, frame = cam.get_frame()   # RGB
        if ret and frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _buf_lock:
                _buf_hik_jpg = enc.tobytes()
                _buf_hik_np  = bgr
            if _ros2_ok:
                _ros2.publish_hik(bgr)   # 캡처 직후 타임스탬프로 퍼블리시
        else:
            time.sleep(0.02)

def _zed_grab_loop():
    global _buf_zed_jpg, _buf_zed_np, _cam_zed_running
    cam = _state["camera_zed"]
    while _cam_zed_running and cam and cam.initialized:
        ret, frame = cam.get_frame()   # RGB
        if ret and frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with _buf_lock:
                _buf_zed_jpg = enc.tobytes()
                _buf_zed_np  = bgr
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
        # Keep auto counters until _on_finished(False) records the failed zone.
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
        # Keep auto counters until _on_finished(False) records the failed zone.
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
        # ros2 + ZED 동시 활성 시: sync callback 이 저장 처리 → legacy tick 건너뜀
        has_zed_now = bool(_state["camera_zed"] and _state["camera_zed"].initialized)
        if not (_ros2_ok and has_zed_now):
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

        # Save raw camera frames as 640x480.
        # Do not crop or resize to 512 here.
        # LeRobot conversion script will handle crop/resize later.
        hik_path = os.path.join(save_dir, "images", "hik", fname)
        hik_save = ensure_640x480_bgr(hik_np)
        cv2.imwrite(hik_path, hik_save)

        has_zed = bool(_state["camera_zed"] and _state["camera_zed"].initialized)
        if has_zed:
            zed_path = os.path.join(save_dir, "images", "zed", fname)
            zed_save = ensure_640x480_bgr(zed_np)
            cv2.imwrite(zed_path, zed_save)

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

class ZoneMoveWithoutSuctionWorker(threading.Thread):
    """초기자세 -> target zone 하강 완료까지만 recording하고, 복귀는 녹화하지 않는다."""
    def __init__(self, robot, zone_id: str, episode_in_zone: int, xy_offset_mm: float = ZONE_XY_OFFSET_MM):
        super().__init__(daemon=True)
        self.robot = robot
        self.zone_id = str(zone_id)
        self.episode_in_zone = int(episode_in_zone)
        self.xy_offset_mm = float(xy_offset_mm)
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
            self._move(ix, iy, iz, irx, iry, irz, velocity=25.0)
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
            x = bx + random.uniform(-self.xy_offset_mm, self.xy_offset_mm)
            y = by + random.uniform(-self.xy_offset_mm, self.xy_offset_mm)
            z = ZONE_TARGET_Z
            rx, ry, rz = base.get_descent_rpy(x, y)
            ok, msg = self.robot.check_ik_solution(x, y, z, rx, ry, rz)
            if ok:
                if attempt > 0:
                    self._log(f"IK target accepted after {attempt + 1} tries")
                return (x, y, z, rx, ry, rz)
            self._log(f"IK reject zone_{self.zone_id} try {attempt + 1}: {msg}")
        return None

    def run(self):
        try:
            if not self.robot or not self.robot.connected:
                self._log("Robot not connected")
                self.finished.emit(False)
                return
            if self.zone_id not in ZONE_POSES:
                self._log(f"Unknown zone_id: {self.zone_id}")
                self.finished.emit(False)
                return

            target_pose = self._sample_target_pose_with_ik()
            if target_pose is None:
                self._log(f"Failed: no IK-valid target for zone_{self.zone_id}")
                self.finished.emit(False)
                return

            ix, iy, iz, irx, iry, irz = ZONE_INIT_POSE
            self._log(f"1) Moving to initial pose for zone_{self.zone_id}...")
            if not self._move(ix, iy, iz, irx, iry, irz, velocity=25.0):
                self._log("Failed: initial pose")
                self.finished.emit(False)
                return

            self._log("2) Holding at initial pose 1s, then start recording...")
            for _ in range(10):
                if self._stop_requested:
                    self.finished.emit(False)
                    return
                time.sleep(0.1)
            self._events.append(("recording_start_at_initial", time.time()))
            self.recording_begin_at_initial.emit()
            if self._stop_requested or not _state.get("recording"):
                self._log("Recording did not start; aborting zone episode")
                self.finished.emit(False)
                return

            tx, ty, tz, trx, try_, trz = target_pose
            travel_z = random.uniform(base.Z_MOVE_MIN, base.Z_MOVE_MAX)
            self._log(
                f"3) Moving above zone_{self.zone_id}: "
                f"X={tx:.1f} Y={ty:.1f} Z={travel_z:.1f}"
            )
            if not self._move(tx, ty, travel_z, base.TRAVEL_RX, base.TRAVEL_RY, base.TRAVEL_RZ, velocity=24.0):
                self._log("Failed: move above target zone")
                self._save_or_discard_recording(False)
                self._return_to_initial_unrecorded()
                self.finished.emit(False)
                return

            self._log(f"4) Descending near zone_{self.zone_id}: Z={tz:.1f} (no suction)")
            if not self._move(tx, ty, base.DESCENT_MID_Z, base.TRAVEL_RX, base.TRAVEL_RY, base.TRAVEL_RZ, velocity=18.0):
                self._log("Failed: descend to mid Z")
                self._save_or_discard_recording(False)
                self._return_to_initial_unrecorded()
                self.finished.emit(False)
                return
            if not self._move(tx, ty, tz, trx, try_, trz, velocity=base.DESCENT_VELOCITY):
                self._log("Failed: descend to target pose")
                self._save_or_discard_recording(False)
                self._return_to_initial_unrecorded()
                self.finished.emit(False)
                return

            self._events.append(("target_zone_reached", time.time()))
            self.episode_vacuum_durations.emit(0.0, 0.0)
            self.episode_meta_ready.emit(self._build_episode_meta(target_pose, travel_z))
            self._log("5) Target reached. Stop/save recording before return.")
            self._save_or_discard_recording(True)

            self._return_to_initial_unrecorded()
            self._log(f"Zone move episode complete: zone_{self.zone_id} ({self.episode_in_zone + 1}/{ZONE_EPISODES_PER_ZONE})")
            self.finished.emit(True)
        except Exception as e:
            self._log(f"Zone worker error: {e}")
            self._save_or_discard_recording(False)
            self._return_to_initial_unrecorded()
            self.finished.emit(False)

# ═══════════════════════════════════════════════════════════════════════════
# Worker 콜백
# ═══════════════════════════════════════════════════════════════════════════

def _on_log(msg):    _log(msg)
def _on_vacuum(ph, pl): _state["vacuum_pick"] = ph; _state["vacuum_place"] = pl
def _on_rec_begin():  _start_recording()
def _on_episode_meta(meta): _state["episode_meta"] = meta

def _on_finished(success: bool):
    w = _state["worker"]
    zone_id = str(getattr(w, "zone_id", _state.get("current_zone_id")))
    episode_in_zone = getattr(w, "episode_in_zone", None)
    auto_target = _state["auto_target"]
    if not getattr(w, "_recording_saved_before_finished", False):
        _stop_and_save(success)
    if w and hasattr(w, 'place_x') and w.place_x is not None:
        _state["last_place_x"] = w.place_x
        _state["last_place_y"] = w.place_y
        _state["pick_section"] = "B" if _state["pick_section"] == "A" else "A"
        _log(f"Place ({w.place_x:.1f},{w.place_y:.1f}) / next: {_state['pick_section']}")

    if _state.get("single_zone_target", 0) > 0:
        zone_stats = _state["zone_stats"].setdefault(
            zone_id, {"success": 0, "fail": 0, "status": "pending"}
        )
        if not success:
            zone_stats["fail"] += 1
            zone_stats["status"] = "failed"
            ep_label = episode_in_zone + 1 if episode_in_zone is not None else "?"
            _state["last_auto_error"] = f"Failed at zone {zone_id}, episode {ep_label}"
            _log("⚠ 4구역 수집 중단됨 (실패/STOP)")
            _state["single_zone_target"] = 0
            return
        zone_stats["success"] += 1
        _state["single_zone_done"] += 1
        done = _state["single_zone_done"]
        target = _state["single_zone_target"]
        zone_stats["status"] = "done" if done >= target else "running"
        _log(f"Zone {zone_id} only collect {done}/{target}")
        if done >= target:
            _state["single_zone_target"] = 0
            _log(f"Zone {zone_id} only collect complete")
        else:
            threading.Timer(0.3, _run_step).start()
        return

    if auto_target > 0:
        zone_stats = _state["zone_stats"].setdefault(
            zone_id, {"success": 0, "fail": 0, "status": "pending"}
        )
        if not success:
            # STOP 또는 실패 → 자동 수집 중단, 카운트 증가 없음
            zone_stats["fail"] += 1
            zone_stats["status"] = "failed"
            ep_label = episode_in_zone + 1 if episode_in_zone is not None else "?"
            _state["last_auto_done"] = _state["auto_done"]
            _state["last_auto_target"] = auto_target
            _state["last_auto_error"] = f"Failed at zone {zone_id}, episode {ep_label}"
            _state["auto_target"] = 0
            _log("⚠ 자동 수집 중단됨 (실패/STOP)")
            return
        zone_stats["success"] += 1
        _state["auto_done"] += 1
        _state["last_auto_done"] = _state["auto_done"]
        _state["last_auto_target"] = auto_target
        zone_stats["status"] = "done" if zone_stats["success"] >= ZONE_EPISODES_PER_ZONE else "running"
        _log(f"Auto {_state['auto_done']}/{_state['auto_target']}")
        if _state["auto_done"] >= auto_target:
            _state["last_auto_done"] = _state["auto_done"]
            _state["last_auto_target"] = auto_target
            _state["last_auto_error"] = ""
            for zid in ZONE_ORDER:
                _state["zone_stats"].setdefault(zid, {"success": 0, "fail": 0, "status": "pending"})
                _state["zone_stats"][zid]["status"] = "done"
            _state["auto_target"] = 0
            _log("Auto collect complete")
        else:
            threading.Timer(0.3, _run_step).start()
    else:
        _log("Step complete")

def _run_step():
    robot   = _state["robot"]
    if not robot or not robot.connected:
        _log("Robot not connected"); return
    if _state["worker"] and _state["worker"].isRunning():
        _log("Worker already running"); return

    if _state.get("single_zone_target", 0) > 0:
        zone_id = str(_state.get("single_zone_id") or "4")
        ep_in_zone = int(_state.get("single_zone_done", 0))
        xy_offset_mm = ZONE4_XY_OFFSET_MM
    elif _state.get("auto_target", 0) > 0:
        seq_idx = min(_state["auto_done"] // ZONE_EPISODES_PER_ZONE, len(ZONE_ORDER) - 1)
        ep_in_zone = _state["auto_done"] % ZONE_EPISODES_PER_ZONE
        zone_id = ZONE_ORDER[seq_idx]
        xy_offset_mm = ZONE_XY_OFFSET_MM
    else:
        seq_idx = _state.get("zone_episode_idx", 0) % len(ZONE_ORDER)
        ep_in_zone = 0
        _state["zone_episode_idx"] = seq_idx + 1
        zone_id = ZONE_ORDER[seq_idx]
        xy_offset_mm = ZONE_XY_OFFSET_MM
    _state["current_zone_id"] = zone_id
    _state["current_zone_episode"] = ep_in_zone + 1
    zone_stats = _state["zone_stats"].setdefault(
        zone_id, {"success": 0, "fail": 0, "status": "pending"}
    )
    if zone_stats["success"] < ZONE_EPISODES_PER_ZONE:
        zone_stats["status"] = "running"

    worker = ZoneMoveWithoutSuctionWorker(
        robot, zone_id=zone_id, episode_in_zone=ep_in_zone, xy_offset_mm=xy_offset_mm
    )
    worker.log_signal.connect(_on_log)
    worker.finished.connect(_on_finished)
    worker.episode_vacuum_durations.connect(_on_vacuum)
    worker.episode_meta_ready.connect(_on_episode_meta)
    worker.recording_begin_at_initial.connect(_on_rec_begin)
    _state["worker"] = worker
    worker.start()

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
        "single_zone_target": _state["single_zone_target"],
        "single_zone_done": _state["single_zone_done"],
        "single_zone_id": _state["single_zone_id"],
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
    def _do():
        ok = robot.move_j(300, 0, 400, 180, 0, 0, coordinate_mode=0, use_waypoint=False)
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

@app.post("/pick-place/step")
def step():
    if _state["worker"] and _state["worker"].isRunning():
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    if not _state["robot"] or not _state["robot"].connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    _state["auto_target"] = 0
    threading.Thread(target=_run_step, daemon=True).start()
    return {"ok": True}

@app.post("/pick-place/auto")
def auto_collect(n: int = ZONE_TASK_TOTAL):
    if _state["worker"] and _state["worker"].isRunning():
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    if not _state["robot"] or not _state["robot"].connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    _state.update(auto_target=n, auto_done=0)
    _log(f"Zone dataset auto collect started: {n} episodes")
    threading.Thread(target=_run_step, daemon=True).start()
    return {"ok": True}

@app.post("/zone-dataset/step")
def zone_dataset_step():
    return step()

@app.post("/zone-dataset/start")
def zone_dataset_start(episodes_per_zone: int = ZONE_EPISODES_PER_ZONE):
    episodes_per_zone = max(1, int(episodes_per_zone))
    total = len(ZONE_ORDER) * episodes_per_zone
    if _state["worker"] and _state["worker"].isRunning():
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    if not _state["robot"] or not _state["robot"].connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    _reset_zone_progress()
    _state["auto_target"] = total
    _state["auto_done"] = 0
    _state["last_auto_target"] = total
    _state["last_auto_done"] = 0
    _log(f"Zone dataset auto collect started: {total} episodes")
    threading.Thread(target=_run_step, daemon=True).start()
    return {"ok": True}

@app.post("/zone4-section/start")
def zone4_section_start(n: int = 10):
    n = 10
    if _state["worker"] and _state["worker"].isRunning():
        return JSONResponse({"ok": False, "msg": "Already running"}, status_code=400)
    if not _state["robot"] or not _state["robot"].connected:
        return JSONResponse({"ok": False, "msg": "Not connected"}, status_code=400)
    _reset_zone_progress()
    _state["auto_target"] = 0
    _state["auto_done"] = 0
    _state["single_zone_id"] = "4"
    _state["single_zone_target"] = n
    _state["single_zone_done"] = 0
    _state["zone_stats"].setdefault("4", {"success": 0, "fail": 0, "status": "pending"})
    _state["zone_stats"]["4"]["status"] = "running"
    _log(f"Zone 4 section collect started: {n} episodes (XY ±{ZONE4_XY_OFFSET_MM}mm)")
    threading.Thread(target=_run_step, daemon=True).start()
    return {"ok": True}

@app.post("/pick-place/stop")
def stop_collect():
    if _state["worker"]: _state["worker"]._stop_requested = True
    _state["auto_target"] = 0
    _state["single_zone_target"] = 0
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
        cam = HikRobotCamera()
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
    <div class="sub">Zone Progress</div>
    <table id="zone-table">
      <thead>
        <tr><th>Zone</th><th>Success</th><th>Fail</th><th>Status</th></tr>
      </thead>
      <tbody id="zone-table-body"></tbody>
    </table>
    <div class="row">
      <button class="btn-g" onclick="api('POST','/zone-dataset/step')">▶ Zone Step</button>
      <input id="auto-n" type="number" value="10" min="1" style="max-width:55px">
      <button class="btn-g" onclick="autoCollect()">▶ Auto</button>
      <button class="btn-g" onclick="collectZone4Only()">▶ 4번 섹션</button>
      <button class="btn-y" onclick="api('POST','/pick-place/stop')">■ Stop</button>
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
  if((s.single_zone_target||0) > 0) {
    $('auto-bar').textContent =
      `Zone 4: ${s.single_zone_done}/${s.single_zone_target} | Current Zone: ${s.current_zone_id||'4'} | Episode: ${s.current_zone_episode||0}/${s.single_zone_target}`;
  } else if(s.auto_target > 0) {
    $('auto-bar').textContent =
      `Auto: ${s.auto_done}/${s.auto_target} | Current Zone: ${s.current_zone_id||'—'} | Episode: ${s.current_zone_episode||0}/${s.zone_episodes_per_zone}`;
  } else if((s.last_auto_target||0) > 0) {
    $('auto-bar').textContent = `Last Auto: ${s.last_auto_done}/${s.last_auto_target}`
      + (s.last_auto_error ? ` | Failed: ${s.last_auto_error}` : ' | Complete');
  } else if(s.worker_running) {
    $('auto-bar').textContent = 'Running…';
  } else {
    $('auto-bar').textContent = 'Ready';
  }
  const zoneBody = $('zone-table-body');
  if(zoneBody) {
    const order = [...(s.zone_order || [])];
    const stats = s.zone_stats || {};
    Object.keys(stats).forEach(zid => { if(!order.includes(zid)) order.push(zid); });
    zoneBody.innerHTML = order.map(zid => {
      const z = stats[zid] || {success:0, fail:0, status:'pending'};
      const status = z.status || 'pending';
      return `<tr><td>${zid}</td><td>${z.success||0}/${s.zone_episodes_per_zone}</td><td>${z.fail||0}</td><td class="zstat ${status}">${status}</td></tr>`;
    }).join('');
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
const autoCollect = () => {
  const n = Math.max(1, parseInt(($('auto-n')?.value || '10'), 10) || 10);
  return api('POST','/zone-dataset/start',{episodes_per_zone:n});
};
const collectZone4Only = () => api('POST','/zone4-section/start',{n:10});
const doEstop = () => { if(confirm('E-STOP?')) api('POST','/estop'); };
const clearAlarm = async () => {
  const d = await api('POST','/clear-alarm');
  if(d&&d.ok) addLog('✓ Alarm cleared');
};
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
