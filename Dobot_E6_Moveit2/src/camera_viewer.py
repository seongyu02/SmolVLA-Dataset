#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Viewer Module for HIKRobot Camera

⚠️ **사본** (2026-09-14). 출처 dobot-xarm-datacollect/UFactory/src/camera_viewer.py
   sha256(앞16) 0fd634b1a6b242cf

   교체 이유: 원래 220줄 버전에는 serial·AOI·gamma·output_size 파라미터가 없어
   (1) 어느 카메라가 열릴지 정할 수 없고 (2) HIK 이 펌웨어에 영구 저장하는 AOI 를
   물려받아 화각이 잘리며 (3) get_frame 이 무조건 640×480 으로 축소했다.

   🔴 출처가 갱신돼도 이 사본은 따라오지 않는다. 이상하면 두 파일을 대조할 것.
      심볼릭 링크로 묶지 말 것(한쪽 수정이 다른 쪽을 조용히 바꾼다).

   공개 API 는 교체 전과 동일: HikRobotCamera / load_calibration / init_camera /
   get_frame / cleanup, .initialized. 인자를 생략하면 동작도 종전과 같다.
"""

import sys
import os
import cv2
import numpy as np
from ctypes import *
import time
from typing import Optional, Tuple

# Windows 콘솔 UTF-8 인코딩 설정
if sys.platform == 'win32':
    try:
        import io
        if not isinstance(sys.stdout, io.TextIOWrapper) or (hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() != 'utf-8'):
            if hasattr(sys.stdout, 'buffer') and not sys.stdout.buffer.closed:
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'buffer') and not sys.stderr.buffer.closed:
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

try:
    from MvImport.MvCameraControl_class import *
except ImportError:
    print("Error: HIKRobot SDK not found. Camera functionality will be disabled.")
    MvCamera = None


class HikRobotCamera:
    """HIKRobot Camera Controller"""
    
    def __init__(
        self,
        calibration_file: Optional[str] = None,
        frame_rate_hz: Optional[float] = None,
        exposure_time_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        auto_exposure: Optional[bool] = None,
        auto_gain: Optional[bool] = None,
        output_size: Optional[Tuple[int, int]] = (640, 480),
        aoi: Optional[Tuple[int, int, int, int]] = None,
        gamma: Optional[float] = None,
        auto_exposure_upper_limit_us: Optional[float] = None,
        auto_gain_upper_limit_db: Optional[float] = None,
        serial: Optional[str] = None,
    ):
        """
        Initialize HIKRobot camera

        Args:
            calibration_file: Path to camera calibration file (.npz)
            frame_rate_hz: Hardware acquisition rate. None preserves the old setting.
            exposure_time_us: Fixed exposure in microseconds. Supplying a value
                disables continuous auto exposure.
            gain_db: Fixed gain in dB. Supplying a value disables continuous auto gain.
            auto_exposure: True=continuous, False=off. None preserves the old
                behavior (continuous) unless exposure_time_us is supplied.
            auto_gain: True=continuous, False=off. None preserves the old
                behavior (continuous) unless gain_db is supplied.
            output_size: (width, height) to resize get_frame() output to. Default
                (640, 480) preserves the original behavior for existing callers.
                Pass None to skip the resize entirely and get the native sensor
                resolution instead (2026-07-23, added for xarm_vla_collector —
                the old hardcoded resize to 640x480 threw away most of the
                camera's real detail before any downstream crop ever ran).
            serial: Bind to a specific camera by serial number. None keeps the old
                behavior (first enumerated device) so existing callers are unaffected.
                ⚠️ With two cameras attached you MUST pass this — enumeration order
                is not guaranteed and the views would swap silently on reboot.
            aoi: (offset_x, offset_y, width, height) hardware Area-of-Interest —
                the sensor only reads out and transfers this region, instead of
                get_frame() returning the full native frame for software cropping
                downstream. None (default) preserves existing behavior for all
                other callers (2026-07-27, added for xarm_vla_collector to cut
                HIK CPU load — verified empirically against the equivalent
                software crop: mean/max pixel diff 0.5/7 out of 255 after the
                same 224x224 resize, i.e. within normal frame-to-frame sensor
                noise, not a systematic shift).
                ⚠️ AOI is persistent camera firmware state across reconnects —
                init_camera() always explicitly sets offset=(0,0) first, then
                width/height, then the final offset, so a leftover AOI from a
                previous run/process can never linger silently.
                ⚠️ offset_x/width must be multiples of 16 and offset_y/height
                must be multiples of 2 (this sensor's GenICam increments) — the
                caller is responsible for passing already-valid values; this
                class does not round anything for you.
            gamma: In-camera gamma correction (2026-07-29, added for
                xarm_vla_collector). None (default) explicitly DISABLES gamma,
                which is what every existing caller already had — but stating it
                explicitly matters because gamma is persistent firmware state
                like AOI, so a leftover value from another process would
                otherwise silently change this camera's appearance.
                A value < 1.0 lifts shadows; measured on this rig at
                exposure=20000us/gain=15dB, the fraction of 224x224 pixels
                crushed to <=5 went 23.06% -> 0.02% at gamma=0.40 while mean
                brightness went 29.1 -> 99.2, with clipping staying at 0.01%
                and the rate staying at 16Hz. Unlike raising exposure this
                costs no frame rate and adds no motion blur.
                Valid range on this sensor: 0.0 ~ 4.0.
            auto_exposure_upper_limit_us: Cap for continuous auto exposure, in us
                (2026-07-29). ONLY applied when auto exposure is active. Without a
                cap, a dark scene stretches exposure past the frame period
                (16Hz -> 62500us) and the rate collapses -- that is exactly why
                auto exposure was disabled on 2026-07-23 (~10Hz observed).
                Measured on this rig: 40000us keeps 16.04Hz.
                ⚠️ Like AOI and gamma this is persistent camera firmware state, so a
                value set here can linger for other processes using this class.
            auto_gain_upper_limit_db: Cap for continuous auto gain, in dB. This
                sensor's max is 15.006dB (probed).
        """
        self.camera = None
        self.deviceList = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_file = calibration_file
        self.frame_rate_hz = frame_rate_hz
        self.exposure_time_us = exposure_time_us
        self.gain_db = gain_db
        self.auto_exposure = auto_exposure
        self.auto_gain = auto_gain
        self.output_size = output_size
        self.serial = serial
        self.aoi = aoi
        self.gamma = gamma
        self.auto_exposure_upper_limit_us = auto_exposure_upper_limit_us
        self.auto_gain_upper_limit_db = auto_gain_upper_limit_db
        self.initialized = False
        # 2448×2048 RGB까지 수용하는 SDK grab 버퍼. 기존 코드는 프레임마다 약
        # 15MB ctypes 배열을 새로 만들었는데, 고정 버퍼를 재사용하면 화각/픽셀
        # 파이프라인을 바꾸지 않고도 할당 비용을 없앨 수 있다.
        self._frame_buffer_size = 2448 * 2048 * 3
        self._frame_buffer = None
        
    def load_calibration(self):
        """Load camera calibration"""
        if self.calibration_file and os.path.exists(self.calibration_file):
            try:
                data = np.load(self.calibration_file)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                print(f"Calibration loaded: {self.calibration_file}")
                return True
            except Exception as e:
                print(f"Failed to load calibration: {e}")
                return False
        return False
    
    def init_camera(self) -> bool:
        """
        Initialize HIKRobot camera
        
        Returns:
            True if successful
        """
        if MvCamera is None:
            print("HIKRobot SDK not available")
            return False
            
        try:
            # SDK 초기화
            ret = MvCamera.MV_CC_Initialize()
            if ret != 0:
                print(f"SDK initialization failed (0x{ret:x})")
                return False
            
            # 카메라 검색
            self.deviceList = MV_CC_DEVICE_INFO_LIST()
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
            
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, self.deviceList)
            if ret != 0:
                print(f"Camera enumeration failed (0x{ret:x})")
                return False
            
            if self.deviceList.nDeviceNum == 0:
                print("No camera found")
                return False
            
            print(f"Camera found: {self.deviceList.nDeviceNum}")

            # 🔴 시리얼로 카메라를 고정한다 (2026-08-05, HIK 2대 구성).
            # 이전엔 `pDeviceInfo[0]` 하드코딩이었다. 카메라가 2대가 되면
            #   ① 두 노드가 같은 카메라를 잡으려 충돌하고
            #   ② **열거 순서는 보장되지 않아** 재부팅 때 작업뷰와 라벨뷰가 뒤바뀐다.
            # ②는 에러 없이 조용히 일어나고, 데이터가 뒤섞인 뒤에야 알게 된다.
            idx = 0
            serial = self.serial
            if serial:
                found = None
                avail = []
                for i in range(self.deviceList.nDeviceNum):
                    di = cast(self.deviceList.pDeviceInfo[i],
                              POINTER(MV_CC_DEVICE_INFO)).contents
                    if di.nTLayerType != MV_USB_DEVICE:
                        continue
                    sn = "".join(
                        chr(c) for c in di.SpecialInfo.stUsb3VInfo.chSerialNumber if c
                    ).strip()
                    avail.append(sn)
                    if sn == serial:
                        found = i
                if found is None:
                    # ⚠️ 인덱스 0으로 폴백하지 않는다 — 엉뚱한 카메라를 조용히 잡느니
                    # 기동에 실패하는 편이 낫다.
                    print(f"Camera serial {serial!r} not found. available={avail}")
                    return False
                idx = found
                print(f"Camera selected by serial {serial} (index {idx})")

            # 카메라 열기
            self.camera = MvCamera()
            stDeviceInfo = cast(self.deviceList.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents
            
            ret = self.camera.MV_CC_CreateHandle(stDeviceInfo)
            if ret != 0:
                print(f"Camera handle creation failed (0x{ret:x})")
                return False
            
            ret = self.camera.MV_CC_OpenDevice()
            if ret != 0:
                print(f"Camera open failed (0x{ret:x})")
                return False
            
            # 카메라 설정
            self.camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

            if self.aoi is not None:
                offset_x, offset_y, width, height = self.aoi
                # offset을 먼저 0으로 낮춰 여유를 만든 뒤 width/height를 줄이고, 그다음
                # 최종 offset을 적용한다 — 순서를 지키지 않으면 offset+width/height가
                # 이전 상태(또는 풀 네이티브)의 sensor 경계를 넘어 SetIntValue가 거부됨
                # (실측 확인됨). AOI는 펌웨어에 영구 저장되므로 매번 명시적으로 전부
                # 설정해 이전 실행의 잔여 상태가 조용히 남지 않도록 한다.
                r0 = self.camera.MV_CC_SetIntValue("OffsetX", 0)
                r1 = self.camera.MV_CC_SetIntValue("OffsetY", 0)
                r2 = self.camera.MV_CC_SetIntValue("Width", int(width))
                r3 = self.camera.MV_CC_SetIntValue("Height", int(height))
                r4 = self.camera.MV_CC_SetIntValue("OffsetX", int(offset_x))
                r5 = self.camera.MV_CC_SetIntValue("OffsetY", int(offset_y))
                print(
                    f"HIK AOI: offset=({offset_x},{offset_y}) size=({width}x{height}) "
                    f"rets=0x{r0:x},0x{r1:x},0x{r2:x},0x{r3:x},0x{r4:x},0x{r5:x}"
                )
                if any(r != 0 for r in (r2, r3, r4, r5)):
                    print("Warning: HIK AOI was not fully applied — check width/height/offset increments")

            if self.frame_rate_hz is not None:
                ret_enable = self.camera.MV_CC_SetBoolValue(
                    "AcquisitionFrameRateEnable", True
                )
                ret_fps = self.camera.MV_CC_SetFloatValue(
                    "AcquisitionFrameRate", float(self.frame_rate_hz)
                )
                print(
                    "HIK frame rate: "
                    f"requested={self.frame_rate_hz:.3f}Hz "
                    f"enable_ret=0x{ret_enable:x} set_ret=0x{ret_fps:x}"
                )
                if ret_enable != 0 or ret_fps != 0:
                    print("Warning: HIK hardware frame-rate setting was not fully applied")

            # 옵션을 생략한 기존 호출은 종전과 같이 Continuous를 사용한다.
            use_auto_exposure = (
                self.auto_exposure
                if self.auto_exposure is not None
                else self.exposure_time_us is None
            )
            use_auto_gain = (
                self.auto_gain
                if self.auto_gain is not None
                else self.gain_db is None
            )

            try:
                ret_auto_exp = self.camera.MV_CC_SetEnumValue(
                    "ExposureAuto", 2 if use_auto_exposure else 0
                )
                if not use_auto_exposure and self.exposure_time_us is not None:
                    ret_exp = self.camera.MV_CC_SetFloatValue(
                        "ExposureTime", float(self.exposure_time_us)
                    )
                    print(
                        "HIK exposure: "
                        f"auto=False requested={self.exposure_time_us:.1f}us "
                        f"auto_ret=0x{ret_auto_exp:x} set_ret=0x{ret_exp:x}"
                    )
                else:
                    print(
                        "HIK exposure: "
                        f"auto={use_auto_exposure} auto_ret=0x{ret_auto_exp:x}"
                    )
                # auto exposure를 쓸 때는 **상한을 반드시 걸어야** 한다.
                # 상한이 없으면 어두운 장면에서 노출이 프레임 주기(16Hz → 62500us)를
                # 넘겨 fps가 무너진다(2026-07-23에 ~10Hz로 떨어져서 auto를 아예 껐던
                # 원인이 이것이었다). 40000us가 실측 16.04Hz 유지 한계선이다.
                if use_auto_exposure and self.auto_exposure_upper_limit_us is not None:
                    ret_lim = self.camera.MV_CC_SetIntValue(
                        "AutoExposureTimeUpperLimit",
                        int(self.auto_exposure_upper_limit_us),
                    )
                    print(
                        "HIK auto exposure upper limit: "
                        f"{int(self.auto_exposure_upper_limit_us)}us "
                        f"set_ret=0x{ret_lim:x}"
                    )
                    if ret_lim != 0:
                        print("Warning: AutoExposureTimeUpperLimit 설정 거부 — "
                              "어두운 장면에서 fps가 떨어질 수 있음")
            except Exception as exc:
                print(f"Warning: HIK exposure setting failed: {exc}")

            try:
                ret_auto_gain = self.camera.MV_CC_SetEnumValue(
                    "GainAuto", 2 if use_auto_gain else 0
                )
                if not use_auto_gain and self.gain_db is not None:
                    ret_gain = self.camera.MV_CC_SetFloatValue(
                        "Gain", float(self.gain_db)
                    )
                    print(
                        "HIK gain: "
                        f"auto=False requested={self.gain_db:.2f}dB "
                        f"auto_ret=0x{ret_auto_gain:x} set_ret=0x{ret_gain:x}"
                    )
                else:
                    print(
                        f"HIK gain: auto={use_auto_gain} auto_ret=0x{ret_auto_gain:x}"
                    )
                if use_auto_gain and self.auto_gain_upper_limit_db is not None:
                    ret_glim = self.camera.MV_CC_SetFloatValue(
                        "AutoGainUpperLimit", float(self.auto_gain_upper_limit_db)
                    )
                    print(
                        "HIK auto gain upper limit: "
                        f"{self.auto_gain_upper_limit_db:.3f}dB set_ret=0x{ret_glim:x}"
                    )
            except Exception as exc:
                print(f"Warning: HIK gain setting failed: {exc}")

            # Gamma는 AOI와 같은 persistent firmware state이므로, 켜든 끄든 항상
            # 명시적으로 설정한다 — 다른 프로세스가 남긴 값이 조용히 이어져
            # 화면 밝기가 달라지는 일이 없도록.
            try:
                if self.gamma is None:
                    ret_en = self.camera.MV_CC_SetBoolValue("GammaEnable", False)
                    print(f"HIK gamma: disabled set_ret=0x{ret_en:x}")
                else:
                    # GammaSelector 1=User(Gamma 값 반영), 2=sRGB(고정 커브)
                    ret_sel = self.camera.MV_CC_SetEnumValue("GammaSelector", 1)
                    ret_en = self.camera.MV_CC_SetBoolValue("GammaEnable", True)
                    ret_val = self.camera.MV_CC_SetFloatValue(
                        "Gamma", float(self.gamma)
                    )
                    print(
                        "HIK gamma: "
                        f"enabled requested={self.gamma:.3f} "
                        f"selector_ret=0x{ret_sel:x} enable_ret=0x{ret_en:x} "
                        f"set_ret=0x{ret_val:x}"
                    )
                    if ret_sel != 0 or ret_en != 0 or ret_val != 0:
                        print(
                            "Warning: HIK gamma partially rejected — 밝기가 "
                            "의도와 다를 수 있음"
                        )
            except Exception as exc:
                print(f"Warning: HIK gamma setting failed: {exc}")

            self._frame_buffer = (c_ubyte * self._frame_buffer_size)()
            
            # 스트리밍 시작
            ret = self.camera.MV_CC_StartGrabbing()
            if ret != 0:
                print(f"Streaming start failed (0x{ret:x})")
                return False
            
            # 캘리브레이션 로드
            self.load_calibration()
            
            self.initialized = True
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get frame from camera
        
        Returns:
            (success, frame) tuple
        """
        if not self.initialized or self.camera is None:
            return False, None
            
        try:
            if self._frame_buffer is None:
                self._frame_buffer = (c_ubyte * self._frame_buffer_size)()
            pData = self._frame_buffer
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
            
            ret = self.camera.MV_CC_GetOneFrameTimeout(
                pData, self._frame_buffer_size, stFrameInfo, 1000
            )
            
            if ret == 0:
                image_data = np.frombuffer(pData, dtype=np.uint8, count=stFrameInfo.nFrameLen)
                
                # Bayer 형식 변환
                if stFrameInfo.enPixelType == PixelType_Gvsp_BayerRG8:
                    image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
                elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerGR8:
                    image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    image = cv2.cvtColor(image, cv2.COLOR_BayerGB2BGR)  # BayerGR2BGR 대신 BayerGB2BGR 사용
                elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerGB8:
                    image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    image = cv2.cvtColor(image, cv2.COLOR_BayerGB2BGR)
                elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerBG8:
                    image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    image = cv2.cvtColor(image, cv2.COLOR_BayerBG2BGR)
                else:
                    if len(image_data) == stFrameInfo.nHeight * stFrameInfo.nWidth:
                        image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    else:
                        image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, -1))
                
                # output_size로 리사이즈 (기본 640x480, None이면 원본 센서 해상도 그대로)
                if self.output_size is not None:
                    image = cv2.resize(image, self.output_size)

                # 캘리브레이션 보정 적용
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
                
                # BGR → RGB 변환
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                return True, image
            else:
                return False, None
                
        except Exception as e:
            return False, None
    
    def cleanup(self):
        """Cleanup camera resources"""
        if self.camera:
            try:
                self.camera.MV_CC_StopGrabbing()
                self.camera.MV_CC_CloseDevice()
                self.camera.MV_CC_DestroyHandle()
            except:
                pass
        
        try:
            if MvCamera:
                MvCamera.MV_CC_Finalize()
        except:
            pass
        
        self.initialized = False
        self._frame_buffer = None
