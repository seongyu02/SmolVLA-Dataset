#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dobot_camera.py — Dobot E6(SmolVLA) 수집용 HIK 카메라 설정의 단일 소유자.

## 왜 이 파일이 있는가

HIK 은 AOI 와 gamma 를 **카메라 펌웨어에 영구 저장**한다. 프로그램을 닫아도, 노드가
죽어도, USB 를 다시 꽂아도 값이 남는다. 그래서 같은 카메라를 쓰는 다른 프로그램이
설정을 남기면 **나중에 여는 쪽이 그 값을 그대로 물려받는다.**

교체 전 `camera_viewer.HikRobotCamera()`(220줄 버전)는 TriggerMode / ExposureAuto /
GainAuto 세 가지만 걸고 AOI·gamma 를 건드리지 않았다. 그래서 같은 카메라를 쓰는 xarm
쪽 `camera_publisher_node` 가 남긴 AOI `(448,130,1808,1808)` 을 상속해 **화각이 가로
70%(1808/2592) 로 잘린 채** 열렸다 — 2026-09-14 실측.

⚠️ 이 유형이 추적하기 어려운 이유: **코드가 한 줄도 바뀌지 않는다.**
   `camera_viewer.py` 는 2026-05-18 최초 커밋 이후 변경 0회였고 640 리사이즈도
   그때부터 그대로였다. 달라진 것은 카메라 안에 남은 값뿐이다. 그래서 "나는 건드린
   적이 없는데 화면만 이상하다" 가 된다.

그래서 이 파일은 **열 때마다 원하는 값을 전부 명시**한다. 카메라에 뭐가 남아 있든
우리 값으로 덮어쓰므로, xarm 을 돌린 직후에 열어도 항상 같은 화각이 나온다.

## 반대 방향은 자동이다 — 챙길 곳은 여기 하나다

xarm `camera_publisher_node` 는 기동할 때마다 자기 AOI 를 명시 설정한다
(`camera_publisher_node.py:180` 기본값, `collect.launch.py:85` 라벨 카메라).
따라서 여기서 전체 센서로 바꿔 놓아도 xarm 을 띄우면 스스로 1808×1808 로 복구한다.
**한쪽만 챙기면 되는 구조이고, 그 한쪽이 여기다.**

## 쓰는 법

    from dobot_camera import make_hik_camera
    cam = make_hik_camera()
    cam.init_camera()

진단(로봇 불필요, 카메라만 있으면 됨):

    python3 dobot_camera.py --list       연결된 카메라와 시리얼 출력 (열지 않음)
    python3 dobot_camera.py --preview    현재 설정으로 한 장 찍어 저장

⚠️ `robot_server.py` 가 카메라를 잡고 있으면 위 명령이 실패한다
   (`0x80000203` = MV_E_ACCESS_DENIED, 권한 문제가 아니라 이미 사용 중이라는 뜻).
   먼저 robot_server 를 내릴 것.
"""

import os
import sys
import time
from typing import List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# 설정 — 값마다 근거를 같이 둔다. 근거 없이 바꾸면 다음 사람이 되돌릴 수 없다.
# ═══════════════════════════════════════════════════════════════════════════

# 어느 물리 카메라인가.
#
# 🔴 아직 확정되지 않았다. HIK 이 2대 물려 있고(둘 다 MV-CE050-30UC) 교체 전 코드는
#    열거 인덱스 0번을 잡았는데, 열거 순서는 USB 재인식 순서에 따라 바뀔 수 있어
#    **어느 날은 다른 카메라가 잡힌다.** `python3 dobot_camera.py --list` 로 시리얼을
#    확인하고 Dobot 리그를 보는 쪽을 여기 적을 것.
#
#    None 이면 교체 전과 같이 열거 0번을 연다(화각·gamma 는 아래 값으로 교정되므로
#    화면을 보고 어느 카메라인지 판단할 수 있다).
#
# ⚠️ 시리얼을 지정하면 못 찾았을 때 **0번으로 폴백하지 않고 기동에 실패**한다.
#    엉뚱한 카메라를 조용히 잡느니 실패하는 편이 낫다는 설계다(camera_viewer 주석).
#
# 2026-09-14 실측. 두 카메라를 각각 열어 본 결과:
#      00DA8057731  열거 0번  → 모니터 쪽을 향함
#      00DA8057763  열거 1번  → xarm 책장 홀더(Science/Humanities/Liberal Arts)
#
# 🟢 2026-09-15 재실측 — **위 관측은 더 이상 맞지 않는다.** 그 사이 731 을 다시
#    조준했다. 두 대에서 동시에 프레임을 받아 나란히 확인한 결과:
#      00DA8057731  → Dobot 작업대 (로봇팔 + 체커보드 + 박스)   ← 이걸 쓴다
#      00DA8057763  → xarm 책장 홀더 (그대로)
#    아래 "둘 다 작업대를 보고 있지 않다" 도 이제 해당하지 않는다.
#
# 🔴 **둘 다 Dobot E6 작업대를 보고 있지 않다.** 과거 E6 수집 프레임
#    (/mnt/robotdata/Dobot/2CAM-Orange-init/1/images/hik/frame_000000.jpg)은
#    **Dobot 팔 + 체커보드 작업대 + 오렌지 박스**를 내려다보는 시점이다. 체커보드
#    리그는 지금도 남아 있다(현재 ZED 프레임에 보인다) — 카메라만 다른 데를 본다.
#    ⇒ **소프트웨어로 못 고친다. 카메라 하나를 작업대로 물리적으로 조준해야 한다.**
#
# 731 을 고른 근거(추론): 과거 이 코드는 pDeviceInfo[0] = 열거 0번을 잡았고 지금
#    0번이 731 이다. 열거 순서가 유지됐다면 과거에도 731 을 쓰고 있었다는 뜻이다.
#    ⚠️ 과거 데이터에 시리얼 기록이 없어(옛 camera_viewer 가 안 남김) **확정은
#       불가능하다.** 물리 조준을 마친 뒤 실제로 쓰는 카메라의 시리얼로 맞출 것.
#
# ⚠️ 731 은 xarm 이 작업뷰로 쓰는 것과 같은 물리 장치다(xarm AOI
#    (448,130,1808,1808)). 763 은 xarm 라벨용이다. HIK 이 2대뿐이라 어느 쪽을
#    쓰든 xarm 과 공유하게 되고, single owner 라 동시 사용은 불가능하다.
# 2026-09-15 731 로 전환. 이전 주석("731 은 벽")은 그 시점 관측이고, 그 뒤 카메라를
# 다시 조준해 지금은 **731 이 작업대(로봇팔·체커보드·박스)를 본다.**
# 763 은 xArm 책장 라벨 리그를 보고 있어 이 수집에는 쓰지 않는다.
# 실측 프레임으로 두 대를 나란히 확인한 결과다.
HIK_SERIAL: Optional[str] = "00DA8057731"   # 작업대를 보는 쪽

# 하드웨어 AOI = (offset_x, offset_y, width, height).
#
# 센서 전체는 2592×1944(MV-CE050-30UC, 5MP). E6 원본 파이프라인이
# `640×480 → 320×240 → crop[16:240,55:279] → 224` 로 **소프트웨어 crop 을 전제**하므로
# 카메라 단에서 화각을 줄이면 안 된다. 전체를 받아야 예전과 같은 그림이 나온다.
HIK_AOI: Optional[Tuple[int, int, int, int]] = (0, 0, 2592, 1944)

# gamma. None 이면 camera_viewer 가 GammaEnable=False 로 **명시적으로 끈다**
# (단순히 "안 건드림"이 아니다). xarm 이 남긴 0.7 을 제거하는 것이 목적이다.
HIK_GAMMA: Optional[float] = None

# get_frame() 출력 크기. None 이면 센서 원본 해상도(2592×1944) 그대로.
#
# 2026-09-14: None(센서 원본 2592×1944). 사용자 요청.
#
# 🔴 처음 None 으로 켰을 때 **robot_server 가 멈췄다.** 그대로 다시 켜면 재발한다.
#
#    실측 증상: 프로세스가 STAT=D(uninterruptible sleep)로 I/O 에 블록돼
#    포트 8000 이 응답 불가, load average 18.33, Ctrl-C 도 안 먹어 강제 종료해야 했다.
#
#    원인: _hik_grab_loop 이 프레임마다 cv2.imencode(실측 44.5ms) 를 돌리고
#    이어서 _ros2.publish_hik(bgr) 로 ROS2 에 그대로 발행하는데, 2592×1944 는
#    프레임당 14.4MB 라 16Hz 면 **약 230MB/s** 가 DDS 로 나간다. 여기서 막혔다.
#
#    ⚠️ 카메라 자체는 2592×1944 에서도 16.04fps 가 나온다(격리 측정). 못 버틴 것은
#       카메라가 아니라 **그 뒤의 인코딩·발행 경로**였다.
#
#    ✅ 그래서 robot_server._hik_grab_loop 을 고쳤다(2026-09-14):
#         화면(MJPEG)  원본 그대로, 단 _HIK_MJPEG_HZ(기본 8Hz)로 주기 제한
#         저장·ROS2    ensure_640x480_bgr 축소본 → DDS 부하 230MB/s 제거
#       저장 결과는 이전과 동일하다(_record_tick 이 어차피 640×480 으로 줄인다).
#
#    ⚠️ 이 값을 None 으로 두려면 **그 수정이 반드시 함께 있어야 한다.**
#       robot_server.py 를 되돌리면 이 값도 (640,480) 으로 되돌릴 것.
#
#    ⚠️ 화각은 이 값과 무관하다 — 화각은 HIK_AOI 가 정하고 이미 센서 전체다.
#       640×480 은 같은 화각을 축소해 담은 것이라 보이는 범위는 동일하다.
#
# ⚠️ 이것이 바꾸는 것과 바꾸지 않는 것을 구분할 것:
#      화각      안 바뀐다. 화각은 위 HIK_AOI 가 정하고 이미 센서 전체다.
#                640×480 도 같은 화각을 축소해 담은 것이라, 끄면 선명해질 뿐
#                더 넓게 보이지는 않는다.
#      MJPEG     원본 해상도로 인코딩된다 → 화면이 선명해진다.
#      저장 이미지 안 바뀐다. robot_server._record_tick 이 ensure_640x480_bgr() 로
#                저장 직전에 다시 640×480(INTER_AREA)으로 줄인다. 원본으로
#                저장하려면 그 함수를 따로 고쳐야 한다 — 여기서는 안 건드렸다.
#
# ⚠️ 비용: _hik_grab_loop 이 매 프레임 cv2.imencode 를 돌리는데 픽셀이 약 16배가
#    된다(640×480 → 2592×1944). ROS2 publish 도 프레임당 약 0.9MB → 5.0MB 로
#    커진다. fps 가 떨어지면 이 값을 (640,480) 으로 되돌릴 것.
#
# ⚠️ `hikrobot_calibration_20260126_143821.txt` 의 내부 파라미터는 640×480 기준이다.
#    `calibration_file=` 을 넘기는 호출부(pick_place_gui*.py, merge.py)는 이 값과
#    스케일이 어긋난다. robot_server 는 캘리브레이션을 로드하지 않아 무관하다.
HIK_OUTPUT_SIZE: Optional[Tuple[int, int]] = None


def make_hik_camera(calibration_file: Optional[str] = None):
    """robot_server 등이 쓰는 진입점. 설정을 전부 명시한 HikRobotCamera 를 만든다.

    ⚠️ 객체를 만들기만 한다. 실제 연결은 호출부가 `init_camera()` 를 불러야 일어나고,
       AOI/gamma 도 그때 적용된다(교체 전 코드와 같은 수명).
    """
    from camera_viewer import HikRobotCamera

    return HikRobotCamera(
        calibration_file=calibration_file,
        serial=HIK_SERIAL,
        aoi=HIK_AOI,
        gamma=HIK_GAMMA,
        output_size=HIK_OUTPUT_SIZE,
    )


def list_cameras() -> List[str]:
    """연결된 HIK 카메라의 시리얼을 반환한다. **카메라를 열지 않는다**(열거만).

    열지 않으므로 다른 프로그램이 카메라를 쓰고 있어도 목록은 나온다.
    """
    import camera_viewer as _cv

    if getattr(_cv, "MvCamera", None) is None:
        print("HIK SDK 를 찾지 못했다 (MvImport import 실패).")
        return []

    MvCamera = _cv.MvCamera
    ret = MvCamera.MV_CC_Initialize()
    if ret != 0:
        print(f"SDK 초기화 실패 (0x{ret:x})")
        return []

    serials: List[str] = []
    try:
        device_list = _cv.MV_CC_DEVICE_INFO_LIST()
        tlayer = _cv.MV_GIGE_DEVICE | _cv.MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayer, device_list)
        if ret != 0:
            print(f"카메라 열거 실패 (0x{ret:x})")
            return []

        for i in range(device_list.nDeviceNum):
            info = _cv.cast(
                device_list.pDeviceInfo[i], _cv.POINTER(_cv.MV_CC_DEVICE_INFO)
            ).contents
            if info.nTLayerType != _cv.MV_USB_DEVICE:
                continue
            sn = "".join(
                chr(c) for c in info.SpecialInfo.stUsb3VInfo.chSerialNumber if c
            ).strip()
            model = "".join(
                chr(c) for c in info.SpecialInfo.stUsb3VInfo.chModelName if c
            ).strip()
            serials.append(sn)
            mark = "  <- HIK_SERIAL 로 지정됨" if sn == HIK_SERIAL else ""
            print(f"  [{i}] serial={sn}  model={model}{mark}")
    finally:
        MvCamera.MV_CC_Finalize()

    if not serials:
        print("  (USB HIK 카메라 없음)")
    return serials


def preview(out_path: Optional[str] = None, warmup: int = 8) -> Optional[str]:
    """현재 설정 그대로 카메라를 열어 한 장 저장한다. 설정 검증용.

    ⚠️ auto exposure 가 안정될 때까지 앞 프레임 몇 장은 버린다. 첫 프레임으로
       밝기를 판단하면 실제보다 어둡거나 밝게 보인다.
    """
    import cv2

    cam = make_hik_camera()
    if not cam.init_camera():
        print("카메라 열기 실패 — robot_server 가 잡고 있는지 확인할 것.")
        return None

    try:
        img = None
        for _ in range(max(1, warmup)):
            ok, frame = cam.get_frame()
            if ok and frame is not None:
                img = frame
            time.sleep(0.05)
        if img is None:
            print("프레임을 받지 못했다.")
            return None

        if out_path is None:
            out_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"preview_{time.strftime('%Y%m%d_%H%M%S')}.jpg",
            )
        cv2.imwrite(out_path, img)
        h, w = img.shape[:2]
        print(f"저장: {out_path}  ({w}x{h})")
        print(f"  AOI={HIK_AOI}  gamma={HIK_GAMMA}  output_size={HIK_OUTPUT_SIZE}")
        return out_path
    finally:
        cam.cleanup()


def _main(argv: List[str]) -> int:
    if "--list" in argv:
        print(f"HIK_SERIAL = {HIK_SERIAL!r}")
        list_cameras()
        return 0
    if "--preview" in argv:
        return 0 if preview() else 1
    print(__doc__.strip().splitlines()[0])
    print()
    print("  python3 dobot_camera.py --list       연결된 카메라와 시리얼")
    print("  python3 dobot_camera.py --preview    현재 설정으로 한 장 저장")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
