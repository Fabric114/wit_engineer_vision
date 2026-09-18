"""Minimal Hikrobot (MVS / MvImport) camera driver.

接口与 daheng_camera.DahengCamera 保持一致，供 camera_node 无差别调用。
底层调用海康 MVS Python SDK（src/camera_pkg/MvImport），抓取原始 Bayer8
数据后用 OpenCV 转成 BGR，转换约定与 daheng 驱动一致。
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# 加载海康 SDK 前先确保动态库路径可用（MvImport 依赖 MVCAM_COMMON_RUNENV）
os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")
_MV_IMPORT_DIR = str(Path(__file__).resolve().parents[1] / "MvImport")
if _MV_IMPORT_DIR not in sys.path:
    sys.path.append(_MV_IMPORT_DIR)

from MvCameraControl_class import *  # noqa: E402,F401,F403

logger = logging.getLogger("HikCamera")

# 相机上报的 Bayer 像素格式 -> OpenCV 转换码。
# OpenCV 两字母 Bayer 常量按模式的第二行/第二列命名，与相机像素格式的
# 左上角命名正好互为相反排列，故此处与 daheng 驱动采用同一套映射。
BAYER_PIXELTYPE_TO_CV = {
    PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerBG2BGR,
    PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerRG2BGR,
    PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGB2BGR,
    PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGR2BGR,
}

# config 中 bayer_mode (0=RG,1=BG,2=GR,3=GB) -> 期望的相机 Bayer8 像素格式
BAYER_MODE_TO_PIXELTYPE = (
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerGB8,
)

TRIGGER_SOURCE_MAP = {
    "software": MV_TRIGGER_SOURCE_SOFTWARE,
    "line0": MV_TRIGGER_SOURCE_LINE0,
    "line1": MV_TRIGGER_SOURCE_LINE1,
    "line2": MV_TRIGGER_SOURCE_LINE2,
    "line3": MV_TRIGGER_SOURCE_LINE3,
}

WB_MODE_MAP = {
    "off": MV_BALANCEWHITE_AUTO_OFF,
    "once": MV_BALANCEWHITE_AUTO_ONCE,
    "continuous": MV_BALANCEWHITE_AUTO_CONTINUOUS,
}

# BalanceRatioSelector 枚举: 0=Red 1=Green 2=Blue
BALANCE_SELECTOR = {"red": 0, "green": 1, "blue": 2}

GAIN_MIN_DB = 0.0
GAIN_MAX_DB = 24.0

_SDK_INIT_LOCK = threading.Lock()
_SDK_INIT_COUNT = 0


def _sdk_initialize() -> None:
    global _SDK_INIT_COUNT
    with _SDK_INIT_LOCK:
        if _SDK_INIT_COUNT == 0:
            MvCamera.MV_CC_Initialize()
        _SDK_INIT_COUNT += 1


def _sdk_finalize() -> None:
    global _SDK_INIT_COUNT
    with _SDK_INIT_LOCK:
        if _SDK_INIT_COUNT > 0:
            _SDK_INIT_COUNT -= 1
            if _SDK_INIT_COUNT == 0:
                MvCamera.MV_CC_Finalize()


def _decode_c_string(buf) -> str:
    return "".join(chr(c) for c in buf if c != 0).strip()


class HikCamera:
    def __init__(
        self,
        sn: str,
        width: int = 960,
        height: int = 720,
        binning: int = 1,
        exposure_us: float = 10000.0,
        gain: float = 19.0,
        bayer_mode: int = 0,
        frame_rate: float = 60.0,
        trigger_enable: bool = False,
        trigger_source: str = "software",
        wb_mode: str = "once",
        wb_r: float = 1.0,
        wb_g: float = 1.0,
        wb_b: float = 1.0,
    ):
        self.device_sn = str(sn).strip()
        self.width = int(width)
        self.height = int(height)
        self.binning = max(1, int(binning))
        self.exposure_us = float(exposure_us)
        self.gain = float(gain)
        self.bayer_mode = int(bayer_mode) % len(BAYER_MODE_TO_PIXELTYPE)
        self.frame_rate = float(frame_rate)
        self.trigger_enable = bool(trigger_enable)
        self.trigger_source = str(trigger_source).strip().lower()
        self.wb_mode = str(wb_mode).strip().lower()
        self.wb_r = float(wb_r)
        self.wb_g = float(wb_g)
        self.wb_b = float(wb_b)

        self.camera: Optional[MvCamera] = None
        self._sdk_ready = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, float]] = None
        self._connected = False

    def open(self) -> bool:
        try:
            _sdk_initialize()
            self._sdk_ready = True

            dev_list = MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, dev_list)
            if ret != 0:
                logger.error("MV_CC_EnumDevices failed: 0x%x", ret)
                return False
            if dev_list.nDeviceNum == 0:
                logger.error("No Hik camera found")
                return False

            available = []
            for i in range(dev_list.nDeviceNum):
                info = ctypes.cast(
                    dev_list.pDeviceInfo[i], ctypes.POINTER(MV_CC_DEVICE_INFO)
                ).contents
                model, sn = self._device_identity(info)
                available.append((sn, info))
                logger.info("Device %d: %s (%s)", i + 1, model, sn)

            sn_list = [sn for sn, _ in available]
            if not self.device_sn:
                # 未配置 SN：自动使用第一台识别到的相机
                self.device_sn, target_info = available[0]
                logger.info("No SN configured, auto-selected first device: %s", self.device_sn)
            else:
                target_info = next((info for sn, info in available if sn == self.device_sn), None)
                if target_info is None:
                    raise RuntimeError(
                        f"Configured SN not found: {self.device_sn}; visible: {sn_list}"
                    )

            cam = MvCamera()
            ret = cam.MV_CC_CreateHandle(target_info)
            if ret != 0:
                raise RuntimeError(f"CreateHandle failed: 0x{ret:x}")
            ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                cam.MV_CC_DestroyHandle()
                raise RuntimeError(f"OpenDevice failed SN={self.device_sn}: 0x{ret:x}")
            self.camera = cam

            # GigE 相机探测最佳包大小（USB 相机返回值无意义，忽略）
            if target_info.nTLayerType == MV_GIGE_DEVICE:
                packet = cam.MV_CC_GetOptimalPacketSize()
                if int(packet) > 0:
                    cam.MV_CC_SetIntValueEx("GevSCPSPacketSize", int(packet))

            self._apply_parameters()

            # 少量缓存 + 只取最新帧，降低延迟
            try:
                cam.MV_CC_SetImageNodeNum(3)
                cam.MV_CC_SetGrabStrategy(MV_GrabStrategy_LatestImagesOnly)
            except Exception:
                pass

            ret = cam.MV_CC_StartGrabbing()
            if ret != 0:
                raise RuntimeError(f"StartGrabbing failed: 0x{ret:x}")

            self._connected = True
            return True
        except Exception:
            logger.exception("Hik camera open failed")
            self._teardown_device()
            self._connected = False
            return False

    @staticmethod
    def _device_identity(info: MV_CC_DEVICE_INFO) -> Tuple[str, str]:
        if info.nTLayerType == MV_USB_DEVICE:
            usb = info.SpecialInfo.stUsb3VInfo
            return _decode_c_string(usb.chModelName), _decode_c_string(usb.chSerialNumber)
        gige = info.SpecialInfo.stGigEInfo
        return _decode_c_string(gige.chModelName), _decode_c_string(gige.chSerialNumber)

    def _set_enum(self, key: str, value: int) -> None:
        ret = self.camera.MV_CC_SetEnumValue(key, value)
        if ret != 0:
            logger.debug("Set enum %s=%d failed: 0x%x", key, value, ret)

    def _set_float(self, key: str, value: float) -> None:
        ret = self.camera.MV_CC_SetFloatValue(key, float(value))
        if ret != 0:
            logger.debug("Set float %s=%s failed: 0x%x", key, value, ret)

    def _set_int(self, key: str, value: int) -> None:
        ret = self.camera.MV_CC_SetIntValueEx(key, int(value))
        if ret != 0:
            logger.debug("Set int %s=%d failed: 0x%x", key, value, ret)

    def _set_bool(self, key: str, value: bool) -> None:
        ret = self.camera.MV_CC_SetBoolValue(key, bool(value))
        if ret != 0:
            logger.debug("Set bool %s=%s failed: 0x%x", key, value, ret)

    def _apply_parameters(self) -> None:
        cam = self.camera
        try:
            # 像素格式：优先请求 bayer_mode 对应的 Bayer8（原始拜耳，带宽最省），
            # 相机若不支持该排列则保持当前原生格式，取流时按实际上报格式转换。
            self._set_enum("PixelFormat", BAYER_MODE_TO_PIXELTYPE[self.bayer_mode])

            # binning
            for attr in ("BinningHorizontal", "BinningVertical"):
                self._set_int(attr, self.binning)

            # ROI：先清零偏移再设分辨率，避免越界
            for off_attr in ("OffsetX", "OffsetY"):
                self._set_int(off_attr, 0)
            self._set_int("Width", self.width)
            self._set_int("Height", self.height)
            self.width = self._get_int("Width", self.width)
            self.height = self._get_int("Height", self.height)

            # 触发
            self._set_enum(
                "TriggerMode",
                MV_TRIGGER_MODE_ON if self.trigger_enable else MV_TRIGGER_MODE_OFF,
            )
            if self.trigger_enable:
                self._set_enum(
                    "TriggerSource",
                    TRIGGER_SOURCE_MAP.get(self.trigger_source, MV_TRIGGER_SOURCE_SOFTWARE),
                )

            # 连续模式下开帧率控制
            if not self.trigger_enable:
                self._set_bool("AcquisitionFrameRateEnable", True)
                self._set_float("AcquisitionFrameRate", self.frame_rate)

            # 曝光 / 增益：关闭自动后手动设定
            self._set_enum("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
            self._set_float("ExposureTime", self.exposure_us)
            self._set_enum("GainAuto", MV_GAIN_MODE_OFF)
            gain = max(GAIN_MIN_DB, min(GAIN_MAX_DB, self.gain))
            self._set_float("Gain", gain)
            self.gain = gain

            # 白平衡
            self._set_enum(
                "BalanceWhiteAuto", WB_MODE_MAP.get(self.wb_mode, MV_BALANCEWHITE_AUTO_ONCE)
            )
            if self.wb_mode == "off":
                for name, value in (("red", self.wb_r), ("green", self.wb_g), ("blue", self.wb_b)):
                    self._set_enum("BalanceRatioSelector", BALANCE_SELECTOR[name])
                    # 海康 BalanceRatio 为整型倍率（1024 = 1.0x）
                    self._set_int("BalanceRatio", int(round(value * 1024)))
        except Exception:
            logger.exception("Failed to apply camera parameters")

    def _get_int(self, key: str, default: int) -> int:
        value = MVCC_INTVALUE_EX()
        ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))
        if self.camera.MV_CC_GetIntValueEx(key, value) == 0:
            return int(value.nCurValue)
        return default

    def _get_float(self, key: str):
        value = MVCC_FLOATVALUE()
        ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))
        if self.camera.MV_CC_GetFloatValue(key, value) == 0:
            return float(value.fMin), float(value.fMax), float(value.fCurValue)
        return None, None, None

    def _grab_bgr(self) -> Optional[np.ndarray]:
        if self.camera is None:
            return None
        frame = MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(frame), 0, ctypes.sizeof(frame))
        ret = self.camera.MV_CC_GetImageBuffer(frame, 1000)
        if ret != 0 or not frame.pBufAddr:
            return None
        try:
            info = frame.stFrameInfo
            width = int(info.nWidth)
            height = int(info.nHeight)
            pixel_type = info.enPixelType
            buf = (ctypes.c_ubyte * info.nFrameLen).from_address(
                ctypes.addressof(frame.pBufAddr.contents)
            )
            raw = np.frombuffer(buf, dtype=np.uint8)

            convert = BAYER_PIXELTYPE_TO_CV.get(pixel_type)
            if convert is not None:
                if raw.size < width * height:
                    return None
                mono = raw[: width * height].reshape((height, width))
                return cv2.cvtColor(mono, convert)
            if pixel_type == PixelType_Gvsp_Mono8:
                if raw.size < width * height:
                    return None
                mono = raw[: width * height].reshape((height, width))
                return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
            if pixel_type == PixelType_Gvsp_BGR8_Packed:
                if raw.size < width * height * 3:
                    return None
                return raw[: width * height * 3].reshape((height, width, 3)).copy()
            if pixel_type == PixelType_Gvsp_RGB8_Packed:
                if raw.size < width * height * 3:
                    return None
                rgb = raw[: width * height * 3].reshape((height, width, 3))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            logger.warning("Unsupported pixel type: 0x%x", pixel_type)
            return None
        finally:
            self.camera.MV_CC_FreeImageBuffer(frame)

    def start_streaming(self) -> bool:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return True
        if not self._connected and not self.open():
            return False
        self._stop_event = threading.Event()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="hik-stream", daemon=True
        )
        self._stream_thread.start()
        return True

    def _stream_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                # 软触发：每帧发一次
                if (
                    self.trigger_enable
                    and self.trigger_source == "software"
                    and self.camera is not None
                ):
                    self.camera.MV_CC_SetCommandValue("TriggerSoftware")

                frame = self._grab_bgr()
                if frame is None:
                    time.sleep(0.001)
                    continue
                with self._lock:
                    self._latest = (frame, time.time())
            except Exception as exc:
                logger.warning("Stream grab error: %s", exc)
                time.sleep(0.01)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._latest is None:
                return None, None
            frame, ts = self._latest
            return frame, ts

    def is_connected(self) -> bool:
        return self._connected

    def get_capabilities(self) -> dict:
        exp_min, exp_max, exp_cur = self._get_float("ExposureTime")
        gain_min, gain_max, gain_cur = self._get_float("Gain")
        fps_min, fps_max, fps_cur = self._get_float("AcquisitionFrameRate")
        return {
            "exposure_min_us": exp_min,
            "exposure_max_us": exp_max,
            "exposure_us": exp_cur if exp_cur is not None else self.exposure_us,
            "gain_min_db": gain_min if gain_min is not None else GAIN_MIN_DB,
            "gain_max_db": gain_max if gain_max is not None else GAIN_MAX_DB,
            "gain_db": gain_cur if gain_cur is not None else self.gain,
            "frame_rate_min": fps_min,
            "frame_rate_max": fps_max,
            "frame_rate": fps_cur if fps_cur is not None else self.frame_rate,
        }

    def _teardown_device(self) -> None:
        if self.camera is not None:
            try:
                self.camera.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                self.camera.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self.camera.MV_CC_DestroyHandle()
            except Exception:
                pass
            self.camera = None
        if self._sdk_ready:
            _sdk_finalize()
            self._sdk_ready = False

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        self._teardown_device()
        self._connected = False
        with self._lock:
            self._latest = None
