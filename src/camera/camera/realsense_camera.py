"""Intel RealSense D435 后端 (pyrealsense2)。

接口与 HikCamera / DahengCamera 一致, camera_node 无差别调用; 另外多两个方法:
    get_latest_rgbd()  -> (color, depth, ts) 同一帧组, 原子取出
    intrinsics()       -> 彩色流出厂内参 dict (深度已对齐到彩色, 共用这一份)

enable_depth=False: 只开彩色流 (tc 兑换站/科技核心, 只要彩色)。
enable_depth=True : 彩色 + 深度, 深度用 rs.align 对齐到彩色 (eu 能量单元)。
    对齐后深度是 uint16 (H, W), 单位见 depth_scale (D435 默认 0.001 m/LSB, 即毫米)。
    开深度时缺任何一路就丢掉这一帧组
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("RealSenseCamera")


def _import_rs():
    """惰性导入: 没装 pyrealsense2 时其它后端照常可用, 选了 realsense 才报错。"""
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "未安装 pyrealsense2: pip install pyrealsense2 (或装 librealsense 的 python 绑定)"
        ) from exc
    return rs


def resolve_enable_depth(depth: object, detector_type: str) -> bool:
    """camera.yaml 的 realsense_camera.depth -> 是否开深度流。

    auto  : 跟 detector.yaml 的 type 走, eu (能量单元) 开, tc (兑换站/科技核心) 不开
    true  / on  : 强制开
    false / off : 强制关
    """
    if isinstance(depth, bool):
        return depth
    value = str(depth).strip().lower()
    if value in ("true", "on", "1"):
        return True
    if value in ("false", "off", "0"):
        return False
    if value in ("", "auto"):
        det = str(detector_type).strip().lower()
        if det == "eu":
            return True
        if det == "tc":
            return False
        raise ValueError(f"depth=auto 但 detector type={detector_type!r} 不是 tc|eu")
    raise ValueError(f"realsense depth={depth!r} 非法 (auto | true | false)")


class RealSenseCamera:
    def __init__(
        self,
        sn: str = "",
        width: int = 640,
        height: int = 480,
        frame_rate: float = 30.0,
        enable_depth: bool = False,
        timeout_ms: int = 1000,
    ):
        self.sn = str(sn or "").strip()
        self.width = int(width)
        self.height = int(height)
        self.frame_rate = float(frame_rate)
        self.enable_depth = bool(enable_depth)
        self.timeout_ms = int(timeout_ms)

        self._rs = None
        self._pipeline = None
        self._align = None
        self._depth_scale: Optional[float] = None
        self._intrinsics: Optional[dict] = None
        self._device_name = ""
        self._device_sn = ""

        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._lock = threading.Lock()
        # (color, depth|None, ts)
        self._latest: Optional[Tuple[np.ndarray, Optional[np.ndarray], float]] = None
        self._connected = False

    def open(self) -> bool:
        rs = self._rs = self._rs or _import_rs()
        config = rs.config()
        if self.sn:
            config.enable_device(self.sn)
        fps = int(round(self.frame_rate))
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)

        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
        except Exception as exc:
            logger.error("RealSense 启动失败 (sn=%s %dx%d@%d depth=%s): %s",
                         self.sn or "<auto>", self.width, self.height, fps,
                         self.enable_depth, exc)
            return False

        device = profile.get_device()
        self._device_name = device.get_info(rs.camera_info.name)
        self._device_sn = device.get_info(rs.camera_info.serial_number)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self._intrinsics = {
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "ppx": float(intr.ppx),
            "ppy": float(intr.ppy),
            "coeffs": [float(c) for c in intr.coeffs],
        }
        if self.enable_depth:
            self._depth_scale = float(device.first_depth_sensor().get_depth_scale())
            self._align = rs.align(rs.stream.color)
        else:
            self._depth_scale = None
            self._align = None

        self._pipeline = pipeline
        self._connected = True
        return True

    def _grab(self) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        assert self._pipeline is not None
        frames = self._pipeline.wait_for_frames(self.timeout_ms)
        if self._align is not None:
            frames = self._align.process(frames)
        color = frames.get_color_frame()
        if not color:
            return None
        color_img = np.asanyarray(color.get_data()).copy()
        if not self.enable_depth:
            return color_img, None
        depth = frames.get_depth_frame()
        if not depth:
            return None
        return color_img, np.asanyarray(depth.get_data()).copy()

    def start_streaming(self) -> bool:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return True
        if not self._connected and not self.open():
            return False
        self._stop_event = threading.Event()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="realsense-stream", daemon=True
        )
        self._stream_thread.start()
        return True

    def _stream_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                grabbed = self._grab()
            except Exception as exc:
                # wait_for_frames 超时 / 拔线; camera_node 的无帧超时负责重启
                logger.warning("RealSense 取帧失败: %s", exc)
                self._stop_event.wait(0.05)
                continue
            if grabbed is None:
                continue
            color, depth = grabbed
            with self._lock:
                self._latest = (color, depth, time.time())

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._latest is None:
                return None, None
            color, _, ts = self._latest
            return color, ts

    def get_latest_rgbd(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """同一帧组的彩色 + 对齐深度; 未开深度时 depth 为 None。"""
        with self._lock:
            if self._latest is None:
                return None, None, None
            return self._latest

    def intrinsics(self) -> Optional[dict]:
        """彩色流出厂内参 (open 之后才有)。"""
        return self._intrinsics

    @property
    def depth_scale(self) -> Optional[float]:
        return self._depth_scale

    def is_connected(self) -> bool:
        return self._connected

    def get_capabilities(self) -> dict:
        caps = {
            "device": self._device_name,
            "sn": self._device_sn,
            "resolution": f"{self.width}x{self.height}",
            "frame_rate": self.frame_rate,
            "depth": self.enable_depth,
        }
        if self._depth_scale is not None:
            caps["depth_scale_m"] = self._depth_scale
        return caps

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:
                logger.warning("RealSense 停止失败: %s", exc)
            self._pipeline = None
        self._align = None
        self._connected = False
        with self._lock:
            self._latest = None
