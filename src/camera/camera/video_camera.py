"""视频文件 "相机" 后端: 用 OpenCV 读本地视频, 按视频帧率 (或配置帧率) 回放。

接口与 HikCamera / DahengCamera 一致, camera_node 无差别调用, 下游
detector / solver 照常从共享内存或 /camera/image_raw 取帧, 不感知来源。

分辨率: width/height 为 0 时用视频原生分辨率; 非 0 时把每帧缩放到该尺寸
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("VideoCamera")


class VideoCamera:
    def __init__(
        self,
        path: str,
        width: int = 0,
        height: int = 0,
        frame_rate: float = 0.0,
        loop: bool = True,
        start_frame: int = 0,
    ):
        self.path = str(Path(str(path)).expanduser())
        self.loop = bool(loop)
        self.start_frame = max(0, int(start_frame))

        # 构造时探测一次视频, 让 camera_node 在建共享内存前就知道输出分辨率
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.path}")
        native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.native_size = (native_w, native_h)
        self.width = int(width) or native_w
        self.height = int(height) or native_h
        # frame_rate<=0 用视频自带帧率; 视频也没写帧率时退回 30
        self.frame_rate = float(frame_rate) if float(frame_rate) > 0 else (native_fps or 30.0)

        self._cap: Optional[cv2.VideoCapture] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, float]] = None
        self._connected = False
        self._finished = False
        self._frame_index = 0

    def open(self) -> bool:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            logger.error("无法打开视频: %s", self.path)
            return False
        if self.start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        self._cap = cap
        self._frame_index = self.start_frame
        self._finished = False
        self._connected = True
        return True

    def _read_bgr(self) -> Optional[np.ndarray]:
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok:
            if not self.loop:
                return None
            # 到尾就回到起始帧继续放
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self._frame_index = self.start_frame
            ok, frame = self._cap.read()
            if not ok:
                return None
        self._frame_index += 1
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def start_streaming(self) -> bool:
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return True
        if not self._connected and not self.open():
            return False
        self._stop_event = threading.Event()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="video-stream", daemon=True
        )
        self._stream_thread.start()
        return True

    def _stream_loop(self) -> None:
        assert self._stop_event is not None
        period = 1.0 / max(self.frame_rate, 1e-3)
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            try:
                frame = self._read_bgr()
            except Exception as exc:
                logger.warning("Video read error: %s", exc)
                frame = None
            if frame is None:
                # 不循环且放完: 停在最后一帧, 不再产出新帧
                self._finished = True
                logger.info("视频播放结束: %s", self.path)
                return
            with self._lock:
                self._latest = (frame, time.time())
            # 按帧率节拍回放 (解码比实时慢时不累积延迟)
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop_event.wait(delay)
            else:
                next_t = time.monotonic()

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._lock:
            if self._latest is None:
                return None, None
            frame, ts = self._latest
            return frame, ts

    def is_connected(self) -> bool:
        return self._connected

    def is_finished(self) -> bool:
        """不循环模式下视频已放完; camera_node 据此不再做无帧重启。"""
        return self._finished

    def get_capabilities(self) -> dict:
        return {
            "path": self.path,
            "native_size": f"{self.native_size[0]}x{self.native_size[1]}",
            "output_size": f"{self.width}x{self.height}",
            "frame_rate": self.frame_rate,
            "frame_count": self.frame_count,
            "loop": self.loop,
        }

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=2.0)
            self._stream_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False
        with self._lock:
            self._latest = None
