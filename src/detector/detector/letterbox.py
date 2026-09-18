"""YOLO letterbox 预处理与坐标反算 —— 纯 numpy/cv2, 零 ROS 依赖, 可 pytest。

等比缩放 + 居中灰边(114) padding 到模型输入尺寸, 记录 scale 与 (pad_x, pad_y),
推理后用同一组参数把框/关键点从模型输入坐标反算回原图像素。这一步必须精确,
否则 PnP 会吃到一个系统性偏移 (见 pose_detect_pkg/detect_roi.py)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxInfo:
    """反算所需的参数: 原图 -> 模型输入的 scale 与居中 padding。"""

    scale: float
    pad_x: float
    pad_y: float

    def undo_points(self, points_xy: np.ndarray) -> np.ndarray:
        """(N, 2) 模型输入坐标 -> 原图像素坐标。"""
        pts = np.asarray(points_xy, dtype=np.float64).copy()
        pts[:, 0] = (pts[:, 0] - self.pad_x) / self.scale
        pts[:, 1] = (pts[:, 1] - self.pad_y) / self.scale
        return pts

    def undo_boxes(self, bbox_xyxy: np.ndarray) -> np.ndarray:
        """(4,) [x1,y1,x2,y2] 模型输入坐标 -> 原图像素坐标。"""
        box = np.asarray(bbox_xyxy, dtype=np.float64).copy()
        box[[0, 2]] = (box[[0, 2]] - self.pad_x) / self.scale
        box[[1, 3]] = (box[[1, 3]] - self.pad_y) / self.scale
        return box


def letterbox(
    image: np.ndarray,
    input_size: Tuple[int, int],
    pad_value: int = 114,
) -> Tuple[np.ndarray, LetterboxInfo]:
    """把 BGR 原图等比缩放 + 居中 padding 到 input_size=(W, H), 返回画布与反算参数。"""
    input_w, input_h = int(input_size[0]), int(input_size[1])
    height, width = image.shape[:2]
    scale = min(input_w / float(width), input_h / float(height))
    resized_w = int(round(width * scale))
    resized_h = int(round(height * scale))
    pad_x = (input_w - resized_w) // 2
    pad_y = (input_h - resized_h) // 2

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_h, input_w, 3), pad_value, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    return canvas, LetterboxInfo(scale=float(scale), pad_x=float(pad_x), pad_y=float(pad_y))


def to_blob(padded_bgr: np.ndarray, swap_rb: bool = True) -> np.ndarray:
    """letterbox 画布 -> [1,3,H,W] FP32 blob。swap_rb=True 时 BGR->RGB, 归一化到 [0,1]。"""
    image = padded_bgr[..., ::-1] if swap_rb else padded_bgr
    blob = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return blob
