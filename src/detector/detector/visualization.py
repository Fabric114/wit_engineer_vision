"""在帧上画检测结果 —— 框 + 标签 + 关键点。纯 cv2/numpy, 供节点做调试可视化。

画在副本上,颜色按类别区分 (BGR), 关键点画点 + 名字, 越界点跳过。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .types import DetectionResult, Instance

# 类别 -> BGR 颜色; 未知类别用白色兜底
_CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "pillar": (0, 255, 0),
    "exchange": (255, 180, 0),
    "dhz": (0, 215, 255),
    "object": (0, 215, 255),
}
_DEFAULT_COLOR = (255, 255, 255)
_KP_COLOR = (0, 0, 255)


def _color_for(class_name: str) -> Tuple[int, int, int]:
    return _CLASS_COLORS.get(class_name, _DEFAULT_COLOR)


def _clamp_box(bbox: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = int(round(max(0.0, min(x1, w - 1))))
    y1 = int(round(max(0.0, min(y1, h - 1))))
    x2 = int(round(max(0.0, min(x2, w - 1))))
    y2 = int(round(max(0.0, min(y2, h - 1))))
    return x1, y1, x2, y2


def draw_instance(canvas: np.ndarray, inst: Instance, kp_score_threshold: float = 0.0) -> None:
    """把单个实例画到 canvas (原地修改)。"""
    h, w = canvas.shape[:2]
    color = _color_for(inst.class_name)
    x1, y1, x2, y2 = _clamp_box(inst.bbox, w, h)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    label = f"{inst.class_name} {inst.score:.2f}"
    cv2.putText(canvas, label, (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    if not inst.has_keypoints():
        return
    names = inst.keypoint_names or [str(i) for i in range(len(inst.keypoints))]
    for idx, point in enumerate(inst.keypoints):
        x, y, score = float(point[0]), float(point[1]), float(point[2])
        if score < kp_score_threshold:
            continue
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        cv2.circle(canvas, (xi, yi), 3, _KP_COLOR, -1)
        name = names[idx] if idx < len(names) else str(idx)
        cv2.putText(canvas, name, (xi + 4, yi - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, _KP_COLOR, 1, cv2.LINE_AA)


def draw_detections(
    image: np.ndarray,
    result: DetectionResult,
    kp_score_threshold: float = 0.0,
    hud: str = "",
) -> np.ndarray:
    """返回带标注的副本 (不改原图)。hud 是左上角叠加的一行状态文字。"""
    canvas = image.copy()
    for inst in result.instances:
        draw_instance(canvas, inst, kp_score_threshold=kp_score_threshold)
    text = hud or f"{len(result)} det  {result.inference_ms:.1f} ms"
    cv2.putText(canvas, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas
