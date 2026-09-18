"""检测结果的公共数据结构 —— 纯数据, 零 ROS / 零推理后端依赖, 可 pytest。

detector_tc (yolo-pose, OpenVINO) 与 detector_eu (yolo detect, ultralytics)
都产出 DetectionResult, 节点层与可视化层只认这一套结构, 不关心后端。

坐标约定: bbox / keypoints 都已反 letterbox 回到**原图像素**坐标系。
- bbox:      np.ndarray shape (4,)  = [x1, y1, x2, y2]
- keypoints: np.ndarray shape (K, 3) = [x, y, score]; 纯检测框 (eu) 时为 None。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Instance:
    """一个检测实例: 框 + 可选关键点。"""

    class_id: int
    class_name: str
    score: float
    bbox: np.ndarray                      # (4,) x1 y1 x2 y2, 原图像素
    keypoints: Optional[np.ndarray] = None  # (K, 3) x y score, 原图像素; 无则 None
    keypoint_names: Optional[List[str]] = None

    def has_keypoints(self) -> bool:
        return self.keypoints is not None and len(self.keypoints) > 0


@dataclass
class DetectionResult:
    """一帧的检测结果。"""

    instances: List[Instance] = field(default_factory=list)
    inference_ms: float = 0.0
    stage_ms: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.instances)
