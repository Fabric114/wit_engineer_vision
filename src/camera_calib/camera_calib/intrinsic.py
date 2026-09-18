"""内参标定的纯算法: 采样质量把关 -> 反复标定剔除坏视图 -> 存 ROS CameraInfo。

参考: BIT charuco_calibration_pkg/calibration.py
零 ROS 依赖, 可用合成图片直接 pytest。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from camera_calib.patterns import Detection, Pattern


@dataclass
class Sample:
    """一张被接受的标定图。"""

    object_points: np.ndarray
    image_points: np.ndarray
    label: str
    sharpness: float = 0.0
    coverage: float = 0.0
    signature: Optional[np.ndarray] = None


@dataclass
class CalibrationResult:
    rms: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    per_view_errors: np.ndarray
    intrinsic_stddev: np.ndarray
    kept: List[int] = field(default_factory=list)
    rejected: List[int] = field(default_factory=list)


# ---------------------------------------------------------------- 采样质量
def image_sharpness(gray: np.ndarray) -> float:
    """Laplacian 方差: 越大越清晰。运动模糊的图会把角点位置带偏。"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def point_coverage(image_points: np.ndarray, image_size: Tuple[int, int]) -> float:
    """标定板在画面里占的面积比, 太小的视图对畸变几乎没贡献。"""
    points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    width, height = image_size
    hull = cv2.convexHull(points)
    return float(cv2.contourArea(hull) / max(width * height, 1))


def view_signature(
    object_points: np.ndarray,
    image_points: np.ndarray,
    image_size: Tuple[int, int],
    board_size_m: Tuple[float, float],
) -> Optional[np.ndarray]:
    """用单应把板子四角投到图像上作为"视角指纹", 用来剔除重复视角。"""
    object_xy = np.asarray(object_points, dtype=np.float32).reshape(-1, 3)[:, :2]
    image_xy = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    if len(object_xy) < 4:
        return None
    homography, _ = cv2.findHomography(object_xy, image_xy, cv2.RANSAC, 2.0)
    if homography is None:
        return None
    width_m, height_m = board_size_m
    outer = np.array(
        [[0.0, 0.0], [width_m, 0.0], [width_m, height_m], [0.0, height_m]],
        dtype=np.float32,
    ).reshape(1, 4, 2)
    projected = cv2.perspectiveTransform(outer, homography).reshape(4, 2)
    projected[:, 0] /= image_size[0]
    projected[:, 1] /= image_size[1]
    return projected.reshape(-1)


def signature_distance(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(4, 2)
    b = np.asarray(second, dtype=np.float64).reshape(4, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


class SampleCollector:
    """在线采样器: 只留清晰、够大、视角不重复的图, 免得凑一堆废图。"""

    def __init__(
        self,
        pattern: Pattern,
        min_points: int = 20,
        min_sharpness: float = 80.0,
        min_coverage: float = 0.035,
        min_signature_distance: float = 0.05,
        max_samples: int = 300,
    ):
        self.pattern = pattern
        self.min_points = int(min_points)
        self.min_sharpness = float(min_sharpness)
        self.min_coverage = float(min_coverage)
        self.min_signature_distance = float(min_signature_distance)
        self.max_samples = int(max_samples)
        self.samples: List[Sample] = []

    def evaluate(self, gray: np.ndarray, label: str = "") -> Tuple[Optional[Sample], str]:
        """返回 (可用样本, 原因)。样本为 None 时 reason 说明被拒的理由。"""
        image_size = (gray.shape[1], gray.shape[0])
        detection: Detection = self.pattern.detect(gray)
        if not detection.found:
            return None, "未检测到标定板"
        if detection.count < self.min_points:
            return None, f"点数不足 {detection.count} < {self.min_points}"
        sharpness = image_sharpness(gray)
        if sharpness < self.min_sharpness:
            return None, f"图像太模糊 {sharpness:.1f} < {self.min_sharpness:.1f}"
        coverage = point_coverage(detection.image_points, image_size)
        if coverage < self.min_coverage:
            return None, f"占画面太小 {coverage:.3f} < {self.min_coverage:.3f}"
        signature = view_signature(
            detection.object_points, detection.image_points, image_size,
            self.pattern.board_size_m,
        )
        if signature is not None:
            for existing in self.samples:
                if existing.signature is None:
                    continue
                if signature_distance(existing.signature, signature) < self.min_signature_distance:
                    return None, "视角与已有样本重复"
        return (
            Sample(
                object_points=detection.object_points,
                image_points=detection.image_points,
                label=label,
                sharpness=sharpness,
                coverage=coverage,
                signature=signature,
            ),
            "ok",
        )

    def add(self, sample: Sample) -> bool:
        if len(self.samples) >= self.max_samples:
            return False
        self.samples.append(sample)
        return True

    def try_add(self, gray: np.ndarray, label: str = "") -> Tuple[bool, str]:
        sample, reason = self.evaluate(gray, label)
        if sample is None:
            return False, reason
        return self.add(sample), reason

    def __len__(self) -> int:
        return len(self.samples)


# ---------------------------------------------------------------- 标定
def _calibrate_once(samples: List[Sample], image_size: Tuple[int, int]):
    return cv2.calibrateCameraExtended(
        [s.object_points for s in samples],
        [s.image_points for s in samples],
        image_size,
        None,
        None,
        flags=0,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-12),
    )


def calibrate(
    samples: List[Sample],
    image_size: Tuple[int, int],
    min_samples: int = 15,
    max_view_error_px: float = 0.8,
    max_rounds: int = 6,
) -> CalibrationResult:
    """标定 -> 剔除重投影误差异常的视图 -> 再标, 直到干净或到轮数上限。

    剔除阈值取 (配置上限, 中位数+3*MAD) 的较小者: 单张坏图不会拖垮整体,
    同时避免在本来就很好的一组样本里过度剔除。
    """
    if len(samples) < min_samples:
        raise ValueError(f"有效样本不足: {len(samples)} < {min_samples}")

    kept = list(range(len(samples)))
    rejected: List[int] = []
    result = None

    for round_index in range(max_rounds + 1):
        result = _calibrate_once([samples[i] for i in kept], image_size)
        errors = np.asarray(result[7], dtype=np.float64).reshape(-1)
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        robust_limit = max(median + 3.0 * 1.4826 * mad, median * 1.25, 0.15)
        limit = min(float(max_view_error_px), robust_limit)
        bad = np.flatnonzero(errors > limit).tolist()
        if not bad or round_index >= max_rounds:
            break
        removable = len(kept) - min_samples
        if removable <= 0:
            break
        bad.sort(key=lambda position: errors[position], reverse=True)
        bad = bad[: min(removable, max(1, len(kept) // 10))]
        removed = {kept[position] for position in bad}
        rejected.extend(sorted(removed))
        kept = [index for index in kept if index not in removed]

    assert result is not None
    return CalibrationResult(
        rms=float(result[0]),
        camera_matrix=np.asarray(result[1], dtype=np.float64),
        dist_coeffs=np.asarray(result[2], dtype=np.float64).reshape(-1),
        per_view_errors=np.asarray(result[7], dtype=np.float64).reshape(-1),
        intrinsic_stddev=np.asarray(result[5], dtype=np.float64).reshape(-1),
        kept=kept,
        rejected=sorted(rejected),
    )


# ---------------------------------------------------------------- 输出
def save_camera_info(
    result: CalibrationResult,
    image_size: Tuple[int, int],
    camera_name: str,
    path,
) -> Path:
    """写标准 ROS CameraInfo YAML —— camera 直接读这个文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = image_size
    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:, :3] = result.camera_matrix
    document = {
        "image_width": int(width),
        "image_height": int(height),
        "camera_name": camera_name,
        "distortion_model": "plumb_bob",
        "camera_matrix": {"rows": 3, "cols": 3,
                          "data": result.camera_matrix.reshape(-1).tolist()},
        "distortion_coefficients": {"rows": 1, "cols": int(result.dist_coeffs.size),
                                    "data": result.dist_coeffs.tolist()},
        "rectification_matrix": {"rows": 3, "cols": 3,
                                 "data": np.eye(3).reshape(-1).tolist()},
        "projection_matrix": {"rows": 3, "cols": 4,
                              "data": projection.reshape(-1).tolist()},
    }
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
    return path


def save_report(
    result: CalibrationResult,
    samples: List[Sample],
    image_size: Tuple[int, int],
    board_description: str,
    path,
) -> Path:
    """存一份可追溯的标定报告: 用了哪些图、每张的误差、剔了哪些。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "board": board_description,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "total_samples": len(samples),
        "used_samples": len(result.kept),
        "rejected_samples": [samples[i].label for i in result.rejected],
        "rms_reprojection_error_px": result.rms,
        "mean_view_error_px": float(np.mean(result.per_view_errors)),
        "max_view_error_px": float(np.max(result.per_view_errors)),
        "camera_matrix": result.camera_matrix.tolist(),
        "distortion_coefficients": result.dist_coeffs.tolist(),
        "intrinsic_stddev": result.intrinsic_stddev.tolist(),
        "per_view_errors_px": {
            samples[index].label: float(error)
            for index, error in zip(result.kept, result.per_view_errors)
        },
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
    return path
