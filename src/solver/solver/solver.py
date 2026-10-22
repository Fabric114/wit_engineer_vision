"""纯 NumPy/OpenCV PnP 解算。"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PoseEstimate:
    T_camera_exchange_station: np.ndarray
    reprojection_error_px: float
    inlier_mask: np.ndarray
    valid: bool
    reason: str

    def __post_init__(self) -> None:
        transform = np.asarray(self.T_camera_exchange_station, dtype=np.float64)
        mask = np.asarray(self.inlier_mask, dtype=bool)
        if transform.shape != (4, 4):
            raise ValueError("pose transform must have shape (4, 4)")
        transform = np.ascontiguousarray(transform)
        mask = np.ascontiguousarray(mask)
        transform.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "T_camera_exchange_station", transform)
        object.__setattr__(self, "inlier_mask", mask)


def _invalid(count: int, reason: str) -> PoseEstimate:
    return PoseEstimate(np.full((4, 4), np.nan), float("inf"), np.zeros(count, dtype=bool), False, reason)


class PnPEstimator:
    def __init__(self, confidence_threshold=0.2, reprojection_error_threshold_px=5.0,
                 min_inliers=4, use_ransac=True):
        self.confidence_threshold = float(confidence_threshold)
        self.reprojection_error_threshold_px = float(reprojection_error_threshold_px)
        self.min_inliers = int(min_inliers)
        if self.min_inliers < 4:
            raise ValueError("min_inliers must be at least four")
        self.use_ransac = bool(use_ransac)

    def estimate(self, object_points_m, image_points_px, camera_matrix,
                 distortion_coeffs, confidence=None) -> PoseEstimate:
        try:
            obj = np.asarray(object_points_m, dtype=np.float64)
            img = np.asarray(image_points_px, dtype=np.float64)
            K = np.asarray(camera_matrix, dtype=np.float64)
            D = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return _invalid(0, "input arrays have invalid shape or type")
        count = obj.shape[0] if obj.ndim >= 1 else 0
        if obj.ndim != 2 or obj.shape[1:] != (3,):
            return _invalid(count, "object points shape is invalid")
        if img.ndim != 2 or img.shape != (count, 2):
            return _invalid(count, "image points shape does not match object points")
        if not np.isfinite(obj).all() or not np.isfinite(img).all():
            return _invalid(count, "point inputs must be finite")
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or K[2, 2] == 0:
            return _invalid(count, "camera matrix is invalid or not finite")
        if D.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(D).all():
            return _invalid(count, "distortion coefficients are invalid")
        if confidence is not None:
            try:
                conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                return _invalid(count, "confidence shape is invalid")
            if conf.shape != (count,) or not np.isfinite(conf).all():
                return _invalid(count, "confidence shape or values are invalid")
            keep = conf >= self.confidence_threshold
            obj, img = obj[keep], img[keep]
            selected = np.flatnonzero(keep)
        else:
            selected = np.arange(count)
        if len(obj) < self.min_inliers:
            return _invalid(count, f"at least four points are required; got {len(obj)}")
        centered = obj - obj.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values.size < 3 or singular_values[0] <= 0:
            return _invalid(count, "object point geometry is degenerate")
        # 共面点集 (如兑换口 7 点) PnP 可解; 只拒绝共线/近共线。
        if singular_values[1] / singular_values[0] < 1e-3:
            return _invalid(count, "object point geometry is degenerate")

        try:
            flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
            if self.use_ransac:
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj, img, K, D, flags=flag, reprojectionError=self.reprojection_error_threshold_px,
                    iterationsCount=100, confidence=0.99)
            else:
                ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=flag)
                inliers = np.arange(len(obj), dtype=np.int32).reshape(-1, 1)
            if not ok or rvec is None or tvec is None:
                return _invalid(count, "PnP did not converge")
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
        except cv2.error as exc:
            return _invalid(count, f"PnP failed: {exc}")

        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = (rotation @ obj.T + tvec).T
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        errors = np.linalg.norm(projected.reshape(-1, 2) - img, axis=1)
        reprojection = float(np.mean(errors))
        mask = np.zeros(count, dtype=bool)
        inlier_indices = np.asarray(inliers).reshape(-1) if inliers is not None else np.arange(len(obj))
        mask[selected[inlier_indices]] = True
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec[:, 0]
        valid = (len(inlier_indices) >= self.min_inliers and np.isfinite(transform).all()
                 and np.all(camera_points[:, 2] > 0) and np.isfinite(reprojection)
                 and reprojection <= self.reprojection_error_threshold_px)
        reason = "ok" if valid else "reprojection error exceeds threshold or pose has invalid depth"
        return PoseEstimate(transform, reprojection, mask, valid, reason)
