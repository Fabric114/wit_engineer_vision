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
    """RANSAC 只负责挑内点; 之后的重解、LM 精化、误差验收都只用内点 (参考 RM2026 pose.py)。

    ``ransac_reprojection_error_px``: RANSAC 判内点的单点门限。
    ``reprojection_error_threshold_px``: 最终验收门限, 比较内点平均重投影误差。
    """

    def __init__(self, confidence_threshold=0.2, reprojection_error_threshold_px=5.0,
                 min_inliers=4, use_ransac=True, ransac_reprojection_error_px=5.0):
        self.confidence_threshold = float(confidence_threshold)
        # padding 槽位置信度为 0, 门限必须为正才能把 (0, 0) 假点挡在外面。
        if not self.confidence_threshold > 0.0:
            raise ValueError("confidence_threshold must be positive")
        self.reprojection_error_threshold_px = float(reprojection_error_threshold_px)
        self.ransac_reprojection_error_px = float(ransac_reprojection_error_px)
        if not self.reprojection_error_threshold_px > 0.0 or not self.ransac_reprojection_error_px > 0.0:
            raise ValueError("reprojection error thresholds must be positive")
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
            keep = conf > self.confidence_threshold
            obj, img = obj[keep], img[keep]
            selected = np.flatnonzero(keep)
        else:
            selected = np.arange(count)
        if len(obj) < 4:
            return _invalid(count, f"at least four points are required; got {len(obj)}")
        if len(obj) < self.min_inliers:
            return _invalid(count, f"{len(obj)} points left after confidence filter; min_inliers={self.min_inliers}")
        centered = obj - obj.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values.size < 3 or singular_values[0] <= 0:
            return _invalid(count, "object point geometry is degenerate")
        # 共面点集 (如兑换口 7 点) PnP 可解; 只拒绝共线/近共线。
        if singular_values[1] / singular_values[0] < 1e-3:
            return _invalid(count, "object point geometry is degenerate")

        flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
        try:
            if self.use_ransac:
                ok, _, _, ransac_inliers = cv2.solvePnPRansac(
                    obj, img, K, D, flags=flag, reprojectionError=self.ransac_reprojection_error_px,
                    iterationsCount=100, confidence=0.99)
                if not ok or ransac_inliers is None:
                    return _invalid(count, "PnP did not converge")
                inliers = np.asarray(ransac_inliers).reshape(-1)
            else:
                inliers = np.arange(len(obj))
            if len(inliers) < self.min_inliers:
                return _invalid(count, f"RANSAC kept {len(inliers)} inliers; min_inliers={self.min_inliers}")
            obj_in, img_in = obj[inliers], img[inliers]
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj_in, img_in, K, D, flags=flag)
            if not ok:
                return _invalid(count, "PnP did not converge")
            best = None
            for rvec, tvec in zip(rvecs, tvecs):
                rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
                tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
                if not _in_front(obj, rvec, tvec):
                    continue
                if hasattr(cv2, "solvePnPRefineLM"):
                    rvec, tvec = cv2.solvePnPRefineLM(obj_in, img_in, K, D, rvec, tvec)
                if not _in_front(obj, rvec, tvec):
                    continue
                projected, _ = cv2.projectPoints(obj_in, rvec, tvec, K, D)
                error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - img_in, axis=1)))
                if best is None or error < best[0]:
                    best = (error, rvec, tvec)
        except cv2.error as exc:
            return _invalid(count, f"PnP failed: {exc}")
        if best is None:
            return _invalid(count, "every PnP solution puts points behind the camera")

        reprojection, rvec, tvec = best
        mask = np.zeros(count, dtype=bool)
        mask[selected[inliers]] = True
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3], _ = cv2.Rodrigues(rvec)
        transform[:3, 3] = tvec[:, 0]
        valid = (np.isfinite(transform).all() and np.isfinite(reprojection)
                 and reprojection <= self.reprojection_error_threshold_px)
        reason = "ok" if valid else (f"inlier reprojection error {reprojection:.2f} px exceeds "
                                     f"{self.reprojection_error_threshold_px:.2f} px")
        return PoseEstimate(transform, reprojection, mask, valid, reason)


def _in_front(object_points, rvec, tvec) -> bool:
    rotation, _ = cv2.Rodrigues(rvec)
    return bool(np.all((object_points @ rotation.T + tvec.reshape(1, 3))[:, 2] > 0))
