"""相机外参与位姿矩阵工具 (零 ROS)。

pnp_node 用它把相机系站姿换到 arm_base 系:

    T_arm_base_exchange_station = T_arm_base_camera · T_camera_exchange_station

四元数一律 ``[w, x, y, z]`` (与 ``planning/transform.py`` 一致)。solver 不依赖 planning,
所以这里用 numpy / cv2 独立实现, 不经 scipy。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def matrix_from_pose(position_xyz_m, quaternion_wxyz) -> np.ndarray:
    """位置 (米) + 四元数 ``[w, x, y, z]`` -> ``(4, 4)`` 齐次变换。四元数会先归一化。"""
    xyz = np.asarray(position_xyz_m, dtype=np.float64).reshape(3)
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-12 or not np.isfinite(xyz).all():
        raise ValueError("pose must be finite with a non-zero quaternion")
    w, x, y, z = q / norm
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    T[:3, 3] = xyz
    return T


def quaternion_wxyz_from_rotation(rotation) -> np.ndarray:
    """``(3, 3)`` 旋转 -> 四元数 ``[w, x, y, z]`` (经 cv2.Rodrigues 轴角)。"""
    import cv2

    rotvec, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotvec[:, 0] / theta
    return np.concatenate(([np.cos(theta / 2.0)], axis * np.sin(theta / 2.0)))


def pose_from_matrix(T) -> tuple[np.ndarray, np.ndarray]:
    """``(4, 4)`` 齐次变换 -> (位置 xyz 米, 四元数 ``[w, x, y, z]``)。"""
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), quaternion_wxyz_from_rotation(T[:3, :3])


def validate_transform(T, name: str, *, atol: float = 1e-6) -> np.ndarray:
    """校验单个齐次变换: 形状 (4, 4)、有限、末行 [0,0,0,1]、旋转正交且 det = +1。"""
    values = np.asarray(T, dtype=np.float64)
    if values.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(values[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} bottom row must be [0, 0, 0, 1]")
    R = values[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError(f"{name} rotation block is not orthonormal")
    if np.linalg.det(R) <= 0.0:
        raise ValueError(f"{name} rotation block must have det = +1")
    return values


def is_identity(T, atol: float = 1e-9) -> bool:
    return bool(np.allclose(np.asarray(T, dtype=np.float64), np.eye(4), atol=atol))


def compose(T_arm_base_camera, T_camera_exchange_station) -> np.ndarray:
    return np.asarray(T_arm_base_camera, dtype=np.float64) @ np.asarray(
        T_camera_exchange_station, dtype=np.float64)


def _matrix_from_entry(entry, name: str) -> np.ndarray:
    if not isinstance(entry, dict) or entry.get("rows") != 4 or entry.get("cols") != 4:
        raise ValueError(f"{name} must be {{rows: 4, cols: 4, data: [...]}}")
    data = np.asarray(entry.get("data"), dtype=np.float64)
    if data.size != 16:
        raise ValueError(f"{name} data must contain 16 values, got {data.size}")
    return validate_transform(data.reshape(4, 4), name)


def load_extrinsic(cfg: dict, base_dir) -> np.ndarray:
    """读 ``T_arm_base_camera``。

    ``extrinsic_file`` 非空时优先读该文件 (``camera_calib.hand_eye.save_extrinsic`` 的输出,
    相对路径相对 ``base_dir``), 否则用 cfg 里内联的 ``arm_base_from_camera``。
    """
    extrinsic_file = cfg.get("extrinsic_file") or ""
    if extrinsic_file:
        import yaml

        path = Path(extrinsic_file)
        if not path.is_absolute():
            path = Path(base_dir) / path
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}
        return _matrix_from_entry(document.get("arm_base_from_camera"), f"arm_base_from_camera ({path})")
    return _matrix_from_entry(cfg.get("arm_base_from_camera"), "arm_base_from_camera")
