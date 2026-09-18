"""camera -> arm_base 外参标定 (相机固定在底盘/机架上, 即 eye-to-hand)。

两条路子, 按你手上有什么选:

A. 单次测量法 `extrinsic_from_measured_board`
   标定板固定在场地上, 你能量出板子在 arm_base 系下的位姿 T_base_board,
   再用 PnP 得到 T_cam_board, 直接 T_base_cam = T_base_board @ inv(T_cam_board)。
   一次搞定, 精度取决于你量得准不准。

B. 多姿态求解 `calibrate_eye_to_hand` (推荐)
   标定板刚性固定在**末端**上, 让臂摆 10+ 个姿态, 每个姿态记
   (FK 得到的 T_base_gripper, PnP 得到的 T_cam_board), 一起解出 T_base_cam。
   不需要量板子的位置, 但需要 FK 准 (依赖真实 DH)。

约定: 全程 SI 单位 (米), 相机系是 OpenCV 光学系 (X 右 / Y 下 / Z 前)。
零 ROS 依赖。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def rt_to_matrix(rvec, tvec) -> np.ndarray:
    """(rvec, tvec) -> 4x4 齐次矩阵。"""
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return matrix


def matrix_to_rt(matrix) -> Tuple[np.ndarray, np.ndarray]:
    """4x4 -> (rvec, tvec)。"""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rvec, _ = cv2.Rodrigues(matrix[:3, :3])
    return rvec.reshape(3), matrix[:3, 3].copy()


def invert(matrix) -> np.ndarray:
    """刚体变换求逆 (不做通用矩阵求逆, 数值更稳)。"""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rotation = matrix[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ matrix[:3, 3]
    return inverse


def board_pose_in_camera(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs,
) -> Optional[np.ndarray]:
    """用标定板角点解 T_cam_board (4x4, 米)。失败返回 None。"""
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(object_points) < 4 or len(object_points) != len(image_points):
        return None
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points,
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(dist_coeffs, dtype=np.float64).reshape(-1),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    return rt_to_matrix(rvec, tvec)


def extrinsic_from_measured_board(
    base_from_board,
    camera_from_board,
) -> np.ndarray:
    """方法 A: 已知板子在 arm_base 下的位姿, 直接算 T_base_cam。"""
    return np.asarray(base_from_board, dtype=np.float64).reshape(4, 4) @ invert(camera_from_board)


def calibrate_eye_to_hand(
    base_from_gripper: Sequence[np.ndarray],
    camera_from_board: Sequence[np.ndarray],
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> np.ndarray:
    """方法 B: 多姿态解 T_base_cam (相机固定, 板子装在末端)。

    OpenCV 的 calibrateHandEye 解的是 eye-in-hand (相机在末端)。把机械臂位姿
    取逆喂进去, 同一套 AX=XB 就变成 eye-to-hand, 输出即 T_base_cam:
        T_gripper_base_i · X · T_cam_board_i = 常量 (板子相对末端固定)
    至少 3 个姿态, 实用上给 10~15 个、旋转角度差别拉开, 否则解不稳。
    """
    if len(base_from_gripper) != len(camera_from_board):
        raise ValueError("机械臂位姿与相机位姿数量不一致")
    if len(base_from_gripper) < 3:
        raise ValueError(f"至少需要 3 个姿态, 当前 {len(base_from_gripper)}")

    rotations_a, translations_a, rotations_b, translations_b = [], [], [], []
    for base_gripper, camera_board in zip(base_from_gripper, camera_from_board):
        gripper_base = invert(base_gripper)
        rotations_a.append(gripper_base[:3, :3])
        translations_a.append(gripper_base[:3, 3].reshape(3, 1))
        camera_board = np.asarray(camera_board, dtype=np.float64).reshape(4, 4)
        rotations_b.append(camera_board[:3, :3])
        translations_b.append(camera_board[:3, 3].reshape(3, 1))

    rotation, translation = cv2.calibrateHandEye(
        rotations_a, translations_a, rotations_b, translations_b, method=method
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return result


def eye_to_hand_residuals(
    base_from_camera,
    base_from_gripper: Sequence[np.ndarray],
    camera_from_board: Sequence[np.ndarray],
) -> dict:
    """一致性校验: 板子相对末端本应固定, 看各姿态算出来的散布有多大。

    位置散布 (米) 和角度散布 (度) 就是这次标定的实际精度上界, 别只看解出来好看。
    """
    base_from_camera = np.asarray(base_from_camera, dtype=np.float64).reshape(4, 4)
    estimates: List[np.ndarray] = [
        invert(base_gripper) @ base_from_camera @ np.asarray(camera_board, dtype=np.float64)
        for base_gripper, camera_board in zip(base_from_gripper, camera_from_board)
    ]
    translations = np.array([estimate[:3, 3] for estimate in estimates])
    mean_translation = translations.mean(axis=0)
    position_errors = np.linalg.norm(translations - mean_translation, axis=1)

    reference = estimates[0][:3, :3]
    angles = []
    for estimate in estimates:
        relative = reference.T @ estimate[:3, :3]
        cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
        angles.append(np.degrees(np.arccos(cosine)))
    return {
        "poses": len(estimates),
        "position_std_m": float(np.std(position_errors)),
        "position_max_m": float(np.max(position_errors)),
        "rotation_max_deg": float(np.max(angles)),
    }


def save_extrinsic(base_from_camera, path, note: str = "") -> str:
    """写 vision_tf 直接读的外参文件 (4x4, 米)。"""
    from pathlib import Path

    import yaml

    matrix = np.asarray(base_from_camera, dtype=np.float64).reshape(4, 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "note": note or "camera_optical -> arm_base, 4x4 齐次矩阵, 单位米",
        "arm_base_from_camera": {
            "rows": 4,
            "cols": 4,
            "data": matrix.reshape(-1).tolist(),
        },
    }
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, allow_unicode=True, sort_keys=False)
    return str(path)
