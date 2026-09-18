"""外参标定的纯算法测试: 造一个真值 T_base_cam, 走一遍 eye-to-hand 看能不能解回来。"""

import cv2
import numpy as np
import pytest

from camera_calib.hand_eye import (
    board_pose_in_camera,
    calibrate_eye_to_hand,
    extrinsic_from_measured_board,
    eye_to_hand_residuals,
    invert,
    matrix_to_rt,
    rt_to_matrix,
    save_extrinsic,
)

K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 512.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)


def make_pose(rpy, xyz):
    rotation = cv2.Rodrigues(np.asarray(rpy, dtype=np.float64))[0]
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz
    return matrix


TRUE_BASE_CAM = make_pose([0.05, -2.2, 0.1], [0.35, -0.02, 0.42])
TRUE_GRIPPER_BOARD = make_pose([0.02, 0.03, 0.6], [0.01, 0.0, 0.06])


def test_matrix_round_trip():
    matrix = make_pose([0.3, -0.2, 0.7], [0.1, 0.2, 0.3])

    rvec, tvec = matrix_to_rt(matrix)

    assert np.allclose(rt_to_matrix(rvec, tvec), matrix, atol=1e-9)


def test_invert_is_rigid_body_inverse():
    matrix = make_pose([0.4, 0.1, -0.9], [0.5, -0.3, 1.2])

    assert np.allclose(invert(matrix) @ matrix, np.eye(4), atol=1e-9)
    # 刚体逆是转置+平移, 不该退化成通用求逆的数值噪声
    assert np.allclose(invert(matrix)[:3, :3], matrix[:3, :3].T, atol=1e-12)


def test_extrinsic_from_measured_board_matches_definition():
    base_board = make_pose([0.0, 0.0, 0.2], [0.6, 0.1, 0.0])
    cam_board = make_pose([0.1, 0.2, 0.0], [0.0, 0.05, 0.5])

    base_cam = extrinsic_from_measured_board(base_board, cam_board)

    assert np.allclose(base_cam @ cam_board, base_board, atol=1e-9)


def test_board_pose_in_camera_recovers_projection_pose():
    object_points = np.zeros((24, 3))
    object_points[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2) * 0.025
    truth = make_pose([0.15, -0.2, 0.05], [-0.05, -0.03, 0.5])
    rvec, tvec = matrix_to_rt(truth)
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, DIST)

    estimated = board_pose_in_camera(object_points, image_points, K, DIST)

    assert estimated is not None
    assert np.allclose(estimated, truth, atol=1e-6)


def test_board_pose_in_camera_returns_none_when_underdetermined():
    assert board_pose_in_camera(np.zeros((3, 3)), np.zeros((3, 2)), K, DIST) is None


def synthetic_eye_to_hand(count=12, noise_m=0.0):
    """相机固定在 arm_base 上, 板子装末端: 摆 count 个姿态生成配对数据。"""
    rng = np.random.default_rng(11)
    base_grippers, camera_boards = [], []
    for _ in range(count):
        base_gripper = make_pose(
            rng.uniform(-0.6, 0.6, size=3),
            [rng.uniform(0.25, 0.45), rng.uniform(-0.15, 0.15), rng.uniform(0.15, 0.4)],
        )
        camera_board = invert(TRUE_BASE_CAM) @ base_gripper @ TRUE_GRIPPER_BOARD
        if noise_m:
            camera_board[:3, 3] += rng.normal(0.0, noise_m, size=3)
        base_grippers.append(base_gripper)
        camera_boards.append(camera_board)
    return base_grippers, camera_boards


def test_calibrate_eye_to_hand_recovers_truth():
    base_grippers, camera_boards = synthetic_eye_to_hand()

    base_cam = calibrate_eye_to_hand(base_grippers, camera_boards)

    assert np.allclose(base_cam[:3, 3], TRUE_BASE_CAM[:3, 3], atol=1e-6)
    assert np.allclose(base_cam[:3, :3], TRUE_BASE_CAM[:3, :3], atol=1e-6)


def test_residuals_are_tiny_on_clean_data_and_grow_with_noise():
    base_grippers, camera_boards = synthetic_eye_to_hand()
    clean = eye_to_hand_residuals(
        calibrate_eye_to_hand(base_grippers, camera_boards), base_grippers, camera_boards
    )

    noisy_arms, noisy_boards = synthetic_eye_to_hand(noise_m=0.004)
    noisy = eye_to_hand_residuals(
        calibrate_eye_to_hand(noisy_arms, noisy_boards), noisy_arms, noisy_boards
    )

    assert clean["position_max_m"] < 1e-6
    assert noisy["position_max_m"] > clean["position_max_m"]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_calibrate_eye_to_hand_needs_three_poses(count):
    base_grippers, camera_boards = synthetic_eye_to_hand(count=max(count, 1))

    with pytest.raises(ValueError):
        calibrate_eye_to_hand(base_grippers[:count], camera_boards[:count])


def test_mismatched_input_lengths_are_rejected():
    base_grippers, camera_boards = synthetic_eye_to_hand()

    with pytest.raises(ValueError, match="数量不一致"):
        calibrate_eye_to_hand(base_grippers, camera_boards[:-1])


def test_save_extrinsic_writes_meters_matrix(tmp_path):
    import yaml

    path = save_extrinsic(TRUE_BASE_CAM, tmp_path / "extrinsic.yaml", note="unit test")

    document = yaml.safe_load(open(path, encoding="utf-8"))
    block = document["arm_base_from_camera"]
    assert (block["rows"], block["cols"]) == (4, 4)
    matrix = np.asarray(block["data"], dtype=np.float64).reshape(4, 4)
    assert np.allclose(matrix, TRUE_BASE_CAM, atol=1e-9)
