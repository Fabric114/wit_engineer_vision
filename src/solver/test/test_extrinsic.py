import numpy as np
import pytest
import yaml

from solver.extrinsic import (
    compose,
    is_identity,
    load_extrinsic,
    matrix_from_pose,
    pose_from_matrix,
    validate_transform,
)


def _rot_y(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _transform(rotation, translation):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def test_pose_matrix_round_trip():
    q_wxyz = np.array([0.8, 0.2, -0.4, 0.4])
    q_wxyz /= np.linalg.norm(q_wxyz)
    T = matrix_from_pose([0.1, -0.2, 0.3], q_wxyz)
    xyz, q_back = pose_from_matrix(T)
    np.testing.assert_allclose(xyz, [0.1, -0.2, 0.3])
    np.testing.assert_allclose(q_back * np.sign(q_back[0]), q_wxyz, atol=1e-9)
    np.testing.assert_allclose(matrix_from_pose(xyz, q_back), T, atol=1e-12)


def test_negated_quaternion_gives_same_matrix():
    q_wxyz = np.array([0.5, 0.5, 0.5, 0.5])
    np.testing.assert_allclose(matrix_from_pose([0, 0, 0], q_wxyz),
                               matrix_from_pose([0, 0, 0], -q_wxyz), atol=1e-12)


def test_quaternion_order_is_wxyz():
    # 绕 z 转 90°: wxyz = (cos45, 0, 0, sin45); 若误按 xyzw 解读会得到绕 x 转 180° 附近的姿态。
    half = np.sqrt(0.5)
    R = matrix_from_pose([0, 0, 0], [half, 0.0, 0.0, half])[:3, :3]
    np.testing.assert_allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


def test_pose_from_matrix_handles_180_degree_rotation():
    T = _transform(np.diag([1.0, -1.0, -1.0]), [0, 0, 0])
    _, q = pose_from_matrix(T)
    np.testing.assert_allclose(matrix_from_pose([0, 0, 0], q), T, atol=1e-9)


def test_compose_hand_computed_example():
    # 相机在 arm_base 下平移 (0.1, 0, 0.3)、绕 y 转 90°; 站体在相机前方 (z) 0.5 m。
    T_base_cam = _transform(_rot_y(np.pi / 2), [0.1, 0.0, 0.3])
    T_cam_station = _transform(np.eye(3), [0.0, 0.0, 0.5])
    T = compose(T_base_cam, T_cam_station)
    np.testing.assert_allclose(T[:3, 3], [0.6, 0.0, 0.3], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], _rot_y(np.pi / 2), atol=1e-12)


def test_compose_matches_sim_truth():
    T_base_cam = _transform(_rot_y(0.3) @ _rot_y(0.0), [0.05, -0.02, 0.4])
    q = np.array([0.9, 0.1, 0.3, -0.2])
    T_base_station_truth = matrix_from_pose([0.55, 0.03, 0.25], q / np.linalg.norm(q))
    T_cam_station = np.linalg.inv(T_base_cam) @ T_base_station_truth
    np.testing.assert_allclose(compose(T_base_cam, T_cam_station), T_base_station_truth, atol=1e-9)


@pytest.mark.parametrize("bad", [
    np.eye(3),                                        # 形状
    _transform(np.diag([1.0, 1.0, -1.0]), [0, 0, 0]),  # det = -1 (镜像)
    _transform(np.diag([1.0, 2.0, 1.0]), [0, 0, 0]),   # 非正交
    np.vstack([np.eye(4)[:3], [0.0, 0.0, 1.0, 1.0]]),  # 末行
    _transform(np.eye(3), [np.nan, 0, 0]),             # NaN
])
def test_validate_transform_rejects_bad_matrices(bad):
    with pytest.raises(ValueError, match="T_test"):
        validate_transform(bad, "T_test")


def test_is_identity():
    assert is_identity(np.eye(4))
    assert not is_identity(_transform(np.eye(3), [0.001, 0, 0]))


def test_load_extrinsic_inline():
    T = _transform(_rot_y(0.2), [0.1, 0.2, 0.3])
    cfg = {"extrinsic_file": "",
           "arm_base_from_camera": {"rows": 4, "cols": 4, "data": T.reshape(-1).tolist()}}
    np.testing.assert_allclose(load_extrinsic(cfg, base_dir="."), T)


def test_load_extrinsic_reads_hand_eye_output(tmp_path, monkeypatch):
    # 直接用标定库写文件, 防止 hand_eye.save_extrinsic 与 load_extrinsic 的格式漂移。
    from pathlib import Path
    import sys

    monkeypatch.setattr(sys, "path", [str(Path(__file__).resolve().parents[2] / "camera_calib"), *sys.path])
    from camera_calib.hand_eye import save_extrinsic

    T = _transform(_rot_y(-0.4), [0.2, 0.0, 0.5])
    save_extrinsic(T, tmp_path / "calib" / "extrinsic.yaml")
    cfg = {"extrinsic_file": "calib/extrinsic.yaml",
           "arm_base_from_camera": {"rows": 4, "cols": 4, "data": np.eye(4).reshape(-1).tolist()}}
    np.testing.assert_allclose(load_extrinsic(cfg, base_dir=tmp_path), T)


def test_load_extrinsic_rejects_invalid_matrix():
    cfg = {"arm_base_from_camera": {"rows": 4, "cols": 4, "data": [2.0] + [0.0] * 15}}
    with pytest.raises(ValueError, match="arm_base_from_camera"):
        load_extrinsic(cfg, base_dir=".")


def test_repo_extrinsic_config_loads():
    from pathlib import Path

    config = Path(__file__).resolve().parents[3] / "config" / "camera_extrinsic.yaml"
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert cfg["camera_frame"] == "camera"
    assert cfg["base_frame"] == "arm_base"
    assert load_extrinsic(cfg, base_dir=config.parent).shape == (4, 4)
