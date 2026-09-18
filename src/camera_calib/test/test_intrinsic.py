"""内参标定的纯算法测试: 用已知 K/畸变投影出合成观测, 看能不能反解回去。"""

import numpy as np
import pytest

from camera_calib.intrinsic import (
    CalibrationResult,
    Sample,
    calibrate,
    image_sharpness,
    point_coverage,
    save_camera_info,
    signature_distance,
    view_signature,
)

IMAGE_SIZE = (1280, 1024)
TRUE_K = np.array([[900.0, 0.0, 640.0], [0.0, 905.0, 512.0], [0.0, 0.0, 1.0]])
TRUE_DIST = np.array([-0.12, 0.05, 0.0, 0.0, 0.0])


def grid_object_points(columns=9, rows=6, spacing=0.025):
    points = np.zeros((rows * columns, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * spacing
    return points


def synthetic_samples(count=14):
    """绕不同角度、不同距离摆板子, 投影成像素点。"""
    import cv2

    object_points = grid_object_points()
    rng = np.random.default_rng(20260914)
    samples = []
    for index in range(count):
        rvec = rng.uniform(-0.35, 0.35, size=3)
        tvec = np.array([
            rng.uniform(-0.06, 0.06) - 0.10,
            rng.uniform(-0.05, 0.05) - 0.07,
            rng.uniform(0.45, 0.75),
        ])
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, TRUE_K, TRUE_DIST)
        samples.append(
            Sample(
                object_points=object_points.reshape(-1, 1, 3).astype(np.float32),
                image_points=projected.astype(np.float32),
                label=f"synthetic_{index:02d}",
            )
        )
    return samples


def test_calibrate_recovers_known_intrinsics():
    result = calibrate(synthetic_samples(16), IMAGE_SIZE, min_samples=10)

    assert isinstance(result, CalibrationResult)
    assert result.rms < 0.05                     # 无噪声数据应该几乎完美
    assert np.allclose(result.camera_matrix, TRUE_K, rtol=0.02)
    assert np.allclose(result.dist_coeffs.reshape(-1)[:2], TRUE_DIST[:2], atol=0.02)
    assert result.rejected == []


def test_calibrate_rejects_a_corrupted_view():
    samples = synthetic_samples(16)
    rng = np.random.default_rng(7)
    samples[5].image_points = samples[5].image_points + rng.normal(0, 12.0, samples[5].image_points.shape).astype(np.float32)

    result = calibrate(samples, IMAGE_SIZE, min_samples=10)

    assert 5 in result.rejected
    assert 5 not in result.kept
    assert np.allclose(result.camera_matrix, TRUE_K, rtol=0.02)


def test_calibrate_needs_enough_samples():
    with pytest.raises(ValueError, match="样本不足"):
        calibrate(synthetic_samples(3), IMAGE_SIZE, min_samples=15)


def test_sharpness_separates_blurred_from_sharp():
    import cv2

    sharp = np.zeros((200, 200), dtype=np.uint8)
    sharp[::10, :] = 255
    blurred = cv2.GaussianBlur(sharp, (21, 21), 8)

    assert image_sharpness(sharp) > image_sharpness(blurred) * 3


def test_coverage_grows_with_board_area():
    small = np.array([[[600.0, 500.0]], [[680.0, 500.0]], [[680.0, 560.0]], [[600.0, 560.0]]])
    large = np.array([[[100.0, 100.0]], [[1100.0, 100.0]], [[1100.0, 900.0]], [[100.0, 900.0]]])

    assert point_coverage(small, IMAGE_SIZE) < 0.02
    assert point_coverage(large, IMAGE_SIZE) > 0.5


def test_signature_distance_flags_repeated_views():
    samples = synthetic_samples(2)
    board_size = (8 * 0.025, 5 * 0.025)

    def signature_of(sample):
        return view_signature(
            sample.object_points, sample.image_points, IMAGE_SIZE, board_size
        )

    first = signature_of(samples[0])
    same = signature_of(samples[0])
    other = signature_of(samples[1])

    assert signature_distance(first, same) < 1e-9
    assert signature_distance(first, other) > 1e-3


def test_save_camera_info_is_readable_by_camera_package(tmp_path):
    import yaml

    result = calibrate(synthetic_samples(14), IMAGE_SIZE, min_samples=10)
    path = save_camera_info(result, IMAGE_SIZE, "engineer_camera", tmp_path / "camera_info.yaml")

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert document["image_width"] == IMAGE_SIZE[0]
    assert document["image_height"] == IMAGE_SIZE[1]
    assert document["camera_name"] == "engineer_camera"
    assert len(document["camera_matrix"]["data"]) == 9
    assert document["camera_matrix"]["data"][8] == pytest.approx(1.0)
