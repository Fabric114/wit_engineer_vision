import cv2
import numpy as np
import pytest

from solver.solver import PnPEstimator, PoseEstimate


@pytest.fixture
def camera_data():
    object_points = np.array(
        [[-0.10, -0.08, 0.0], [0.10, -0.08, 0.0], [0.10, 0.08, 0.0],
         [-0.10, 0.08, 0.0], [0.0, 0.0, 0.05]], dtype=np.float64)
    camera_matrix = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    distortion = np.zeros(5, dtype=np.float64)
    rvec = np.array([[0.1], [-0.08], [0.2]])
    tvec = np.array([[0.03], [-0.02], [0.8]])
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    return object_points, image_points.reshape(-1, 2), camera_matrix, distortion, tvec[:, 0]


def test_synthetic_projection_recovers_pose(camera_data):
    object_points, image_points, K, D, expected_t = camera_data
    result = PnPEstimator(reprojection_error_threshold_px=1.0).estimate(
        object_points, image_points, K, D
    )
    assert result.valid
    assert result.reprojection_error_px < 1.0
    np.testing.assert_allclose(result.T_camera_exchange_station[:3, 3], expected_t, atol=2e-3)


def test_validation_failures_are_diagnostic(camera_data):
    object_points, image_points, K, D, _ = camera_data
    estimator = PnPEstimator()
    cases = [
        (object_points[:3], image_points[:3], K, "at least four"),
        (object_points, image_points[:-1], K, "shape"),
        (object_points, image_points, np.full((3, 3), np.nan), "finite"),
        (object_points, image_points, np.zeros((3, 3)), "camera matrix"),
    ]
    for points, pixels, matrix, expected in cases:
        result = estimator.estimate(points, pixels, matrix, D)
        assert isinstance(result, PoseEstimate)
        assert not result.valid
        assert result.reason
        assert expected.lower() in result.reason.lower()


def test_confidence_filtering_requires_four_points(camera_data):
    object_points, image_points, K, D, _ = camera_data
    confidence = np.array([1.0, 1.0, 1.0, 0.1, 0.1])
    result = PnPEstimator(confidence_threshold=0.5).estimate(
        object_points, image_points, K, D, confidence
    )
    assert not result.valid
    assert "four" in result.reason.lower()


def test_rejects_degenerate_geometry(camera_data):
    _, image_points, K, D, _ = camera_data
    points = np.zeros((5, 3), dtype=np.float64)
    result = PnPEstimator().estimate(points, image_points, K, D)
    assert not result.valid
    assert "degenerate" in result.reason.lower()


def test_accepts_coplanar_geometry(camera_data):
    _, _, K, D, expected_t = camera_data
    # 兑换口 7 点全部共面; 4 个以上不共线的共面点 PnP 可解。
    points = np.array([
        [-0.1, -0.08, 0.0], [0.1, -0.08, 0.0], [0.1, 0.08, 0.0],
        [-0.1, 0.08, 0.0], [0.03, 0.01, 0.0],
    ])
    pixels, _ = cv2.projectPoints(points, np.array([[0.1], [-0.08], [0.2]]), expected_t.reshape(3, 1), K, D)
    result = PnPEstimator(reprojection_error_threshold_px=1.0).estimate(points, pixels.reshape(-1, 2), K, D)
    assert result.valid, result.reason
    np.testing.assert_allclose(result.T_camera_exchange_station[:3, 3], expected_t, atol=2e-3)


def test_rejects_collinear_geometry(camera_data):
    _, image_points, K, D, _ = camera_data
    points = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0],
                       [0.15, 0.0, 0.0], [0.2, 0.0, 0.0]])
    result = PnPEstimator().estimate(points, image_points, K, D)
    assert not result.valid
    assert "degenerate" in result.reason.lower()


def test_rejects_min_inliers_below_four():
    with pytest.raises(ValueError, match="min_inliers"):
        PnPEstimator(min_inliers=3)


def test_rejects_excessive_reprojection_error(camera_data):
    object_points, image_points, K, D, _ = camera_data
    noisy = image_points.copy()
    noisy[0] += [100.0, -80.0]
    result = PnPEstimator(reprojection_error_threshold_px=1.0).estimate(
        object_points, noisy, K, D
    )
    assert not result.valid
    assert "reprojection" in result.reason.lower()
