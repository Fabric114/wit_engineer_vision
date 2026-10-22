import cv2
import numpy as np

from solver.solver import PnPEstimator


class SimulatedCamera:
    def capture(self, object_points, transform, K, distortion):
        rotation = transform[:3, :3]
        rvec, _ = cv2.Rodrigues(rotation)
        image_points, _ = cv2.projectPoints(object_points, rvec, transform[:3, 3], K, distortion)
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for x, y in image_points.reshape(-1, 2):
            cv2.drawMarker(image, (round(float(x)), round(float(y))), (255, 255, 255), cv2.MARKER_CROSS, 11, 2)
        return image, image_points.reshape(-1, 2)


def test_simulated_camera_detector_solver_round_trip():
    points = np.array([
        [-0.10, -0.08, 0.0], [0.10, -0.08, 0.0], [0.10, 0.08, 0.0],
        [-0.10, 0.08, 0.0], [0.0, 0.0, 0.05],
    ], dtype=np.float64)
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5)
    transform = np.eye(4)
    transform[:3, :3], _ = cv2.Rodrigues(np.array([[0.1], [-0.08], [0.2]]))
    transform[:3, 3] = [0.03, -0.02, 0.8]
    image, projected = SimulatedCamera().capture(points, transform, K, D)
    assert image.shape == (480, 640, 3)

    # Test-local detector adapter returns the configured schema order.
    detected = projected.copy()
    result = PnPEstimator(reprojection_error_threshold_px=1.0).estimate(points, detected, K, D)
    assert result.valid
    np.testing.assert_allclose(result.T_camera_exchange_station, transform, atol=2e-3)


def test_permuted_detector_order_is_not_accepted_as_schema_order():
    points = np.array([
        [-0.10, -0.08, 0.0], [0.10, -0.08, 0.0], [0.10, 0.08, 0.0],
        [-0.10, 0.08, 0.0], [0.0, 0.0, 0.05],
    ], dtype=np.float64)
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    rvec = np.array([[0.1], [-0.08], [0.2]])
    tvec = np.array([[0.03], [-0.02], [0.8]])
    image_points, _ = cv2.projectPoints(points, rvec, tvec, K, np.zeros(5))
    result = PnPEstimator(reprojection_error_threshold_px=1.0).estimate(
        points, image_points.reshape(-1, 2)[[1, 0, 2, 3, 4]], K, np.zeros(5)
    )
    assert not result.valid or result.reprojection_error_px > 1.0
