"""仓库根 config/ 中真实 schema 与 detector 配置的一致性检查。"""

from pathlib import Path

import cv2
import numpy as np
import yaml

from solver.schema import load_schema
from solver.solver import PnPEstimator

CONFIG = Path(__file__).resolve().parents[3] / "config"


def _tc_config():
    with open(CONFIG / "detector.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["tc"]


def test_detector_schema_matches_keypoint_order_and_classes():
    tc = _tc_config()
    schema = load_schema(CONFIG / "keypoint_schema.yaml", tc["schema"])
    assert list(schema.names) == list(tc["keypoint_names"])
    assert set(schema.parts) == set(tc["class_names"])
    names = {v: k for k, v in tc["class_names"].items()}
    assert [schema.names[i] for i in schema.parts[names["pillar"]]] == ["TL", "TR", "BL", "BR", "ring"]
    assert len(schema.parts[names["exchange"]]) == 7


def test_detector_schema_recovers_synthetic_pose():
    schema = load_schema(CONFIG / "keypoint_schema.yaml", _tc_config()["schema"])
    K = np.array([[1200.0, 0.0, 720.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5)
    rotation = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    rvec, _ = cv2.Rodrigues(rotation)
    tvec = np.array([[0.02], [-0.05], [0.9]])
    pixels, _ = cv2.projectPoints(schema.object_points, rvec, tvec, K, D)

    result = PnPEstimator(reprojection_error_threshold_px=1.0).estimate(
        schema.object_points, pixels.reshape(-1, 2), K, D
    )

    assert result.valid, result.reason
    np.testing.assert_allclose(result.T_camera_exchange_station[:3, 3], tvec[:, 0], atol=2e-3)
