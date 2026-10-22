import cv2
import numpy as np
import pytest

from solver.pairing import PartObservation, merge_candidates, solve_best_candidate
from solver.schema import KeypointSchema
from solver.solver import PnPEstimator

K = np.array([[1200.0, 0.0, 720.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]])
D = np.zeros(5)
NAMES = ("TL", "TR", "BL", "BR", "ring",
         "light_BR", "light_TR", "shell_R", "shell_M", "shell_L", "light_TL", "light_BL")
POINTS = np.array([
    [-0.1, -0.04, 0.04], [-0.1, 0.04, 0.04], [-0.1, -0.04, -0.04], [-0.1, 0.04, -0.04], [0.0, 0.0, 0.0],
    [-0.112, 0.11487, 0.035953], [-0.112, 0.11487, 0.167312], [-0.112, 0.084871, 0.276449],
    [-0.112, 0.0, 0.325449], [-0.112, -0.084871, 0.276449], [-0.112, -0.11487, 0.167312],
    [-0.112, -0.11487, 0.035953],
])
PILLAR, EXCHANGE = 0, 1


@pytest.fixture
def schema():
    return KeypointSchema("exchange_station", None, NAMES, POINTS,
                          {PILLAR: tuple(range(5)), EXCHANGE: tuple(range(5, 12))})


def _pose(tx=0.02, ty=-0.05, tz=0.9):
    # 相机光轴大致沿 exchange_station 的 +X, 兑换面朝向相机。
    rotation = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    rvec, _ = cv2.Rodrigues(rotation @ cv2.Rodrigues(np.array([0.05, -0.08, 0.03]))[0])
    return rvec, np.array([[tx], [ty], [tz]])


def _instance(class_id, part, rvec, tvec):
    """模拟 tc 实例: 12 槽全部输出, 只有本类别段有效, 其余槽是低置信度 padding。"""
    pixels, _ = cv2.projectPoints(POINTS, rvec, tvec, K, D)
    pixels = pixels.reshape(-1, 2)
    confidence = np.full(12, 0.01)
    confidence[list(part)] = 0.9
    padded = np.where(confidence[:, None] > 0.5, pixels, 0.0)
    return PartObservation(class_id, padded, confidence)


def test_merges_pillar_and_exchange_into_full_schema(schema):
    rvec, tvec = _pose()
    obs = [_instance(PILLAR, range(5), rvec, tvec), _instance(EXCHANGE, range(5, 12), rvec, tvec)]

    best = merge_candidates(schema, obs)[0]

    assert best.sources == (0, 1)
    assert np.all(best.confidence > 0.5)


def test_merged_candidate_recovers_pose(schema):
    rvec, tvec = _pose()
    obs = [_instance(EXCHANGE, range(5, 12), rvec, tvec), _instance(PILLAR, range(5), rvec, tvec)]

    solution = solve_best_candidate(schema, obs, PnPEstimator(reprojection_error_threshold_px=1.0), K, D)

    assert solution.estimate is not None, solution.reasons
    assert solution.candidate.sources == (1, 0)
    assert solution.estimate.inlier_mask.all()
    np.testing.assert_allclose(solution.estimate.T_camera_exchange_station[:3, 3], tvec[:, 0], atol=2e-3)


def test_single_part_falls_back_to_available_points(schema):
    rvec, tvec = _pose()
    obs = [_instance(EXCHANGE, range(5, 12), rvec, tvec)]

    solution = solve_best_candidate(schema, obs, PnPEstimator(reprojection_error_threshold_px=1.0), K, D)

    assert solution.estimate is not None, solution.reasons
    assert solution.candidate.sources == (None, 0)
    assert not solution.estimate.inlier_mask[:5].any()
    np.testing.assert_allclose(solution.estimate.T_camera_exchange_station[:3, 3], tvec[:, 0], atol=5e-3)


def test_pairs_consistent_instances_when_two_stations_are_visible(schema):
    near, far = _pose(), _pose(tx=0.35, tz=1.4)
    obs = [
        _instance(PILLAR, range(5), *far),
        _instance(PILLAR, range(5), *near),
        _instance(EXCHANGE, range(5, 12), *near),
    ]

    solution = solve_best_candidate(schema, obs, PnPEstimator(reprojection_error_threshold_px=1.0), K, D)

    assert solution.estimate is not None, solution.reasons
    assert solution.candidate.sources == (1, 2)


def test_ignores_unknown_class_and_reports_when_nothing_solves(schema):
    rvec, tvec = _pose()
    stray = _instance(7, range(5), rvec, tvec)

    assert merge_candidates(schema, [stray]) == []
    solution = solve_best_candidate(schema, [stray], PnPEstimator(), K, D)
    assert solution.estimate is None
    assert solution.reasons


def test_rejects_observation_with_wrong_slot_count(schema):
    bad = PartObservation(PILLAR, np.zeros((5, 2)), np.ones(5))
    with pytest.raises(ValueError, match="slot"):
        merge_candidates(schema, [bad])
