"""RMUC2026 规则位姿 -> 站体位姿 的换算 (sim/station_pose.py)。

    PYTHONPATH=src/runtime/sim python3 -m pytest src/runtime/sim/test/test_station_pose.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.station_pose import base_T_station, check_level3, station_in_O

INITIAL = {"x": -200, "y": 200, "z": 800, "theta": 0, "phi": 90, "alpha": 0}


def test_initial_pose_is_upright_facing_plus_x():
    """规则初始位姿: 装配柱水平朝 +X (朝机器人), 平移方向竖直向上。"""
    transform = station_in_O(INITIAL)
    np.testing.assert_allclose(transform[:3, 3], [-0.2, 0.2, 0.8])
    np.testing.assert_allclose(transform[:3, 0], [1, 0, 0], atol=1e-12)   # x = e
    np.testing.assert_allclose(transform[:3, 2], [0, 0, 1], atol=1e-12)   # z = 平移方向


@pytest.mark.parametrize("theta,phi", [(0, 0), (30, 60), (-45, 45), (90, 90)])
def test_insertion_axis_matches_rule_spherical_angles(theta, phi):
    e = station_in_O(dict(INITIAL, theta=theta, phi=phi))[:3, 0]
    t, p = np.deg2rad(theta), np.deg2rad(phi)
    np.testing.assert_allclose(e, [np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)], atol=1e-12)


def test_alpha_rotates_about_insertion_axis():
    a = station_in_O(dict(INITIAL, alpha=0))
    b = station_in_O(dict(INITIAL, alpha=30))
    np.testing.assert_allclose(a[:3, 0], b[:3, 0], atol=1e-12)
    assert np.degrees(np.arccos(a[:3, 2] @ b[:3, 2])) == pytest.approx(30.0)


def test_level3_range_check():
    assert check_level3({"x": -50, "y": 200, "z": 600, "theta": 0, "phi": 90, "alpha": 0}) == []
    assert check_level3(INITIAL)                       # 初始位姿 x=-200 / z=800 不在三级范围
    assert any("alpha" in p for p in check_level3({"x": -50, "y": 200, "z": 600, "theta": 0, "phi": 90, "alpha": 90}))


def test_station_in_base_frame_when_facing_core():
    """车头正对科技核心 (yaw=180): 站体在车头前方, 装配柱朝车。"""
    pose = {"x": -50, "y": 200, "z": 600, "theta": 0, "phi": 90, "alpha": 0}
    station = base_T_station(pose, (0.5, 0.2, 180.0), 0.155)
    np.testing.assert_allclose(station[:3, 3], [0.55, 0.0, 0.445], atol=1e-12)
    np.testing.assert_allclose(station[:3, 0], [-1, 0, 0], atol=1e-12)   # 装配柱指向车 (-x_base)
