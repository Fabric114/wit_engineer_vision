"""RMUC2026 规则位姿 -> 站体位姿 的换算 (sim/station_pose.py)。

    PYTHONPATH=src/runtime/sim python3 -m pytest src/runtime/sim/test/test_station_pose.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path

from sim.station_pose import base_T_station, check_level3, station_in_O

MODEL = Path(__file__).resolve().parents[1] / "model"

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


LEVEL_KEYS = ("x", "y", "z", "theta", "phi", "alpha")


@pytest.fixture(scope="module")
def scene():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL / "scene.xml"))
    adr = [int(model.jnt_qposadr[model.joint(f"level_{k}").id]) for k in LEVEL_KEYS]
    return mujoco, model, adr


def _station_in_O_from_joints(mujoco, model, data, adr, pose: dict) -> np.ndarray:
    """把规则值原样写进 6 个 level 关节, 读 E 在 O_frame 下的位姿。"""
    data.qpos[adr] = [pose["x"] / 1000, pose["y"] / 1000, pose["z"] / 1000,
                      *np.deg2rad([pose["theta"], pose["phi"], pose["alpha"]])]
    mujoco.mj_kinematics(model, data)

    def pose_of(name):
        T = np.eye(4)
        T[:3, :3] = data.body(name).xmat.reshape(3, 3)
        T[:3, 3] = data.body(name).xpos
        return T

    return np.linalg.inv(pose_of("O_frame")) @ pose_of("exchange_station")


def test_level_joint_ranges_are_rule_level3(scene):
    """6 个 level 关节的 range = 表 5-10 三级范围 (换成 m / rad)。"""
    mujoco, model, _ = scene
    from sim.station_pose import LEVEL3_RANGES
    for k in LEVEL_KEYS:
        lo, hi = LEVEL3_RANGES[k]
        scale = 1e-3 if k in ("x", "y", "z") else np.pi / 180
        np.testing.assert_allclose(model.jnt_range[model.joint(f"level_{k}").id], (lo * scale, hi * scale), atol=1e-9)


def test_level_qpos0_is_rule_initial_orientation_inside_range(scene):
    """qpos0 (= ref) 合法: mj_resetData 后关节不越限, 姿态是规则初始位姿 θ=0 φ=90 α=0。"""
    mujoco, model, adr = scene
    q0 = model.qpos0[adr]
    np.testing.assert_allclose(q0, [-0.05, 0.2, 0.6, 0.0, np.pi / 2, 0.0], atol=1e-12)
    ids = [model.joint(f"level_{k}").id for k in LEVEL_KEYS]
    assert np.all(model.jnt_range[ids, 0] <= q0) and np.all(q0 <= model.jnt_range[ids, 1])


def test_level_joints_match_station_in_O(scene):
    """核心: 关节 qpos 直接填规则值, 摆出的 E 与 station_in_O 一致 (尤其 φ 不需要 90°-φ 换算)。"""
    mujoco, model, adr = scene
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    from sim.station_pose import LEVEL3_RANGES
    poses = [dict(x=0, y=200, z=600, theta=0, phi=90, alpha=0),     # 初始: 装配柱水平朝 +X
             dict(x=0, y=200, z=600, theta=0, phi=0, alpha=0)]      # e 竖直朝上
    poses += [{k: rng.uniform(*LEVEL3_RANGES[k]) for k in LEVEL_KEYS} for _ in range(500)]
    for pose in poses:
        np.testing.assert_allclose(_station_in_O_from_joints(mujoco, model, data, adr, pose),
                                   station_in_O(pose), atol=1e-9, err_msg=str(pose))


def test_station_holds_still_under_gravity(scene):
    """gravcomp: 只在启动时写一次 qpos, 之后站体不会在重力下滑走。"""
    mujoco, model, adr = scene
    data = mujoco.MjData(model)
    q = np.array([-0.08, 0.15, 0.55, 0.3, 0.4, -0.2])
    data.qpos[adr] = q
    for _ in range(3000):
        mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos[adr], q, atol=1e-6)


def test_station_in_base_frame_when_facing_core():
    """车头正对科技核心 (yaw=180): 站体在车头前方, 装配柱朝车。"""
    pose = {"x": -50, "y": 200, "z": 600, "theta": 0, "phi": 90, "alpha": 0}
    station = base_T_station(pose, (0.5, 0.2, 180.0), 0.155)
    np.testing.assert_allclose(station[:3, 3], [0.55, 0.0, 0.445], atol=1e-12)
    np.testing.assert_allclose(station[:3, 0], [-1, 0, 0], atol=1e-12)   # 装配柱指向车 (-x_base)
