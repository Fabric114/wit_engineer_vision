"""科技核心 (exchange_column) 的 level 3 关节与"跟随 TCP"逻辑。

纯 MuJoCo + numpy, 不需要 ROS:
    PYTHONPATH=src/runtime/sim python3 -m pytest src/runtime/sim/test/test_exchange_column.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sim.column import ColumnFollower

MODEL = Path(__file__).resolve().parents[1] / "model"
SLIDE_RANGE = (0.0, 0.12)


def _follower() -> ColumnFollower:
    return ColumnFollower(engage_radius_m=0.01, release_radius_m=0.03, slide_range=SLIDE_RANGE)


def test_not_engaged_until_inserted():
    """矿石还在站外 (轴向退开 / 侧偏) 时, 科技核心不动。"""
    f = _follower()
    assert f.update(np.array([0.15, 0.0, 0.05])) == 0.0
    assert f.update(np.array([0.0, 0.05, 0.05])) == 0.0
    assert not f.engaged


def test_follows_slide_once_inserted():
    f = _follower()
    assert f.update(np.array([0.002, -0.001, 0.0])) == 0.0
    assert f.engaged
    assert f.update(np.array([0.003, 0.002, 0.06])) == pytest.approx(0.06)
    assert f.update(np.array([0.0, 0.0, 0.104])) == pytest.approx(0.104)


def test_clipped_to_joint_range():
    f = _follower()
    f.update(np.array([0.0, 0.0, 0.0]))
    assert f.update(np.array([0.0, 0.0, -0.02])) == 0.0
    assert f.update(np.array([0.0, 0.0, 0.5])) == pytest.approx(SLIDE_RANGE[1])


def test_hysteresis_tolerates_tracking_error_then_releases():
    """跟踪误差 (超 engage、没超 release) 不脱开; 真拔出来 (超 release) 才脱开并停在原位。"""
    f = _follower()
    f.update(np.array([0.0, 0.0, 0.0]))
    assert f.update(np.array([0.02, 0.0, 0.05])) == pytest.approx(0.05)
    assert f.engaged
    assert f.update(np.array([0.08, 0.0, 0.09])) == pytest.approx(0.05)
    assert not f.engaged


def test_rejects_bad_radii():
    with pytest.raises(ValueError):
        ColumnFollower(engage_radius_m=0.03, release_radius_m=0.01, slide_range=SLIDE_RANGE)


@pytest.fixture(scope="module")
def scene():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(MODEL / "scene.xml"))
    return mujoco, model


def test_scene_has_level3_column_joints(scene):
    mujoco, model = scene
    expected = {"step3_slide": SLIDE_RANGE, "p_axis": (0.0, 1.5708), "q_axis": (-1.5708, 1.5708)}
    for name, (lo, hi) in expected.items():
        joint = model.joint(name)
        np.testing.assert_allclose(model.jnt_range[joint.id], (lo, hi), atol=1e-9)


def test_column_slides_along_station_z(scene):
    mujoco, model = scene
    data = mujoco.MjData(model)
    station, column = model.body("exchange_station").id, model.body("exchange_column").id

    def column_in_station():
        mujoco.mj_forward(model, data)
        rotation = data.xmat[station].reshape(3, 3)
        return rotation.T @ (data.xpos[column] - data.xpos[station])

    np.testing.assert_allclose(column_in_station(), 0.0, atol=1e-9)   # 全 0 = 原静止位置
    data.qpos[model.joint("step3_slide").qposadr[0]] = 0.08
    np.testing.assert_allclose(column_in_station(), [0.0, 0.0, 0.08], atol=1e-9)


def test_column_holds_still_without_driver(scene):
    """gravcomp: 没人写 qpos 时科技核心也不会自己掉下去。"""
    mujoco, model = scene
    data = mujoco.MjData(model)
    adr = model.joint("step3_slide").qposadr[0]
    data.qpos[adr] = 0.06
    for _ in range(1000):
        mujoco.mj_step(model, data)
    assert abs(data.qpos[adr] - 0.06) < 1e-4


def test_ore_is_attached_to_link6(scene):
    mujoco, model = scene
    geom = model.geom("energy_unit")
    assert model.geom_bodyid[geom.id] == model.body("link_6").id
    assert model.geom_contype[geom.id] == 0 and model.geom_conaffinity[geom.id] == 0
