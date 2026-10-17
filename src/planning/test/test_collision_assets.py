# test_collision_assets.py — 站体碰撞网格能加载, 碰撞判定与 hppfcl 精确距离一致。
"""test_collision_assets.py — 站体碰撞网格能加载, 且碰撞判定与精确距离一致。

站体 .obj 在 src/runtime/sim/model/exchange_station/ (config 的 collision.station.asset_dir)。
这个测试防的是"资产又丢了": planning_node 构不出碰撞模型时只打 error 日志然后关掉碰撞,
不会让任何东西失败, 所以得有一条测试守着。
"""

from __future__ import annotations

import numpy as np
import pytest

hppfcl = pytest.importorskip("hppfcl")

from planning import ArmModel, load_config  # noqa: E402
from planning.collision import CollisionChecker, CollisionModel  # noqa: E402


# fixture: (ArmModel, CollisionModel, 原始 config), 整模块共用。
@pytest.fixture(scope="module")
def _setup():
    config = load_config()
    return ArmModel.from_config(config["arm"]), CollisionModel.from_config(config["collision"]), config


# 期望: 站体网格非空且文件都存在; 局部交换网格 (桩体, 允许为空) 的文件也都存在。
def test_station_assets_load(_setup):
    _, model, _ = _setup
    assert model.station_meshes and all(path.is_file() for path in model.station_meshes)
    assert all(path.is_file() for path in model.local_exchange_meshes)


# 期望: 零位下站体放在远处 (3,3,3) 不碰撞; 站体原点放到 link_4 原点处必碰撞。
def test_far_station_is_free_and_overlapping_station_collides(_setup):
    arm, model, _ = _setup
    # 远处站体: 应判无碰撞
    joints = np.zeros((1, 6))
    far = np.eye(4)
    far[:3, 3] = [3.0, 3.0, 3.0]
    assert not CollisionChecker(arm, [(model.station_meshes, far)], model.arm_capsules).check_configs(joints)[0]
    # 站体原点放到 link_4 的 DH 原点上: 站体包住了这段臂, 必撞
    overlap = np.eye(4)
    overlap[:3, 3] = arm.forward_kinematics(joints).link_transforms[0, 4][:3, 3]
    checker = CollisionChecker(arm, [(model.station_meshes, overlap)], model.arm_capsules)
    assert checker.check_configs(joints)[0]


# 期望: 60 个随机样本逐个满足 collides == (精确最小距离 <= 0); 无碰撞时 lower_bound <= 精确距离。
def test_collision_flag_matches_exact_distance(_setup):
    """随机 (q, 站体位姿) 下, collide 的判定与 hppfcl.distance 一致, 且距离下界不高估。"""
    arm, model, config = _setup
    limits = config["arm"]["joint_limits"]
    rng = np.random.default_rng(1)
    joints = rng.uniform(limits["lower"], limits["upper"], (60, 6))
    for index in range(len(joints)):
        # 随机站体位姿: 平移随机 + 绕 z 随机转角
        station = np.eye(4)
        station[:3, 3] = rng.uniform([-0.1, -0.5, -0.2], [0.6, 0.5, 0.6])
        angle = rng.uniform(-np.pi, np.pi)
        station[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        checker = CollisionChecker(arm, [(model.station_meshes, station)], model.arm_capsules)
        # 被测: 内部 _evaluate 给出的碰撞判定与距离下界
        collides, lower_bound = checker._evaluate(joints[index : index + 1])
        # 参照: 逐根胶囊调 hppfcl.distance, 取最小值作精确距离
        exact = np.inf
        transform = hppfcl.Transform3f()
        for capsule, midpoints, rotations in checker._capsule_geometry(joints[index : index + 1], 0.0):
            transform.setTransform(rotations[0], midpoints[0])
            result = hppfcl.DistanceResult()
            hppfcl.distance(capsule, transform, checker._bvh, checker._station_transform,
                            hppfcl.DistanceRequest(), result)
            exact = min(exact, result.min_distance)
        # 判定须与精确距离一致; 无碰撞时下界不得高估
        assert bool(collides[0]) == (exact <= 0.0), f"样本 {index}: 判定与精确距离 {exact:.4f} 不一致"
        if not collides[0]:
            assert lower_bound[0] <= exact + 1e-9, f"样本 {index}: 下界 {lower_bound[0]:.4f} > 精确 {exact:.4f}"
