# test_planning_safety.py — 规划安全回归 (近似 IK 验收 / 五次插值保形 / 首点衔接)。
"""规划安全回归: 2026-10-09 审计复现的三项问题 (docs/audits/2026-10-09/)。

1. Type III 近似 IK 残差超阈值必须判失败 (P0-1);
2. 五次插值不得越出路点区间, 最终轨迹要能被校验 (P0-2);
3. 装配首点必须与当前关节衔接, 否则判失败 (P0-5)。

验收判据见 docs/audits/2026-10-09/planning-safety-fix-spec.md。
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from planning import ArmModel, FixedDurationParameterizer, Type3Planner, load_config
from planning.trajectory import validate_trajectory
from planning.type3 import AssemblyPath, AssemblyState

REPO_ROOT = Path(__file__).resolve().parents[3]
#: 2026-10-09 审计用的站姿 (当时 config/sim.yaml 的 station.from_joint, 站体 z 竖直)。
#: sim 演示已换站姿, 这里固定住旧值, 让审计复现的场景不随 sim 配置漂移。
AUDIT_STATION_JOINTS = np.array([-0.25, 1.971, 1.341, 0.364, 0.257, 0.293])


# 模块级共享: 默认 config。
@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


# 模块级共享: 由 config['arm'] 构造的机械臂模型。
@pytest.fixture(scope="module")
def arm(config) -> ArmModel:
    return ArmModel.from_config(config["arm"])


# helper: 按 config 的 exchange.stage_path 参数构造 Type3Planner。
# branch 覆盖 IK 分支, best_effort 覆盖修复开关, best_overrides 覆盖 best_ik 键 (值为 None 则删除该键)。
def _planner(arm, config, *, branch=None, best_effort=None, max_step="config", **best_overrides) -> Type3Planner:
    exchange = config["planning"]["exchange"]
    jp = exchange["stage_path"]["joint_path"]
    repair = copy.deepcopy(exchange["stage_path"]["best_ik"])
    if best_effort is not None:
        repair["enabled"] = best_effort
    for key, value in best_overrides.items():
        if value is None:
            repair.pop(key, None)
        else:
            repair[key] = value
    return Type3Planner(
        arm,
        geometry=config["planning"]["type3"]["exchange_trajectory"],
        ik_branch=exchange["ik_branches"][0] if branch is None else branch,
        roll_sample_step_deg=jp["roll_sample_step_deg"],
        bandwidth=jp["bandwidth"],
        collision_soft_margin=jp["collision_soft_margin"],
        collision_soft_weight=jp["collision_soft_weight"],
        motion_l1_weight=jp["motion_l1_weight"],
        motion_l2_weight=jp["motion_l2_weight"],
        joint_limit_margin_rad=jp["joint_limit_margin_rad"],
        joint_limit_weight=jp["joint_limit_weight"],
        best_effort=repair,
        start_joint_tolerance_rad=jp["start_joint_tolerance_rad"],
        max_joint_step_rad=jp["max_joint_step_rad"] if max_step == "config" else max_step,
    )


# ---------------------------------------------------------------- P0-1 近似 IK 验收
# P0-1: 完全不可达站位即使开尽力修复也必须判失败, 且不输出路点。
def test_unreachable_station_is_rejected_even_with_best_effort(arm, config):
    """站位 (2,0,0) m 完全不可达: 修复残差 >1 m, 不能报成功。"""
    # 构造: 站位放到 2 m 外, 装配路径滑移 0~0.1 m, 任取初始关节角
    station = np.eye(4)
    station[:3, 3] = [2.0, 0.0, 0.0]
    path = AssemblyPath.between(AssemblyState(), AssemblyState(slide_m=0.1), samples=5)
    initial = np.array([0.3, 0.8, 1.0, 0.2, 0.3, 0.1])
    result = _planner(arm, config, best_effort=True).plan(path, station, initial, None)
    # 断言: 无严格 IK 节点; 修复残差 >1 m 且被判失败; 整体失败并提示 tolerance
    assert result.diagnostics["ik_valid_nodes"] == 0
    assert result.diagnostics["repair"] is not None
    assert result.diagnostics["repair"]["max_pos_error_m"] > 1.0
    assert not result.diagnostics["repair"]["success"]
    assert not result.success
    assert "tolerance" in result.message
    assert len(result.waypoints) == 0


# fixture: 审计站位下的插入任务, 返回 (装配路径, 站位, 按分支取待插位关节角的函数)。
@pytest.fixture(scope="module")
def sim_insertion(arm):
    """审计站姿: 退开 0.15 m -> 插到底; 分支 0 第 8 层起无严格解, 修复残差约 0.24 mm / 0.4°。"""
    # 站位 = AUDIT_STATION_JOINTS 的 FK; 待插位 = 沿站 x 轴退开 0.15 m
    station = arm.forward_kinematics(AUDIT_STATION_JOINTS[None]).tcp_transforms[0]
    path = AssemblyPath.between(AssemblyState(axial_offset_m=0.15), AssemblyState(), samples=10)
    start_pose = station.copy()
    start_pose[:3, 3] += 0.15 * station[:3, 0]
    joints, feasible = arm.solve_ik(start_pose[None])

    def start(branch: int) -> np.ndarray:
        """待插位在指定分支上的关节角 (接近段应落在装配段同一分支上)。"""
        assert feasible[0, branch]
        return joints[0, branch]

    return path, station, start


# P0-1: 修复点是否接受由 best_ik 的位置/轴向残差阈值决定; 阈值键缺失视为拒绝。
def test_repair_acceptance_follows_configured_thresholds(arm, config, sim_insertion):
    """同一组修复点: 阈值放宽到残差之上则接受, 用默认阈值或缺阈值键则拒绝。"""
    path, station, start_of = sim_insertion
    start = start_of(0)
    # 放宽阈值 (1 mm / 1°): 修复残差在阈值内 -> 成功
    loose = _planner(
        arm, config, branch=0, best_effort=True,
        max_position_error_m=0.001, max_axis_error_deg=1.0,
    ).plan(path, station, start, None)
    repair = loose.diagnostics["repair"]
    assert repair is not None and repair["repair_count"] > 0
    assert repair["max_pos_error_m"] <= 0.001
    assert np.rad2deg(repair["max_axis_error_rad"]) <= 1.0
    assert loose.success, loose.message

    # 默认阈值 0.1° 小于该修复的轴向残差 -> 拒绝
    assert np.rad2deg(repair["max_axis_error_rad"]) > config["planning"]["exchange"][
        "stage_path"]["best_ik"]["max_axis_error_deg"]
    strict = _planner(arm, config, branch=0, best_effort=True).plan(path, station, start, None)
    assert not strict.success

    # 删除阈值键 -> 拒绝 (缺配置不得默认放行)
    missing = _planner(
        arm, config, branch=0, best_effort=True,
        max_position_error_m=None, max_axis_error_deg=None,
    ).plan(path, station, start, None)
    assert not missing.success


# 回归: 分支 4 每层都有严格解, 但最后一层构型翻转 (实测单步 2.58 rad) —— 跳变门槛必须拒;
# 关掉门槛就是旧行为 (只算代价, 照样"成功")。 起点落在其他分支上则因首点不衔接被拒。
def test_branch_flip_inside_insertion_is_rejected(arm, config, sim_insertion):
    path, station, start_of = sim_insertion
    result = _planner(arm, config, branch=4, best_effort=False).plan(path, station, start_of(4), None)
    assert not result.success
    assert "every layer has candidates" in result.message
    legacy = _planner(arm, config, branch=4, best_effort=False, max_step=None).plan(
        path, station, start_of(4), None
    )
    assert legacy.success, legacy.message
    assert legacy.diagnostics["repair"] is None
    assert legacy.diagnostics["max_joint_step_rad"] > config["planning"]["exchange"][
        "stage_path"]["joint_path"]["max_joint_step_rad"]
    # 起点落在另一分支 (分支 0) 上则衔接不上, 必须拒绝
    crossed = _planner(arm, config, branch=4, best_effort=False, max_step=None).plan(
        path, station, start_of(0), None
    )
    assert not crossed.success
    assert "start" in crossed.message


# ---------------------------------------------------------------- P0-2 插值与最终校验
# helper: 各关节限位中点。
def _mid(arm):
    return (arm.joint_space.lower + arm.joint_space.upper) / 2.0


# P0-2: J1 从下限跳到上限后保持, 五次插值不得越过上限, 首末点精确保留。
def test_quintic_does_not_overshoot_joint_limit(arm):
    """审计复现用例: J1 [-0.52, 1.91, 1.91, 1.91], 2 s / 1 ms; 旧实现峰值 2.1105 rad, 666 点越限。"""
    # 构造: 其余关节取中点, J1 = [下限, 上限, 上限, 上限]
    points = np.tile(_mid(arm), (4, 1))
    points[:, 0] = [arm.joint_space.lower[0]] + [arm.joint_space.upper[0]] * 3
    trajectory = FixedDurationParameterizer(2.0).parameterize(points, sample_dt=0.001)
    # 断言: 全程在限位内, 首末点等于路点, 且通过 validate_trajectory
    assert np.all(trajectory.positions >= arm.joint_space.lower - 1e-12)
    assert np.all(trajectory.positions <= arm.joint_space.upper + 1e-12)
    np.testing.assert_array_equal(trajectory.positions[0], points[0])
    np.testing.assert_array_equal(trajectory.positions[-1], points[-1])
    validate_trajectory(trajectory, arm.joint_space, waypoints=points)


# P0-2: 随机路点的插值保形 (不越出所在段端点区间), 端点速度/加速度为零。
def test_quintic_samples_stay_inside_each_segment(arm):
    """随机合法路点: 每个采样都落在所在段两端路点之间 (保形), 端点导数为零。"""
    rng = np.random.default_rng(20261009)
    for _ in range(20):
        count = int(rng.integers(3, 12))
        points = rng.uniform(arm.joint_space.lower, arm.joint_space.upper, (count, 6))
        trajectory = FixedDurationParameterizer(3.0).parameterize(points, sample_dt=0.002)
        # 按时间戳反推每个采样所在段 (路点在总时长上均匀分布), 求该段两端路点的上下界
        times = np.linspace(0.0, 3.0, count)
        segments = np.clip(
            np.searchsorted(times, trajectory.timestamps, side="right") - 1, 0, count - 2
        )
        low = np.minimum(points[segments], points[segments + 1])
        high = np.maximum(points[segments], points[segments + 1])
        assert np.all(trajectory.positions >= low - 1e-12)
        assert np.all(trajectory.positions <= high + 1e-12)
        # 端点导数为零, 且通过最终校验
        np.testing.assert_array_equal(trajectory.velocities[[0, -1]], 0.0)
        np.testing.assert_array_equal(trajectory.accelerations[[0, -1]], 0.0)
        validate_trajectory(trajectory, arm.joint_space, waypoints=points)


# P0-2: 保形不应过度: 单调路点的中间段速度保持为正, 不被迫停顿。
def test_monotone_waypoints_keep_nonzero_interior_velocity(arm):
    """保形只在极值/越界处置零; 单调路点中间不应被迫停顿。"""
    points = np.tile(_mid(arm), (5, 1))
    points[:, 0] = np.linspace(-0.4, 1.6, 5)
    trajectory = FixedDurationParameterizer(2.0).parameterize(points, sample_dt=0.001)
    # 去掉首末 0.1 s (端点速度本就为零) 后检查 J1 速度
    interior = (trajectory.timestamps > 0.1) & (trajectory.timestamps < 1.9)
    assert np.all(trajectory.velocities[interior, 0] > 0.0)


# validate_trajectory 应识别: 越限 / 非有限值 / 首点与路点不符 / 超速 / 超加速度。
def test_validate_trajectory_rejects_bad_trajectories(arm):
    # 构造 J2 三路点的正常轨迹, 应通过校验
    points = np.tile(_mid(arm), (3, 1))
    points[:, 1] = [0.0, 1.0, 2.0]
    good = FixedDurationParameterizer(2.0).parameterize(points, sample_dt=0.01)
    validate_trajectory(good, arm.joint_space, waypoints=points)

    # helper: 只替换 positions, 其余字段沿用 good
    def with_positions(positions):
        return type(good)(positions, good.velocities, good.accelerations, good.timestamps)

    # 越上限 -> 报 limit
    outside = good.positions.copy()
    outside[5, 0] = arm.joint_space.upper[0] + 0.01
    with pytest.raises(ValueError, match="limit"):
        validate_trajectory(with_positions(outside), arm.joint_space, waypoints=points)

    # NaN -> 报 finite
    nan = good.positions.copy()
    nan[3, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_trajectory(with_positions(nan), arm.joint_space, waypoints=points)

    # 路点首点偏移 -> 报 first
    shifted = points.copy()
    shifted[0, 3] += 0.1
    with pytest.raises(ValueError, match="first"):
        validate_trajectory(good, arm.joint_space, waypoints=shifted)

    # 速度/加速度上限设为实际峰值一半 -> 分别报 velocity / acceleration
    peak = float(np.max(np.abs(good.velocities)))
    validate_trajectory(good, arm.joint_space, waypoints=points)  # 不给上限: 不查速度
    with pytest.raises(ValueError, match="velocity"):
        validate_trajectory(
            good, arm.joint_space, waypoints=points, velocity_limits=np.full(6, 0.5 * peak)
        )
    accel = float(np.max(np.abs(good.accelerations)))
    with pytest.raises(ValueError, match="acceleration"):
        validate_trajectory(
            good, arm.joint_space, waypoints=points, acceleration_limits=np.full(6, 0.5 * accel)
        )


# ---------------------------------------------------------------- P0-5 首点衔接
# fixture: seed=12 随机采样中首个分支 0 可解的任务位姿; 装配路径为三个相同状态 (原地不动)。
@pytest.fixture(scope="module")
def reachable_task(arm):
    """固定任务位姿 (审计 seed=12 的首个分支 0 可解目标), 三个相同状态。"""
    sampled = np.random.default_rng(12).uniform(arm.joint_space.lower, arm.joint_space.upper, (1000, 6))
    targets = arm.forward_kinematics(sampled).tcp_transforms
    _, valid = arm.solve_ik(targets)
    target = targets[int(np.flatnonzero(valid[:, 0])[0])]
    path = AssemblyPath.between(AssemblyState(), AssemblyState(), samples=3)
    return path, target


# P0-5: 初始关节与流形首点相差过大时必须失败, 并在 diagnostics 给出 start_delta_rad。
def test_mismatched_start_is_rejected(arm, config, reachable_task):
    """审计复现: 从关节下限起步, 首点差 4.08 rad, 旧实现返回 ok。"""
    path, target = reachable_task
    initial = arm.joint_space.lower.copy()
    result = _planner(arm, config, branch=0, best_effort=False).plan(path, target, initial, None)
    assert not result.success
    assert "start" in result.message
    assert result.diagnostics["start_delta_rad"] > 1.0


# P0-5: 首点偏差在 start_joint_tolerance_rad 内时, 轨迹以实测关节角开头再接流形首点。
def test_near_start_is_connected_from_measured_joints(arm, config, reachable_task):
    """首点在阈值内: 轨迹第 0 点就是实测关节角, 再接流形首点。"""
    path, target = reachable_task
    planner = _planner(arm, config, branch=0, best_effort=False)
    # 用首层的一个有效节点当"流形首点", 实测关节角只在 J1 上偏半个阈值
    graph, _ = planner._build_graph(path, target, None)
    first = graph.joints[0][graph.valid[0]][0]
    tolerance = config["planning"]["exchange"]["stage_path"]["joint_path"]["start_joint_tolerance_rad"]
    initial = first.copy()
    initial[0] += 0.5 * tolerance if first[0] < arm.joint_space.upper[0] - tolerance else -0.5 * tolerance
    result = planner.plan(path, target, initial, None)
    # 断言: 成功且标记 start_connected; 路点比 path 多一个首点; 衔接段偏差不超阈值
    assert result.success, result.message
    assert result.diagnostics["start_connected"]
    np.testing.assert_array_equal(result.waypoints[0], initial)
    assert len(result.waypoints) == len(result.path) + 1 == len(result.rolls)
    delta = np.max(np.abs(arm.joint_space.delta(result.waypoints[1:2], initial[None])))
    assert delta <= tolerance


# P0-5: 初始关节恰为流形首点时不插入衔接点, 路点数等于 path 长度。
def test_exact_start_needs_no_connection(arm, config, reachable_task):
    path, target = reachable_task
    planner = _planner(arm, config, branch=0, best_effort=False)
    graph, _ = planner._build_graph(path, target, None)
    initial = graph.joints[0][graph.valid[0]][0].copy()
    result = planner.plan(path, target, initial, None)
    assert result.success, result.message
    assert not result.diagnostics["start_connected"]
    assert len(result.waypoints) == len(result.path)
    np.testing.assert_allclose(result.waypoints[0], initial, atol=1e-12)
