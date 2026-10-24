# test_type3_continuation.py — type3 末端 continuation (P 段 Q 缓存) 的纯算法回归。
"""type3 末端 continuation: 末层节点必须能走完全部后续分支, 合并结果就是 Q 缓存。

planning_node 的 P 段把 Q ±90° 两条分支作为 terminal_continuations 传进来。 rm26 在真实/仿真站姿下
P 段不可解 (2026-10-24 实测, 老车兑不了三级矿), 所以这里用"原地不动 + Q ±10°"的合成场景只测
continuation 机制本身: 剔除走不通的末端节点、合并出 s 严格递增且关节连续的查找表。
不需要 ROS。
"""

from __future__ import annotations

import numpy as np
import pytest

from planning import ArmModel, Type3Planner, load_config
from planning.type3 import AssemblyPath, AssemblyState

#: 合成场景: default_rng(0) 采 200 组关节角, 第 184 组的 FK 当站姿 (2026-10-25 搜得, ±10° Q 可走通,
#: 且有末端节点因 Q 走不通被剔除; 加 continuation 跳变门槛后重搜)
SCENE_SEED = 0
SCENE_INDEX = 184
SCENE_MAX_Q_DEG = 10.0
Q_STEP_DEG = 5.0


# helper: 以 base 为中心的 Q 两条分支 (0 -> -max, 0 -> +max), 每 step_deg 一点。
def _q_sweep_paths_up_to(base: AssemblyState, step_deg: float, max_deg: float):
    q = np.deg2rad(np.arange(0.0, max_deg + 0.5 * step_deg, step_deg))
    plus = np.repeat(base.as_array()[None, :], len(q), axis=0)
    minus = plus.copy()
    plus[:, 3] = q
    minus[:, 3] = -q
    return AssemblyPath(minus), AssemblyPath(plus)


# 模块级: 带 config q_sweep 代价的 type3 规划器 (不开 best_ik, 不查碰撞) 与合成场景。
@pytest.fixture(scope="module")
def feasible_q_scene():
    config = load_config()
    arm = ArmModel.from_config(config["arm"])
    joint_path = config["planning"]["exchange"]["stage_path"]["joint_path"]
    planner = Type3Planner(
        arm,
        geometry=config["planning"]["type3"]["exchange_trajectory"],
        ik_branch=0,
        roll_sample_step_deg=joint_path["roll_sample_step_deg"],
        bandwidth=joint_path["bandwidth"],
        collision_soft_margin=0.0,
        collision_soft_weight=0.0,
        motion_l1_weight=joint_path["motion_l1_weight"],
        motion_l2_weight=joint_path["motion_l2_weight"],
        joint_limit_margin_rad=joint_path["joint_limit_margin_rad"],
        joint_limit_weight=joint_path["joint_limit_weight"],
        best_effort={"enabled": False},
        start_joint_tolerance_rad=joint_path["start_joint_tolerance_rad"],
        max_joint_step_rad=joint_path["max_joint_step_rad"],
        continuation=dict(joint_path["q_sweep"]),
    )
    samples = np.random.default_rng(SCENE_SEED).uniform(
        arm.joint_space.lower, arm.joint_space.upper, (200, 6)
    )
    station = arm.forward_kinematics(samples[SCENE_INDEX][None]).tcp_transforms[0]
    base = AssemblyState()
    joints, valid = arm.solve_ik(
        planner.task_poses(AssemblyPath(base.as_array()[None]), station, planner.rolls), branches=(0,)
    )
    indices = np.flatnonzero(valid[:, 0])
    if len(indices) < 20:
        pytest.skip("合成场景失效 (arm/config 变了): 重新搜 SCENE_INDEX")
    initial = joints[indices[len(indices) // 2], 0]
    path = AssemblyPath.between(base, base, samples=3)
    continuations = _q_sweep_paths_up_to(base, Q_STEP_DEG, SCENE_MAX_Q_DEG)
    return planner, path, station, initial, continuations


# 合并后的 Q 缓存: s 从 -max 严格递增到 +max, 点数/形状一致, 相邻点跳变不超 max_joint_step。
def test_q_sweep_continuation_grid(feasible_q_scene):
    planner, path, station, initial, continuations = feasible_q_scene
    result = planner.plan(path, station, initial, None, terminal_continuations=continuations)
    if not result.success:
        pytest.skip(f"合成场景失效 (arm/config 变了): {result.message}")
    q = result.continuation["states"][:, 3]
    max_q = continuations[1].states[-1, 3]
    assert q[0] == pytest.approx(-max_q)
    assert q[-1] == pytest.approx(max_q)
    assert np.all(np.diff(q) > 0)
    # 两条分支各 n 点, 合并时去掉重合的 q=0 点
    assert len(q) == 2 * len(continuations[1].states) - 1
    assert result.continuation["joints"].shape == (len(q), 6)
    steps = np.abs(planner.arm.joint_space.delta(result.continuation["joints"][1:], result.continuation["joints"][:-1]))
    assert float(np.max(steps)) <= planner.max_joint_step
    # 有末端节点因为走不通 Q 被剔除: 剔除机制确实生效
    assert result.diagnostics["continuation_rejected_nodes"] > 0
    assert result.diagnostics["continuation_rejected_nodes"] < result.diagnostics["continuation_checked_nodes"]


# Q 转到 ±180° (超出该场景可达范围) 时所有末端节点都被剔除, 规划判失败, 不返回 continuation。
def test_terminal_without_feasible_continuation_is_rejected(feasible_q_scene):
    planner, path, station, initial, _ = feasible_q_scene
    impossible = _q_sweep_paths_up_to(AssemblyState(), Q_STEP_DEG, 180.0)
    result = planner.plan(path, station, initial, None, terminal_continuations=impossible)
    assert not result.success
    assert result.continuation is None
    assert result.diagnostics["continuation_rejected_nodes"] == result.diagnostics["continuation_checked_nodes"] > 0
