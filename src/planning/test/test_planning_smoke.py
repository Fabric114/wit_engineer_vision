"""planning 移植冒烟测试: 验证导入链、FK、type3 全流程可跑通。

config/planning.example.yaml 的 arm.dh 已换成 rm26_arm 的实测 DH。 DH 的数值正确性
由 test_dh_mujoco.py 拿 MuJoCo 回代验证; 本文件只验证代码通路与结构性结论。
rm26_arm 是偏置腕 (非球腕), 解析 solve_ik 不适用 —— 见 README 第六节; config 里
arm.use_analytic_ik=false 时 solve_ik 自动走数值兜底, 正逆解算往返仍能闭环。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from planning import ArmModel, Type3Planner, load_config
from planning.type3 import AssemblyPath, AssemblyState, Type3PlanResult


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def arm(config) -> ArmModel:
    return ArmModel.from_config(config["arm"])


def test_import_does_not_require_optional_backends():
    """导入 planning 不应依赖 ompl / hppfcl (它们是可选后端)。"""
    import planning  # noqa: F401  仅确认可导入


def test_forward_kinematics_shape(arm: ArmModel):
    q = np.zeros((1, 6), dtype=float)
    fk = arm.forward_kinematics(q)
    assert fk.tcp_transforms.shape == (1, 4, 4)
    # 齐次矩阵最后一行为 [0,0,0,1]
    np.testing.assert_allclose(fk.tcp_transforms[0, 3], [0, 0, 0, 1], atol=1e-9)


def test_ik_round_trip_numeric(arm: ArmModel):
    """正逆解算往返: FK 得目标位姿 -> solve_ik 反解关节角 -> 再 FK 应复现同一位姿。

    rm26_arm 是偏置腕 (后三轴不汇交), 解析 IK 的球腕假设不成立; config 里
    ``arm.use_analytic_ik=false`` 时 solve_ik 自动退化为数值解 (scipy least_squares,
    README 第六节 6.3 迁移路线)。 本测试验证这条数值通路能把位姿解回来。
    """
    q = np.array([0.3, 0.8, 1.0, 0.2, 0.3, 0.1], dtype=float)
    target = arm.forward_kinematics(q[None, :]).tcp_transforms  # (1,4,4)

    # 请求 4 个分支 (不同初值撒开), 只要有一个既有效又能复现即算往返成功。
    joints, valid = arm.solve_ik(target, branches=list(range(4)))
    assert joints.shape == (1, 4, 6)
    assert valid[0].any(), "数值 IK 未能解出任何有效关节角"

    reproduced = False
    for k in range(4):
        if not valid[0, k]:
            continue
        fk = arm.forward_kinematics(joints[0, k][None, :]).tcp_transforms[0]
        position_ok = np.linalg.norm(fk[:3, 3] - target[0][:3, 3]) < 1e-4
        rotation_ok = (
            np.linalg.norm(Rotation.from_matrix(target[0][:3, :3].T @ fk[:3, :3]).as_rotvec())
            < 1e-3
        )
        if position_ok and rotation_ok:
            reproduced = True
            break
    assert reproduced, "数值 IK 解出的关节角未能复现目标位姿 (正逆解算未闭环)"


def _make_planner(arm: ArmModel, config: dict) -> Type3Planner:
    jp = config["planning"]["exchange"]["stage_path"]["joint_path"]
    best_ik = config["planning"]["exchange"]["stage_path"]["best_ik"]
    return Type3Planner(
        arm,
        geometry=config["planning"]["type3"]["exchange_trajectory"],
        ik_branch=config["planning"]["exchange"]["ik_branches"][0],
        roll_sample_step_deg=jp["roll_sample_step_deg"],
        bandwidth=jp["bandwidth"],
        collision_soft_margin=jp["collision_soft_margin"],
        collision_soft_weight=jp["collision_soft_weight"],
        motion_l1_weight=jp["motion_l1_weight"],
        motion_l2_weight=jp["motion_l2_weight"],
        joint_limit_margin_rad=jp["joint_limit_margin_rad"],
        joint_limit_weight=jp["joint_limit_weight"],
        best_effort=best_ik,
    )


def test_type3_plan_runs_end_to_end(arm: ArmModel, config: dict):
    """type3 完整流程 (建图 -> Viterbi -> 尽力修复) 应无异常返回结果对象。

    这里不保证一定规划成功 (station_transform 是任取的), 只验证代码通路。
    """
    planner = _make_planner(arm, config)
    path = AssemblyPath.between(
        AssemblyState(slide_m=0.0),
        AssemblyState(slide_m=0.10),
        samples=5,
    )
    # 任取一个在臂前方的兑换站位姿 (arm_base 系)
    station = np.eye(4)
    station[:3, 3] = [0.45, 0.0, 0.30]
    initial = np.array([0.3, 0.8, 1.0, 0.2, 0.3, 0.1], dtype=float)

    result = planner.plan(path, station, initial, collision_checker=None)
    assert isinstance(result, Type3PlanResult)
    assert isinstance(result.message, str)
    if result.success:
        assert result.waypoints.shape[1] == 6
        assert len(result.waypoints) == len(result.rolls)


def test_type2_bitstar_optional():
    """type2 依赖 OMPL; 装了就跑一次点到点规划, 没装则跳过。"""
    pytest.importorskip("ompl")
    from planning.type2 import plan_joint_path_bitstar

    start = np.zeros(6)
    goal = np.array([0.5, 0.5, 0.5, 0.0, 0.3, 0.0])
    lower = np.array([-3.14, 0.0, 0.0, -3.14, -1.83, -3.14])
    upper = np.array([3.14, 2.20, 2.31, 3.14, 1.83, 3.14])
    waypoints = plan_joint_path_bitstar(
        start,
        goal,
        validity_fn=lambda q: True,
        joint_lower=lower,
        joint_upper=upper,
        timeout_s=1.0,
    )
    assert waypoints is None or (waypoints.ndim == 2 and waypoints.shape[1] == 6)


def test_collision_module_imports_optional():
    """collision 依赖 hppfcl; 装了就确认能构造 CapsuleSpec, 没装则跳过。"""
    pytest.importorskip("hppfcl")
    from planning.collision import CapsuleSpec

    spec = CapsuleSpec(
        name="c0", frame=2,
        point_from=np.zeros(3), point_to=np.array([0.3, 0.0, 0.0]), radius=0.03,
    )
    assert spec.segment_length == pytest.approx(0.3)
