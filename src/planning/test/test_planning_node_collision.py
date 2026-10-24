# test_planning_node_collision.py — planning_node 碰撞行为 (启动校验 / 起点碰撞 / 发布前复检) 与 request_id 回显。
"""planning_node 碰撞行为 (第二轮, docs/audits/2026-10-09/collision-fix-spec.md 第 3 节 4~8)。

需要 rclpy 与已构建的 interfaces (先 source install/setup.bash); 缺了就整体 skip。
只建节点、直接调规划方法, 不 spin、不发任何话题。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("interfaces.msg")
pytest.importorskip("hppfcl")

from interfaces.msg import ApproachRequest, ApproachResult, AssemblyRequest, AssemblyResult  # noqa: E402

from planning import ArmModel, FixedDurationParameterizer, load_config  # noqa: E402
from planning.type3 import AssemblyPath, AssemblyState  # noqa: E402
from planning import planning_node as node_module  # noqa: E402
from planning.planning_node import PlanningNode  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


# 模块内所有测试共用一个 rclpy 上下文。
@pytest.fixture(scope="module", autouse=True)
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


# fixture: PlanningNode 工厂。collision_cfg 不为哨兵 ... 时替换 config 的 collision 段
# (None 表示该段缺失); 测试结束统一销毁节点。
@pytest.fixture
def node_factory(monkeypatch):
    created = []

    def factory(collision_cfg=...):
        if collision_cfg is not ...:
            config = load_config()
            config = dict(config, collision=collision_cfg)
            monkeypatch.setattr(node_module, "load_config", lambda: config)
        node = PlanningNode()
        created.append(node)
        return node

    yield factory
    for node in created:
        node.destroy_node()


# 模块级: 默认 config 的机械臂模型。
@pytest.fixture(scope="module")
def arm() -> ArmModel:
    return ArmModel.from_config(load_config()["arm"])


# 模块级: sim.yaml 中 station.from_joint 的 FK 作为站体位姿 (arm_base 系)。
@pytest.fixture(scope="module")
def station(arm) -> np.ndarray:
    with (REPO_ROOT / "config/sim.yaml").open() as stream:
        sim = yaml.safe_load(stream)
    return arm.forward_kinematics(
        np.asarray(sim["station"]["from_joint"], dtype=float)[None]
    ).tcp_transforms[0]


# helper: 把 4x4 齐次变换写入 Pose (四元数按 w,x,y,z 取出)。
def _set_pose(pose, transform: np.ndarray) -> None:
    from planning import quaternions_from_rotations

    w, x, y, z = quaternions_from_rotations(transform[None, :3, :3])[0]
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in transform[:3, 3])
    pose.orientation.w, pose.orientation.x = float(w), float(x)
    pose.orientation.y, pose.orientation.z = float(y), float(z)


# helper: 构造 ApproachRequest; fill_station=False 时 station_pose 留默认值。
def _approach_request(arm, station, start, *, fill_station=True) -> ApproachRequest:
    """接近点 = 站体沿 +x 退 0.15 m (与装配首状态一致)。"""
    target = station.copy()
    target[:3, 3] += 0.15 * station[:3, 0]
    msg = ApproachRequest()
    msg.header.frame_id = "arm_base"
    msg.start_joint = [float(v) for v in start]
    _set_pose(msg.target_pose, target)
    if fill_station:
        _set_pose(msg.station_pose, station)
    msg.timeout = 2.0
    return msg


# helper: 直接调 node._plan_approach, 返回 (是否成功, message 或异常文本)。
def _approach(node, msg):
    result = ApproachResult()
    result.header.stamp = node.get_clock().now().to_msg()
    try:
        return True, node._plan_approach(msg, result)
    except Exception as exc:  # 与 _on_approach_request 一致: 任何异常都是失败
        return False, f"{type(exc).__name__}: {exc}"


# 一个远离站体、限位内的起点
SAFE_START = np.array([0.3, 0.8, 1.0, 0.2, 0.3, 0.1])


# collision 段损坏或缺失, 节点构造必须抛异常 (缺失时为 RuntimeError 且提及 collision)。
def test_broken_collision_model_refuses_to_start(node_factory):
    # TODO(注释): 损坏模型只断言 Exception, 具体异常类型/消息未约束, 是否有意放宽?
    with pytest.raises(Exception):
        node_factory(collision_cfg={"station": {}, "arm": {"capsules": []}})
    with pytest.raises(RuntimeError, match="collision"):
        node_factory(collision_cfg=None)


# 没装 OMPL 时节点构造必须失败, 不退化成直线插值。
def test_missing_ompl_refuses_to_start(node_factory, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def no_ompl(name, *args, **kwargs):
        if name == "ompl" or name.startswith("ompl."):
            raise ImportError("no ompl (test)")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ompl)
    with pytest.raises(RuntimeError, match="OMPL"):
        node_factory()


# ApproachRequest 未填 station_pose 必须失败, 错误提及 station_pose。
def test_approach_requires_station_pose(node_factory, arm, station):
    node = node_factory()
    ok, message = _approach(node, _approach_request(arm, station, SAFE_START, fill_station=False))
    assert not ok
    assert "station_pose" in message


# 起点与站体碰撞时接近段规划必须拒绝, 错误提示含"起点"。
def test_approach_rejects_colliding_start(node_factory, arm, station):
    from planning.collision import CollisionChecker

    node = node_factory()
    # 用节点的碰撞模型 (站体 + 局部兑换桩体) 搭检查器, 从随机采样里挑首个碰撞构型当起点
    checker = CollisionChecker(
        arm,
        [(node.collision_model.station_meshes, station),
         (node.collision_model.local_exchange_meshes, station)],
        node.collision_model.arm_capsules,
    )
    samples = np.random.default_rng(5).uniform(arm.joint_space.lower, arm.joint_space.upper, (2000, 6))
    colliding = samples[checker.check_configs(samples)][0]
    ok, message = _approach(node, _approach_request(arm, station, colliding))
    assert not ok
    assert "起点" in message


# 正常接近请求应成功。
def test_approach_succeeds(node_factory, arm, station):
    node = node_factory()
    ok, message = _approach(node, _approach_request(arm, station, SAFE_START))
    assert ok, message


# 发布前稠密碰撞复检: 轨迹中点被替换成碰撞构型时必须拒绝, message 含 collides。
# TODO(注释): monkeypatch 参数未使用, 是否为遗留?
def test_dense_collision_fails_final_check(node_factory, arm, station, monkeypatch):
    """注入一条必撞的稠密轨迹: 即使规划器给出路点, 发布前的复检也必须拒。"""
    from planning.collision import CollisionChecker

    node = node_factory()
    # 挑一个与站体碰撞的随机构型 hit (只用 station_meshes)
    checker = CollisionChecker(
        arm, [(node.collision_model.station_meshes, station)], node.collision_model.arm_capsules
    )
    samples = np.random.default_rng(9).uniform(arm.joint_space.lower, arm.joint_space.upper, (2000, 6))
    hit = samples[checker.check_configs(samples)][0]

    # 包装参数化器: 正常插值后把中间采样点换成 hit, 模拟规划器漏检
    class Through(FixedDurationParameterizer):
        def parameterize(self, waypoints, **kwargs):
            trajectory = super().parameterize(waypoints, **kwargs)
            positions = trajectory.positions.copy()
            positions[len(positions) // 2] = hit
            return type(trajectory)(
                positions, trajectory.velocities, trajectory.accelerations, trajectory.timestamps
            )

    node.approach_parameterizer = Through(node.approach_duration_s)
    ok, message = _approach(node, _approach_request(arm, station, SAFE_START))
    assert not ok
    assert "collides" in message


# 两段回调都把请求的 request_id 原样写进结果, 成功和失败都一样。
def test_results_echo_request_id(node_factory, arm, station, monkeypatch):
    node = node_factory()
    published = []
    monkeypatch.setattr(node.approach_result_pub, "publish", published.append)
    monkeypatch.setattr(node.assembly_result_pub, "publish", published.append)

    ok_request = _approach_request(arm, station, SAFE_START)
    ok_request.request_id = "approach-ok"
    node._on_approach_request(ok_request)
    bad_request = _approach_request(arm, station, SAFE_START, fill_station=False)
    bad_request.request_id = "approach-bad"
    node._on_approach_request(bad_request)
    assembly_request = AssemblyRequest()
    assembly_request.header.frame_id = "arm_base"
    assembly_request.request_id = "assembly-bad"  # station_pose 没填, 必失败
    node._on_assembly_request(assembly_request)

    assert [(r.request_id, r.success) for r in published] == [
        ("approach-ok", True),
        ("approach-bad", False),
        ("assembly-bad", False),
    ]
    # 结构化诊断: 成功为空串 + 0; 失败给稳定分类, 不再只靠 message 字符串
    assert [(r.error_code, r.repaired_nodes) for r in published] == [
        ("", 0),
        (node_module.ERROR_INVALID_REQUEST, 0),
        (node_module.ERROR_INVALID_REQUEST, 0),
    ]


# 装配段成功路径: 最终 validate_trajectory 必须收到 collision_checker。
def test_assembly_success_runs_dense_collision_check(node_factory, arm, station, monkeypatch):
    """装配段成功路径也必须经过带碰撞检查器的最终校验。"""
    node = node_factory()
    # spy: 记录 validate_trajectory 收到的 collision_checker 后转调原函数
    seen = {}
    original = node_module.validate_trajectory

    def spy(*args, **kwargs):
        seen["checker"] = kwargs.get("collision_checker")
        return original(*args, **kwargs)

    monkeypatch.setattr(node_module, "validate_trajectory", spy)
    # 起点: 沿站 x 轴退开 0.15 m 的待插位 (与 config insert_slide_mag 同值) 在装配分支上的解
    branch = node.type3_planner.ik_branch
    start_pose = station.copy()
    start_pose[:3, 3] += 0.15 * station[:3, 0]
    joints, feasible = arm.solve_ik(start_pose[None], branches=(branch,))
    assert feasible[0, 0]
    # 构造 AssemblyRequest 并直接调 _plan_assembly
    msg = AssemblyRequest()
    msg.header.frame_id = "arm_base"
    _set_pose(msg.station_pose, station)
    msg.start_joint = [float(v) for v in joints[0, 0]]
    result = AssemblyResult()
    result.header.stamp = node.get_clock().now().to_msg()
    success, message = node._plan_assembly(msg, result)
    # 断言: 成功、复检带检查器、轨迹非空、结构化诊断字段已填
    assert success, message
    assert seen.get("checker") is not None
    assert len(result.trajectory.points) > 0
    assert result.error_code == ""
    assert result.repaired_nodes == 0


# 接近段默认只用装配段的 IK 分支 (两段衔接); 接近终点必须落在该分支上。
def test_approach_uses_assembly_branch(node_factory, arm, station):
    node = node_factory()
    assert node.approach_ik_branches == (node.type3_planner.ik_branch,)
    ok, message = _approach(node, _approach_request(arm, station, SAFE_START))
    assert ok, message
    target = station.copy()
    target[:3, 3] += 0.15 * station[:3, 0]
    joints, valid = arm.solve_ik(target[None], branches=(node.type3_planner.ik_branch,))
    assert valid[0, 0]


# 滑动段: use_start=true 从"插到底"出发, 走装配场景 (cc_link6 vs 桩体豁免) 并过稠密复检。
def test_slide_from_inserted_state(node_factory, arm, station):
    node = node_factory()
    branch = node.type3_planner.ik_branch
    inserted = node.type3_planner.task_poses(
        AssemblyPath(AssemblyState().as_array()[None]), station, np.zeros(1)
    )
    joints, feasible = arm.solve_ik(inserted, branches=(branch,))
    assert feasible[0, 0]
    msg = AssemblyRequest()
    msg.header.frame_id = "arm_base"
    _set_pose(msg.station_pose, station)
    msg.start_joint = [float(v) for v in joints[0, 0]]
    msg.use_start = True
    msg.goal.slide = float(load_config()["planning"]["type3"]["exchange_trajectory"]["slide_z_mag"])
    result = AssemblyResult()
    result.header.stamp = node.get_clock().now().to_msg()
    success, message = node._plan_assembly(msg, result)
    assert success, message
    assert len(result.trajectory.points) > 0
    # 不给 use_start 时起点是退开 0.15 m 的待插状态, 与"插到底"的实测关节角衔接不上
    msg.use_start = False
    success, message = node._plan_assembly(msg, AssemblyResult())
    assert not success
    assert "start" in message

