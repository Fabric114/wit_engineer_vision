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
from geometry_msgs.msg import PoseStamped  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from planning import ArmModel, FixedDurationParameterizer, load_config  # noqa: E402
from planning.type3 import AssemblyPath, AssemblyState, Type3PlanResult  # noqa: E402
from planning import planning_node as node_module  # noqa: E402
from planning.planning_node import JOINT_NAMES, PlanningNode  # noqa: E402

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


# helper: 关节角 -> JointState (默认名字 J1..J6, 顺序与 q 对应)。
def _joint_state(q, names=JOINT_NAMES) -> JointState:
    msg = JointState()
    msg.name = list(names)
    msg.position = [float(v) for v in q]
    return msg


# helper: 4x4 齐次变换 -> PoseStamped (默认 frame_id = arm_base)。
def _pose_stamped(transform: np.ndarray, frame: str = "arm_base") -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame
    _set_pose(msg.pose, transform)
    return msg


# helper: 构造 ApproachRequest; frame 是 station_pose_arm_base 的 frame_id。 目标由节点自选。
def _approach_request(arm, station, start, *, frame="arm_base") -> ApproachRequest:
    msg = ApproachRequest()
    msg.header.frame_id = "arm_base"
    msg.initial_joint_state = _joint_state(start)
    msg.station_pose_arm_base = _pose_stamped(station, frame)
    return msg


# helper: 插到底 (AssemblyState(), roll=0) 在装配分支上的关节角。
def _inserted_joints(arm, station, node) -> np.ndarray:
    branch = node.type3_planner.ik_branch
    inserted = node.type3_planner.task_poses(
        AssemblyPath(AssemblyState().as_array()[None]), station, np.zeros(1)
    )
    joints, feasible = arm.solve_ik(inserted, branches=(branch,))
    assert feasible[0, 0]
    return joints[0, 0]


# helper: 滑动段请求 (插到底 -> slide_z_mag), 起点为插到底关节角。
def _slide_request(arm, station, node) -> AssemblyRequest:
    msg = AssemblyRequest()
    msg.header.frame_id = "arm_base"
    msg.station_pose_arm_base = _pose_stamped(station)
    msg.initial_joint_state = _joint_state(_inserted_joints(arm, station, node))
    msg.path_name = AssemblyRequest.PATH_SLIDE
    return msg


# helper: 直接调 node._plan_approach, 返回 (是否成功, message 或异常文本)。
def _approach(node, msg):
    result = ApproachResult()
    result.header.stamp = node.get_clock().now().to_msg()
    try:
        message, _cost = node._plan_approach(msg, result)
        return True, message
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


# initial_joint_state 按名字取 J1..J6, 不按下标: 打乱顺序也要规划成功。
def test_joint_state_reordered_by_name(node_factory, arm, station):
    msg = _approach_request(arm, station, SAFE_START)
    order = [2, 0, 1, 5, 4, 3]
    msg.initial_joint_state = _joint_state(SAFE_START[order], names=[JOINT_NAMES[i] for i in order])
    ok, message = _approach(node_factory(), msg)
    assert ok, message


# 缺关节必须失败, 错误里点名缺的关节。
def test_joint_state_missing_joint_rejected(node_factory, arm, station):
    msg = _approach_request(arm, station, SAFE_START)
    msg.initial_joint_state = _joint_state(SAFE_START[:5], names=JOINT_NAMES[:5])
    ok, message = _approach(node_factory(), msg)
    assert not ok
    assert "J6" in message


# 多出来的关节 (如升降轴 line) 忽略。
def test_joint_state_extra_joint_ignored(node_factory, arm, station):
    msg = _approach_request(arm, station, SAFE_START)
    msg.initial_joint_state = _joint_state(np.append(SAFE_START, 0.1), names=[*JOINT_NAMES, "line"])
    ok, message = _approach(node_factory(), msg)
    assert ok, message


# 站姿 frame_id 不是 arm_base (含空串) 时两段都必须失败, 不能告警后照常规划。
@pytest.mark.parametrize("frame", ["", "camera"])
def test_station_frame_must_be_arm_base(node_factory, arm, station, frame):
    node = node_factory()
    ok, message = _approach(node, _approach_request(arm, station, SAFE_START, frame=frame))
    assert not ok
    assert "frame_id" in message
    assembly = _slide_request(arm, station, node)
    assembly.station_pose_arm_base.header.frame_id = frame
    with pytest.raises(ValueError, match="frame_id"):
        node._plan_assembly(assembly, AssemblyResult())


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
    bad_request = _approach_request(arm, station, SAFE_START, frame="")
    bad_request.request_id = "approach-bad"
    node._on_approach_request(bad_request)
    assembly_request = _slide_request(arm, station, node)
    assembly_request.station_pose_arm_base.header.frame_id = ""  # 站姿 frame 不对, 必失败
    assembly_request.request_id = "assembly-bad"
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
    msg = _slide_request(arm, station, node)
    result = AssemblyResult()
    result.header.stamp = node.get_clock().now().to_msg()
    success, message = node._plan_assembly(msg, result)
    # 断言: 成功、复检带检查器、轨迹非空、结构化诊断字段已填
    assert success, message
    assert seen.get("checker") is not None
    assert len(result.trajectory.points) > 0
    assert result.error_code == ""
    assert result.repaired_nodes == 0


# 接近目标由节点按站姿自选: 终点 TCP 落在沿站 +x 退开 insert_slide_mag 的待插位, 分支默认同装配段。
def test_approach_goal_is_insert_standoff_on_assembly_branch(node_factory, arm, station):
    node = node_factory()
    assert node.approach_ik_branches == (node.type3_planner.ik_branch,)
    result = ApproachResult()
    message, cost = node._plan_approach(_approach_request(arm, station, SAFE_START), result)
    goal = np.asarray(result.trajectory.points[-1].positions)
    tcp = arm.forward_kinematics(goal[None]).tcp_transforms[0]
    standoff = station[:3, 3] + node.approach_axial_m * station[:3, 0]
    assert np.linalg.norm(tcp[:3, 3] - standoff) < 1e-3
    # 插入轴 (TCP x) 与站轴平行
    assert abs(float(np.dot(tcp[:3, 0], station[:3, 0]))) > 1.0 - 1e-6
    assert np.isfinite(cost) and cost >= 0.0


# 接近段选的 roll 必须落在 type3 的 roll 网格上: 操作手在这个 roll 上直插到底后, 滑动段首层要有
# 同 roll 的节点, 否则起点差超 start_joint_tolerance_rad 被拒 (1° 扫 vs 5° 网格曾经对不上)。
@pytest.mark.parametrize("start", [np.full(6, 0.1), SAFE_START])
def test_approach_roll_is_on_assembly_grid(node_factory, station, start):
    node = node_factory()
    checker = node._make_collision_checker(station, stage="approach")
    _, _, roll = node._select_approach_goal(station, node._to_type2_bounds(start)[0], checker)
    grid = np.mod(node.type3_planner.rolls, 2.0 * np.pi)
    gap = np.abs((np.mod(roll, 2.0 * np.pi) - grid + np.pi) % (2.0 * np.pi) - np.pi)
    assert float(np.min(gap)) < 1e-9, f"roll {np.rad2deg(roll):.1f}° 不在 type3 网格上"


# 失败结果的 cost 为 inf; 成功结果 cost 有限。
def test_approach_result_cost(node_factory, arm, station, monkeypatch):
    node = node_factory()
    published = []
    monkeypatch.setattr(node.approach_result_pub, "publish", published.append)
    node._on_approach_request(_approach_request(arm, station, SAFE_START))
    node._on_approach_request(_approach_request(arm, station, SAFE_START, frame="camera"))
    assert published[0].success and np.isfinite(published[0].cost)
    assert not published[1].success and published[1].cost == float("inf")


# PATH_SLIDE: 从"插到底"出发滑到 slide_z_mag, 走装配场景 (cc_link6 vs 桩体豁免) 并过稠密复检。
def test_slide_path_from_inserted_state(node_factory, arm, station):
    node = node_factory()
    result = AssemblyResult()
    result.header.stamp = node.get_clock().now().to_msg()
    success, message = node._plan_assembly(_slide_request(arm, station, node), result)
    assert success, message
    assert len(result.trajectory.points) > 0
    assert result.selected_ik_branch == node.type3_planner.ik_branch
    assert np.isfinite(result.cost)
    assert len(result.q_sweep_s) == 0 and len(result.q_sweep_trajectory.points) == 0


# 未知 path_name: INVALID_REQUEST, cost=inf, selected_ik_branch=-1。
def test_unknown_path_name_is_invalid_request(node_factory, arm, station, monkeypatch):
    node = node_factory()
    published = []
    monkeypatch.setattr(node.assembly_result_pub, "publish", published.append)
    msg = _slide_request(arm, station, node)
    msg.path_name = 7
    node._on_assembly_request(msg)
    assert not published[0].success
    assert published[0].error_code == node_module.ERROR_INVALID_REQUEST
    assert published[0].cost == float("inf") and published[0].selected_ik_branch == -1


# 结果回显 path_name, 成功和失败都一样。
def test_assembly_result_echoes_path_name(node_factory, arm, station, monkeypatch):
    node = node_factory()
    published = []
    monkeypatch.setattr(node.assembly_result_pub, "publish", published.append)
    good = _slide_request(arm, station, node)
    bad = _slide_request(arm, station, node)
    bad.station_pose_arm_base.header.frame_id = ""
    node._on_assembly_request(good)
    node._on_assembly_request(bad)
    assert [(r.path_name, r.success) for r in published] == [
        (AssemblyRequest.PATH_SLIDE, True),
        (AssemblyRequest.PATH_SLIDE, False),
    ]



# ---------------------------------------------------------------- P 段 / Q 缓存
# rm26 在 sim 站姿下 P 段不可解 (2026-10-24 实测), 这里打桩 type3_planner.plan 只测节点的接线:
# 传给 type3 的路径与 Q 分支、结果里 Q 缓存 / cost / roll 的填法。 type3 本身见 test_type3_continuation.py。

#: 打桩 plan 结果的 diagnostics (只含 _log_assembly_diagnostics 读的键)
_FAKE_DIAGNOSTICS = {
    "final_valid_nodes": 3,
    "total_nodes": 3,
    "ik_valid_nodes": 3,
    "collision_rejected_nodes": 0,
    "path_samples": 3,
    "roll_count": 1,
    "ik_branch": 0,
    "final_valid_by_step": np.ones(3, dtype=int),
    "repair": None,
}


# 打桩 Q 缓存: q = -90° / 0 / +90°, 关节角各不相同, 方便核对写进消息的顺序。
def _fake_sweep() -> dict:
    states = np.zeros((3, 5))
    states[:, 1] = 0.104
    states[:, 2] = np.pi / 2
    states[:, 3] = [-np.pi / 2, 0.0, np.pi / 2]
    return {"states": states, "joints": SAFE_START + np.arange(3)[:, None] * 0.01, "cost": 0.0}


def _fake_plan(continuation) -> Type3PlanResult:
    return Type3PlanResult(
        success=True,
        message="ok",
        cost=1.5,
        waypoints=np.tile(SAFE_START, (3, 1)),
        rolls=np.array([0.0, 0.0, 0.3]),
        path=[(0, 0), (1, 0), (2, 0)],
        continuation=continuation,
        diagnostics=dict(_FAKE_DIAGNOSTICS),
    )


# helper: P 段请求 (起点关节随便给, plan 是打桩的)。
def _p_request(station) -> AssemblyRequest:
    msg = AssemblyRequest()
    msg.header.frame_id = "arm_base"
    msg.station_pose_arm_base = _pose_stamped(station)
    msg.initial_joint_state = _joint_state(SAFE_START)
    msg.path_name = AssemblyRequest.PATH_P
    return msg


# PATH_P: 路径 = 滑动终点 -> p=90°, continuation = Q 0->-90° / 0->+90°; 结果填 Q 缓存 / cost / roll。
def test_p_path_passes_q_continuations(node_factory, station, monkeypatch):
    node = node_factory()
    seen = {}

    def spy(path, station_transform, *, initial_joints, collision_checker, terminal_continuations):
        seen["path"], seen["continuations"] = path, terminal_continuations
        return _fake_plan(_fake_sweep())

    monkeypatch.setattr(node.type3_planner, "plan", spy)
    monkeypatch.setattr(node_module, "_checked_trajectory", lambda *args, **kwargs: "peaks")
    result = AssemblyResult()
    success, message = node._plan_assembly(_p_request(station), result)
    assert success, message
    # 路径: 起点 slide=slide_z_mag p=0, 终点 p=90°
    assert seen["path"].states[0, 1] == pytest.approx(node.assembly_slide_m)
    assert seen["path"].states[0, 2] == pytest.approx(0.0)
    assert seen["path"].states[-1, 2] == pytest.approx(np.pi / 2)
    # Q 两条分支: 都从 q=0 出发, 分别到 -90° / +90°, 其余分量 = P 终点
    minus, plus = seen["continuations"]
    assert minus.states[0, 3] == pytest.approx(0.0) and minus.states[-1, 3] == pytest.approx(-np.pi / 2)
    assert plus.states[0, 3] == pytest.approx(0.0) and plus.states[-1, 3] == pytest.approx(np.pi / 2)
    assert np.allclose(plus.states[:, 2], np.pi / 2) and np.allclose(plus.states[:, 1], node.assembly_slide_m)
    step = np.deg2rad(node.type3_planner.continuation_config["sample_step_deg"])
    assert np.allclose(np.diff(plus.states[:, 3]), step)
    # 结果: Q 缓存 (s = q / 90°), 查找表 time_from_start = 序号, 速度 0
    assert list(result.q_sweep_s) == pytest.approx([-1.0, 0.0, 1.0])
    points = result.q_sweep_trajectory.points
    assert len(points) == 3
    assert list(result.q_sweep_trajectory.joint_names) == list(JOINT_NAMES)
    assert np.allclose(points[2].positions, SAFE_START + 0.02)
    assert points[2].time_from_start.sec == 2 and not any(points[2].velocities)
    assert result.selected_roll == pytest.approx(0.3)
    assert result.cost == pytest.approx(1.5)


# PATH_P 成功却没有 continuation: 不得发出空 Q 缓存的成功结果。
def test_p_success_without_continuation_is_failure(node_factory, station, monkeypatch):
    node = node_factory()
    monkeypatch.setattr(node.type3_planner, "plan", lambda *args, **kwargs: _fake_plan(None))
    monkeypatch.setattr(node_module, "_checked_trajectory", lambda *args, **kwargs: "peaks")
    result = AssemblyResult()
    success, message = node._plan_assembly(_p_request(station), result)
    assert not success
    assert result.error_code == node_module.ERROR_PLANNER_FAILED
    assert len(result.q_sweep_s) == 0
