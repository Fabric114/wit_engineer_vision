# test_collision_safety.py — 稠密轨迹碰撞复检、桩体网格入场景、装配场景已知缺口的回归。
"""碰撞安全回归 (第二轮, docs/audits/2026-10-09/collision-fix-spec.md)。

1. 稠密轨迹逐点碰撞复检 (validate_trajectory 的 collision_checker);
2. 局部桩体网格确实能进入场景;
3. 装配场景不含桩体 —— 锁定"已知缺口"的现状。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("hppfcl")

from planning import ArmModel, FixedDurationParameterizer, Type3Planner, load_config  # noqa: E402
from planning.collision import CollisionChecker, CollisionModel  # noqa: E402
from planning.trajectory import validate_trajectory  # noqa: E402
from planning.type3 import AssemblyPath, AssemblyState  # noqa: E402

# 仓库根目录 (test -> planning -> src -> 根), 用于读 config/sim.yaml。
REPO_ROOT = Path(__file__).resolve().parents[3]
#: 2026-10-09 审计用的站姿 (当时 config/sim.yaml 的 station.from_joint, 站体 z 竖直)。
#: sim 演示已换站姿, 这里固定住旧值, 让审计复现的场景不随 sim 配置漂移。
AUDIT_STATION_JOINTS = np.array([-0.25, 1.971, 1.341, 0.364, 0.257, 0.293])
#: 随机搜出的构型 (seed=3): 只碰桩体、不碰站体
COLUMN_ONLY_JOINTS = np.array([0.2884, 2.5832, 1.4259, 0.6715, -0.2435, -1.1883])


# fixture: 加载默认规划 config。
@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


# fixture: 由 config 构造臂模型。
@pytest.fixture(scope="module")
def arm(config) -> ArmModel:
    return ArmModel.from_config(config["arm"])


# fixture: 由 config 构造碰撞模型 (站体/桩体网格 + 臂胶囊)。
@pytest.fixture(scope="module")
def model(config) -> CollisionModel:
    return CollisionModel.from_config(config["collision"])


# fixture: AUDIT_STATION_JOINTS 的 FK 得到的 TCP 位姿, 用作站体位姿。
@pytest.fixture(scope="module")
def station(arm) -> np.ndarray:
    return arm.forward_kinematics(AUDIT_STATION_JOINTS[None]).tcp_transforms[0]


# fixture: 只含站体网格的碰撞检查器。
@pytest.fixture(scope="module")
def station_checker(arm, model, station) -> CollisionChecker:
    return CollisionChecker(arm, [(model.station_meshes, station)], model.arm_capsules)


# fixture: 站体 + 局部桩体网格 (同一位姿) 的碰撞检查器。
@pytest.fixture(scope="module")
def column_checker(arm, model, station) -> CollisionChecker:
    return CollisionChecker(
        arm,
        [(model.station_meshes, station), (model.local_exchange_meshes, station)],
        model.arm_capsules,
    )


# helper: 按 config 的 exchange.stage_path.joint_path 参数构造 Type3Planner, 指定 IK 分支, 关闭 best_effort。
def _planner(arm, config, branch: int) -> Type3Planner:
    exchange = config["planning"]["exchange"]
    jp = exchange["stage_path"]["joint_path"]
    return Type3Planner(
        arm,
        geometry=config["planning"]["type3"]["exchange_trajectory"],
        ik_branch=branch,
        roll_sample_step_deg=jp["roll_sample_step_deg"],
        bandwidth=jp["bandwidth"],
        collision_soft_margin=jp["collision_soft_margin"],
        collision_soft_weight=jp["collision_soft_weight"],
        motion_l1_weight=jp["motion_l1_weight"],
        motion_l2_weight=jp["motion_l2_weight"],
        joint_limit_margin_rad=jp["joint_limit_margin_rad"],
        joint_limit_weight=jp["joint_limit_weight"],
        best_effort=dict(exchange["stage_path"]["best_ik"], enabled=False),
        start_joint_tolerance_rad=jp["start_joint_tolerance_rad"],
        max_joint_step_rad=jp["max_joint_step_rad"],
    )


# fixture: 返回 (退开 0.15 m -> 插到底 的 10 点装配路径, 起点在 IK 分支 4 下的关节角)。
@pytest.fixture(scope="module")
def insertion(arm, station):
    """sim 站姿: 退开 0.15 m -> 插到底, 分支 4 起点。"""
    path = AssemblyPath.between(AssemblyState(axial_offset_m=0.15), AssemblyState(), samples=10)
    # 起点 = 站体位姿沿其 x 轴退开 0.15 m; 要求分支 4 可解
    start_pose = station.copy()
    start_pose[:3, 3] += 0.15 * station[:3, 0]
    joints, feasible = arm.solve_ik(start_pose[None])
    assert feasible[0, 4]
    return path, joints[0, 4]


# 期望: 不传 collision_checker 时 validate_trajectory 通过; 传入后抛含 "collides" 的 ValueError。
def test_dense_trajectory_collision_is_rejected(arm, station_checker):
    """两端无碰撞、中间穿过站体的轨迹: 路点检查过, 稠密复检必须拒。"""
    # 随机采 3000 个构型, 分成碰撞 / 无碰撞两类
    rng = np.random.default_rng(7)
    samples = rng.uniform(arm.joint_space.lower, arm.joint_space.upper, (3000, 6))
    collides = station_checker.check_configs(samples)
    free = samples[~collides]

    # 以碰撞构型 hit 为中点镜像: mirror = 2*hit - q, 要求 mirror 在限位内且无碰撞
    def endpoint_pair():
        """找一对无碰撞端点, 其中点 (两点五次插值 t=T/2 处) 是碰撞构型。"""
        for hit in samples[collides]:
            mirrors = 2.0 * hit - free
            inside = np.all(
                (mirrors >= arm.joint_space.lower) & (mirrors <= arm.joint_space.upper), axis=1
            )
            for q, mirror in zip(free[inside][:50], mirrors[inside][:50]):
                if not station_checker.check_configs(mirror[None])[0]:
                    return q, mirror
        pytest.fail("no free endpoint pair around a colliding midpoint")

    # 端点本身都无碰撞; 2 s 五次插值、10 ms 采样得到稠密轨迹
    points = np.stack(endpoint_pair())
    assert not np.any(station_checker.check_configs(points))
    trajectory = FixedDurationParameterizer(2.0).parameterize(points, sample_dt=0.01)
    validate_trajectory(trajectory, arm.joint_space, waypoints=points)  # 不给检查器: 不查碰撞
    with pytest.raises(ValueError, match="collides"):
        validate_trajectory(
            trajectory, arm.joint_space, waypoints=points, collision_checker=station_checker
        )


# 期望: COLUMN_ONLY_JOINTS 在 station_checker 下无碰撞, 在 column_checker 下碰撞。
def test_column_mesh_enters_scene(station_checker, column_checker):
    """桩体网格必须真的参与检查: 只碰桩体的构型, 不含桩体判无碰撞, 含桩体判碰撞。"""
    assert not station_checker.check_configs(COLUMN_ONLY_JOINTS[None])[0]
    assert column_checker.check_configs(COLUMN_ONLY_JOINTS[None])[0]


# 审计站姿的分支 4 插入 (2026-10-10 时这里断言"成功 + 稠密复检通过"): 最后一层构型翻转 2.58 rad,
# 碰撞检查只看离散路点和稠密采样, 拦不住这种跳变; 现在由 max_joint_step_rad 拒掉, 与碰撞场景无关。
def test_audit_insertion_branch_flip_rejected_regardless_of_scene(
    arm, config, station, station_checker, insertion
):
    path, start = insertion
    result = _planner(arm, config, 4).plan(path, station, start, station_checker)
    assert not result.success
    assert "max_joint_step" in result.message


# 允许接触规则 (取代 2026-10-10 锁定的"已知缺口"): 在 sim 演示站姿 (config/sim.yaml) 上,
# 桩体全查 (接近段场景) 时插入后段 / 滑动全程被 cc_link6 vs 桩体拒掉; 装配段场景豁免 cc_link6 后
# 插入和滑动都能规划, 且稠密轨迹复检通过。 其他胶囊 vs 桩体、全部胶囊 vs 站体 / 车体照查。
@pytest.mark.parametrize("stage", ["insert", "slide"])
def test_assembly_allowed_contact_rule(arm, config, model, stage):
    with (REPO_ROOT / "config/sim.yaml").open() as stream:
        demo_station = arm.forward_kinematics(
            np.asarray(yaml.safe_load(stream)["station"]["from_joint"], dtype=float)[None]
        ).tcp_transforms[0]
    slide = float(config["planning"]["type3"]["exchange_trajectory"]["slide_z_mag"])
    start_state, goal_state = (
        (AssemblyState(axial_offset_m=0.15), AssemblyState())
        if stage == "insert"
        else (AssemblyState(), AssemblyState(slide_m=slide))
    )
    branch = int(config["planning"]["exchange"]["ik_branches"][0])
    planner = _planner(arm, config, branch)
    pose = planner.task_poses(AssemblyPath(start_state.as_array()[None]), demo_station, np.zeros(1))
    joints, feasible = arm.solve_ik(pose, branches=(branch,))
    assert feasible[0, 0]
    path = AssemblyPath.between(start_state, goal_state, samples=10)

    def checker(scene):
        return CollisionChecker(arm, model.scene_groups(demo_station, stage=scene), model.arm_capsules)

    strict = planner.plan(path, demo_station, joints[0, 0], checker("approach"))
    assert not strict.success
    assert int(np.sum(strict.diagnostics["collision_rejected_by_step"])) > 0
    assembly_checker = checker("assembly")
    result = planner.plan(path, demo_station, joints[0, 0], assembly_checker)
    assert result.success, result.message
    trajectory = FixedDurationParameterizer(6.0).parameterize(result.waypoints, sample_dt=0.001)
    validate_trajectory(
        trajectory, arm.joint_space, waypoints=result.waypoints, collision_checker=assembly_checker
    )


# ---------------------------------------------------------------- 车体 / 地面 / 允许接触规则
def _body_only_checker(arm, model, index):
    return CollisionChecker(arm, [model.body_groups[index]], model.arm_capsules)


def test_body_and_ground_are_in_model(model):
    """config 的 robot_body (底盘 / 滑台 / 4 轮) 与 ground 都进了碰撞模型。"""
    assert len(model.body_groups) == 7
    assert model.assembly_allowed_contact == frozenset({"cc_link6", "cc_ore"})


def test_line_height_matches_sim():
    """planning 的车体 / 地面按 collision.line_m 下移, 必须和 sim 锁定的 line 高度一致。"""
    with (REPO_ROOT / "config/sim.yaml").open() as stream:
        sim_line = float(yaml.safe_load(stream)["arm"]["line_m"])
    assert float(load_config()["collision"]["line_m"]) == pytest.approx(sim_line)


def test_line_lowers_chassis_but_not_lift_carriage(config):
    """line 升高: 底盘 / 轮子 / 地面在 arm_base 系里下移, 升降滑台 line_link 不动。"""
    import copy

    low = CollisionModel.from_config(dict(copy.deepcopy(config["collision"]), line_m=0.0))
    high = CollisionModel.from_config(dict(copy.deepcopy(config["collision"]), line_m=0.38))
    for a, b in zip(low.body_groups, high.body_groups):
        dz = float(np.asarray(b[1])[2, 3] - np.asarray(a[1])[2, 3])
        name = str(a[0][0]).rsplit("/", 1)[-1] if not isinstance(a[0][0], tuple) else "ground"
        expected = 0.0 if name == "line_link.obj" else -0.38
        assert dz == pytest.approx(expected), name


def test_home_pose_is_free_of_robot_body(arm, model):
    """sim 的 home 构型不能被车体误判 (cc_link1 贴着底盘 / 滑台装, 对这两块豁免)。"""
    with (REPO_ROOT / "config/sim.yaml").open() as stream:
        home = np.asarray(yaml.safe_load(stream)["arm"]["home"], dtype=float)
    checker = CollisionChecker(arm, list(model.body_groups), model.arm_capsules)
    assert not checker.check_configs(home[None])[0]


def test_robot_body_detects_contact(arm, model):
    """随机构型里确实有撞底盘的, 完整场景必须把它们判成碰撞。"""
    samples = np.random.default_rng(21).uniform(
        arm.joint_space.lower, arm.joint_space.upper, (3000, 6)
    )
    hits = _body_only_checker(arm, model, 0).check_configs(samples)
    assert np.any(hits), "no random config touches the chassis"
    full = CollisionChecker(arm, list(model.body_groups), model.arm_capsules)
    assert np.all(full.check_configs(samples[hits][:20]))


def test_ground_box_detects_capsules_below(arm, model):
    """地面盒的机制: line=0 时臂最低点离地约 0.33 m, 真实地面碰不到, 所以把地面抬到 TCP 附近验证;
    胶囊整段沉到地面以下也要判碰撞 (实心盒, 不是单个平面)。"""
    from planning.collision import _body_groups

    q = np.array([0.3, 0.8, 1.0, 0.2, 0.3, 0.1])
    fk = arm.forward_kinematics(q[None])
    lowest = float(fk.link_origins[0, 1:, 2].min())
    names = {c.name for c in model.arm_capsules}
    below = _body_groups(None, {"z_m": lowest + 0.02, "thickness_m": 2.0}, names)
    above = _body_groups(None, {"z_m": lowest - 0.3}, names)
    assert CollisionChecker(arm, list(below), model.arm_capsules).check_configs(q[None])[0]
    assert not CollisionChecker(arm, list(above), model.arm_capsules).check_configs(q[None])[0]


def test_allowed_contact_only_exempts_listed_capsules(arm, model, station):
    """允许接触规则只豁免列出的胶囊: cc_link6 vs 桩体不查, 其他胶囊 vs 桩体照查。"""
    exempt = CollisionChecker(
        arm, [(model.local_exchange_meshes, station, {"cc_link6"})], model.arm_capsules
    )
    strict = CollisionChecker(arm, [(model.local_exchange_meshes, station)], model.arm_capsules)
    samples = np.random.default_rng(3).uniform(
        arm.joint_space.lower, arm.joint_space.upper, (4000, 6)
    )
    only6 = CollisionChecker(arm, [(model.local_exchange_meshes, station)],
                             [c for c in model.arm_capsules if c.name == "cc_link6"])
    others = CollisionChecker(arm, [(model.local_exchange_meshes, station)],
                              [c for c in model.arm_capsules if c.name != "cc_link6"])
    h6, ho = only6.check_configs(samples), others.check_configs(samples)
    assert np.any(h6 & ~ho) and np.any(ho)
    np.testing.assert_array_equal(exempt.check_configs(samples), ho)
    np.testing.assert_array_equal(strict.check_configs(samples), h6 | ho)
    # config 笔误 (不存在的胶囊名) 在加载时就报错
    bad = load_config()
    bad = dict(bad["collision"], station=dict(bad["collision"]["station"], assembly_allowed_contact=["no_such"]))
    with pytest.raises(ValueError, match="unknown capsules"):
        CollisionModel.from_config(bad)
