# planning_node.py — 规划主节点 (接近段 BIT* / 装配段 type3 两条请求-结果通道的 ROS 薄层)。
"""轨迹规划部分主节点 (薄 ROS 层): 把 planning 纯算法库接到两条规划通道上。

订阅 -> 发布 (话题名见 README_PRE.md 第五节, 与 task_node 的约定):
    /plan/approach/request (interfaces/ApproachRequest) -> /plan/approach/result (ApproachResult)
    /plan/assembly/request (interfaces/AssemblyRequest) -> /plan/assembly/result (AssemblyResult)

两段的算法分工 (仿 RM2026-Engineer-Assembly-Algorithm 的 arm_exchange_host/planning_node.py):
    接近段 type2: 对请求里的 target_pose 求 IK, 在候选解里挑"离起始关节角最省力"的一组,
        再用 OMPL BIT* 在关节空间搜一条不越限、不碰撞的稀疏路径 (planning/type2.py);
        没装 OMPL 时节点起不来, 不退化成直线。
        碰撞场景按请求里的 station_pose_arm_base 摆 (站体 + 局部桩体), 起点碰撞直接拒绝。
    装配段 type3: 把 AssemblyState 目标铺成流形上的一串状态, 按 roll 离散建分层图,
        Viterbi 选一串连续关节路点, 必要时走 best_ik 尽力修复 (planning/type3.py)。

两段最后都用 FixedDurationParameterizer 做定时长五次多项式采样, 出稠密
trajectory_msgs/JointTrajectory (位置/速度/加速度/time_from_start 齐全)。
发布前过 validate_trajectory: 有限值 / 首末点 / 限位, 并逐个稠密采样点查碰撞。
碰撞检测没有开关, 构不出碰撞模型 (hpp-fcl / 网格 / config 缺失) 时节点起不来。

请求带 request_id, 结果原样回显 (同 RM2026); 请求方据此丢弃迟到/作废的结果。

位姿一律在 arm_base 系; 关节名取 rm26_arm 的 J1..J6 (与 URDF/MJCF 一致)。

【性能】config 里 arm.ik_type=rm26_arm (专用闭式解, 单目标单分支约 7 ms, 批量每支
约 0.1 ms)。 装配段一次请求要解 path_samples × roll 数 次 IK, 批量一次解完;
ik_type=numeric 时约 16 ms/次, 默认 10 × 72 ≈ 15 s。

超参默认全部来自仓库根 config/planning.yaml, 可用 ROS 参数逐项覆盖:
    ros2 run planning planning_node --ros-args -p assembly.path_samples:=6
"""

from __future__ import annotations

# 标准库 / 第三方 / ROS 消息依赖
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory as JointTrajectoryMsg
from trajectory_msgs.msg import JointTrajectoryPoint

# 本仓库 interfaces 包: 与 task_node 约定的请求/结果消息
from interfaces.msg import (
    ApproachRequest,
    ApproachResult,
    AssemblyRequest,
    AssemblyResult,
)
from interfaces.msg import AssemblyState as AssemblyStateMsg

# planning 纯算法库 (与 ROS 无关): 臂模型、type3 规划器、时间参数化、轨迹校验
from planning import (
    ArmModel,
    AssemblyPath,
    AssemblyState,
    FixedDurationParameterizer,
    JointTrajectory as CoreJointTrajectory,
    Type3Planner,
    load_config,
    quaternions_from_rotations,
    rotations_from_quaternions,
    validate_trajectory,
)

#: rm26_arm 的 6 个臂关节 (URDF/MJCF 里的名字; 第 7 轴 line 升降不在本库范围内)
JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
# 所有位姿与轨迹消息的参考坐标系
FRAME_ARM_BASE = "arm_base"
#: 起始关节角落在限位外多少弧度内算"测量噪声", 直接夹回去而不报错
BOUND_TOLERANCE_RAD = 1e-3
#: 接近段选目标时绕插入轴扫的 roll 个数 (每 1°, 同 RM2026)
APPROACH_ROLL_COUNT = 360

# 结果里的机器可读失败分类。空串表示成功; 其余值与 message 文案解耦, 执行侧只认这些值。
ERROR_INVALID_REQUEST = "INVALID_REQUEST"
ERROR_IK_UNREACHABLE = "IK_UNREACHABLE"
ERROR_COLLISION = "COLLISION"
ERROR_TRAJECTORY_INVALID = "TRAJECTORY_INVALID"
ERROR_PLANNER_FAILED = "PLANNER_FAILED"
ERROR_INTERNAL = "INTERNAL"


def _classify_exception(exc: Exception) -> str:
    """把规划异常归到稳定的 error_code (message 仍保留完整上下文给日志/人看)。"""
    message = str(exc)
    # 先判碰撞/轨迹校验: validate_trajectory 抛的是 ValueError, 但语义不是"输入非法"。
    if "碰撞" in message or "collision" in message.lower():
        return ERROR_COLLISION
    if "trajectory" in message.lower() or "校验" in message:
        return ERROR_TRAJECTORY_INVALID
    if isinstance(exc, ValueError):
        return ERROR_INVALID_REQUEST
    if not isinstance(exc, RuntimeError):
        return ERROR_INTERNAL
    # 先判碰撞: "target_pose 的所有 IK 解都与兑换站碰撞" 同时含 IK 和碰撞, 碰撞更具体。
    if "无解" in message or "IK" in message:
        return ERROR_IK_UNREACHABLE
    return ERROR_PLANNER_FAILED


def _classify_plan_failure(plan) -> str:
    """把 type3 的 success=False 结果归到 error_code。"""
    message = str(plan.message)
    if "collision" in message.lower() or "碰撞" in message:
        return ERROR_COLLISION
    if "ik" in message.lower() or "无解" in message:
        return ERROR_IK_UNREACHABLE
    if "start" in message.lower() or "mismatch" in message.lower():
        return ERROR_INVALID_REQUEST
    return ERROR_PLANNER_FAILED


# Pose (位置 m + 四元数) 转 4x4 齐次矩阵; 四元数先归一化, 零/非有限值直接拒绝
def _transform_from_pose(pose: Pose) -> np.ndarray:
    """geometry_msgs/Pose -> 4x4 齐次变换 (arm_base 系)。"""
    # 组成 (1, 4) 的 wxyz 四元数, 匹配 rotations_from_quaternions 的批量输入约定
    quaternion = np.asarray(
        [[pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]],
        dtype=float,
    )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("Pose.orientation 四元数为零或非有限值, 无法转成旋转矩阵")
    # 旋转块取批量结果的第 0 个, 平移块直接写入位置 (m)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotations_from_quaternions(quaternion / norm)[0]
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return transform


# 把消息里的关节角数组转成 (6,) 向量并校验维度与有限性 (单位 rad)
def _joint_positions_from_joint_state(msg: JointState) -> np.ndarray:
    """JointState -> (6,) float64 关节角, 按名字取 J1..J6 (同 RM2026 ros_utils); 多余关节忽略。"""
    by_name = {name: index for index, name in enumerate(msg.name)}
    missing = [name for name in JOINT_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"initial_joint_state 缺关节 {missing} (收到 name={list(msg.name)})")
    if len(msg.position) != len(msg.name):
        raise ValueError(
            f"initial_joint_state 的 position 长度 {len(msg.position)} 与 name 长度 {len(msg.name)} 不一致"
        )
    joints = np.asarray([msg.position[by_name[name]] for name in JOINT_NAMES], dtype=float)
    if not np.all(np.isfinite(joints)):
        raise ValueError(f"initial_joint_state 含非有限值: {joints.tolist()}")
    return joints


# PoseStamped -> 4x4; frame_id 必须是 arm_base (含空串都拒绝), 不再告警后照常解释
def _transform_from_pose_stamped(msg: PoseStamped, label: str) -> np.ndarray:
    """站姿 PoseStamped -> 4x4 齐次变换 (arm_base 系)。"""
    if msg.header.frame_id != FRAME_ARM_BASE:
        raise ValueError(
            f"{label}.header.frame_id 必须是 {FRAME_ARM_BASE!r}, 收到 {msg.header.frame_id!r}"
        )
    return _transform_from_pose(msg.pose)


# 浮点秒拆成 sec + nanosec; 四舍五入可能让 nanosec 进位到 1e9, 需手动进位
def _seconds_to_duration(value: float) -> Duration:
    """秒 -> builtin_interfaces/Duration。"""
    whole = int(value)
    message = Duration()
    message.sec = whole
    message.nanosec = int(round((float(value) - whole) * 1e9))
    if message.nanosec >= 1_000_000_000:
        message.sec += 1
        message.nanosec -= 1_000_000_000
    return message


# 稠密轨迹逐点转成 JointTrajectoryPoint (位置/速度/加速度 + time_from_start)
def _dense_trajectory_to_msg(trajectory: CoreJointTrajectory, *, stamp) -> JointTrajectoryMsg:
    """稠密轨迹 (planning.JointTrajectory) -> trajectory_msgs/JointTrajectory。"""
    # 消息头: 时间戳沿用 result 头, frame_id 固定 arm_base, 关节名 J1..J6
    message = JointTrajectoryMsg()
    message.header.stamp = stamp
    message.header.frame_id = FRAME_ARM_BASE
    message.joint_names = list(JOINT_NAMES)
    # 形状校验: positions 应为 (N, 6), N 为稠密采样点数
    expected = (trajectory.timestamps.shape[0], len(JOINT_NAMES))
    if trajectory.positions.shape != expected:
        raise ValueError(f"稠密轨迹形状应为 {expected}, 实际 {trajectory.positions.shape}")
    # 逐采样点组装; strict=True 保证四个数组长度一致
    for position, velocity, acceleration, timestamp in zip(
        trajectory.positions,
        trajectory.velocities,
        trajectory.accelerations,
        trajectory.timestamps,
        strict=True,
    ):
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in position]
        point.velocities = [float(value) for value in velocity]
        point.accelerations = [float(value) for value in acceleration]
        point.time_from_start = _seconds_to_duration(float(timestamp))
        message.points.append(point)
    return message


# 发布前最后一道关: 调 validate_trajectory, 通过后只汇报峰值速度/加速度
def _checked_trajectory(
    trajectory: CoreJointTrajectory, joint_space, waypoints: np.ndarray, collision_checker
) -> str:
    """发布前的最终校验 (不过则抛 ValueError), 返回峰值速度/加速度摘要。

    逐个稠密采样点查碰撞。 config 没有真实速度/加速度上限, 这里不查,
    只把峰值写进 message 供人工核对。
    """
    # 有限值 / 首末点对齐 waypoints / 关节限位 / 逐点碰撞, 任一不过抛 ValueError
    validate_trajectory(
        trajectory, joint_space, waypoints=waypoints, collision_checker=collision_checker
    )
    # 峰值取所有关节、所有采样点的绝对值最大
    peak_velocity = float(np.max(np.abs(trajectory.velocities)))
    peak_acceleration = float(np.max(np.abs(trajectory.accelerations)))
    return f"peak_vel={peak_velocity:.3f}rad/s peak_acc={peak_acceleration:.3f}rad/s2 (未校验速度/加速度上限)"


# 日志用: 位置保留 4 位小数, 姿态转回 wxyz 四元数
def _pose_summary(transform: np.ndarray) -> str:
    """位姿的一行摘要 (位置 + 四元数), 只用于日志。"""
    quaternion = quaternions_from_rotations(transform[None, :3, :3])[0]
    position = np.round(transform[:3, 3], 4).tolist()
    return f"pos={position} quat_wxyz={np.round(quaternion, 4).tolist()}"


# 日志用: 汇总 type3 分层图每层有效节点数的分布
def _step_counts_summary(counts) -> str:
    """分层图每层有效节点数的紧凑摘要 (min/median/max)。"""
    values = np.asarray(counts, dtype=int).reshape(-1)
    if not values.size:
        return "n/a"
    return f"min={int(values.min())} med={int(np.median(values))} max={int(values.max())}"


# ROS2 节点: 读 config + ROS 参数覆盖, 构造算法对象, 挂两组订阅/发布
class PlanningNode(Node):
    """规划节点: 接近段 (BIT*) 与装配段 (type3 流形) 两条请求/结果通道。"""

    # 初始化: 读配置 -> 声明参数 -> 构造规划器/碰撞模型 -> 建话题
    def __init__(self) -> None:
        super().__init__("planning_node")

        # config 默认来自仓库根 config/planning.yaml; 按用途取出各子段
        config = load_config()
        arm_cfg = config["arm"]
        exchange_cfg = config["planning"]["exchange"]
        geometry_cfg = config["planning"]["type3"]["exchange_trajectory"]
        stage_path_cfg = exchange_cfg["stage_path"]
        joint_path_cfg = stage_path_cfg["joint_path"]
        trajectory_cfg = stage_path_cfg["trajectory"]
        approach_cfg = exchange_cfg["approach"]

        # ---- ROS 参数 (默认值一律取自 config, 这里只做"可覆盖") ----
        # 接近段: 轨迹总时长 (s)、稠密采样步长 (s)、BIT* 默认超时 (s)
        self.approach_duration_s = self._declare_float(
            "approach.duration_s", approach_cfg["duration_s"]
        )
        self.approach_sample_dt = self._declare_float(
            "approach.sample_dt", approach_cfg["sample_dt"]
        )
        self.approach_bit_timeout_s = self._declare_float(
            "approach.bit_timeout_s", approach_cfg["bit_timeout_s"]
        )
        # 接近段目标: 沿站轴退开 insert_slide_mag 的待插状态 (= 装配手动插入的起点)
        self.approach_axial_m = float(geometry_cfg["insert_slide_mag"])
        # 装配段: 轨迹总时长 (s)、稠密采样步长 (s)
        self.assembly_duration_s = self._declare_float(
            "assembly.duration_s", trajectory_cfg["duration_s"]
        )
        self.assembly_sample_dt = self._declare_float(
            "assembly.sample_dt", trajectory_cfg["sample_dt"]
        )
        # 流形上插值多少个状态: 直接决定 type3 建图的 IK 次数 (samples × roll 数)
        self.declare_parameter("assembly.path_samples", 10)
        self.assembly_path_samples = max(2, int(self.get_parameter("assembly.path_samples").value))
        # 装配段的起点状态: 默认取几何里的插入行程, 即"沿站轴退开 insert_slide_mag 的待插位姿"
        # —— 正好是接近段 target_pose 该给的那个点, 两段因此能首尾相接。
        self.assembly_start_axial_m = self._declare_float(
            "assembly.start_axial_m", geometry_cfg["insert_slide_mag"]
        )
        # goal.release=true 时给流形的释放位移 (m); 默认 0 表示几何上不动,
        # 真正的吸盘/夹爪动作由执行侧 (gripper/arm_host) 负责。
        self.assembly_release_m = self._declare_float("assembly.release_m", 0.0)
        # type3 分层图按 roll 离散的步长 (deg), 决定 roll 层数
        self.assembly_roll_step_deg = self._declare_float(
            "assembly.roll_step_deg", joint_path_cfg["roll_sample_step_deg"]
        )
        # 接近段 IK 的分支: 默认与装配段同一套 (config 的 planning.exchange.ik_branches), 保证接近段
        # 停下的构型就是装配段首层的候选, 两段能衔接 (RM2026 同做法); 在这些分支的有效无碰撞解里
        # 挑运动代价最小的。 ROS 参数可覆盖, 但覆盖成别的分支, 装配段的首点衔接检查多半会拒。
        self.declare_parameter(
            "approach.ik_branches", [int(value) for value in exchange_cfg["ik_branches"]]
        )
        self.approach_ik_branches = tuple(
            int(value) for value in self.get_parameter("approach.ik_branches").value
        )
        if not self.approach_ik_branches:
            raise ValueError("approach.ik_branches 不能为空")
        # ---- 算法对象 ----
        # 臂模型 (FK/IK/关节空间), ik_type 由 config 的 arm.ik_type 决定
        self.arm = ArmModel.from_config(arm_cfg)
        # 运动代价权重: 接近段挑 IK 解与 type3 建图共用同一组 L1/L2 权重
        self.motion_l1_weight = float(joint_path_cfg["motion_l1_weight"])
        self.motion_l2_weight = float(joint_path_cfg["motion_l2_weight"])
        # 装配段 IK 分支来自 config, 必须恰好一个
        ik_branches = [int(value) for value in exchange_cfg["ik_branches"]]
        if len(ik_branches) != 1:
            raise ValueError("type3 建图只支持一个 IK 分支, 请检查 planning.exchange.ik_branches")
        # type3 规划器: 几何、roll 离散、Viterbi 带宽、碰撞软约束、限位软约束、best_ik 修复参数全从 config 注入
        self.type3_planner = Type3Planner(
            self.arm,
            geometry=geometry_cfg,
            ik_branch=ik_branches[0],
            roll_sample_step_deg=self.assembly_roll_step_deg,
            bandwidth=int(joint_path_cfg["bandwidth"]),
            collision_soft_margin=float(joint_path_cfg["collision_soft_margin"]),
            collision_soft_weight=float(joint_path_cfg["collision_soft_weight"]),
            motion_l1_weight=self.motion_l1_weight,
            motion_l2_weight=self.motion_l2_weight,
            joint_limit_margin_rad=float(joint_path_cfg["joint_limit_margin_rad"]),
            joint_limit_weight=float(joint_path_cfg["joint_limit_weight"]),
            best_effort=dict(stage_path_cfg.get("best_ik", {})),
            start_joint_tolerance_rad=joint_path_cfg.get("start_joint_tolerance_rad"),
            max_joint_step_rad=joint_path_cfg.get("max_joint_step_rad"),
        )
        # 两段各一个定时长五次多项式参数化器
        self.approach_parameterizer = FixedDurationParameterizer(self.approach_duration_s)
        self.assembly_parameterizer = FixedDurationParameterizer(self.assembly_duration_s)
        # 碰撞模型: 依赖 hpp-fcl 与站体网格 (config 的 collision.station.asset_dir,
        # 即 src/runtime/sim/model/exchange_station/); 构造失败直接让节点起不来
        self.collision_model = self._load_collision_model(config.get("collision"))
        # OMPL 是接近段唯一的路径规划后端, 启动时就确认装了, 不等第一次请求才失败
        try:
            import ompl.geometric  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(f"接近段需要 OMPL (BIT*), 导入失败: {exc}") from exc

        # BIT* 的采样边界: 连续关节 (config 里限位写 null) 折到 [-pi, pi]
        self.joint_lower = np.where(
            self.arm.joint_space.continuous, -np.pi, self.arm.joint_space.lower
        )
        self.joint_upper = np.where(
            self.arm.joint_space.continuous, np.pi, self.arm.joint_space.upper
        )

        # ---- 话题 ----
        # 接近段: 订阅 /plan/approach/request, 结果发到 /plan/approach/result (QoS depth=10)
        self.approach_result_pub = self.create_publisher(
            ApproachResult, "/plan/approach/result", 10
        )
        self.create_subscription(
            ApproachRequest, "/plan/approach/request", self._on_approach_request, 10
        )
        # 装配段: 订阅 /plan/assembly/request, 结果发到 /plan/assembly/result
        self.assembly_result_pub = self.create_publisher(
            AssemblyResult, "/plan/assembly/result", 10
        )
        self.create_subscription(
            AssemblyRequest, "/plan/assembly/request", self._on_assembly_request, 10
        )

        # 启动日志: 打印关键超参, 方便现场核对实际生效的配置
        self.get_logger().info(
            "planning_node 就绪 "
            f"ik_type={self.arm.ik_type} "
            f"type3_ik_branch={ik_branches[0]} "
            f"roll_count={len(self.type3_planner.rolls)} "
            f"assembly_path_samples={self.assembly_path_samples} "
            f"assembly_start_axial_m={self.assembly_start_axial_m:.4f} "
            f"approach_ik_branches={list(self.approach_ik_branches)}"
        )

    # 声明一个 float 型 ROS 参数 (默认值来自 config) 并返回实际生效值
    def _declare_float(self, name: str, default) -> float:
        self.declare_parameter(name, float(default))
        return float(self.get_parameter(name).value)

    # 构造全局碰撞模型 (站体网格 + 局部桩体网格 + 臂胶囊体), 整个节点生命周期复用
    def _load_collision_model(self, collision_cfg):
        """构造碰撞模型; 构不出来 (hpp-fcl / 网格 / config 缺失) 直接抛异常, 不允许降级成"不查碰撞"。"""
        # 缺配置/依赖: 抛异常让节点启动失败; hpp-fcl 依赖延迟导入
        if not collision_cfg:
            raise RuntimeError("config 里没有 collision 段, 规划必须查碰撞")
        from planning.collision import CollisionModel

        return CollisionModel.from_config(collision_cfg)

    # 按本次请求的站位姿 (arm_base 系 4x4) 摆放网格, 建一次性检查器
    def _make_collision_checker(self, station_transform: np.ndarray, *, stage: str):
        """给定兑换站位姿建一次性碰撞检查器。

        场景见 CollisionModel.scene_groups: 站体 + 桩体 + 车体 / 地面; stage="assembly" 时桩体对
        config 的 collision.station.assembly_allowed_contact 豁免 (允许接触规则)。
        """
        from planning.collision import CollisionChecker

        return CollisionChecker(
            self.arm,
            self.collision_model.scene_groups(station_transform, stage=stage),
            self.collision_model.arm_capsules,
        )

    # 输入 (6,) 或 (B, 6), 输出 (B, 6); 超限超过 BOUND_TOLERANCE_RAD 才报错, 噪声范围内夹回
    def _to_type2_bounds(self, joints: np.ndarray) -> np.ndarray:
        """把关节角折进 BIT* 的采样盒: 连续关节折到 [-pi, pi], 有限位关节夹到限位内。"""
        # 先把连续关节 wrap 到 [-pi, pi]
        values = self.arm.joint_space.wrap(np.atleast_2d(np.asarray(joints, dtype=float)))
        bounded = self.arm.joint_space.bounded
        # 只对有限位关节判超限 (容差 BOUND_TOLERANCE_RAD)
        outside = bounded & (
            (values < self.joint_lower - BOUND_TOLERANCE_RAD)
            | (values > self.joint_upper + BOUND_TOLERANCE_RAD)
        )
        if np.any(outside):
            raise ValueError(
                f"关节角超出限位: {np.round(values[np.any(outside, axis=1)], 4).tolist()}"
            )
        # 容差内的轻微越界夹回限位, 保证 BIT* 起终点落在采样盒内
        values[..., bounded] = np.clip(
            values[..., bounded], self.joint_lower[bounded], self.joint_upper[bounded]
        )
        return values

    # candidates (B, 6) 相对 reference (6,) 的代价: w1*sum|d| + w2*sum d^2, 返回 (B,)
    def _motion_cost(self, candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """候选关节角相对参考位姿的运动代价 (连续关节按最近折算, 权重取自 config)。"""
        delta = self.arm.joint_space.delta(candidates, reference[None, :])
        return self.motion_l1_weight * np.sum(np.abs(delta), axis=-1) + (
            self.motion_l2_weight * np.sum(delta * delta, axis=-1)
        )

    # frame_id 非空且不是 arm_base 时只告警, 不做坐标变换
    def _check_frame(self, frame_id: str, label: str) -> None:
        if frame_id and frame_id != FRAME_ARM_BASE:
            self.get_logger().warning(
                f"{label} 的 frame_id={frame_id!r} 不是 {FRAME_ARM_BASE!r}, 仍按 {FRAME_ARM_BASE} 解释"
            )

    # ------------------------------------------------------------------ 接近段
    # 接近段回调: 调 _plan_approach, 无论成败都发布一条 ApproachResult
    def _on_approach_request(self, msg: ApproachRequest) -> None:
        # 结果头: 当前时间 + arm_base, 稠密轨迹消息复用此时间戳
        result = ApproachResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = FRAME_ARM_BASE
        result.request_id = msg.request_id
        result.repaired_nodes = 0  # 接近段没有 best_ik 修复
        result.error_code = ""
        started = time.perf_counter()
        try:
            result.message, result.cost = self._plan_approach(msg, result)
            result.success = True
        except Exception as exc:  # 任何一步失败都回一条失败结果, 不让节点挂掉
            result.success = False
            result.message = f"{type(exc).__name__}: {exc}"
            result.cost = float("inf")
            result.error_code = _classify_exception(exc)
            result.trajectory = JointTrajectoryMsg()
            self.get_logger().warning(
                f"接近段规划失败 id={msg.request_id} error_code={result.error_code}: {result.message}"
            )
        # 成功分支: 记录稠密点数、轨迹时长、规划耗时
        else:
            self.get_logger().info(
                "接近段规划成功 "
                f"id={msg.request_id} "
                f"dense_steps={len(result.trajectory.points)} "
                f"duration_s={self.approach_duration_s:.3f} "
                f"elapsed_s={time.perf_counter() - started:.3f} "
                f"message={result.message}"
            )
        self.approach_result_pub.publish(result)

    # 流程: 输入校验 -> 碰撞场景 -> 扫 roll 选终点 -> BIT* 路径 -> 参数化 -> 校验 -> 填消息
    def _plan_approach(self, msg: ApproachRequest, result: ApproachResult) -> tuple[str, float]:
        """规划接近段并填好 result.trajectory, 返回 (result.message 的说明, 终点运动代价)。"""
        # 输入校验与预处理: 起始关节角折进采样盒, 站位姿转 4x4
        self._check_frame(msg.header.frame_id, "ApproachRequest.header")
        start_joint = self._to_type2_bounds(
            _joint_positions_from_joint_state(msg.initial_joint_state)
        )[0]
        station = _transform_from_pose_stamped(
            msg.station_pose_arm_base, "ApproachRequest.station_pose_arm_base"
        )
        self.get_logger().info(
            "接近段请求 "
            f"start_joint={np.round(start_joint, 4).tolist()} "
            f"station_{_pose_summary(station)} "
            f"axial_m={self.approach_axial_m:.4f} "
            f"timeout_s={self.approach_bit_timeout_s:.3f}"
        )

        # 按站位姿建检查器 (含局部桩体), 并先验起点不碰撞
        collision_checker = self._make_collision_checker(station, stage="approach")
        if bool(collision_checker.check_configs(start_joint[None, :])[0]):
            raise RuntimeError("接近段起点 start_joint 已与兑换站碰撞")
        # 在待插位 × 360 roll 的 IK 候选里选终点关节角 (有效 + 无碰撞 + 运动代价最小)
        goal_joint, goal_cost, goal_roll = self._select_approach_goal(
            station, start_joint, collision_checker
        )
        self.get_logger().info(
            "接近段目标关节角 "
            f"goal_joint={np.round(goal_joint, 4).tolist()} motion_cost={goal_cost:.6f} "
            f"goal_roll_deg={np.rad2deg(goal_roll):.1f}"
        )

        # BIT* 的状态合法性回调: 单个构型查碰撞
        def validity_fn(joints: np.ndarray) -> bool:
            return not bool(
                collision_checker.check_configs(np.asarray(joints, dtype=float)[None, :])[0]
            )

        # OMPL BIT* 搜稀疏路径 waypoints (K, 6); OMPL 已在启动时确认可用
        from planning.type2 import plan_joint_path_bitstar

        waypoints = plan_joint_path_bitstar(
            start_joint,
            goal_joint,
            validity_fn=validity_fn,
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
            timeout_s=self.approach_bit_timeout_s,
        )

        # 稀疏路径 -> 定时长稠密轨迹, 校验通过后转成消息填入 result
        trajectory = self.approach_parameterizer.parameterize(
            waypoints,
            sample_dt=self.approach_sample_dt,
            duration_s=self.approach_duration_s,
        )
        peaks = _checked_trajectory(trajectory, self.arm.joint_space, waypoints, collision_checker)
        result.trajectory = _dense_trajectory_to_msg(trajectory, stamp=result.header.stamp)
        self.get_logger().info(
            f"接近段路径 sparse_steps={int(waypoints.shape[0])} "
            f"dense_steps={int(trajectory.positions.shape[0])} "
            f"sample_dt={self.approach_sample_dt:.4f}"
        )
        # message = 规划说明 + 峰值摘要
        return f"ok {peaks}", goal_cost

    def _select_approach_goal(
        self,
        station: np.ndarray,
        start_joint: np.ndarray,
        collision_checker,
    ) -> tuple[np.ndarray, float, float]:
        """待插位绕插入轴扫 360 个 roll 求 IK, 在有效且无碰撞的候选里挑运动代价最小的一组。

        返回 (终点关节角 (6,), 运动代价, roll [rad])。 同 RM2026 _select_approach_goal;
        不检查同一 roll 下"插到底"是否可解 (见 todo.md §3)。
        """
        rolls = np.arange(APPROACH_ROLL_COUNT, dtype=float) * (2.0 * np.pi / APPROACH_ROLL_COUNT)
        approach_state = AssemblyState(axial_offset_m=self.approach_axial_m)
        targets = self.type3_planner.task_poses(
            AssemblyPath(approach_state.as_array()[None, :]), station, rolls
        )
        # 批量 IK: joints (R, n_branches, 6) / valid (R, n_branches); 展平后下标 // n_branches = roll 下标
        joints, valid = self.arm.solve_ik(targets, branches=self.approach_ik_branches)
        flat_index = np.flatnonzero(valid.reshape(-1))
        if not len(flat_index):
            raise RuntimeError(
                f"接近段在 IK 分支 {list(self.approach_ik_branches)} 上 {APPROACH_ROLL_COUNT} 个 roll 均无解 "
                f"(ik_type={self.arm.ik_type}; 待插位 axial={self.approach_axial_m:.3f} m), 站姿可能超出工作空间"
            )
        # 候选折进采样盒 (超限会抛错), 再剔除碰撞解
        candidates = self._to_type2_bounds(joints.reshape(-1, 6)[flat_index])
        free = ~collision_checker.check_configs(candidates)
        if not np.any(free):
            raise RuntimeError("接近段所有 IK 候选都与兑换站碰撞")
        candidates = candidates[free]
        flat_index = flat_index[free]
        # 按相对起始关节角的运动代价挑最省力的一组
        costs = self._motion_cost(candidates, start_joint)
        best = int(np.argmin(costs))
        roll = float(rolls[int(flat_index[best]) // valid.shape[1]])
        return candidates[best], float(costs[best]), roll

    # ------------------------------------------------------------------ 装配段
    # 装配段回调: 规划失败分两类 (plan.success=False 返回 / 抛异常), 都发布结果
    def _on_assembly_request(self, msg: AssemblyRequest) -> None:
        result = AssemblyResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = FRAME_ARM_BASE
        result.request_id = msg.request_id
        result.repaired_nodes = 0
        result.error_code = ""
        started = time.perf_counter()
        try:
            success, message = self._plan_assembly(msg, result)
        except Exception as exc:
            success, message = False, f"{type(exc).__name__}: {exc}"
            result.error_code = _classify_exception(exc)
            result.trajectory = JointTrajectoryMsg()
            self.get_logger().warning(
                f"装配段规划异常 id={msg.request_id} error_code={result.error_code}: {message}"
            )
        # 统一写 success/message, 按成败选日志级别, 最后发布
        result.success = success
        result.message = message
        level = self.get_logger().info if success else self.get_logger().warning
        level(
            f"装配段规划{'成功' if success else '失败'} "
            f"id={msg.request_id} "
            f"error_code={result.error_code} "
            f"repaired_nodes={result.repaired_nodes} "
            f"dense_steps={len(result.trajectory.points)} "
            f"elapsed_s={time.perf_counter() - started:.3f} "
            f"message={message}"
        )
        self.assembly_result_pub.publish(result)

    # 流程: 输入校验 -> 构造流形路径 -> type3 建图规划 -> 参数化 -> 校验 -> 填消息
    def _plan_assembly(self, msg: AssemblyRequest, result: AssemblyResult) -> tuple[bool, str]:
        """在装配流形上规划, 填好 result.trajectory, 返回 (是否成功, 说明)。"""
        self._check_frame(msg.header.frame_id, "AssemblyRequest.header")
        # 输入解析: 起始关节角 (6,)、站位姿 4x4 (arm_base 系)、目标流形状态
        start_joint = _joint_positions_from_joint_state(msg.initial_joint_state)
        station = _transform_from_pose_stamped(
            msg.station_pose_arm_base, "AssemblyRequest.station_pose_arm_base"
        )
        goal_state = self._assembly_state(msg.goal)
        # 起点: use_start=false 取"沿站轴退开 start_axial_m 的待插状态", 与接近段的落点衔接;
        # use_start=true 取请求给的 start (如"已插到底"再接滑动段)。
        # 终点由请求给定, 中间线性铺开 path_samples 个状态交给 type3 建图。
        start_state = (
            self._assembly_state(msg.start)
            if bool(msg.use_start)
            else AssemblyState(axial_offset_m=self.assembly_start_axial_m)
        )
        path = AssemblyPath.between(start_state, goal_state, samples=self.assembly_path_samples)
        # 请求摘要日志: 角度类状态转 deg 显示
        self.get_logger().info(
            "装配段请求 "
            f"start_joint={np.round(start_joint, 4).tolist()} "
            f"station_{_pose_summary(station)} "
            f"goal=(axial={goal_state.axial_offset_m:.4f} slide={goal_state.slide_m:.4f} "
            f"p={np.rad2deg(goal_state.p_angle_rad):.2f}deg "
            f"q={np.rad2deg(goal_state.q_angle_rad):.2f}deg "
            f"release={goal_state.release_m:.4f}) "
            f"path_samples={self.assembly_path_samples} "
            f"roll_count={len(self.type3_planner.rolls)}"
        )

        # 装配段检查器**放**局部桩体, 但对 assembly_allowed_contact 里的胶囊豁免
        # (原因见 _make_collision_checker)
        collision_checker = self._make_collision_checker(station, stage="assembly")
        plan = self.type3_planner.plan(
            path,
            station,
            initial_joints=start_joint,
            collision_checker=collision_checker,
        )
        # 先打诊断再判成败, 失败时日志里能看出是 IK 不可达还是碰撞挡住
        self._log_assembly_diagnostics(plan)
        # best_ik 修复节点数: 严格 Viterbi 成功时为 0; 失败结果也保留尝试修复的数量供诊断
        repair = plan.diagnostics.get("repair")
        if repair is not None:
            result.repaired_nodes = int(repair.get("repair_count", 0))
        if not plan.success:
            result.error_code = _classify_plan_failure(plan)
            return False, plan.message

        # Viterbi 路点 -> 定时长稠密轨迹; 最终校验失败按规划失败返回而非抛异常
        trajectory = self.assembly_parameterizer.parameterize(
            plan.waypoints,
            sample_dt=self.assembly_sample_dt,
            duration_s=self.assembly_duration_s,
        )
        try:
            peaks = _checked_trajectory(
                trajectory, self.arm.joint_space, plan.waypoints, collision_checker
            )
        except ValueError as exc:
            result.error_code = ERROR_TRAJECTORY_INVALID
            return False, f"final trajectory check failed: {exc}"
        # 填消息并记录路点数、稠密点数、图代价与每个路点的 roll (deg)
        result.trajectory = _dense_trajectory_to_msg(trajectory, stamp=result.header.stamp)
        self.get_logger().info(
            f"装配段路径 waypoint_steps={int(plan.waypoints.shape[0])} "
            f"dense_steps={int(trajectory.positions.shape[0])} "
            f"cost={float(plan.cost):.6f} "
            f"repaired_nodes={result.repaired_nodes} "
            f"roll_deg={np.round(np.rad2deg(plan.rolls), 2).tolist()}"
        )
        result.error_code = ""
        return True, f"{plan.message} {peaks}"

    def _assembly_state(self, goal: AssemblyStateMsg) -> AssemblyState:
        """interfaces/AssemblyState -> planning 的流形状态。

        msg 里 release 是布尔 (吸盘要不要松), 流形几何要的是"释放位移";
        用 ROS 参数 assembly.release_m 给这个位移, 默认 0 = 几何上不额外让开。
        """
        # 单位: axial/slide/release 为 m, p/q 角为 rad (直接沿用消息字段, 未做换算)
        return AssemblyState(
            axial_offset_m=float(goal.axial),
            slide_m=float(goal.slide),
            p_angle_rad=float(goal.p_angle),
            q_angle_rad=float(goal.q_angle),
            release_m=self.assembly_release_m if bool(goal.release) else 0.0,
        )

    def _log_assembly_diagnostics(self, plan) -> None:
        """打印分层图/尽力修复的诊断, 失败时据此判断是 IK 不可达还是碰撞挡住。"""
        # 建图统计: 总节点 / IK 有效 / 碰撞剔除 / 最终有效, 以及每层有效数分布
        diagnostics = plan.diagnostics
        self.get_logger().info(
            "装配段建图 "
            f"valid_nodes={int(diagnostics['final_valid_nodes'])}/{int(diagnostics['total_nodes'])} "
            f"ik_valid_nodes={int(diagnostics['ik_valid_nodes'])} "
            f"collision_rejected={int(diagnostics['collision_rejected_nodes'])} "
            f"path_samples={int(diagnostics['path_samples'])} "
            f"roll_count={int(diagnostics['roll_count'])} "
            f"ik_branch={int(diagnostics['ik_branch'])} "
            f"per_step_valid({_step_counts_summary(diagnostics['final_valid_by_step'])})"
        )
        # 无 repair 说明严格 Viterbi 已成功, 未走 best_ik
        repair = diagnostics.get("repair")
        if repair is None:
            return
        # repair 失败: 打印拒绝原因与最大位置 (mm) / 轴向 (deg) 误差
        if not bool(repair["success"]):
            self.get_logger().warning(
                f"best_ik 修复被拒 reason={repair['reject_reason']} "
                f"max_pos_mm={float(repair['max_pos_error_m']) * 1000.0:.3f} "
                f"max_axis_deg={np.rad2deg(float(repair['max_axis_error_rad'])):.3f}"
            )
        # repair 成功: 打印修复节点数与各项最大误差, 用 warning 提示该结果非严格解
        else:
            self.get_logger().warning(
                "严格 Viterbi 失败, 已用 best_ik 尽力修复 "
                f"repaired_nodes={int(repair['repair_count'])} "
                f"max_pos_mm={float(repair['max_pos_error_m']) * 1000.0:.3f} "
                f"max_axis_deg={np.rad2deg(float(repair['max_axis_error_rad'])):.3f} "
                f"max_roll_deg={np.rad2deg(float(repair['max_roll_error_rad'])):.3f}"
            )


# 入口: 初始化 rclpy, spin 节点, Ctrl-C 退出时清理
def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()




