"""轨迹规划部分主节点 (薄 ROS 层): 把 planning 纯算法库接到两条规划通道上。

订阅 -> 发布 (话题名见 README_PRE.md 第五节, 与 task_node 的约定):
    /plan/approach/request (interfaces/ApproachRequest) -> /plan/approach/result (ApproachResult)
    /plan/assembly/request (interfaces/AssemblyRequest) -> /plan/assembly/result (AssemblyResult)

两段的算法分工 (仿 RM2026-Engineer-Assembly-Algorithm 的 arm_exchange_host/planning_node.py):
    接近段 type2: 对请求里的 target_pose 求 IK, 在候选解里挑"离起始关节角最省力"的一组,
        再用 OMPL BIT* 在关节空间搜一条不越限、不碰撞的稀疏路径 (planning/type2.py);
        没装 OMPL 时退化成 起点->终点 直线插值, 并在 result.message 里注明。
    装配段 type3: 把 AssemblyState 目标铺成流形上的一串状态, 按 roll 离散建分层图,
        Viterbi 选一串连续关节路点, 必要时走 best_ik 尽力修复 (planning/type3.py)。

两段最后都用 FixedDurationParameterizer 做定时长五次多项式采样, 出稠密
trajectory_msgs/JointTrajectory (位置/速度/加速度/time_from_start 齐全)。

位姿一律在 arm_base 系; 关节名取 rm26_arm 的 J1..J6 (与 URDF/MJCF 一致)。

【性能】config 里 arm.use_analytic_ik=false (rm26_arm 是偏置腕, 解析 IK 不适用),
数值 IK 约 16 ms/次。 装配段一次请求要解 path_samples × roll 数 次 IK, 默认
10 × 72 ≈ 15 s; 嫌慢就调小 assembly.path_samples 或调大 assembly.roll_step_deg。

超参默认全部来自 config/planning.example.yaml, 可用 ROS 参数逐项覆盖:
    ros2 run planning planning_node --ros-args -p assembly.path_samples:=6
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory as JointTrajectoryMsg
from trajectory_msgs.msg import JointTrajectoryPoint

from interfaces.msg import (
    ApproachRequest,
    ApproachResult,
    AssemblyRequest,
    AssemblyResult,
)
from interfaces.msg import AssemblyState as AssemblyStateMsg

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
)

#: rm26_arm 的 6 个臂关节 (URDF/MJCF 里的名字; 第 7 轴 line 升降不在本库范围内)
JOINT_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6")
FRAME_ARM_BASE = "arm_base"
#: 起始关节角落在限位外多少弧度内算"测量噪声", 直接夹回去而不报错
BOUND_TOLERANCE_RAD = 1e-3

def _transform_from_pose(pose: Pose) -> np.ndarray:
    """geometry_msgs/Pose -> 4x4 齐次变换 (arm_base 系)。"""
    quaternion = np.asarray(
        [[pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]],
        dtype=float,
    )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("Pose.orientation 四元数为零或非有限值, 无法转成旋转矩阵")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotations_from_quaternions(quaternion / norm)[0]
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return transform


def _joint_array(values: Sequence[float]) -> np.ndarray:
    """float32[6] -> (6,) float64 关节角向量。"""
    joints = np.asarray(values, dtype=float).reshape(-1)
    if joints.shape != (len(JOINT_NAMES),):
        raise ValueError(f"关节角必须是 {len(JOINT_NAMES)} 维, 收到 {joints.shape}")
    if not np.all(np.isfinite(joints)):
        raise ValueError("关节角含非有限值")
    return joints


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


def _dense_trajectory_to_msg(trajectory: CoreJointTrajectory, *, stamp) -> JointTrajectoryMsg:
    """稠密轨迹 (planning.JointTrajectory) -> trajectory_msgs/JointTrajectory。"""
    message = JointTrajectoryMsg()
    message.header.stamp = stamp
    message.header.frame_id = FRAME_ARM_BASE
    message.joint_names = list(JOINT_NAMES)
    expected = (trajectory.timestamps.shape[0], len(JOINT_NAMES))
    if trajectory.positions.shape != expected:
        raise ValueError(f"稠密轨迹形状应为 {expected}, 实际 {trajectory.positions.shape}")
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


def _pose_summary(transform: np.ndarray) -> str:
    """位姿的一行摘要 (位置 + 四元数), 只用于日志。"""
    quaternion = quaternions_from_rotations(transform[None, :3, :3])[0]
    position = np.round(transform[:3, 3], 4).tolist()
    return f"pos={position} quat_wxyz={np.round(quaternion, 4).tolist()}"


def _step_counts_summary(counts) -> str:
    """分层图每层有效节点数的紧凑摘要 (min/median/max)。"""
    values = np.asarray(counts, dtype=int).reshape(-1)
    if not values.size:
        return "n/a"
    return f"min={int(values.min())} med={int(np.median(values))} max={int(values.max())}"


class PlanningNode(Node):
    """规划节点: 接近段 (BIT*) 与装配段 (type3 流形) 两条请求/结果通道。"""

    def __init__(self) -> None:
        super().__init__("planning_node")

        config = load_config()
        arm_cfg = config["arm"]
        exchange_cfg = config["planning"]["exchange"]
        geometry_cfg = config["planning"]["type3"]["exchange_trajectory"]
        stage_path_cfg = exchange_cfg["stage_path"]
        joint_path_cfg = stage_path_cfg["joint_path"]
        trajectory_cfg = stage_path_cfg["trajectory"]
        approach_cfg = exchange_cfg["approach"]

        # ---- ROS 参数 (默认值一律取自 config, 这里只做"可覆盖") ----
        self.approach_duration_s = self._declare_float(
            "approach.duration_s", approach_cfg["duration_s"]
        )
        self.approach_sample_dt = self._declare_float(
            "approach.sample_dt", approach_cfg["sample_dt"]
        )
        self.approach_bit_timeout_s = self._declare_float(
            "approach.bit_timeout_s", approach_cfg["bit_timeout_s"]
        )
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
        self.assembly_roll_step_deg = self._declare_float(
            "assembly.roll_step_deg", joint_path_cfg["roll_sample_step_deg"]
        )
        # 接近段 IK 的分支: 数值 IK 下"分支"只是不同初值, 多撒几个初值成功率明显更高,
        # 故默认 8 个全上 (一次约 0.2 s); 装配段建图只吃单分支, 用 config 的 ik_branches。
        self.declare_parameter("approach.ik_branches", [0, 1, 2, 3, 4, 5, 6, 7])
        self.approach_ik_branches = tuple(
            int(value) for value in self.get_parameter("approach.ik_branches").value
        )
        if not self.approach_ik_branches:
            raise ValueError("approach.ik_branches 不能为空")
        # 碰撞检测默认关: 依赖 hpp-fcl 与 planning/assets/collision/*.obj,
        # 本仓库暂无这些网格资产 (见 config 的 collision.station), 开了会直接构不出来。
        self.declare_parameter("collision.enabled", False)
        self.collision_enabled = bool(self.get_parameter("collision.enabled").value)

        # ---- 算法对象 ----
        self.arm = ArmModel.from_config(arm_cfg)
        self.motion_l1_weight = float(joint_path_cfg["motion_l1_weight"])
        self.motion_l2_weight = float(joint_path_cfg["motion_l2_weight"])
        ik_branches = [int(value) for value in exchange_cfg["ik_branches"]]
        if len(ik_branches) != 1:
            raise ValueError("type3 建图只支持一个 IK 分支, 请检查 planning.exchange.ik_branches")
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
        )
        self.approach_parameterizer = FixedDurationParameterizer(self.approach_duration_s)
        self.assembly_parameterizer = FixedDurationParameterizer(self.assembly_duration_s)
        self.collision_model = self._load_collision_model(config.get("collision"))

        # BIT* 的采样边界: 连续关节 (config 里限位写 null) 折到 [-pi, pi]
        self.joint_lower = np.where(
            self.arm.joint_space.continuous, -np.pi, self.arm.joint_space.lower
        )
        self.joint_upper = np.where(
            self.arm.joint_space.continuous, np.pi, self.arm.joint_space.upper
        )

        # ---- 话题 ----
        self.approach_result_pub = self.create_publisher(
            ApproachResult, "/plan/approach/result", 10
        )
        self.create_subscription(
            ApproachRequest, "/plan/approach/request", self._on_approach_request, 10
        )
        self.assembly_result_pub = self.create_publisher(
            AssemblyResult, "/plan/assembly/result", 10
        )
        self.create_subscription(
            AssemblyRequest, "/plan/assembly/request", self._on_assembly_request, 10
        )

        self.get_logger().info(
            "planning_node 就绪 "
            f"use_analytic_ik={bool(arm_cfg.get('use_analytic_ik', True))} "
            f"type3_ik_branch={ik_branches[0]} "
            f"roll_count={len(self.type3_planner.rolls)} "
            f"assembly_path_samples={self.assembly_path_samples} "
            f"assembly_start_axial_m={self.assembly_start_axial_m:.4f} "
            f"approach_ik_branches={list(self.approach_ik_branches)} "
            f"collision={'on' if self.collision_model is not None else 'off'}"
        )

    def _declare_float(self, name: str, default) -> float:
        self.declare_parameter(name, float(default))
        return float(self.get_parameter(name).value)

    def _load_collision_model(self, collision_cfg):
        """按需构造碰撞模型; 关着或构不出来时返回 None (节点照常跑, 只是不查碰撞)。"""
        if not self.collision_enabled:
            return None
        if not collision_cfg:
            self.get_logger().warning("collision.enabled=true 但 config 里没有 collision 段, 已关闭碰撞检测")
            return None
        try:
            from planning.collision import CollisionModel

            return CollisionModel.from_config(collision_cfg)
        except Exception as exc:  # hpp-fcl 缺失 / 网格资产缺失 都在这里兜住
            self.get_logger().error(f"碰撞模型构造失败, 已关闭碰撞检测: {exc}")
            return None

    def _make_collision_checker(self, station_transform: np.ndarray):
        """给定兑换站位姿建一次性碰撞检查器; 没有碰撞模型时返回 None。"""
        if self.collision_model is None:
            return None
        from planning.collision import CollisionChecker

        return CollisionChecker(
            self.arm,
            [(self.collision_model.station_meshes, station_transform)],
            self.collision_model.arm_capsules,
        )

    def _to_type2_bounds(self, joints: np.ndarray) -> np.ndarray:
        """把关节角折进 BIT* 的采样盒: 连续关节折到 [-pi, pi], 有限位关节夹到限位内。"""
        values = self.arm.joint_space.wrap(np.atleast_2d(np.asarray(joints, dtype=float)))
        bounded = self.arm.joint_space.bounded
        outside = bounded & (
            (values < self.joint_lower - BOUND_TOLERANCE_RAD)
            | (values > self.joint_upper + BOUND_TOLERANCE_RAD)
        )
        if np.any(outside):
            raise ValueError(
                f"关节角超出限位: {np.round(values[np.any(outside, axis=1)], 4).tolist()}"
            )
        values[..., bounded] = np.clip(
            values[..., bounded], self.joint_lower[bounded], self.joint_upper[bounded]
        )
        return values

    def _motion_cost(self, candidates: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """候选关节角相对参考位姿的运动代价 (连续关节按最近折算, 权重取自 config)。"""
        delta = self.arm.joint_space.delta(candidates, reference[None, :])
        return self.motion_l1_weight * np.sum(np.abs(delta), axis=-1) + (
            self.motion_l2_weight * np.sum(delta * delta, axis=-1)
        )

    def _check_frame(self, frame_id: str, label: str) -> None:
        if frame_id and frame_id != FRAME_ARM_BASE:
            self.get_logger().warning(
                f"{label} 的 frame_id={frame_id!r} 不是 {FRAME_ARM_BASE!r}, 仍按 {FRAME_ARM_BASE} 解释"
            )

    # ------------------------------------------------------------------ 接近段
    def _on_approach_request(self, msg: ApproachRequest) -> None:
        result = ApproachResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = FRAME_ARM_BASE
        started = time.perf_counter()
        try:
            result.message = self._plan_approach(msg, result)
            result.success = True
        except Exception as exc:  # 任何一步失败都回一条失败结果, 不让节点挂掉
            result.success = False
            result.message = f"{type(exc).__name__}: {exc}"
            result.trajectory = JointTrajectoryMsg()
            self.get_logger().warning(f"接近段规划失败: {result.message}")
        else:
            self.get_logger().info(
                "接近段规划成功 "
                f"dense_steps={len(result.trajectory.points)} "
                f"duration_s={self.approach_duration_s:.3f} "
                f"elapsed_s={time.perf_counter() - started:.3f} "
                f"message={result.message}"
            )
        self.approach_result_pub.publish(result)

    def _plan_approach(self, msg: ApproachRequest, result: ApproachResult) -> str:
        """规划接近段并填好 result.trajectory, 返回给 result.message 的说明。"""
        self._check_frame(msg.header.frame_id, "ApproachRequest.header")
        start_joint = self._to_type2_bounds(_joint_array(msg.start_joint))[0]
        target = _transform_from_pose(msg.target_pose)
        timeout_s = float(msg.timeout) if float(msg.timeout) > 0.0 else self.approach_bit_timeout_s
        self.get_logger().info(
            "接近段请求 "
            f"start_joint={np.round(start_joint, 4).tolist()} "
            f"target_{_pose_summary(target)} "
            f"timeout_s={timeout_s:.3f}"
        )

        collision_checker = self._make_collision_checker(target)
        goal_joint, goal_cost = self._select_approach_goal(target, start_joint, collision_checker)
        self.get_logger().info(
            "接近段目标关节角 "
            f"goal_joint={np.round(goal_joint, 4).tolist()} motion_cost={goal_cost:.6f}"
        )

        if collision_checker is None:
            def validity_fn(joints: np.ndarray) -> bool:
                return True
        else:
            def validity_fn(joints: np.ndarray) -> bool:
                return not bool(
                    collision_checker.check_configs(np.asarray(joints, dtype=float)[None, :])[0]
                )

        note = "ok"
        try:
            from planning.type2 import plan_joint_path_bitstar

            waypoints = plan_joint_path_bitstar(
                start_joint,
                goal_joint,
                validity_fn=validity_fn,
                joint_lower=self.joint_lower,
                joint_upper=self.joint_upper,
                timeout_s=timeout_s,
            )
        except ImportError as exc:
            # OMPL 是系统级可选后端; 没装就退化成两点直线插值, 让通路仍然闭环。
            self.get_logger().warning(f"OMPL 不可用, 接近段退化为关节空间直线插值: {exc}")
            waypoints = np.stack((start_joint, goal_joint))
            note = "ok degraded=straight_line (OMPL 不可用)"

        trajectory = self.approach_parameterizer.parameterize(
            waypoints,
            sample_dt=self.approach_sample_dt,
            duration_s=self.approach_duration_s,
        )
        result.trajectory = _dense_trajectory_to_msg(trajectory, stamp=result.header.stamp)
        self.get_logger().info(
            f"接近段路径 sparse_steps={int(waypoints.shape[0])} "
            f"dense_steps={int(trajectory.positions.shape[0])} "
            f"sample_dt={self.approach_sample_dt:.4f}"
        )
        return note

    def _select_approach_goal(
        self,
        target: np.ndarray,
        start_joint: np.ndarray,
        collision_checker,
    ) -> tuple[np.ndarray, float]:
        """对 target_pose 求 IK, 在有效且无碰撞的候选里挑运动代价最小的一组关节角。"""
        joints, valid = self.arm.solve_ik(target[None, :, :], branches=self.approach_ik_branches)
        candidates = joints[0][valid[0]]
        if not len(candidates):
            raise RuntimeError(
                f"target_pose 无 IK 解 (试了 {len(self.approach_ik_branches)} 个初值分支), "
                "该位姿可能超出工作空间或姿态不可达"
            )
        candidates = self._to_type2_bounds(candidates)
        if collision_checker is not None:
            free = ~collision_checker.check_configs(candidates)
            if not np.any(free):
                raise RuntimeError("target_pose 的所有 IK 解都与兑换站碰撞")
            candidates = candidates[free]
        costs = self._motion_cost(candidates, start_joint)
        best = int(np.argmin(costs))
        return candidates[best], float(costs[best])

    # ------------------------------------------------------------------ 装配段
    def _on_assembly_request(self, msg: AssemblyRequest) -> None:
        result = AssemblyResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = FRAME_ARM_BASE
        started = time.perf_counter()
        try:
            success, message = self._plan_assembly(msg, result)
        except Exception as exc:
            success, message = False, f"{type(exc).__name__}: {exc}"
            result.trajectory = JointTrajectoryMsg()
            self.get_logger().warning(f"装配段规划异常: {message}")
        result.success = success
        result.message = message
        level = self.get_logger().info if success else self.get_logger().warning
        level(
            f"装配段规划{'成功' if success else '失败'} "
            f"dense_steps={len(result.trajectory.points)} "
            f"elapsed_s={time.perf_counter() - started:.3f} "
            f"message={message}"
        )
        self.assembly_result_pub.publish(result)

    def _plan_assembly(self, msg: AssemblyRequest, result: AssemblyResult) -> tuple[bool, str]:
        """在装配流形上规划, 填好 result.trajectory, 返回 (是否成功, 说明)。"""
        self._check_frame(msg.header.frame_id, "AssemblyRequest.header")
        start_joint = _joint_array(msg.start_joint)
        station = _transform_from_pose(msg.station_pose)
        goal_state = self._assembly_state(msg.goal)
        # 起点取"沿站轴退开 start_axial_m 的待插状态", 与接近段的落点衔接;
        # 终点由请求给定, 中间线性铺开 path_samples 个状态交给 type3 建图。
        start_state = AssemblyState(axial_offset_m=self.assembly_start_axial_m)
        path = AssemblyPath.between(start_state, goal_state, samples=self.assembly_path_samples)
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

        plan = self.type3_planner.plan(
            path,
            station,
            initial_joints=start_joint,
            collision_checker=self._make_collision_checker(station),
        )
        self._log_assembly_diagnostics(plan)
        if not plan.success:
            return False, plan.message

        trajectory = self.assembly_parameterizer.parameterize(
            plan.waypoints,
            sample_dt=self.assembly_sample_dt,
            duration_s=self.assembly_duration_s,
        )
        result.trajectory = _dense_trajectory_to_msg(trajectory, stamp=result.header.stamp)
        self.get_logger().info(
            f"装配段路径 waypoint_steps={int(plan.waypoints.shape[0])} "
            f"dense_steps={int(trajectory.positions.shape[0])} "
            f"cost={float(plan.cost):.6f} "
            f"roll_deg={np.round(np.rad2deg(plan.rolls), 2).tolist()}"
        )
        return True, plan.message

    def _assembly_state(self, goal: AssemblyStateMsg) -> AssemblyState:
        """interfaces/AssemblyState -> planning 的流形状态。

        msg 里 release 是布尔 (吸盘要不要松), 流形几何要的是"释放位移";
        用 ROS 参数 assembly.release_m 给这个位移, 默认 0 = 几何上不额外让开。
        """
        return AssemblyState(
            axial_offset_m=float(goal.axial),
            slide_m=float(goal.slide),
            p_angle_rad=float(goal.p_angle),
            q_angle_rad=float(goal.q_angle),
            release_m=self.assembly_release_m if bool(goal.release) else 0.0,
        )

    def _log_assembly_diagnostics(self, plan) -> None:
        """打印分层图/尽力修复的诊断, 失败时据此判断是 IK 不可达还是碰撞挡住。"""
        diagnostics = plan.diagnostics
        self.get_logger().info(
            "装配段建图 "
            f"valid_nodes={int(diagnostics['final_valid_nodes'])}/{int(diagnostics['total_nodes'])} "
            f"ik_valid_nodes={int(diagnostics['ik_valid_nodes'])} "
            f"collision={'on' if self.collision_model is not None else 'off'} "
            f"collision_rejected={int(diagnostics['collision_rejected_nodes'])} "
            f"path_samples={int(diagnostics['path_samples'])} "
            f"roll_count={int(diagnostics['roll_count'])} "
            f"ik_branch={int(diagnostics['ik_branch'])} "
            f"per_step_valid({_step_counts_summary(diagnostics['final_valid_by_step'])})"
        )
        repair = diagnostics.get("repair")
        if repair is None:
            return
        if bool(repair["success"]):
            self.get_logger().warning(
                "严格 Viterbi 失败, 已用 best_ik 尽力修复 "
                f"repaired_nodes={int(repair['repair_count'])} "
                f"max_pos_mm={float(repair['max_pos_error_m']) * 1000.0:.3f} "
                f"max_axis_deg={np.rad2deg(float(repair['max_axis_error_rad'])):.3f} "
                f"max_roll_deg={np.rad2deg(float(repair['max_roll_error_rad'])):.3f}"
            )
        else:
            self.get_logger().warning(
                f"best_ik 修复失败 collisions={int(repair['collision_count'])}"
            )


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




