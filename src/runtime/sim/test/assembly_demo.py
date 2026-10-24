#!/usr/bin/env python3
"""sim 演示 (RM2026 level 3 形态): 接近段 -> (按键) 直接跳到插到底 -> 滑动 (推动) 段, 碰撞检测开着。

level (--level, 默认 3): 1 = 接近 + 插到底; 2 = 再加滑动; 3 = 滑动后接 P 段 (rm26 兑不了三级矿, 预期失败)。
科技核心 (sim 里的 exchange_column) 带 step3_slide 关节, 由 sim 的 column 插件跟随 TCP 移动;
本脚本订阅 /sim/exchange_column/joint_states 核对: 插到底后滑动量 ≈ 0, 滑动完 ≈ slide_z_mag。

流程:
    1. 等 /mcu/arm/state (起始关节) 和 /vision/exchange_pose (sim 真值站姿, arm_base 系)
    2. 接近段: 发 ApproachRequest (initial_joint_state + station_pose_arm_base); 待插目标
       (沿站轴退开 insert_slide_mag) 由 planning_node 扫 roll 自选。 收到结果后按 time_from_start 回放
    3. 按 Enter: **直接跳到插到底** —— 代替操作手手动插入 (RM2026 里插入就是手动的)。
       目标关节 = 插到底位姿在装配分支上、离接近终点最近的 IK 解, 关节空间线性插值过去。
       ⚠️ 这一步不经过规划、不查碰撞, 只是演示用的"瞬移"
    4. 滑动段: 发 AssemblyRequest(path_name=PATH_SLIDE), 起点 = 跳完后的实测关节角; 收到结果后回放
    5. level 3: 发 PATH_P (附 Q 缓存)。 rm26 兑不了三级矿, sim 站姿下预期规划失败

跑法 (三个终端, 碰撞检测是 planning_node 的默认, 不用加参数):
    ros2 run sim sim_node
    ros2 run planning planning_node
    python3 src/runtime/sim/test/assembly_demo.py            # 加 --no-pause 不等按键, --level 1/2/3

退出码: 0 成功; 1 等待超时; 2 规划失败 / 科技核心没跟上 / 配置缺矿石胶囊。

依赖 planning 包 (算待插 / 插到底的 IK, 与 planning_node 用同一份 config/planning.yaml)。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import rclpy
from interfaces.msg import (ApproachRequest, ApproachResult, ArmHostCommand, ArmMcuState,
                            AssemblyRequest, AssemblyResult, ExchangeStationPose)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_planning_demo import joint_state_msg, pose_to_mat, sample, station_pose_msg  # noqa: E402

from planning import ArmModel, Type3Planner, load_config  # noqa: E402
from planning.type3 import AssemblyPath, AssemblyState  # noqa: E402

#: 科技核心滑动量与期望值的允许偏差 (m)
COLUMN_TOLERANCE_M = 0.003
#: 规划侧必须带上的矿石胶囊名 (config/planning.yaml 的 collision.arm.capsules)
ORE_CAPSULE = "cc_ore"


class AssemblyDemo(Node):
    def __init__(self) -> None:
        super().__init__("assembly_demo")
        self._state: ArmMcuState | None = None
        self._station: ExchangeStationPose | None = None
        self._column: JointState | None = None
        self._results: dict[str, object] = {}
        self._pending: dict[str, str] = {}  # kind -> 最新发出的 request_id
        self._events = {name: threading.Event() for name in ("state", "station", "approach", "assembly", "column")}
        self.create_subscription(ArmMcuState, "/mcu/arm/state", self._on_state, 10)
        self.create_subscription(JointState, "/sim/exchange_column/joint_states", self._on_column, 10)
        self.create_subscription(ExchangeStationPose, "/vision/exchange_pose", self._on_station, 10)
        self.create_subscription(ApproachResult, "/plan/approach/result",
                                 lambda msg: self._on_result("approach", msg), 10)
        self.create_subscription(AssemblyResult, "/plan/assembly/result",
                                 lambda msg: self._on_result("assembly", msg), 10)
        self.approach_pub = self.create_publisher(ApproachRequest, "/plan/approach/request", 10)
        self.assembly_pub = self.create_publisher(AssemblyRequest, "/plan/assembly/request", 10)
        self.cmd_pub = self.create_publisher(ArmHostCommand, "/host/arm/command", 10)

    def _on_state(self, msg: ArmMcuState) -> None:
        self._state = msg
        self._events["state"].set()

    def _on_column(self, msg: JointState) -> None:
        self._column = msg
        self._events["column"].set()

    @property
    def column_slide(self) -> float:
        """科技核心 step3_slide 当前值 (m)。"""
        return float(self._column.position[list(self._column.name).index("step3_slide")])

    def _on_station(self, msg: ExchangeStationPose) -> None:
        if msg.valid:
            self._station = msg
            self._events["station"].set()

    def _on_result(self, kind: str, msg) -> None:
        # 只认最新发出的那个请求; 超时后迟到的旧结果直接丢 (同 RM2026 task_node)
        if msg.request_id != self._pending.get(kind):
            self.get_logger().warning(f"丢弃过期 {kind} 结果 id={msg.request_id}")
            return
        self._results[kind] = msg
        self._events[kind].set()

    def request(self, kind: str, req) -> None:
        """给请求配新 request_id 并发出; 之前未到的同类结果从此作废。"""
        req.request_id = uuid.uuid4().hex
        self._pending[kind] = req.request_id
        self._events[kind].clear()
        self._results.pop(kind, None)
        (self.approach_pub if kind == "approach" else self.assembly_pub).publish(req)

    def wait(self, name: str, timeout: float) -> bool:
        return self._events[name].wait(timeout)

    def result(self, kind: str):
        self._events[kind].clear()
        return self._results.pop(kind)

    @property
    def joints(self) -> np.ndarray:
        return np.asarray(self._state.joint_pos, dtype=float)

    @property
    def station(self) -> np.ndarray:
        return pose_to_mat(self._station.pose)

    @property
    def station_stamp(self):
        return self._station.header.stamp

    @property
    def joints_msg(self):
        """当前关节角的 JointState (stamp 取 ArmMcuState 的采样时间)。"""
        return joint_state_msg(self.joints, self._state.header.stamp)

    def send_command(self, q: np.ndarray) -> None:
        msg = ArmHostCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_target = np.asarray(q, dtype=np.float32).tolist()
        msg.joint_vel_ff = [0.0] * 6       # 语义未定, 恒 0 (理由见 replay_planning_demo.py)
        msg.gripper_cmd = 2                # hold
        msg.control_mode = 1               # position
        self.cmd_pub.publish(msg)

    def play(self, traj, rate_hz: float, label: str) -> None:
        """按墙钟回放稠密轨迹, 终点保持 1 s, 打印跟踪误差。"""
        total = traj.points[-1].time_from_start.sec + traj.points[-1].time_from_start.nanosec * 1e-9
        period = 1.0 / rate_hz
        start = time.perf_counter()
        worst = 0.0
        while True:
            t = time.perf_counter() - start
            q, _ = sample(traj, t)
            self.send_command(q)
            worst = max(worst, float(np.max(np.abs(self.joints - q))))
            if t > total:
                break
            time.sleep(period)
        self.hold(np.asarray(traj.points[-1].positions), 1.0, rate_hz)
        end_error = float(np.max(np.abs(self.joints - np.asarray(traj.points[-1].positions))))
        self.get_logger().info(
            f"[{label}] 回放完成 {total:.2f} s / {len(traj.points)} 点: 过程最大误差 "
            f"{np.degrees(worst):.2f}°, 终点误差 {np.degrees(end_error):.3f}°")

    def move_linear(self, goal: np.ndarray, duration_s: float, rate_hz: float) -> None:
        """关节空间线性插值到 goal (不经规划、不查碰撞), 再保持 1 s 让 PD 收敛。"""
        start = self.joints.copy()
        steps = max(1, int(duration_s * rate_hz))
        for k in range(1, steps + 1):
            self.send_command(start + (goal - start) * (k / steps))
            time.sleep(1.0 / rate_hz)
        self.hold(goal, 1.0, rate_hz)

    def hold(self, q: np.ndarray, seconds: float, rate_hz: float) -> None:
        for _ in range(int(seconds * rate_hz)):
            self.send_command(q)
            time.sleep(1.0 / rate_hz)


class Geometry:
    """与 planning_node 同一份 config 的装配几何 / IK, 只用来算演示的待插与插到底关节角。"""

    def __init__(self) -> None:
        config = load_config()
        self.collision = config.get("collision") or {}
        exchange = config["planning"]["exchange"]
        joint_path = exchange["stage_path"]["joint_path"]
        self.geometry = config["planning"]["type3"]["exchange_trajectory"]
        self.branch = int(exchange["ik_branches"][0])
        self.arm = ArmModel.from_config(config["arm"])
        self.planner = Type3Planner(
            self.arm, geometry=self.geometry, ik_branch=self.branch,
            roll_sample_step_deg=joint_path["roll_sample_step_deg"], bandwidth=0,
            collision_soft_margin=0.0, collision_soft_weight=0.0, motion_l1_weight=1.0,
            motion_l2_weight=0.0, joint_limit_margin_rad=0.0, joint_limit_weight=0.0,
        )

    def ore_capsule_problem(self) -> str | None:
        """规划侧必须带矿石胶囊, 且装配段对桩体豁免; 不满足返回原因。"""
        names = [c["name"] for c in (self.collision.get("arm") or {}).get("capsules", [])]
        if ORE_CAPSULE not in names:
            return f"config 的 collision.arm.capsules 里没有 {ORE_CAPSULE} (规划看不见矿石)"
        allowed = (self.collision.get("station") or {}).get("assembly_allowed_contact") or []
        if ORE_CAPSULE not in allowed:
            return f"{ORE_CAPSULE} 不在 collision.station.assembly_allowed_contact 里 (插入后必然判碰撞)"
        return None

    def nearest_inserted(self, station: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
        """插到底 (AssemblyState()) 在装配分支上的 IK 解里离 reference (接近终点) 最近的那个; 全无解返回 None。

        接近目标的 roll 由 planning_node 选, 脚本不知道; 这里扫 type3 的 roll 网格反找。
        """
        poses = self.planner.task_poses(AssemblyPath(AssemblyState().as_array()[None]), station, self.planner.rolls)
        joints, valid = self.arm.solve_ik(poses, branches=(self.branch,), reference_joints=reference)
        candidates = joints[valid[:, 0], 0]
        if not len(candidates):
            return None
        jumps = np.max(np.abs(self.arm.joint_space.delta(candidates, np.asarray(reference)[None, :])), axis=1)
        return candidates[int(np.argmin(jumps))]


def check_column(node: AssemblyDemo, expected_m: float, label: str) -> bool:
    """等一帧新的科技核心状态 (插件 20 Hz 发布), 核对滑动量。"""
    time.sleep(0.2)
    actual = node.column_slide
    ok = abs(actual - expected_m) <= COLUMN_TOLERANCE_M
    report = node.get_logger().info if ok else node.get_logger().error
    report(f"[{label}] 科技核心 step3_slide = {actual * 1000:.1f} mm (期望 {expected_m * 1000:.1f} mm, "
           f"容差 {COLUMN_TOLERANCE_M * 1000:.0f} mm) {'OK' if ok else '没跟上'}")
    return ok


def plan_assembly(node: AssemblyDemo, station: np.ndarray, path_name: int, timeout: float, label: str):
    """发一条装配段请求 (起点 = 此刻实测关节角), 成功返回结果, 失败/超时打日志返回 None。"""
    log = node.get_logger()
    req = AssemblyRequest()
    req.header.stamp = node.get_clock().now().to_msg()
    req.header.frame_id = "arm_base"
    req.initial_joint_state = node.joints_msg
    req.station_pose_arm_base = station_pose_msg(station, node.station_stamp)
    req.path_name = path_name
    node.request("assembly", req)
    log.info(f"[{label}] 已发请求 (path_name={path_name}), 等结果 ...")
    if not node.wait("assembly", timeout):
        log.error(f"[{label}] 等结果超时")
        return None
    res = node.result("assembly")
    if not res.success:
        log.error(f"[{label}] 规划失败 ({res.error_code}): {res.message}")
        return None
    log.info(f"[{label}] 规划成功 cost={res.cost:.4f} roll={np.degrees(res.selected_roll):.1f}°: {res.message}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="sim 演示 (RM2026 level 3): 接近 -> 跳到插到底 -> 滑动 -> P")
    ap.add_argument("--rate", type=float, default=100.0, help="回放频率 Hz")
    ap.add_argument("--plan-timeout", type=float, default=30.0, help="等规划结果的秒数")
    ap.add_argument("--jump-s", type=float, default=1.5, help="跳到插到底的插值时长 s")
    ap.add_argument("--no-pause", action="store_true", help="不等按键, 接近完直接跳")
    ap.add_argument("--level", type=int, choices=(1, 2, 3), default=3,
                    help="1 接近+插到底; 2 再加滑动; 3 滑动后接 P 段 (rm26 兑不了三级矿, P 段预期失败)")
    args = ap.parse_args()

    geometry = Geometry()
    problem = geometry.ore_capsule_problem()
    if problem:
        print(f"[配置] {problem}", file=sys.stderr)
        return 2
    slide = float(geometry.geometry["slide_z_mag"])

    rclpy.init()
    node = AssemblyDemo()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    log = node.get_logger()
    try:
        if not node.wait("state", 10.0) or not node.wait("station", 10.0) or not node.wait("column", 10.0):
            log.error("10 s 没收到 /mcu/arm/state、/vision/exchange_pose 或 /sim/exchange_column/joint_states "
                      "—— sim_node 起了吗? (科技核心关节要新版 exchange_station.xml)")
            return 1
        capsules = [c["name"] for c in geometry.collision["arm"]["capsules"]]
        log.info(f"level {args.level}; 规划碰撞胶囊 {capsules}; 装配段豁免桩体 "
                 f"{geometry.collision['station']['assembly_allowed_contact']}")
        station = node.station
        log.info(f"站姿 {np.round(station[:3, 3], 4).tolist()}  装配分支 {geometry.branch} "
                 "(待插目标由 planning_node 自选)")

        # ---- 1. 接近段 ----
        req = ApproachRequest()
        req.header.stamp = node.get_clock().now().to_msg()
        req.header.frame_id = "arm_base"
        req.initial_joint_state = node.joints_msg
        req.station_pose_arm_base = station_pose_msg(station, node.station_stamp)
        node.request("approach", req)
        log.info("[接近] 已发请求, 等结果 ...")
        if not node.wait("approach", args.plan_timeout):
            log.error("[接近] 等结果超时 —— planning_node 起了吗?")
            return 1
        res = node.result("approach")
        if not res.success:
            log.error(f"[接近] 规划失败 ({res.error_code}): {res.message}")
            return 2
        log.info(f"[接近] 规划成功 cost={res.cost:.4f}: {res.message}")
        node.play(res.trajectory, args.rate, "接近")

        # ---- 2. 跳到插到底 (代替手动插入, 不查碰撞) ----
        if not args.no_pause:
            input("\n接近段完成。 按 Enter 直接跳到插到底 (代替操作手手动插入) ...")
        inserted = geometry.nearest_inserted(station, node.joints)
        if inserted is None:
            log.error("接近终点插不到底: 装配分支上所有 roll 的插到底位姿都无 IK 解 "
                      "(见 todo.md §3 接近段 roll 筛选)")
            return 2
        jump = float(np.max(np.abs(inserted - node.joints)))
        log.info(f"[跳转] 关节空间线性插值到插到底 (最大单轴 {jump:.3f} rad, 不查碰撞)")
        node.move_linear(inserted, args.jump_s, args.rate)
        if not check_column(node, 0.0, "插到底"):
            return 2
        if args.level == 1:
            log.info("level 1 完成 (只插入)")
            return 0

        # ---- 3. 滑动段 (PATH_SLIDE, 目标固定 slide_z_mag, 失败就报失败) ----
        res = plan_assembly(node, station, AssemblyRequest.PATH_SLIDE, args.plan_timeout, "滑动")
        if res is None:
            return 2
        node.play(res.trajectory, args.rate, "滑动")
        if not check_column(node, slide, "滑动"):
            return 2
        if args.level == 2:
            log.info("演示完成")
            return 0

        # ---- 4. P 段 (PATH_P, 附 Q 缓存); rm26 兑不了三级矿, sim 站姿下预期规划失败 ----
        res = plan_assembly(node, station, AssemblyRequest.PATH_P, args.plan_timeout, "P")
        if res is None:
            log.warning("[P] rm26 兑不了三级矿, P 段在 sim 站姿下预期失败")
            return 2
        log.info(f"[P] Q 缓存 {len(res.q_sweep_s)} 点 (sim 科技核心 p_axis/q_axis 固定 0, 站体不跟转)")
        node.play(res.trajectory, args.rate, "P")
        log.info("演示完成")
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
