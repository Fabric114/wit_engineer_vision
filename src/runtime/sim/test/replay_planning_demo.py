#!/usr/bin/env python3
"""粗接 planning 的演示脚本: 仿真真值站姿 -> 规划 -> 回放到 sim, 看 MuJoCo 画面.

流程 (设计文档 §九):
    1. 等 /mcu/arm/state            -> start_joint
    2. 等 /vision/exchange_pose     -> 站姿 (frame_id=arm_base, 正是 planning 要的系)
    3. 站姿沿自身 -z 退 retract 米作为待插位姿, 发 ApproachRequest
    4. 收到 /plan/approach/result 后, 按 time_from_start 以 100 Hz 采样,
       逐点发 /host/arm/command (control_mode=1)
    5. 打印跟踪误差

为什么回放归脚本而不归 sim: ArmHostCommand 只有单点 joint_target, 没有轨迹字段,
本轮不动契约, 所以"谁要执行轨迹谁负责按时间采样下发"。 以后 task_node 接管这段。

跑法 (三个终端):
    ros2 run sim sim_node
    ros2 run planning planning_node
    python3 src/runtime/sim/test/replay_planning_demo.py
"""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import rclpy
from interfaces.msg import (ApproachRequest, ApproachResult, ArmHostCommand,
                            ArmMcuState, ExchangeStationPose)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

def quat_to_mat(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """返回 [w, x, y, z]。 用迹最大的分支, 数值上稳。"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def pose_to_mat(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(pose.orientation.w, pose.orientation.x,
                            pose.orientation.y, pose.orientation.z)
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


def mat_to_pose(T: np.ndarray, pose) -> None:
    pose.position.x, pose.position.y, pose.position.z = [float(v) for v in T[:3, 3]]
    q = mat_to_quat(T[:3, :3])
    pose.orientation.w, pose.orientation.x = float(q[0]), float(q[1])
    pose.orientation.y, pose.orientation.z = float(q[2]), float(q[3])


class DemoNode(Node):
    def __init__(self) -> None:
        super().__init__("replay_planning_demo")
        self._state: ArmMcuState | None = None
        self._station: ExchangeStationPose | None = None
        self._result: ApproachResult | None = None
        self.got_state = threading.Event()
        self.got_station = threading.Event()
        self.got_result = threading.Event()

        self.create_subscription(ArmMcuState, "/mcu/arm/state", self._on_state, 10)
        self.create_subscription(ExchangeStationPose, "/vision/exchange_pose",
                                 self._on_station, 10)
        self.create_subscription(ApproachResult, "/plan/approach/result",
                                 self._on_result, 10)
        self.req_pub = self.create_publisher(ApproachRequest, "/plan/approach/request", 10)
        self.cmd_pub = self.create_publisher(ArmHostCommand, "/host/arm/command", 10)

    def _on_state(self, msg: ArmMcuState) -> None:
        self._state = msg
        self.got_state.set()

    def _on_station(self, msg: ExchangeStationPose) -> None:
        if msg.valid:
            self._station = msg
            self.got_station.set()

    def _on_result(self, msg: ApproachResult) -> None:
        self._result = msg
        self.got_result.set()

    @property
    def joints(self) -> np.ndarray:
        return np.asarray(self._state.joint_pos, dtype=float)

    @property
    def station(self) -> ExchangeStationPose:
        return self._station

    @property
    def result(self) -> ApproachResult:
        return self._result

    def send_command(self, q: np.ndarray) -> None:
        msg = ArmHostCommand()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_target = np.asarray(q, dtype=np.float32).tolist()
        # ⚠️ joint_vel_ff 恒 0 —— 不是"还没实现", 是仓里的硬约束:
        # 这个字段全仓零生产零消费、语义未定 (谁负责插补还没和电控确认),
        # README_PRE 17.2 结论出来之前一律置 0 (CLAUDE.md:119 / README_PRE.md:859)。
        # 反正这里是 100 Hz 逐点下发的稠密轨迹, 位置项本身就够密, 不缺速度前馈。
        msg.joint_vel_ff = [0.0] * 6
        msg.gripper_cmd = 2          # hold
        msg.control_mode = 1         # position
        self.cmd_pub.publish(msg)


def tool_offset() -> np.ndarray:
    """planning 配置里的 tcp_offset_6_tcp (第 6 DH 系 -> TCP 的平移)。

    读 planning 自己的配置而不是在 config/sim.yaml 里抄一份 —— 工具偏置是规划侧的
    标定量, 一处一个真值。 读不到就退回 0, 并在日志里说清楚。
    """
    try:
        from planning import load_config
        return np.asarray(load_config()["arm"]["tool"]["tcp_offset_6_tcp"], dtype=float)
    except Exception as exc:                                     # noqa: BLE001
        print(f"[warn] 读不到 planning 的 tcp_offset_6_tcp ({exc}), 按 0 处理")
        return np.zeros(3)


def sample(traj, t: float) -> tuple[np.ndarray, np.ndarray]:
    """按 time_from_start 在稠密轨迹上线性插值出 (位置, 速度)。

    速度只用于打印/排查, **不会进 joint_vel_ff** (理由见 DemoNode.send_command)。
    """
    pts = traj.points
    times = [p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in pts]
    if t <= times[0]:
        p = pts[0]
        return np.asarray(p.positions), np.asarray(p.velocities or np.zeros(6))
    if t >= times[-1]:
        p = pts[-1]
        return np.asarray(p.positions), np.zeros(6)
    i = int(np.searchsorted(times, t))
    t0, t1 = times[i - 1], times[i]
    a = (t - t0) / max(1e-9, t1 - t0)
    p0, p1 = pts[i - 1], pts[i]
    q = (1 - a) * np.asarray(p0.positions) + a * np.asarray(p1.positions)
    v = (np.asarray(p0.velocities) if p0.velocities else np.zeros(6))
    return q, v


def replay(node: DemoNode, traj, rate_hz: float) -> float:
    """按墙钟回放, 返回最大跟踪误差 (rad)。"""
    total = (traj.points[-1].time_from_start.sec
             + traj.points[-1].time_from_start.nanosec * 1e-9)
    period = 1.0 / rate_hz
    t_start = time.perf_counter()
    worst = 0.0
    while True:
        t = time.perf_counter() - t_start
        q, _ = sample(traj, t)
        node.send_command(q)
        worst = max(worst, float(np.max(np.abs(node.joints - q))))
        if t > total:
            break
        time.sleep(period)

    # 终点保持一会儿, 让 PD 收敛, 再量一次静态误差
    q_end = np.asarray(traj.points[-1].positions)
    for _ in range(int(1.5 * rate_hz)):
        node.send_command(q_end)
        time.sleep(period)
    node.get_logger().info(
        f"回放完成 {total:.2f} s / {len(traj.points)} 点: "
        f"过程最大误差 {np.degrees(worst):.2f}°, "
        f"终点静态误差 {np.degrees(np.max(np.abs(node.joints - q_end))):.3f}°")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description="sim + planning 粗接演示")
    ap.add_argument("--retract", type=float, default=0.25,
                    help="待插位姿: 沿站体 -z 退多少米 (默认与 sim 的 standoff_m 一致)")
    ap.add_argument("--rate", type=float, default=100.0, help="回放频率 Hz")
    ap.add_argument("--plan-timeout", type=float, default=60.0, help="等规划结果秒数")
    args = ap.parse_args()

    rclpy.init()
    node = DemoNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    log = node.get_logger()

    try:
        if not node.got_state.wait(10.0):
            log.error("10 s 没收到 /mcu/arm/state —— sim_node 起了吗?")
            return 1
        if not node.got_station.wait(10.0):
            log.error("10 s 没收到 /vision/exchange_pose —— "
                      "检查 config/sim.yaml 的 station.publish_pose")
            return 1

        start = node.joints
        T_station = pose_to_mat(node.station.pose)
        retract = np.eye(4)
        retract[2, 3] = -args.retract
        T_frame6 = T_station @ retract          # 待插位姿, 按第 6 DH 系

        # ⚠️ ApproachRequest.target_pose 是 **TCP** 位姿, 而 sim 的站姿是按第 6 DH 系
        # 定的 (sim 只认 MuJoCo, 不该知道 planning 的工具参数)。 差的这个工具偏置
        # 属于 planning 的标定量, 所以在消费侧 (这里) 补上。 不补就会 IK 无解。
        T_target = T_frame6.copy()
        T_target[:3, 3] += T_frame6[:3, :3] @ tool_offset()

        log.info(f"起始关节 {np.round(start, 3).tolist()}")
        log.info(f"站姿 (arm_base) {np.round(T_station[:3, 3], 4).tolist()}, "
                 f"待插 frame6 {np.round(T_frame6[:3, 3], 4).tolist()}, "
                 f"目标 TCP {np.round(T_target[:3, 3], 4).tolist()}")

        req = ApproachRequest()
        req.header.stamp = node.get_clock().now().to_msg()
        req.header.frame_id = "arm_base"
        req.start_joint = np.asarray(start, dtype=np.float32).tolist()
        mat_to_pose(T_target, req.target_pose)
        req.timeout = float(args.plan_timeout)
        node.req_pub.publish(req)
        log.info("已发 ApproachRequest, 等 /plan/approach/result ...")

        if not node.got_result.wait(args.plan_timeout):
            log.error(f"{args.plan_timeout:.0f} s 没收到结果 —— planning_node 起了吗?")
            return 1
        res = node.result
        if not res.success:
            log.error(f"规划失败: {res.message}")
            return 2
        log.info(f"规划成功: {res.message} ({len(res.trajectory.points)} 点)")

        replay(node, res.trajectory, args.rate)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
