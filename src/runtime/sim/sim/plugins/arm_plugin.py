"""臂插件: 订阅 /host/arm/command 算 PD 力矩, 100 Hz 发 /mcu/arm/state.

为什么插件自己算力矩, 不用 MJCF 的 actuator:
    rm26_arm.xml 里 11 个执行器都是 `<motor gear="1" ctrlrange="-1 1">`,
    是力矩源且上限 1 N·m —— 做不了位置控制。 所以绕过 actuator,
    直接写 data.qfrc_applied。

PD 必须带重力补偿 (实测 kp=80 时稳态误差 1.74° -> 0.14°)。 补偿量用 MuJoCo 自己的
data.qfrc_bias (重力 + 科氏), 而不是 planning 的 RNEA —— 同一套模型算的, 不引入
DH/惯量标定误差。
"""

from __future__ import annotations

import threading

import numpy as np
from interfaces.msg import ArmHostCommand, ArmMcuState

from sim.mujoco_engine import SimContext, SimPlugin

_STATUS_AT_LIMIT = 1 << 0        # status_code bit0: 有关节贴着限位

# ArmHostCommand.control_mode
_MODE_IDLE = 0
_MODE_POSITION = 1
_MODE_TRAJECTORY = 2


class ArmPlugin(SimPlugin):
    name = "arm"

    def __init__(self, engine, gripper=None) -> None:
        self._engine = engine
        self._gripper = gripper          # 只为了把夹爪状态填进 ArmMcuState
        self._cmd_lock = threading.Lock()

        self._q_target: np.ndarray | None = None
        self._vel_ff = np.zeros(6)
        self._mode = _MODE_IDLE
        self._warned_traj = False
        self._cmd_count = 0

    # ---------- 初始化 ----------

    def setup(self, ctx: SimContext) -> None:
        node = ctx.node
        cfg = ctx.config.get("arm", {}) or {}

        names = list(cfg.get("joint_names") or ["J1", "J2", "J3", "J4", "J5", "J6"])
        self._jids, qadr, dadr = self._engine.joint_addr(names)
        # 公开给 sim_node 用 (它算站姿真值时要临时设一次 qpos)
        self.qadr = self._qadr = np.array(qadr, dtype=int)
        self.dadr = self._dadr = np.array(dadr, dtype=int)
        self._names = names

        self._kp = np.asarray(cfg.get("kp") or [400, 400, 200, 80, 60, 30], dtype=float)
        kd = cfg.get("kd")
        self._kd = np.asarray(kd, dtype=float) if kd else self._kp * 0.1
        self._tau_max = float(cfg.get("tau_max", 50.0))

        # 目标值一律按 MJCF 的 range 夹 —— 以模型为准。
        # (planning 配置的 arm.joint_limits 已按同一份 URDF/MJCF 填好, 两边数值一致;
        #  这里仍然夹一次, 保证模型是最后一道关)
        self.lo = self._lo = np.array([ctx.model.jnt_range[j, 0] for j in self._jids])
        self.hi = self._hi = np.array([ctx.model.jnt_range[j, 1] for j in self._jids])

        # 初始位姿: 把 home 写进 qpos, 并让 q_target 从这里起步 (否则第一帧会甩)
        home = np.asarray(cfg.get("home") or np.zeros(6), dtype=float)
        home = np.clip(home, self._lo, self._hi)
        ctx.data.qpos[self._qadr] = home
        ctx.data.qvel[self._dadr] = 0.0
        self._q_target = home.copy()

        # line 升降轴: 本轮不驱动, 用一个软 PD 锁在当前位置。
        # 它必须不动, 否则 line_link 位姿变了, arm_base 的安装常量 F0 就不成立。
        self._line_dadr = self._line_qadr = None
        try:
            _, lq, ld = self._engine.joint_addr(["line"])
            self._line_qadr, self._line_dadr = lq[0], ld[0]
            self._line_hold = float(ctx.data.qpos[self._line_qadr])
        except RuntimeError:
            pass

        self._sub = node.create_subscription(
            ArmHostCommand, "/host/arm/command", self._on_command, 10)
        self._pub = node.create_publisher(ArmMcuState, "/mcu/arm/state", 10)

        node.get_logger().info(
            f"arm 插件: {names} qpos={self._qadr.tolist()} dof={self._dadr.tolist()}, "
            f"home={np.round(home, 3).tolist()}")

    # ---------- 下行: 命令 -> 力矩 ----------

    def _on_command(self, msg: ArmHostCommand) -> None:
        """ROS 执行器线程。 只更新命令缓存, 不碰 data。"""
        target = np.clip(np.asarray(msg.joint_target, dtype=float), self._lo, self._hi)
        mode = int(msg.control_mode)
        if mode == _MODE_TRAJECTORY and not self._warned_traj:
            self._warned_traj = True
            self._engine.ctx.node.get_logger().warn(
                "control_mode=2 (trajectory) 未实现: ArmHostCommand 里没有轨迹载荷, "
                "轨迹回放归请求方 (见设计文档 §九)。 本次按 position 处理。")
        with self._cmd_lock:
            self._mode = _MODE_POSITION if mode == _MODE_TRAJECTORY else mode
            self._cmd_count += 1
            if self._mode == _MODE_IDLE:
                # idle 不更新目标, 保持上一个目标不动 (不是"卸力", 理由见 on_step)
                self._vel_ff = np.zeros(6)
                return
            self._q_target = target
            self._vel_ff = np.asarray(msg.joint_vel_ff, dtype=float)

    def on_step(self, ctx: SimContext, dt: float) -> None:
        d = ctx.data
        bias = d.qfrc_bias[self._dadr]

        with self._cmd_lock:
            q_target = self._q_target
            vel_ff = self._vel_ff

        # 任何模式下都跑 PD 保持。 idle 只是"不接受新目标", 不是卸力 ——
        # 光给 qfrc_bias 在数值上是中性稳定的, 实测 4 s 就滑到限位
        # (J2/J3 直接顶到 2.61/3.14)。 真机 idle 下电会塌, 但那对演示没意义。
        err = q_target - d.qpos[self._qadr]
        derr = vel_ff - d.qvel[self._dadr]
        tau = self._kp * err + self._kd * derr + bias
        d.qfrc_applied[self._dadr] = np.clip(tau, -self._tau_max, self._tau_max)

        if self._line_dadr is not None:
            # 软 PD + 重力补偿, 把升降台钉在初始高度
            d.qfrc_applied[self._line_dadr] = (
                2000.0 * (self._line_hold - d.qpos[self._line_qadr])
                - 200.0 * d.qvel[self._line_dadr]
                + d.qfrc_bias[self._line_dadr])

    # ---------- 上行: 100 Hz 状态 ----------

    def on_publish(self, ctx: SimContext) -> None:
        d = ctx.data
        q = np.asarray(d.qpos[self._qadr], dtype=float)
        v = np.asarray(d.qvel[self._dadr], dtype=float)

        msg = ArmMcuState()
        msg.header.stamp = ctx.node.get_clock().now().to_msg()
        msg.header.frame_id = "arm_base"
        msg.joint_pos = q.astype(np.float32).tolist()
        msg.joint_vel = v.astype(np.float32).tolist()

        if self._gripper is not None:
            msg.gripper_state = self._gripper.state
            msg.gripper_force = float(self._gripper.force)

        status = 0
        if np.any(q <= self._lo + 1e-3) or np.any(q >= self._hi - 1e-3):
            status |= _STATUS_AT_LIMIT
        msg.status_code = status

        self._pub.publish(msg)
