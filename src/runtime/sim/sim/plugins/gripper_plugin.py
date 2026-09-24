"""夹爪/吸盘插件: 只走状态通道, 吸附物理留钩子.

本轮场景里没有矿石 body, 所以这里不做真吸附, 只把 ArmHostCommand.gripper_cmd
翻译成一个带过渡时间的状态机, 由 arm_plugin 填进 ArmMcuState:
    0 open / 1 close -> 过渡 close_time_s 期间报 2 (moving), 到位后报 0 / 1
    2 hold           -> 保持不变

以后要真吸附: 场景里加动态矿石 body + 一条 `<weld>` equality, 闭合时
`data.eq_active[eq_id] = 1`、松开置 0 —— attach()/detach() 两个钩子已经留好。
"""

from __future__ import annotations

import threading

from interfaces.msg import ArmHostCommand

from sim.mujoco_engine import SimContext, SimPlugin

# ArmHostCommand.gripper_cmd / ArmMcuState.gripper_state
CMD_OPEN, CMD_CLOSE, CMD_HOLD = 0, 1, 2
ST_OPEN, ST_CLOSED, ST_MOVING = 0, 1, 2


class GripperPlugin(SimPlugin):
    name = "gripper"

    def __init__(self, engine) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._state = ST_OPEN
        self._goal = ST_OPEN
        self._t_arrive = 0.0          # 仿真时刻: 到这个时间算开合到位
        self._close_time = 0.4
        self._hold_force = 20.0

    # arm_plugin 读这两个属性填状态消息
    @property
    def state(self) -> int:
        with self._lock:
            return self._state

    @property
    def force(self) -> float:
        with self._lock:
            return self._hold_force if self._state == ST_CLOSED else 0.0

    def setup(self, ctx: SimContext) -> None:
        cfg = ctx.config.get("gripper", {}) or {}
        self._close_time = float(cfg.get("close_time_s", 0.4))
        self._hold_force = float(cfg.get("hold_force", 20.0))
        self._sub = ctx.node.create_subscription(
            ArmHostCommand, "/host/arm/command", self._on_command, 10)

    def _on_command(self, msg: ArmHostCommand) -> None:
        """ROS 执行器线程。 只改状态机的目标, 不碰 data。"""
        cmd = int(msg.gripper_cmd)
        if cmd == CMD_HOLD:
            return
        goal = ST_CLOSED if cmd == CMD_CLOSE else ST_OPEN
        with self._lock:
            if goal == self._goal:
                return
            self._goal = goal
            self._state = ST_MOVING
            self._t_arrive = float(self._engine.data.time) + self._close_time

    def on_step(self, ctx: SimContext, dt: float) -> None:
        with self._lock:
            if self._state == ST_MOVING and ctx.data.time >= self._t_arrive:
                self._state = self._goal
                if self._goal == ST_CLOSED:
                    self.attach(ctx)
                else:
                    self.detach(ctx)

    # ---------- 吸附钩子 (本轮空实现) ----------

    def attach(self, ctx: SimContext) -> None:
        """闭合到位: 以后在这里激活矿石的 weld equality。"""

    def detach(self, ctx: SimContext) -> None:
        """张开到位: 以后在这里断开 weld。"""
