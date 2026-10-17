"""科技核心插件: 按"跟随 TCP"运动学地驱动 step3_slide, 并发布科技核心关节状态.

站体 / 科技核心 geom 全 contype=0, 没有接触, 所以不是矿石"推"动科技核心, 而是每个物理步:
TCP 在站体系下的位置 -> sim.column.ColumnFollower -> 直接写 step3_slide 的 qpos (qvel 置 0).
p_axis / q_axis 本轮固定 0 (level 3 的 P/Q 段还没做).

站体位姿与 TCP 换算 (C、工具偏置) 由 sim_node 在 _setup_station 之后调 configure() 注入;
注入前 on_step 什么都不做.

发布 /sim/exchange_column/joint_states (sensor_msgs/JointState, 关节名 step3_slide/p_axis/q_axis),
演示脚本 assembly_demo.py 用它核对科技核心有没有跟上.
"""

from __future__ import annotations

import mujoco
import numpy as np
from sensor_msgs.msg import JointState

from sim.column import ColumnFollower
from sim.mujoco_engine import SimContext, SimPlugin

JOINTS = ("step3_slide", "p_axis", "q_axis")


class ColumnPlugin(SimPlugin):
    name = "column"

    def __init__(self, engine) -> None:
        self._engine = engine
        self._enabled = False
        self._station_T_world = None
        self._link6_T_tcp = None
        self._publish_every = 5
        self._publish_count = 0

    def setup(self, ctx: SimContext) -> None:
        cfg = ((ctx.config.get("station", {}) or {}).get("column", {}) or {})
        m = ctx.model
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
        self._link6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        if min(ids) < 0 or self._link6 < 0:
            ctx.node.get_logger().warn("场景里没有科技核心关节 (step3_slide/p_axis/q_axis) 或 link_6, column 插件停用")
            return
        self._qadr = [int(m.jnt_qposadr[i]) for i in ids]
        self._dadr = [int(m.jnt_dofadr[i]) for i in ids]
        lo, hi = m.jnt_range[ids[0]]
        self._follow = bool(cfg.get("follow_tcp", True))
        self._follower = ColumnFollower(
            engage_radius_m=float(cfg.get("engage_radius_m", 0.01)),
            release_radius_m=float(cfg.get("release_radius_m", 0.03)),
            slide_range=(float(lo), float(hi)),
        )
        self._publish_every = max(1, int(cfg.get("publish_every", 5)))
        self._pub = ctx.node.create_publisher(JointState, "/sim/exchange_column/joint_states", 10)
        self._enabled = True

    def configure(self, world_T_station: np.ndarray, frame6_T_link6: np.ndarray, tool_offset: np.ndarray) -> None:
        """sim_node 在站体位姿确定后调用 (此时物理线程还没起)。"""
        self._station_T_world = np.linalg.inv(np.asarray(world_T_station, dtype=float))
        frame6_T_tcp = np.eye(4)
        frame6_T_tcp[:3, 3] = np.asarray(tool_offset, dtype=float)
        self._link6_T_tcp = np.linalg.inv(np.asarray(frame6_T_link6, dtype=float)) @ frame6_T_tcp

    def on_step(self, ctx: SimContext, dt: float) -> None:
        if not self._enabled or self._station_T_world is None:
            return
        d = ctx.data
        if self._follow:
            world_T_link6 = np.eye(4)
            world_T_link6[:3, :3] = d.xmat[self._link6].reshape(3, 3)
            world_T_link6[:3, 3] = d.xpos[self._link6]
            tcp = self._station_T_world @ world_T_link6 @ self._link6_T_tcp
            slide = self._follower.update(tcp[:3, 3])
        else:
            slide = self._follower.slide
        d.qpos[self._qadr] = (slide, 0.0, 0.0)
        d.qvel[self._dadr] = 0.0

    def on_publish(self, ctx: SimContext) -> None:
        if not self._enabled:
            return
        self._publish_count += 1
        if self._publish_count % self._publish_every:
            return
        msg = JointState()
        msg.header.stamp = ctx.node.get_clock().now().to_msg()
        msg.name = list(JOINTS)
        msg.position = [float(v) for v in ctx.data.qpos[self._qadr]]
        self._pub.publish(msg)
