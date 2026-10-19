"""仿真后端节点: 起 MuJoCo, 注册插件, 与真机 firmware bridge 发同一套 topic.

    订阅  /host/arm/command   (interfaces/ArmHostCommand)
    发布  /mcu/arm/state      (interfaces/ArmMcuState)      100 Hz
    发布  /vision/exchange_pose (interfaces/ExchangeStationPose) 5 Hz, 真值,
          可用 station.publish_pose 关掉

参数来自仓库根 config/sim.yaml, 每个叶子都能用同名 ROS 参数 (点分路径) 覆盖:
    ros2 run sim sim_node --ros-args -p viewer.enable:=false

"""

from __future__ import annotations

import os
import threading

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from interfaces.msg import ExchangeStationPose
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor

import mujoco

from sim.mujoco_engine import MujocoEngine
from sim.plugins.arm_plugin import ArmPlugin
from sim.plugins.column_plugin import ColumnPlugin
from sim.station_pose import base_in_O, base_T_station, check_level3
from sim.plugins.gripper_plugin import GripperPlugin


def _flatten(cfg: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (cfg or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _assign(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def _mat_from_flat(flat) -> np.ndarray:
    return np.asarray(flat, dtype=float).reshape(4, 4)


def _pose_of(data, body: str) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(data.body(body).xmat).reshape(3, 3)
    T[:3, 3] = np.asarray(data.body(body).xpos)
    return T


def _find_repo_config(share: str, name: str = "sim.yaml") -> str:
    """从 install/sim/share/sim 往上找仓库根的 config/<name>; 找不到就返回 cwd 下的猜测。

    配置的唯一真源是仓库根 config/, 不随包安装, 所以只能这么定位。
    """
    for base in (share, os.getcwd()):
        cur = os.path.abspath(base)
        while True:
            cand = os.path.join(cur, "config", name)
            if os.path.isfile(cand):
                return cand
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return os.path.join(os.getcwd(), "config", name)


def _rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


class SimNode(Node):
    def __init__(self) -> None:
        super().__init__("sim_node")

        share = get_package_share_directory("sim")
        self.declare_parameter("config_path", _find_repo_config(share))
        self.declare_parameter("model_path", "")

        cfg_path = self.get_parameter("config_path").value
        self.config = self._load_config(cfg_path)

        scene = self.get_parameter("model_path").value or os.path.join(
            share, "model", (self.config.get("model", {}) or {}).get("scene", "scene.xml"))
        if not os.path.isfile(scene):
            raise RuntimeError(f"找不到场景文件: {scene}")
        self.get_logger().info(f"场景: {scene}")

        if (self.config.get("publish", {}) or {}).get("use_sim_time"):
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

        self.engine = MujocoEngine(scene, self.config, logger=self.get_logger())
        self.engine.ctx.node = self

        self.gripper = GripperPlugin(self.engine)
        self.arm = ArmPlugin(self.engine, gripper=self.gripper)
        self.column = ColumnPlugin(self.engine)
        self.engine.register(self.gripper)
        self.engine.register(self.arm)
        self.engine.register(self.column)
        self.engine.setup()

        self._setup_station()
        self.column.configure(self._world_T_station, self._C, self._tool_offset())

    # ---------- 配置 ----------

    def _load_config(self, path: str) -> dict:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fp:
                cfg = yaml.safe_load(fp) or {}
            self.get_logger().info(f"配置: {path}")
        else:
            cfg = {}
            self.get_logger().warn(f"没找到配置 {path}, 全部走代码内默认值")

        # 每个叶子声明成同名 ROS 参数 (dynamic_typing 才能容纳 yaml 里的 null)
        desc = ParameterDescriptor(dynamic_typing=True)
        for key, val in _flatten(cfg).items():
            self.declare_parameter(key, val, descriptor=desc)
            got = self.get_parameter(key).value
            if got != val:
                self.get_logger().info(f"参数覆盖: {key} = {got}")
            _assign(cfg, key, got)
        return cfg

    # ---------- 兑换站真值 ----------

    def _setup_station(self) -> None:
        cfg = self.config.get("station", {}) or {}
        mount = self.config.get("mount", {}) or {}
        m, d = self.engine.model, self.engine.data

        self._F0 = _mat_from_flat(mount.get("line_link_T_arm_base"))
        self._C = _mat_from_flat(mount.get("frame6_T_link6"))

        explicit = cfg.get("pose_xyz_rpy")
        rmuc = cfg.get("rmuc", {}) or {}
        if explicit:
            world_T_station = np.eye(4)
            world_T_station[:3, 3] = np.asarray(explicit[:3], dtype=float)
            world_T_station[:3, :3] = _rpy_to_mat(*[float(v) for v in explicit[3:6]])
        elif rmuc.get("enabled"):
            world_T_station = self._station_from_rmuc(rmuc)
        else:
            world_T_station = self._station_from_joint(
                np.asarray(cfg.get("from_joint") or np.zeros(6), dtype=float))

        # 覆写静态 body 的 pos/quat (静态 body 的 xpos 每次 mj_forward 由 model 算出)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "exchange_station")
        if bid < 0:
            raise RuntimeError("场景里没有 exchange_station body")
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, world_T_station[:3, :3].flatten())
        with self.engine.lock:
            m.body_pos[bid] = world_T_station[:3, 3]
            m.body_quat[bid] = quat
            mujoco.mj_forward(m, d)

             # 摆底座平台 (只有 RMUC 摆法才有 O 系) ↓↓↓
        if rmuc.get("enabled") and not explicit:
            with self.engine.lock:
                world_T_base = _pose_of(self.engine.data, "root")
            world_T_O = world_T_base @ np.linalg.inv(
                base_in_O(rmuc["chassis_in_O"], float(world_T_base[2, 3])))
            pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "assembly_platform")
            if pid < 0:
                raise RuntimeError("场景里没有 assembly_platform body")
            pquat = np.zeros(4)
            mujoco.mju_mat2Quat(pquat, world_T_O[:3, :3].flatten())
            with self.engine.lock:
                m.body_pos[pid] = world_T_O[:3, 3]
                m.body_quat[pid] = pquat
                mujoco.mj_forward(m, d)

        self._world_T_station = world_T_station
        self._station_pub = None
        if cfg.get("publish_pose", True):
            self._station_pub = self.create_publisher(
                ExchangeStationPose, "/vision/exchange_pose", 10)
            hz = float(cfg.get("pose_hz", 5.0))
            self.create_timer(1.0 / max(0.1, hz), self._publish_station)

        arm_base = self._world_T_arm_base()
        rel = np.linalg.inv(arm_base) @ world_T_station
        self.get_logger().info(
            f"兑换站 world 位置 {np.round(world_T_station[:3, 3], 4).tolist()}, "
            f"相对 arm_base {np.round(rel[:3, 3], 4).tolist()} "
            f"(距离 {np.linalg.norm(rel[:3, 3]):.3f} m)")

    def _station_from_rmuc(self, rmuc: dict) -> np.ndarray:
        """按 RMUC2026 规则的科技核心位姿 + 底盘在装配坐标系里的摆放, 算站体 world 位姿。

        底盘焊在 world, 所以反过来摆站体: world_T_station = world_T_base @ base_T_station。
        地面 = world z 0 = 机器人所站的装配区地面 (规则 z 是相对它量的)。
        """
        pose = dict(zip(("x", "y", "z", "theta", "phi", "alpha"), (float(v) for v in rmuc["pose"])))
        for problem in check_level3(pose):
            self.get_logger().warn(f"station.rmuc.pose 不是三级难度合规位姿: {problem}")
        with self.engine.lock:
            world_T_base = _pose_of(self.engine.data, "root")
        base_height = float(world_T_base[2, 3])
        world_T_station = world_T_base @ base_T_station(pose, rmuc["chassis_in_O"], base_height)
        self.get_logger().info(
            f"站体位姿: pose(x,y,z mm / θ,φ,α °)={list(rmuc['pose'])} "
            f"底盘在装配坐标系 (x,y m, yaw °)={list(rmuc['chassis_in_O'])}")
        return world_T_station

    def _station_from_joint(self, q: np.ndarray) -> np.ndarray:
        """用一组演示关节角做 FK, 取那一刻的 TCP 位姿当站体系。

        与 planning 的约定一致: 站体系 = 矿石"完全插入"时的 TCP 位姿 (type3 状态 0),
        x 轴是插入轴。 所以 from_joint 就是一组"插到底"的关节角, 站姿必然可达。
        TCP = 第 6 DH 系 + 工具偏置 (旋转相同), 工具偏置读 planning 的配置 (见 _tool_offset)。
        """
        m, d = self.engine.model, self.engine.data
        with self.engine.lock:
            qsave = d.qpos.copy()
            vsave = d.qvel.copy()
            d.qpos[self.arm.qadr] = np.clip(q, self.arm.lo, self.arm.hi)
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            world_T_link6 = _pose_of(d, "link_6")
            d.qpos[:] = qsave
            d.qvel[:] = vsave
            mujoco.mj_forward(m, d)

        world_T_frame6 = world_T_link6 @ np.linalg.inv(self._C)
        frame6_T_tcp = np.eye(4)
        frame6_T_tcp[:3, 3] = self._tool_offset()
        return world_T_frame6 @ frame6_T_tcp

    def _tool_offset(self) -> np.ndarray:
        """planning 配置里的 tcp_offset_6_tcp (第 6 DH 系 -> TCP 的平移)。

        读 planning 自己的配置, 不在 config/sim.yaml 里抄一份 —— 工具偏置是规划侧的
        标定量, 一处一个真值。 读不到 (没装 planning) 就按 0, 站姿会差一个工具偏置。
        """
        try:
            from planning import load_config
            return np.asarray(load_config()["arm"]["tool"]["tcp_offset_6_tcp"], dtype=float)
        except Exception as exc:                                 # noqa: BLE001
            self.get_logger().warn(
                f"读不到 planning 的 tcp_offset_6_tcp ({exc}), 按 0 处理: "
                "站姿会偏一个工具偏置, planning 侧可能 IK 无解")
            return np.zeros(3)

    def _world_T_arm_base(self) -> np.ndarray:
        """arm_base 在 J1 转轴上, 不是 base_link; line = 0 时是常量 (见设计文档 §六)。"""
        with self.engine.lock:
            return _pose_of(self.engine.data, "line_link") @ self._F0

    def _publish_station(self) -> None:
        rel = np.linalg.inv(self._world_T_arm_base()) @ self._world_T_station
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rel[:3, :3].flatten())

        msg = ExchangeStationPose()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "arm_base"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = rel[:3, 3]
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        msg.confidence = 1.0
        msg.station_id = 0
        msg.valid = True
        self._station_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True, name="ros_spin")
    spin_thread.start()

    node.engine.start()

    # viewer 必须在主线程 (部分 GL 驱动只允许主线程建窗口), 所以 spin 反过来放后台
    viewer_cfg = node.config.get("viewer", {}) or {}
    try:
        if viewer_cfg.get("enable", True):
            node.engine.run_viewer(rclpy.ok, fps=float(viewer_cfg.get("fps", 60.0)))
        else:
            node.engine.run_headless(rclpy.ok)
    except KeyboardInterrupt:
        pass
    finally:
        node.engine.stop()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
