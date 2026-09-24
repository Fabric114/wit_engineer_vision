"""插件式 MuJoCo 引擎: 只管加载 / 步进 / 渲染 / 调度, 行为全部做成插件.
线程模型 :
    主线程     : viewer (launch_passive) —— 部分 GL 驱动只允许主线程建窗口
    物理线程   : 本文件的 _physics_loop, 按墙钟推进
    ROS 执行器 : 由 sim_node 放到另一个线程 spin

一把 RLock 护住 data: 物理线程写 qfrc_applied/步进, ROS 回调写命令缓存,
100 Hz timer 读 qpos/qvel。 锁粒度是"一批 substeps", 不是一步一锁。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import mujoco


@dataclass
class SimContext:
    """插件能看到的一切。 engine 持有它, 每个回调都传进去。"""

    model: Any                       # mujoco.MjModel
    data: Any                        # mujoco.MjData
    lock: threading.RLock
    node: Any = None                 # rclpy Node, 由 sim_node 注入 (引擎自己不用)
    config: dict = field(default_factory=dict)

    def sim_time(self) -> float:
        return float(self.data.time)


class SimPlugin:
    """插件基类。 鸭子类型即可, 继承只是为了少写空方法。

    调用约定:
      - setup()      : 引擎启动前, 单线程, 可以随便建 pub/sub、查模型索引。
      - on_step()    : 每个物理步, **已在锁内**, 插件不要再取锁, 也别做 IO。
      - on_publish() : 状态发布频率 (默认 100 Hz), 也在锁内调用。
      - shutdown()   : 退出时, 单线程。
    """

    name = "plugin"

    def setup(self, ctx: SimContext) -> None: ...

    def on_step(self, ctx: SimContext, dt: float) -> None: ...

    def on_publish(self, ctx: SimContext) -> None: ...

    def shutdown(self, ctx: SimContext) -> None: ...


class MujocoEngine:
    """加载模型 + 跑物理线程 + 调度插件。"""

    def __init__(self, scene_path: str, config: dict, logger: Any = None) -> None:
        self._log = logger
        self.config = config or {}

        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)

        phys = self.config.get("physics", {}) or {}
        if phys.get("timestep"):
            self.model.opt.timestep = float(phys["timestep"])
        if phys.get("disable_contact"):
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
            self._info("已关闭接触 (mjDSBL_CONTACT)")

        self.timestep = float(self.model.opt.timestep)
        self.substeps = max(1, int(phys.get("substeps", 10) or 10))
        self.realtime_factor = float(phys.get("realtime_factor", 1.0) or 1.0)

        self.lock = threading.RLock()
        self.ctx = SimContext(model=self.model, data=self.data,
                              lock=self.lock, config=self.config)

        self._plugins: list[SimPlugin] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        publish_hz = float((self.config.get("publish", {}) or {}).get("state_hz", 100.0))
        self._publish_period = 1.0 / max(1e-6, publish_hz)
        self._next_publish = 0.0

        self.step_count = 0

    # ---------- 生命周期 ----------

    def register(self, plugin: SimPlugin) -> None:
        self._plugins.append(plugin)

    def setup(self) -> None:
        """跑一次 mj_forward 让 xpos 等派生量可用, 然后初始化所有插件。"""
        mujoco.mj_forward(self.model, self.data)
        for p in self._plugins:
            p.setup(self.ctx)
        mujoco.mj_forward(self.model, self.data)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._physics_loop,
                                        name="mujoco_physics", daemon=True)
        self._thread.start()
        self._info(f"物理线程启动: timestep={self.timestep * 1e3:.1f} ms, "
                   f"substeps={self.substeps}, realtime={self.realtime_factor}x")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for p in self._plugins:
            try:
                p.shutdown(self.ctx)
            except Exception as exc:                     # noqa: BLE001
                self._warn(f"插件 {p.name} shutdown 异常: {exc}")

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    # ---------- 物理 ----------

    def _physics_loop(self) -> None:
        wall_start = time.perf_counter()
        sim_start = self.data.time
        batch_dt = self.timestep * self.substeps

        while not self._stop.is_set():
            with self.lock:
                for _ in range(self.substeps):
                    self._step_once()

            # 对齐墙钟: 仿真已经跑到 sim_elapsed, 墙钟该走到 sim_elapsed / factor
            sim_elapsed = self.data.time - sim_start
            target_wall = wall_start + sim_elapsed / self.realtime_factor
            sleep = target_wall - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -1.0:
                # 跑不动了 (通常是插件里做了重活), 重新对齐, 别把欠账越积越多
                wall_start = time.perf_counter()
                sim_start = self.data.time
                self._warn(f"物理线程落后 > 1 s, 已重新对齐墙钟 (batch={batch_dt * 1e3:.0f} ms)")

    def _step_once(self) -> None:
        for p in self._plugins:
            p.on_step(self.ctx, self.timestep)
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        if self.data.time >= self._next_publish:
            self._next_publish = self.data.time + self._publish_period
            for p in self._plugins:
                p.on_publish(self.ctx)

    # ---------- 可视化 (必须在主线程调用) ----------

    def run_viewer(self, should_run: Callable[[], bool], fps: float = 60.0) -> None:
        """阻塞在主线程跑 passive viewer, 直到窗口关闭或 should_run() 变假。"""
        import mujoco.viewer

        period = 1.0 / max(1.0, fps)
        with mujoco.viewer.launch_passive(self.model, self.data,
                                         show_left_ui=False, show_right_ui=False) as viewer:
            while viewer.is_running() and should_run() and not self._stop.is_set():
                with self.lock:
                    viewer.sync()
                time.sleep(period)

    def run_headless(self, should_run: Callable[[], bool]) -> None:
        while should_run() and not self._stop.is_set():
            time.sleep(0.1)

    # ---------- 小工具 ----------

    def joint_addr(self, names: list[str]) -> tuple[list[int], list[int], list[int]]:
        """按关节名查 (joint id, qpos 地址, dof 地址)。 名字写错直接报错, 别静默跑偏。"""
        jids, qadr, dadr = [], [], []
        for n in names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid < 0:
                raise RuntimeError(f"模型里没有关节 '{n}'")
            jids.append(jid)
            qadr.append(int(self.model.jnt_qposadr[jid]))
            dadr.append(int(self.model.jnt_dofadr[jid]))
        return jids, qadr, dadr

    def _info(self, msg: str) -> None:
        (self._log.info if self._log else print)(msg)

    def _warn(self, msg: str) -> None:
        (self._log.warn if self._log else print)(msg)
