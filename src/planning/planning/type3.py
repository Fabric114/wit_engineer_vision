# type3.py — 装配段规划 (约束流形上的插入轨迹)。
"""type3.py — 装配段规划: 约束流形上的插入轨迹 (本库最复杂的一块)。

【这个文件干什么】
插入/装配阶段, 末端不能像接近段那样自由乱走, 必须贴着兑换站定义的"装配
流形"运动 —— 也就是只允许 沿轴向进给 / 滑动 / 两个方向的小旋转 这几种受约束
的动作。 本文件负责把这样一段装配动作规划成一条可执行的关节轨迹。

【它的整体流程 (读懂这 5 步就懂了 type3)】
  1. 把装配动作离散成一串任务空间位姿 (AssemblyState -> 末端目标位姿序列);
  2. 对每个位姿, 绕插入轴在一圈 roll 角上采样, 每个 roll 用解析 IK 解出候选
     关节配置 (这就形成了 viterbi.py 里说的"分层网格": 层=路径点, 节点=候选);
  3. (可选) 用碰撞检测给每个候选打分/过滤;
  4. 用 Viterbi 在这些候选里选出一条平滑、代价最小的关节序列;
  5. 若严格解不出来, 用 scipy 最小二乘 (least_squares) "尽力修复", 找一个
     尽量满足约束的近似解; 残差超 best_ik.max_position_error_m / max_axis_error_deg
     的修复点一律判失败 (sigma 只是权重, 不是验收阈值)。
  6. 首个路点必须与实测起点 initial_joints 衔接: 差值 ≤ start_joint_tolerance_rad
     时在前面插入 initial 作为第 0 点, 否则判失败。
  最后再做时间参数化 (trajectory.py) 就得到可执行轨迹。

【要懂的概念】
- AssemblyState: 描述"装到哪一步"的物理坐标 (轴向偏移/滑动/两个旋转/释放),
  AssemblyPath 就是这些状态的有序序列 (.between() 可在两个状态间线性取样)。
- station_transform: 兑换站相对 arm_base 的 4x4 位姿 —— 正常由视觉给出;
  你们没相机时, 从仿真里直接读 ground truth 得到 (见 README)。

【依赖】 arm_model (IK) + viterbi + scipy。 几何参数 (r_p/r_q/phi 等) 在
config 的 planning.type3.exchange_trajectory 里, 是兑换站的装配尺寸, 需按实物核对。
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .arm_model import ArmModel
from .viterbi import solve_viterbi
from .transform import validate_transforms

#: 起点与首点差小于此值视为同一点, 不插连接段
START_EXACT_TOLERANCE_RAD = 1e-9
#: 连接段碰撞检查的插值步长 (rad, 最大单轴)
START_CONNECTION_STEP_RAD = 0.01


# 装配流形上的一个物理状态 (5 维): 轴向偏移/滑动 [m], 绕 p/q 两轴的旋转 [rad], 释放后退量 [m]。
@dataclass(frozen=True, slots=True)
class AssemblyState:
    """Physical coordinates that define one pose on the assembly manifold."""

    axial_offset_m: float = 0.0
    slide_m: float = 0.0
    p_angle_rad: float = 0.0
    q_angle_rad: float = 0.0
    release_m: float = 0.0

    # 按 (axial, slide, p, q, release) 顺序转成 (5,) 数组, 与 AssemblyPath.states 的列顺序一致。
    def as_array(self) -> np.ndarray:
        return np.asarray(
            (
                self.axial_offset_m,
                self.slide_m,
                self.p_angle_rad,
                self.q_angle_rad,
                self.release_m,
            ),
            dtype=float,
        )


# 装配状态序列, states 形状 (N, 5), 列含义同 AssemblyState.as_array()。
@dataclass(frozen=True, slots=True)
class AssemblyPath:
    """Ordered task-space states; rows are physical assembly coordinates."""

    states: np.ndarray

    # 构造后校验形状与有限性, 并把 states 规范为 float ndarray (frozen dataclass 需用 object.__setattr__ 回写)。
    def __post_init__(self) -> None:
        values = np.asarray(self.states, dtype=float)
        if values.ndim != 2 or values.shape[1] != 5 or len(values) < 1:
            raise ValueError(f"AssemblyPath.states must have shape (N, 5), N >= 1; got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("AssemblyPath.states must be finite")
        object.__setattr__(self, "states", values)

    # 在 start 与 goal 两个状态间按 samples 个点线性插值 (含两端)。
    @classmethod
    def between(
        cls,
        start: AssemblyState,
        goal: AssemblyState,
        samples: int = 100,
    ) -> "AssemblyPath":
        if samples < 2:
            raise ValueError("AssemblyPath requires at least two samples")
        # fraction: (samples, 1), 广播后得到 (samples, 5) 的状态序列。
        fraction = np.linspace(0.0, 1.0, int(samples))[:, None]
        start_values = start.as_array()
        return cls(start_values + fraction * (goal.as_array() - start_values))


# type3 规划结果。waypoints: (M, 6) 关节轨迹 [rad]; rolls: (M,) 每点所选 roll [rad, 已 unwrap];
# path: Viterbi 选中的 (层, roll 索引) 序列; continuation: 末端后续分支结果; 多数失败情况下 waypoints 为空。
@dataclass(frozen=True, slots=True)
class Type3PlanResult:
    success: bool
    message: str
    cost: float
    waypoints: np.ndarray
    rolls: np.ndarray
    path: list[tuple[int, int]] | None
    continuation: dict[str, object] | None
    diagnostics: dict[str, object]


# 分层图 (层 = 路径采样点 N, 节点 = roll 采样 R)。
# targets: (N, R, 4, 4) TCP 目标位姿 (arm_base 系); joints: (N, R, 6) [rad]; valid/penalties: (N, R)。
@dataclass(slots=True)
class _LayeredGraph:
    rolls: np.ndarray
    targets: np.ndarray
    joints: np.ndarray
    valid: np.ndarray
    penalties: np.ndarray


# type3 规划器: 固定一个 IK 分支, 在 roll 离散化后的装配路径上建分层图, 用 Viterbi 求最优关节序列。
class Type3Planner:
    """Plan one IK branch over a roll-discretized assembly path."""

    # 保存装配几何、roll 采样网格与各项代价权重。
    def __init__(
        self,
        arm: ArmModel,
        *,
        geometry: dict,
        ik_branch: int,
        roll_sample_step_deg: float,
        bandwidth: int,
        collision_soft_margin: float,
        collision_soft_weight: float,
        motion_l1_weight: float,
        motion_l2_weight: float,
        joint_limit_margin_rad: float,
        joint_limit_weight: float,
        best_effort: dict | None = None,
        continuation: dict | None = None,
        start_joint_tolerance_rad: float | None = None,
        max_joint_step_rad: float | None = None,
    ) -> None:
        self.arm = arm
        # 兑换站装配几何: r_p/r_q 为 p/q 旋转的回转半径 [m], phi 为 p 旋转的相位偏置 [rad] (用法见 _poses)。
        self.p_radius = float(geometry["r_p"])
        self.q_radius = float(geometry["r_q"])
        self.p_offset = float(geometry["phi"])
        # IK 分支号限定 0..7 (8 分支 IK, 具体含义随 arm 的 ik_type 而定)。
        self.ik_branch = int(ik_branch)
        if self.ik_branch not in range(8):
            raise ValueError("ik_branch must be in [0, 7]")
        # roll 采样网格: [0, 360) 度按步长等分并转为 rad, 形状 (R,)。
        self.rolls = np.deg2rad(
            np.arange(0.0, 360.0, float(roll_sample_step_deg), dtype=float)
        )
        if len(self.rolls) == 0:
            raise ValueError("roll_sample_step_deg produced an empty roll grid")
        # Viterbi 带宽: 相邻层 roll 索引最多变化 ±bandwidth (环形)。
        self.bandwidth = int(bandwidth)
        if self.bandwidth < 0:
            raise ValueError("bandwidth must be non-negative")
        # 代价权重: 碰撞软间隔惩罚、关节运动 L1/L2、关节限位软惩罚。
        self.soft_margin = float(collision_soft_margin)
        self.soft_weight = float(collision_soft_weight)
        self.motion_l1_weight = float(motion_l1_weight)
        self.motion_l2_weight = float(motion_l2_weight)
        self.limit_margin = float(joint_limit_margin_rad)
        self.limit_weight = float(joint_limit_weight)
        # best_effort: 严格搜索失败后的修复配置; continuation: 末端后续分支的代价配置。
        self.best_effort = dict(best_effort or {})
        self.continuation_config = dict(continuation or {})
        # 实测起点与流形首点允许的最大单轴差; 不给则只接受精确衔接
        self.start_tolerance = (
            START_EXACT_TOLERANCE_RAD
            if start_joint_tolerance_rad is None
            else float(start_joint_tolerance_rad)
        )
        # 相邻路点最大单轴关节变化 (rad) 的硬门槛; None = 不限 (只靠运动代价, 旧行为)
        self.max_joint_step = None if max_joint_step_rad is None else float(max_joint_step_rad)

    # 主入口: 建图 -> (可选) 过滤末端后续分支 -> Viterbi 搜索 -> (可选) best-effort 修复 -> 起点衔接检查 -> 组装结果。
    # station_transform: (4, 4) 兑换站在 arm_base 下的位姿; initial_joints: (6,) 实测关节角 [rad]。
    def plan(
        self,
        path: AssemblyPath,
        station_transform: np.ndarray,
        initial_joints: np.ndarray,
        collision_checker,
        *,
        terminal_continuations: tuple[AssemblyPath, ...] = (),
    ) -> Type3PlanResult:
        # 输入校验: 站位姿须为合法齐次变换, 起始关节须为 (6,)。
        station = validate_transforms(
            np.asarray(station_transform, dtype=float)[None, :, :],
            name="station_transform",
        )[0]
        initial = np.asarray(initial_joints, dtype=float)
        if initial.shape != (6,):
            raise ValueError(f"initial_joints must have shape (6,), got {initial.shape}")

        # 建分层图; 若给了末端后续分支, 剔除末端层中走不通后续分支的节点, 并按 roll 索引缓存后续结果。
        graph, diagnostics = self._build_graph(path, station, collision_checker)
        continuation_cache: dict[int, dict[str, object]] = {}
        if terminal_continuations:
            continuation_cache, continuation_diagnostics = self._filter_terminal_continuations(
                graph,
                terminal_continuations,
                station,
                collision_checker,
            )
            diagnostics.update(continuation_diagnostics)

        # 先做严格搜索 (只经过 IK 与碰撞均有效的节点)。
        path_indices, cost = self._search(graph, initial, allow_invalid=False)
        repair_diagnostics = None
        waypoints = None
        # 严格失败且启用 best_effort: 允许经过无效节点 (固定惩罚) 再搜一次, 再对无效节点做数值 IK 修复;
        # 修复不达标仍判失败。
        if path_indices is None and bool(self.best_effort.get("enabled", False)):
            path_indices, cost = self._search(graph, initial, allow_invalid=True)
            if path_indices is not None:
                waypoints, repair_diagnostics = self._repair(
                    graph,
                    path_indices,
                    initial,
                    collision_checker,
                )
                if not bool(repair_diagnostics["success"]):
                    path_indices = None

        # 失败分支: 拼接失败原因 (含修复被拒原因), 返回空轨迹。
        diagnostics["repair"] = repair_diagnostics
        if path_indices is None:
            message = self._failure_message(graph, diagnostics)
            if repair_diagnostics is not None and not bool(repair_diagnostics["success"]):
                message += f" repair_rejected={repair_diagnostics['reject_reason']}"
            return Type3PlanResult(
                success=False,
                message=message,
                cost=float(cost),
                waypoints=np.zeros((0, 6), dtype=float),
                rolls=np.zeros(0, dtype=float),
                path=None,
                continuation=None,
                diagnostics=diagnostics,
            )

        # 严格搜索成功时直接取图中关节解, 连续关节 unwrap 并对齐到 initial 附近的 2π 圈。
        if waypoints is None:
            raw = np.asarray([graph.joints[i, j] for i, j in path_indices])
            waypoints = self.arm.joint_space.align_trajectory(raw, initial[None, :])
        # 所选 roll 序列做 unwrap, 避免 0/2π 处跳变。
        selected_rolls = np.unwrap(np.asarray([graph.rolls[j] for _, j in path_indices]))
        # 相邻路点跳变复查 (修复出来的点没经过搜索里的门槛), 再做首点与实测起点的衔接检查。
        jump_reason = self._check_steps(waypoints, diagnostics)
        start_ok, start_reason = (
            (False, jump_reason)
            if jump_reason is not None
            else self._check_start(waypoints[0], initial, collision_checker, diagnostics)
        )
        if not start_ok:
            return Type3PlanResult(
                success=False,
                message=start_reason,
                cost=float(cost),
                waypoints=np.zeros((0, 6), dtype=float),
                rolls=np.zeros(0, dtype=float),
                path=None,
                continuation=None,
                diagnostics=diagnostics,
            )
        # 末端后续分支: 走过修复时末端关节可能已变, 需用实际末端关节重新评估; 否则直接取建图时的缓存。
        continuation = None
        if terminal_continuations and repair_diagnostics is not None:
            branches = tuple(
                self._evaluate_continuation(
                    candidate,
                    station,
                    float(graph.rolls[path_indices[-1][1]]),
                    waypoints[-1],
                    collision_checker,
                )[0]
                for candidate in terminal_continuations
            )
            if all(branch is not None for branch in branches):
                continuation = self._combine_continuations(
                    terminal_continuations,
                    branches,
                )
        elif terminal_continuations:
            continuation = continuation_cache.get(path_indices[-1][1])
        # 要求后续分支却无可行解: 判失败, 但仍返回已规划轨迹供诊断。
        if terminal_continuations:
            if continuation is None:
                return Type3PlanResult(
                    success=False,
                    message="selected terminal has no feasible continuation",
                    cost=float(cost),
                    waypoints=waypoints,
                    rolls=selected_rolls,
                    path=path_indices,
                    continuation=None,
                    diagnostics=diagnostics,
                )
        if diagnostics["start_connected"]:
            # 轨迹从实测关节角出发: 第 0 点 = initial, 再接流形首点
            waypoints = np.vstack((initial[None, :], waypoints))
            selected_rolls = np.concatenate((selected_rolls[:1], selected_rolls))
        # 成功: 组装结果, message 中注明 best-effort 修复的节点数。
        repair_count = 0 if repair_diagnostics is None else int(repair_diagnostics["repair_count"])
        message = "ok" if repair_count == 0 else f"ok best_effort repaired_nodes={repair_count}"
        return Type3PlanResult(
            success=True,
            message=message,
            cost=float(cost),
            waypoints=waypoints,
            rolls=selected_rolls,
            path=path_indices,
            continuation=continuation,
            diagnostics=diagnostics,
        )

    # 公开接口: 由装配状态与 roll 计算 TCP 目标位姿 (arm_base 系), 返回 (B, 4, 4)。
    # states 与 rolls 的 batch 长度须相等, 或其中一方为 1 (广播)。
    def task_poses(
        self,
        path: AssemblyPath,
        station_transform: np.ndarray,
        rolls: np.ndarray,
    ) -> np.ndarray:
        # 统一 batch 大小并广播。
        states = np.asarray(path.states, dtype=float)
        roll = np.asarray(rolls, dtype=float).reshape(-1)
        batch_size = max(len(states), len(roll))
        if len(states) not in (1, batch_size) or len(roll) not in (1, batch_size):
            raise ValueError("path states and rolls must have equal batch sizes or a singleton batch")
        states = np.broadcast_to(states, (batch_size, 5))
        roll = np.broadcast_to(roll, (batch_size,))
        # 校验站位姿后计算位姿。
        station = validate_transforms(
            np.asarray(station_transform, dtype=float)[None, :, :],
            name="station_transform",
        )[0]
        return self._poses(states, roll, station)

    # 建分层图: 对每个 (路径点, roll) 求目标位姿与 IK, 再做碰撞过滤与软惩罚, 返回 (graph, diagnostics)。
    def _build_graph(self, path: AssemblyPath, station: np.ndarray, collision_checker):
        # 展平成 N*R 个 (state, roll) 组合, 行序为 [层][roll]: state 按 roll 数重复, roll 网格按层平铺。
        n_steps = len(path.states)
        n_rolls = len(self.rolls)
        states = np.repeat(path.states, n_rolls, axis=0)
        rolls = np.tile(self.rolls, n_steps)
        targets = self._poses(states, rolls, station)
        # 批量 IK, 只取配置的单一分支; 结果 reshape 成 (N, R, 6) / (N, R)。
        joints, valid = self.arm.solve_ik(targets, branches=(self.ik_branch,))
        joints = joints[:, 0].reshape(n_steps, n_rolls, 6)
        ik_valid = valid[:, 0].reshape(n_steps, n_rolls)
        valid = ik_valid.copy()
        penalties = np.zeros((n_steps, n_rolls), dtype=float)

        # 碰撞检查只对 IK 有效节点做: 碰撞则置无效, 未碰撞则记录软间隔惩罚。
        # valid_flat 是 valid 的 view, 原地修改即同步到 valid。
        valid_flat = valid.reshape(-1)
        selected = np.flatnonzero(valid_flat)
        collision_rejected = np.zeros(valid_flat.shape, dtype=bool)
        if collision_checker is not None and len(selected):
            collides, collision_penalty = collision_checker.check_configs_with_penalty(
                joints.reshape(-1, 6)[selected],
                self.soft_margin,
                self.soft_weight,
            )
            collision_rejected[selected[collides]] = True
            valid_flat[selected[collides]] = False
            penalties.reshape(-1)[selected[~collides]] = collision_penalty[~collides]
        valid = valid_flat.reshape(n_steps, n_rolls)
        # 叠加关节限位软惩罚 (进入限位 margin 内按平方增长)。
        penalties += self.arm.joint_space.limit_penalty(
            joints,
            self.limit_margin,
            self.limit_weight,
        )
        # 组装图与诊断统计 (各阶段有效节点数, 总计与逐层)。
        graph = _LayeredGraph(
            rolls=self.rolls,
            targets=targets.reshape(n_steps, n_rolls, 4, 4),
            joints=joints,
            valid=valid,
            penalties=penalties,
        )
        diagnostics = {
            "total_nodes": int(valid.size),
            "ik_valid_nodes": int(np.count_nonzero(ik_valid)),
            "final_valid_nodes": int(np.count_nonzero(valid)),
            "collision_checked_nodes": int(len(selected)),
            "collision_rejected_nodes": int(np.count_nonzero(collision_rejected)),
            "ik_valid_by_step": np.count_nonzero(ik_valid, axis=1),
            "final_valid_by_step": np.count_nonzero(valid, axis=1),
            "collision_rejected_by_step": np.count_nonzero(
                collision_rejected.reshape(n_steps, n_rolls), axis=1
            ),
            "roll_count": n_rolls,
            "path_samples": n_steps,
            "ik_branch": self.ik_branch,
        }
        return graph, diagnostics

    # Viterbi 搜索最优 roll 序列。allow_invalid=False: 无效节点代价为 inf;
    # True (best_effort): 无效节点代价为 invalid_node_penalty, 并额外按 roll 索引变化计代价。返回 ([(层, roll 索引)], cost)。
    def _search(self, graph: _LayeredGraph, initial: np.ndarray, *, allow_invalid: bool):
        # 环形带宽邻域: previous_indices[j] 为当前层第 j 个 roll 在上一层可接的 roll 索引, 形状 (R, 2*bandwidth+1)。
        n_steps, n_rolls = graph.valid.shape
        offsets = np.arange(-self.bandwidth, self.bandwidth + 1)
        previous_indices = (np.arange(n_rolls)[:, None] + offsets[None, :]) % n_rolls
        # best_effort 模式的无效节点惩罚与 roll 变化权重。
        invalid_penalty = float(self.best_effort.get("invalid_node_penalty", 1000.0))
        roll_weight = float(self.best_effort.get("roll_motion_weight", 1.0))

        # 单层节点代价 (R,): 有效节点取碰撞+限位惩罚, 无效节点取 inf 或固定惩罚。
        def node_cost(layer: int) -> np.ndarray:
            if allow_invalid:
                return np.where(graph.valid[layer], graph.penalties[layer], invalid_penalty)
            return np.where(graph.valid[layer], graph.penalties[layer], np.inf)

        # 首层代价 = 节点代价 + 从 initial 到该节点的关节运动代价 (仅有效节点计运动)。
        initial_delta = self.arm.joint_space.delta(
            graph.joints[0],
            np.broadcast_to(initial, graph.joints[0].shape),
        )
        initial_motion = self._motion_cost(initial_delta)
        initial_cost = node_cost(0) + np.where(graph.valid[0], initial_motion, 0.0)
        # 首层有有效节点时, 只保留离 initial 运动代价最小的那个 roll 作为起点 (其余置 inf),
        # 保证轨迹从最接近实测起点的解出发。
        if np.any(graph.valid[0]):
            nearest = int(np.argmin(np.where(graph.valid[0], initial_motion, np.inf)))
            locked = np.full(n_rolls, np.inf)
            locked[nearest] = initial_cost[nearest]
            initial_cost = locked

        # 状态转移: 对当前层每个 roll, 在带宽邻域内选累计代价最小的前驱; 返回 (累计代价 (R,), 前驱索引 (R,))。
        def step(layer: int, previous_cost: np.ndarray):
            previous_joints = graph.joints[layer - 1][previous_indices]
            both_valid = graph.valid[layer, :, None] & graph.valid[layer - 1][previous_indices]
            delta = self.arm.joint_space.delta(
                graph.joints[layer, :, None, :],
                previous_joints,
            )
            # 两端都有效才计关节运动代价; 涉及无效节点时运动项为 0。
            transition = np.where(both_valid, self._motion_cost(delta), 0.0)
            # 硬门槛: 两端都有效但单轴跳变超限的边不可走 (分支内跨腕奇异 / 构型翻转)。
            if self.max_joint_step is not None:
                jump = both_valid & (np.max(np.abs(delta), axis=-1) > self.max_joint_step)
                transition = np.where(jump, np.inf, transition)
            # best_effort 下额外惩罚 roll 索引的环形距离, 防止经过无效节点时 roll 乱跳。
            if allow_invalid:
                roll_delta = np.abs(np.arange(n_rolls)[:, None] - previous_indices)
                transition += roll_weight * np.minimum(roll_delta, n_rolls - roll_delta)
            # 累计代价 (R, K), 逐行取最小。
            total = previous_cost[previous_indices] + transition + node_cost(layer)[:, None]
            best = np.argmin(total, axis=1)
            return total[np.arange(n_rolls), best], previous_indices[np.arange(n_rolls), best]

        # 交给通用 Viterbi 求解并回溯; 无解返回 (None, inf)。
        result = solve_viterbi(initial_cost, n_steps, step)
        if not result.success:
            return None, float("inf")
        return [(i, int(j)) for i, j in enumerate(result.path_indices)], float(result.cost)

    def _check_steps(self, waypoints, diagnostics):
        """相邻路点最大单轴变化; 超过 max_joint_step 返回失败说明, 否则 None。"""
        steps = (
            np.max(np.abs(self.arm.joint_space.delta(waypoints[1:], waypoints[:-1])), axis=1)
            if len(waypoints) > 1
            else np.zeros(0)
        )
        diagnostics["max_joint_step_rad"] = float(steps.max()) if len(steps) else 0.0
        if self.max_joint_step is None or not len(steps) or steps.max() <= self.max_joint_step:
            return None
        layer = int(np.argmax(steps))
        return (
            f"joint jump between waypoints {layer}->{layer + 1} "
            f"max_step_rad={float(steps[layer]):.4f} limit_rad={self.max_joint_step:.4f}"
        )

    def _check_start(self, first, initial, collision_checker, diagnostics):
        """实测起点 -> 流形首点: 差值超阈值则拒绝; 阈值内则连接段逐点查碰撞。"""
        # 最大单轴关节差 (连续关节按最短角距)。
        delta = self.arm.joint_space.delta(first[None, :], initial[None, :])[0]
        max_delta = float(np.max(np.abs(delta)))
        diagnostics["start_delta_rad"] = max_delta
        diagnostics["start_connected"] = False
        # 超出容差或非有限: 直接拒绝。
        if not np.isfinite(max_delta) or max_delta > self.start_tolerance:
            return False, (
                f"start joints do not match first assembly waypoint "
                f"max_delta_rad={max_delta:.4f} tolerance_rad={self.start_tolerance:.4f}"
            )
        # 几乎重合: 不需要连接段。
        if max_delta <= START_EXACT_TOLERANCE_RAD:
            return True, "ok"
        # initial -> 首点按 START_CONNECTION_STEP_RAD 线性插值, 逐点查碰撞。
        if collision_checker is not None:
            count = max(2, int(np.ceil(max_delta / START_CONNECTION_STEP_RAD)) + 1)
            fraction = np.linspace(0.0, 1.0, count)[:, None]
            connection = initial[None, :] + fraction * delta[None, :]
            if np.any(collision_checker.check_configs(connection)):
                return False, "start connection to first assembly waypoint collides"
        # 标记需在轨迹前插入 initial 作为第 0 点 (见 plan)。
        diagnostics["start_connected"] = True
        return True, "ok"

    # 返回 None 表示接受, 否则返回拒绝原因字符串。
    def _repair_accepted(self, q, metric):
        """近似 IK 节点的验收: 有限、在限位内、位置/轴向 (roll 不自由时含 roll) 残差不超阈值。

        阈值键缺失时一律不接受 (fail closed); position_sigma_m / axis_sigma_deg 只是优化权重。
        """
        # 阈值未配置: fail closed。
        config = self.best_effort
        if "max_position_error_m" not in config or "max_axis_error_deg" not in config:
            return "tolerance_not_configured"
        # 有限性检查 (metric 前 3 项: 位置/轴向/roll 误差)。
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(metric[:3])):
            return "non_finite"
        # 有界关节须在限位内。
        bounded = self.arm.joint_space.bounded
        if np.any(q[bounded] < self.arm.joint_space.lower[bounded]) or np.any(
            q[bounded] > self.arm.joint_space.upper[bounded]
        ):
            return "joint_limit"
        # 位置误差 [m] 与轴向误差 [rad] 阈值检查; roll 不自由时 roll 误差也套用轴向阈值。
        if metric[0] > float(config["max_position_error_m"]):
            return "position_tolerance"
        max_rotation = np.deg2rad(float(config["max_axis_error_deg"]))
        if metric[1] > max_rotation:
            return "axis_tolerance"
        if not bool(config.get("roll_free", True)) and metric[2] > max_rotation:
            return "roll_tolerance"
        return None

    # best-effort 修复: 沿 Viterbi 路径逐点处理, 有效节点用图中解, 无效节点用 _approximate_ik 近似,
    # 每点以前一点为参考做 2π 对齐, 最后整体查碰撞。返回 (joints (N, 6), 修复诊断)。
    def _repair(self, graph, path, initial, collision_checker):
        repaired = []
        q_ref = np.asarray(initial, dtype=float)
        invalid_mask = []
        metrics = []
        reject_reasons = []
        # 逐层前推; q_ref 是上一点的关节, 作为数值 IK 初值与对齐参考。
        for layer, roll_index in path:
            if graph.valid[layer, roll_index]:
                q = graph.joints[layer, roll_index]
                metric = None
            else:
                # 无效节点: 数值 IK 近似, 不达标只记录原因不中断, 以便收集所有失败点。
                q, metric = self._approximate_ik(graph.targets[layer, roll_index], q_ref)
                reason = self._repair_accepted(q, metric)
                if reason is not None:
                    reject_reasons.append(f"{reason}@{layer}")
            # 连续关节做 2π 对齐到参考点附近, 再更新参考。
            q = self.arm.joint_space.align_to_reference(q[None, :], q_ref[None, :])[0]
            repaired.append(q)
            invalid_mask.append(not bool(graph.valid[layer, roll_index]))
            if metric is not None:
                metrics.append(metric)
            q_ref = q
        # 整条修复后轨迹做碰撞检查。
        joints = np.asarray(repaired)
        collisions = (
            np.zeros(len(joints), dtype=bool)
            if collision_checker is None
            else collision_checker.check_configs(joints)
        )
        if np.any(collisions):
            reject_reasons.append(f"collision@{np.flatnonzero(collisions).tolist()}")
        # 汇总: 有任何拒绝原因即失败; 误差统计只针对被修复的节点。
        return joints, {
            "success": not reject_reasons,
            "reject_reason": ",".join(reject_reasons) or None,
            "repair_count": int(np.count_nonzero(invalid_mask)),
            "collision_count": int(np.count_nonzero(collisions)),
            "max_pos_error_m": max((m[0] for m in metrics), default=0.0),
            "max_axis_error_rad": max((m[1] for m in metrics), default=0.0),
            "max_roll_error_rad": max((m[2] for m in metrics), default=0.0),
            "max_runtime_ms": max((m[3] for m in metrics), default=0.0),
        }

    # 数值近似 IK: least_squares 在关节限位内最小化 位置/轴向 (可选 roll) 误差 + 偏离 q_ref 的正则项。
    # 返回 (q (6,), metric=(位置误差 m, 轴向误差 rad, roll 误差 rad, 耗时 ms))。
    def _approximate_ik(self, target: np.ndarray, q_ref: np.ndarray):
        # sigma 为各残差项的归一化尺度 (即权重), 不是验收阈值。
        config = self.best_effort
        position_sigma = float(config.get("position_sigma_m", 0.005))
        axis_sigma = np.deg2rad(float(config.get("axis_sigma_deg", 5.0)))
        reference_sigma = np.deg2rad(float(config.get("joint_reference_sigma_deg", 30.0)))
        roll_free = bool(config.get("roll_free", True))

        # FK 求 TCP 并算姿态误差: 误差旋转在目标 TCP 系下取 rotvec, x 分量视为绕插入轴的 roll 误差 (小角度近似),
        # 其余分量为插入轴指向误差 (TCP x 轴即插入轴, 见 _poses)。
        def errors(q):
            tcp = self.arm.forward_kinematics(np.asarray(q)[None, :]).tcp_transforms[0]
            rotation_error = Rotation.from_matrix(target[:3, :3].T @ tcp[:3, :3]).as_rotvec()
            roll_error = float(rotation_error[0])
            axis_error = rotation_error.copy()
            axis_error[0] = 0.0
            return tcp, axis_error, roll_error

        # 残差: 位置/轴向误差按 sigma 归一化, roll 不自由时加 roll 项, 最后加关节偏离 q_ref 的正则。
        def residual(q):
            tcp, axis_error, roll_error = errors(q)
            terms = [(tcp[:3, 3] - target[:3, 3]) / position_sigma, axis_error / axis_sigma]
            if not roll_free:
                # TODO(注释): roll 项归一化尺度写死为 90°, 未走 config, 不确定是否为有意的弱约束取值。
                terms.append(np.asarray([roll_error / np.deg2rad(90.0)]))
            delta = self.arm.joint_space.delta(q[None, :], q_ref[None, :])[0]
            terms.append(delta / reference_sigma)
            return np.concatenate(terms)

        # 初值取 q_ref 并裁剪进限位 (trf 要求初值在 bounds 内)。
        initial = np.clip(
            np.asarray(q_ref, dtype=float),
            self.arm.joint_space.lower,
            self.arm.joint_space.upper,
        )
        # 有界 TRF 求解并计时; max_nfev 默认 40, 限制单点开销。
        started = time.perf_counter()
        solution = least_squares(
            residual,
            initial,
            bounds=(self.arm.joint_space.lower, self.arm.joint_space.upper),
            method="trf",
            max_nfev=int(config.get("max_nfev", 40)),
            x_scale="jac",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        # 用最终解重算误差, 组装 metric。
        q = np.asarray(solution.x)
        tcp, axis_error, roll_error = errors(q)
        metric = (
            float(np.linalg.norm(tcp[:3, 3] - target[:3, 3])),
            float(np.linalg.norm(axis_error)),
            float(abs(roll_error)),
            (time.perf_counter() - started) * 1000.0,
        )
        return q, metric

    # 对末端层每个有效 roll 节点评估全部后续分支: 任一分支不可行则把该节点置无效 (原地改 graph);
    # 全部可行则缓存合并结果, 并把后续代价加到该节点惩罚上。返回 (cache: roll 索引 -> 合并结果, 诊断)。
    def _filter_terminal_continuations(
        self,
        graph,
        continuations: tuple[AssemblyPath, ...],
        station,
        collision_checker,
    ):
        cache = {}
        checked = 0
        rejected = 0
        reasons: dict[str, int] = {}
        terminal = len(graph.valid) - 1
        for roll_index in np.flatnonzero(graph.valid[terminal]):
            checked += 1
            branches = []
            branch_reasons = []
            for continuation in continuations:
                result, reason = self._evaluate_continuation(
                    continuation,
                    station,
                    float(graph.rolls[roll_index]),
                    graph.joints[terminal, roll_index],
                    collision_checker,
                )
                branches.append(result)
                branch_reasons.append(reason)
            # 任一分支失败: 剔除该末端节点, 按 branch_{序号}_{原因} 计数。
            if any(result is None for result in branches):
                graph.valid[terminal, roll_index] = False
                rejected += 1
                for index, (result, reason) in enumerate(zip(branches, branch_reasons, strict=True)):
                    if result is None:
                        key = f"branch_{index}_{reason}"
                        reasons[key] = reasons.get(key, 0) + 1
            # 全部可行: 合并并缓存; 后续代价计入末端节点惩罚, 使 Viterbi 偏好后续更顺的末端 roll。
            else:
                combined = self._combine_continuations(
                    continuations,
                    tuple(branches),
                )
                cache[int(roll_index)] = combined
                graph.penalties[terminal, roll_index] += float(combined["cost"])
        # 诊断中覆盖 final_valid_* (过滤后有效节点数已变化)。
        return cache, {
            "continuation_checked_nodes": checked,
            "continuation_rejected_nodes": rejected,
            "continuation_reject_reasons": reasons,
            "final_valid_nodes": int(np.count_nonzero(graph.valid)),
            "final_valid_by_step": np.count_nonzero(graph.valid, axis=1),
        }

    # 合并末端后续分支: 单分支直接返回副本; 双分支时第一条反转并去掉与装配末端重合的首点, 再接第二条,
    # 得到 '分支1 末端 -> 装配末端 -> 分支2 末端' 的连续序列, 代价取两者最大值。
    # TODO(注释): 双分支合并的物理意图不确定 (是否表示先沿分支1 退回再走分支2?), 代价取 max 而非 sum 的原因待确认。
    @staticmethod
    def _combine_continuations(paths, results):
        if len(results) == 1:
            return dict(results[0])
        if len(results) != 2:
            raise ValueError("combining more than two terminal continuation branches is unsupported")
        first_path, second_path = paths
        first, second = results
        return {
            "states": np.vstack((first_path.states[:0:-1], second_path.states)),
            "joints": np.vstack((first["joints"][:0:-1], second["joints"])),
            "cost": max(float(first["cost"]), float(second["cost"])),
        }

    # 评估一条后续分支: 固定末端 roll 与 IK 分支, 从 start 关节接续求整段关节解, 查碰撞并计算代价。
    # 返回 (结果 dict 或 None, 原因: ok / ik / repair_tolerance / collision)。
    def _evaluate_continuation(self, path, station, roll, start, collision_checker):
        config = self.continuation_config
        # 整段使用同一 roll 计算目标位姿并做 IK。
        targets = self._poses(path.states, np.full(len(path.states), roll), station)
        joints, valid = self.arm.solve_ik(targets, branches=(self.ik_branch,))
        joints = joints[:, 0]
        valid = valid[:, 0]
        # 有 IK 无效点且未启用 best_effort: 直接失败。
        if not np.all(valid) and not bool(self.best_effort.get("enabled", False)):
            return None, "ik"
        # 启用 best_effort: 逐点用数值 IK 修复无效点 (不达标即失败), 并逐点 2π 对齐到前一点。
        if not np.all(valid):
            q_ref = np.asarray(start)
            for index in range(len(joints)):
                if not valid[index]:
                    joints[index], metric = self._approximate_ik(targets[index], q_ref)
                    if self._repair_accepted(joints[index], metric) is not None:
                        return None, "repair_tolerance"
                joints[index] = self.arm.joint_space.align_to_reference(
                    joints[index][None, :], q_ref[None, :]
                )[0]
                q_ref = joints[index]
        # 整段连续关节 unwrap 并对齐到 start 附近。
        joints = self.arm.joint_space.align_trajectory(joints, np.asarray(start)[None, :])
        # 碰撞检查: 任一点碰撞即失败, 否则取软间隔惩罚。
        if collision_checker is None:
            collision_penalty = np.zeros(len(joints))
        else:
            collides, collision_penalty = collision_checker.check_configs_with_penalty(
                joints,
                float(config.get("collision_soft_margin", 0.0)),
                float(config.get("collision_soft_weight", 0.0)),
            )
            if np.any(collides):
                return None, "collision"
        # 代价 = motion_weight * (start->首点 + 相邻点) 运动代价 + 碰撞软惩罚 + 限位软惩罚。
        deltas = self.arm.joint_space.delta(joints[1:], joints[:-1])
        start_delta = self.arm.joint_space.delta(joints[:1], np.asarray(start)[None, :])
        motion = float(np.sum(self._motion_cost(np.vstack((start_delta, deltas)))))
        limit = float(
            np.sum(
                self.arm.joint_space.limit_penalty(
                    joints,
                    float(config.get("limit_margin_rad", 0.0)),
                    float(config.get("limit_weight", 0.0)),
                )
            )
        )
        return {
            "states": path.states,
            "joints": joints,
            "cost": float(config.get("motion_weight", 1.0)) * motion
            + float(np.sum(collision_penalty))
            + limit,
        }, "ok"

    # 装配状态 + roll -> TCP 目标位姿 (arm_base 系), 返回 (B, 4, 4)。
    # 兑换站系下姿态 R = Rx(q)·Ry(-p)·Rx(roll): TCP x 轴为插入轴, roll 绕它旋转。
    def _poses(self, states: np.ndarray, rolls: np.ndarray, station: np.ndarray) -> np.ndarray:
        # 拆出各状态分量, 每个形状 (B,)。
        axial, slide, p_angle, q_angle, release = np.asarray(states, dtype=float).T
        # p/q 旋转后的姿态基向量 (未加 roll), 各 (B, 3), 即 Rx(q)·Ry(-p) 的三列。
        cos_q, sin_q = np.cos(q_angle), np.sin(q_angle)
        cos_p, sin_p = np.cos(p_angle), np.sin(p_angle)
        x_axis = np.column_stack((cos_p, -sin_q * sin_p, cos_q * sin_p))
        y_axis = np.column_stack((np.zeros(len(states)), cos_q, sin_q))
        z_axis = np.column_stack((-sin_p, -sin_q * cos_p, cos_q * cos_p))
        # 兑换站系下位置 [m]: 轴向偏移/滑动, 叠加 p 旋转 (半径 r_p, 相位 phi) 与 q 旋转 (半径 r_q) 引起的圆弧位移, p=q=0 时附加项为 0。
        # TODO(注释): r_p/phi 对应的回转中心几何 (为何是 cos(p-phi)-cos(phi) 形式) 需对照兑换站图纸确认。
        local_position = np.column_stack(
            (
                axial + self.p_radius * (np.cos(p_angle - self.p_offset) - np.cos(self.p_offset)),
                -self.q_radius * np.sin(q_angle),
                slide
                + self.p_radius * (np.sin(p_angle - self.p_offset) + np.sin(self.p_offset))
                + self.q_radius * (np.cos(q_angle) - 1.0),
            )
        )
        # 释放: 沿 roll 后 TCP z 轴的反方向后退 release 距离。
        cos_roll, sin_roll = np.cos(rolls), np.sin(rolls)
        local_position += release[:, None] * (
            sin_roll[:, None] * y_axis - cos_roll[:, None] * z_axis
        )
        # 组装局部旋转: 绕 x 轴 (插入轴) 施加 roll, 即右乘 Rx(roll)。
        local_rotation = np.empty((len(states), 3, 3))
        local_rotation[:, :, 0] = x_axis
        local_rotation[:, :, 1] = cos_roll[:, None] * y_axis + sin_roll[:, None] * z_axis
        local_rotation[:, :, 2] = -sin_roll[:, None] * y_axis + cos_roll[:, None] * z_axis
        # 变换到 arm_base 系: T = station · T_local。
        poses = np.broadcast_to(np.eye(4), (len(states), 4, 4)).copy()
        poses[:, :3, :3] = station[:3, :3] @ local_rotation
        poses[:, :3, 3] = local_position @ station[:3, :3].T + station[:3, 3]
        return poses

    # 关节运动代价: L1 + L2 加权, 对最后一维 (6 关节) 求和, 输出形状 delta.shape[:-1]。
    def _motion_cost(self, delta: np.ndarray) -> np.ndarray:
        return self.motion_l1_weight * np.sum(np.abs(delta), axis=-1) + self.motion_l2_weight * np.sum(
            delta * delta, axis=-1
        )

    # 失败说明: 末端层有效节点数、首个全无效层号、后续分支被拒原因统计。
    @staticmethod
    def _failure_message(graph, diagnostics):
        valid_counts = np.count_nonzero(graph.valid, axis=1)
        first_zero = np.flatnonzero(valid_counts == 0)
        first_zero_text = "none" if len(first_zero) == 0 else str(int(first_zero[0]))
        reasons = diagnostics.get("continuation_reject_reasons", {})
        reason_text = ",".join(f"{key}:{value}" for key, value in sorted(reasons.items())) or "none"
        return (
            "type3 planner found no path "
            f"terminal_valid={int(valid_counts[-1])} "
            f"first_zero_step={first_zero_text} "
            f"continuation_reasons={reason_text}"
            + (
                " (every layer has candidates: blocked by bandwidth / max_joint_step)"
                if len(first_zero) == 0
                else ""
            )
        )
