# trajectory.py — 轨迹时间参数化 (把路径变成轨迹)。
"""trajectory.py — 轨迹时间参数化 (把"路径"变成"轨迹")。

【这个文件干什么】
规划器 (type2/type3) 输出的只是一串"路点" (关节角序列), 是纯几何的路径,
不带时间, 也没有速度/加速度。 这个文件把这串路点在给定总时长内用五次多项式
(quintic) 平滑插值, 产出每个时刻的 位置/速度/加速度/时间戳, 并保证起点和
终点的速度、加速度都为 0 (平顺起停)。
插值是保形的: 每个采样点每个关节都落在所在段两端路点之间 (极值处停住、过冲段置零导数),
所以路点在限位内则轨迹在限位内。 发布前还要过 validate_trajectory (有限值/首末点/限位,
给了速度/加速度上限才查上限 —— config 目前没有真实上限)。

【要懂的概念】
- "路径 (path)" = 只有形状, 一串点; "轨迹 (trajectory)" = 路径 + 时间,
  即"几点几分到哪、速度多少"。 机器人执行需要的是轨迹。
- FixedDurationParameterizer: 固定总时长的参数化器, 是这里的主入口。
这个文件是通用数学, 换臂不用改 (只是配置里可调总时长/采样步长)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 判定"越出路点区间"时容忍的浮点舍入 (rad)
_OVERSHOOT_TOLERANCE = 1e-12


# 稠密轨迹: positions/velocities/accelerations 形状 (B, dof), 单位 rad, rad/s, rad/s^2; timestamps (B,) 单位 s。
@dataclass(frozen=True, slots=True)
class JointTrajectory:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    timestamps: np.ndarray

    def __post_init__(self) -> None:
        # 转 float 数组并校验三者同形、时间戳与点数一致且严格递增; frozen 需用 object.__setattr__ 回写。
        positions = np.asarray(self.positions, dtype=float)
        velocities = np.asarray(self.velocities, dtype=float)
        accelerations = np.asarray(self.accelerations, dtype=float)
        timestamps = np.asarray(self.timestamps, dtype=float)
        if positions.ndim != 2:
            raise ValueError(f"positions must have shape (B, dof), got {positions.shape}")
        if velocities.shape != positions.shape or accelerations.shape != positions.shape:
            raise ValueError("trajectory positions, velocities, and accelerations must have identical shapes")
        if timestamps.shape != (len(positions),) or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("timestamps must be strictly increasing with one value per trajectory point")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "velocities", velocities)
        object.__setattr__(self, "accelerations", accelerations)
        object.__setattr__(self, "timestamps", timestamps)


# 固定总时长参数化器: 路点在 [0, duration] 上等时间间隔分布, 再做五次插值采样。
class FixedDurationParameterizer:
    # duration_s 可在构造时给默认值。
    # TODO(注释): **_ 静默吞掉其余参数, 推测是为兼容其他参数化器的统一构造接口, 待确认。
    def __init__(self, duration_s: float | None = None, **_: object) -> None:
        self.duration_s = None if duration_s is None else float(duration_s)

    # 把 (B, dof) 路点参数化为轨迹; 调用时传的 duration_s 优先于构造时的默认值。
    def parameterize(
        self,
        waypoints: np.ndarray,
        *,
        sample_dt: float,
        duration_s: float | None = None,
        **_: object,
    ) -> JointTrajectory:
        # 确定总时长并校验为正; 路点至少 2 个。
        duration = self.duration_s if duration_s is None else float(duration_s)
        if duration is None or duration <= 0.0:
            raise ValueError("duration_s must be positive")
        points = np.asarray(waypoints, dtype=float)
        if points.ndim != 2 or len(points) < 2:
            raise ValueError(f"waypoints must have shape (B, dof) with B >= 2, got {points.shape}")
        # 路点时刻均匀分布 (不按各段关节位移分配时间), 再交给五次插值采样。
        return sample_quintic_trajectory(
            points,
            np.linspace(0.0, duration, len(points)),
            sample_dt,
        )


# 分段五次多项式插值: 给定路点及其时刻, 按 sample_dt 采样, 输出保形的稠密轨迹。
def sample_quintic_trajectory(
    waypoints: np.ndarray,
    waypoint_times: np.ndarray,
    sample_dt: float,
) -> JointTrajectory:
    # 输入校验: 路点 (B, dof) 且 B >= 2, 时间严格递增, 采样步长为正。
    positions = np.asarray(waypoints, dtype=float)
    times = np.asarray(waypoint_times, dtype=float)
    dt = float(sample_dt)
    if positions.ndim != 2 or len(positions) < 2:
        raise ValueError(f"waypoints must have shape (B, dof) with B >= 2, got {positions.shape}")
    if times.shape != (len(positions),) or np.any(np.diff(times) <= 0.0):
        raise ValueError("waypoint_times must be strictly increasing with one value per waypoint")
    if dt <= 0.0:
        raise ValueError("sample_dt must be positive")

    # 估计各路点速度/加速度 (首末为 0), 生成采样时刻, 并定位每个采样时刻所在段 (0..B-2)。
    waypoint_velocities, waypoint_accelerations = _estimate_derivatives(positions, times)
    sample_times = _sample_times(times, dt)
    segments = np.clip(np.searchsorted(times, sample_times, side="right") - 1, 0, len(positions) - 2)
    # 每个采样点所在段两端路点的逐关节区间 [low, high], 用于判定过冲。
    low = np.minimum(positions[segments], positions[segments + 1])
    high = np.maximum(positions[segments], positions[segments + 1])
    # 保形: 某段某关节的采样越出两端路点区间 (五次曲线过冲) 时, 把该段两端该关节的
    # 速度/加速度置零再算。 全零时每段是单调的最小 jerk 曲线, 所以循环必然结束。
    while True:
        sampled = _evaluate_quintic(
            positions, times, waypoint_velocities, waypoint_accelerations, segments, sample_times
        )
        # 找出过冲的 (采样点, 关节), 其所在段两端路点即需要清零导数的位置。
        outside = (sampled[0] < low - _OVERSHOOT_TOLERANCE) | (sampled[0] > high + _OVERSHOOT_TOLERANCE)
        rows, joints = np.nonzero(outside)
        ends = np.concatenate((segments[rows], segments[rows] + 1))
        joints = np.concatenate((joints, joints))
        if not np.any(waypoint_velocities[ends, joints]) and not np.any(waypoint_accelerations[ends, joints]):
            break  # 无过冲, 或只剩全零导数下的舍入误差
        waypoint_velocities[ends, joints] = 0.0
        waypoint_accelerations[ends, joints] = 0.0
    # 末点强制等于最后路点 (消除浮点误差), 末端速度/加速度取路点值 (即 0)。
    sampled_positions, sampled_velocities, sampled_accelerations = sampled
    sampled_positions[-1] = positions[-1]
    sampled_velocities[-1] = waypoint_velocities[-1]
    sampled_accelerations[-1] = waypoint_accelerations[-1]
    return JointTrajectory(sampled_positions, sampled_velocities, sampled_accelerations, sample_times)


# 批量求值五次多项式: 每段用两端 (位置, 速度, 加速度) 共 6 个边界条件定系数,
# 在段内归一化时间 u∈[0,1] 上求值; 返回 (位置, 速度, 加速度), 形状均为 (S, dof)。
def _evaluate_quintic(positions, times, waypoint_velocities, waypoint_accelerations, segments, sample_times):
    # 每个采样点的段起点时刻、段时长 h 与段内归一化进度 u。
    start_times = times[segments]
    durations = times[segments + 1] - start_times
    progress = (sample_times - start_times) / durations
    # 段两端的边界条件。
    q0 = positions[segments]
    q1 = positions[segments + 1]
    v0 = waypoint_velocities[segments]
    v1 = waypoint_velocities[segments + 1]
    a0 = waypoint_accelerations[segments]
    a1 = waypoint_accelerations[segments + 1]
    # 归一化后 q(u)=Σ c_k u^k: c0..c2 由起点条件直接得到, c3..c5 由终点位置/速度/加速度残差解出。
    h = durations[:, None]
    c0 = q0
    c1 = h * v0
    c2 = 0.5 * h**2 * a0
    displacement = q1 - (c0 + c1 + c2)
    velocity_delta = h * v1 - (c1 + 2.0 * c2)
    acceleration_delta = h**2 * a1 - 2.0 * c2
    c5 = 0.5 * (acceleration_delta + 12.0 * displacement - 6.0 * velocity_delta)
    c4 = velocity_delta - 3.0 * displacement - 2.0 * c5
    c3 = displacement - c4 - c5
    # 求值; 对 u 的导数除以 h / h^2 还原为对时间的速度/加速度。
    u = progress[:, None]
    sampled_positions = c0 + c1 * u + c2 * u**2 + c3 * u**3 + c4 * u**4 + c5 * u**5
    sampled_velocities = (
        c1 + 2.0 * c2 * u + 3.0 * c3 * u**2 + 4.0 * c4 * u**3 + 5.0 * c5 * u**4
    ) / h
    sampled_accelerations = (
        2.0 * c2 + 6.0 * c3 * u + 12.0 * c4 * u**2 + 20.0 * c5 * u**3
    ) / h**2
    return sampled_positions, sampled_velocities, sampled_accelerations


# 估计内部路点的速度/加速度 (非均匀时间三点差分); 首末路点保持 0, 保证平顺起停。
def _estimate_derivatives(positions: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocities = np.zeros_like(positions)
    accelerations = np.zeros_like(positions)
    if len(positions) > 2:
        # 速度: 中心差分; 加速度: 前后割线斜率之差除以半跨度 (二阶差分)。
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (
            timestamps[2:] - timestamps[:-2]
        )[:, None]
        accelerations[1:-1] = 2.0 * (
            (positions[2:] - positions[1:-1]) / (timestamps[2:] - timestamps[1:-1])[:, None]
            - (positions[1:-1] - positions[:-2]) / (timestamps[1:-1] - timestamps[:-2])[:, None]
        ) / (timestamps[2:] - timestamps[:-2])[:, None]
        # 局部极值/平台 (两侧割线异号或其一为零) 处停住, 否则中心差分会冲过路点
        before = positions[1:-1] - positions[:-2]
        after = positions[2:] - positions[1:-1]
        turning = before * after <= 0.0
        velocities[1:-1][turning] = 0.0
        accelerations[1:-1][turning] = 0.0
    return velocities, accelerations


def validate_trajectory(
    trajectory: JointTrajectory,
    joint_space,
    *,
    waypoints: np.ndarray,
    velocity_limits: np.ndarray | None = None,
    acceleration_limits: np.ndarray | None = None,
    collision_checker=None,
    position_tolerance: float = 1e-9,
) -> None:
    """发布前的最终轨迹校验, 不通过抛 ValueError。

    查: 有限值、时间戳严格递增、首末点等于首末路点、有限位关节不越限;
    速度/加速度上限只在调用方给出时检查 (config 目前没有真实上限, 不在这里编默认值);
    给了 collision_checker 就对每个稠密采样点查碰撞 (离散采样, 不是连续碰撞检测)。
    """
    # 维度校验: 轨迹与路点关节数须等于 joint_space.dof。
    positions = trajectory.positions
    points = np.asarray(waypoints, dtype=float)
    if positions.shape[1] != joint_space.dof or points.ndim != 2 or points.shape[1] != joint_space.dof:
        raise ValueError(f"trajectory/waypoints must have {joint_space.dof} joints")
    # 有限值与时间戳单调性。
    values = (positions, trajectory.velocities, trajectory.accelerations, trajectory.timestamps)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("trajectory contains non-finite values")
    if np.any(np.diff(trajectory.timestamps) <= 0.0):
        raise ValueError("trajectory timestamps must be strictly increasing")
    # 首末点须与首末路点一致 (容差 position_tolerance)。
    if np.max(np.abs(positions[0] - points[0])) > position_tolerance:
        raise ValueError("trajectory first point does not match the first waypoint")
    if np.max(np.abs(positions[-1] - points[-1])) > position_tolerance:
        raise ValueError("trajectory last point does not match the last waypoint")
    # 有限位关节不得越限 (连续关节不查)。
    bounded = joint_space.bounded
    outside = (positions[:, bounded] < joint_space.lower[bounded] - position_tolerance) | (
        positions[:, bounded] > joint_space.upper[bounded] + position_tolerance
    )
    if np.any(outside):
        raise ValueError(f"trajectory leaves joint limits at {int(np.count_nonzero(outside.any(axis=1)))} samples")
    # 可选: 逐关节峰值速度/加速度与上限比较。
    for name, data, limits in (
        ("velocity", trajectory.velocities, velocity_limits),
        ("acceleration", trajectory.accelerations, acceleration_limits),
    ):
        if limits is None:
            continue
        peak = np.max(np.abs(data), axis=0)
        exceeded = peak > np.asarray(limits, dtype=float)
        if np.any(exceeded):
            raise ValueError(
                f"trajectory {name} exceeds limit on joints {np.flatnonzero(exceeded).tolist()}: "
                f"peak={np.round(peak, 4).tolist()}"
            )
    # 可选: 对每个稠密采样点做离散碰撞检查, 报告首个碰撞时刻。
    if collision_checker is not None:
        collides = np.asarray(collision_checker.check_configs(positions), dtype=bool)
        if np.any(collides):
            first = int(np.flatnonzero(collides)[0])
            raise ValueError(
                f"trajectory collides at {int(np.count_nonzero(collides))} samples, "
                f"first t={float(trajectory.timestamps[first]):.3f}s"
            )


# 生成采样时刻: 从起点按 sample_dt 等步长, 保证包含起点和终点 (最后一步可能短于 sample_dt)。
def _sample_times(waypoint_times: np.ndarray, sample_dt: float) -> np.ndarray:
    start = float(waypoint_times[0])
    end = float(waypoint_times[-1])
    samples = np.arange(start, end, sample_dt, dtype=float)
    if samples.size == 0 or samples[0] != start:
        samples = np.concatenate(([start], samples))
    if samples[-1] != end:
        samples = np.concatenate((samples, [end]))
    return samples
