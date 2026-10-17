# joint_space.py — 关节空间 (限位与角度处理)。


from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# 把角度折回 [-pi, pi)。
def _wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


# 关节空间: 每关节上下限 (rad) + 连续关节掩码 (无限位, 按 2pi 周期处理), 各数组形状 (dof,)。
@dataclass(frozen=True, slots=True)
class JointSpace:
    lower: np.ndarray
    upper: np.ndarray
    continuous: np.ndarray
    dof: int = field(init=False)

    # 统一转成 ndarray 并校验形状与限位合法性。
    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        continuous = np.asarray(self.continuous, dtype=bool)
        # 校验: 三者一维等长; 有限位关节上下限必须有限且 lower <= upper。
        if lower.ndim != 1 or upper.shape != lower.shape or continuous.shape != lower.shape:
            raise ValueError("joint limits and continuous mask must be one-dimensional and equally sized")
        if np.any(~continuous & (~np.isfinite(lower) | ~np.isfinite(upper))):
            raise ValueError("bounded joints require finite lower and upper limits")
        if np.any(~continuous & (lower > upper)):
            raise ValueError("joint lower limits must not exceed upper limits")
        # frozen dataclass 只能用 object.__setattr__ 回写规范化后的数组, 并派生 dof。
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "continuous", continuous)
        object.__setattr__(self, "dof", len(lower))

    # 从配置构造: lower/upper 同为 None 的关节视为连续关节, 限位记为 ±inf。
    @classmethod
    def from_config(cls, config: dict) -> "JointSpace":
        lower_raw = tuple(config["lower"])
        upper_raw = tuple(config["upper"])
        if len(lower_raw) != len(upper_raw):
            raise ValueError("joint lower and upper limits must have equal length")
        # 连续关节掩码 + None 替换为 ±inf。
        continuous = np.asarray(
            [lower is None and upper is None for lower, upper in zip(lower_raw, upper_raw, strict=True)]
        )
        lower = np.asarray([-np.inf if value is None else float(value) for value in lower_raw])
        upper = np.asarray([np.inf if value is None else float(value) for value in upper_raw])
        return cls(lower=lower, upper=upper, continuous=continuous)

    # 有限位关节掩码 (continuous 取反)。
    @property
    def bounded(self) -> np.ndarray:
        return ~self.continuous

    # 校验关节数组形状为 (..., B, dof) 且至少二维, 返回 float 数组。
    def _validate(self, joints: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(joints, dtype=float)
        if values.ndim < 2 or values.shape[-1] != self.dof:
            raise ValueError(f"{name} must have shape (..., B, {self.dof}), got {values.shape}")
        return values

    # 关节差 target - source, 形状 (..., B, dof); 连续关节取最短角距 (折回 [-pi, pi))。
    def delta(self, target: np.ndarray, source: np.ndarray) -> np.ndarray:
        target_values = self._validate(target, "target")
        source_values = self._validate(source, "source")
        delta = np.asarray(target_values - source_values, dtype=float)
        delta[..., self.continuous] = _wrap_pi(delta[..., self.continuous])
        return delta

    # 连续关节逐点加 2pi 整数倍, 使其最接近 reference 对应值; 有限位关节不动。
    def align_to_reference(self, joints: np.ndarray, reference: np.ndarray) -> np.ndarray:
        aligned = self._validate(joints, "joints").copy()
        reference_values = self._validate(reference, "reference")
        aligned[..., self.continuous] += 2.0 * np.pi * np.round(
            (reference_values[..., self.continuous] - aligned[..., self.continuous]) / (2.0 * np.pi)
        )
        return aligned

    # 对 (B, dof) 轨迹: 连续关节先沿时间 unwrap 去掉 2pi 跳变,
    # 再整体平移 2pi 整数倍让首点贴近 reference (1, dof); 有限位关节不动。
    def align_trajectory(self, trajectory: np.ndarray, reference: np.ndarray) -> np.ndarray:
        aligned = self._validate(trajectory, "trajectory").copy()
        # 形状校验: trajectory 必须 (B, dof), reference 必须 (1, dof)。
        if aligned.ndim != 2:
            raise ValueError(f"trajectory must have shape (B, {self.dof}), got {aligned.shape}")
        reference_values = self._validate(reference, "reference")
        if reference_values.shape != (1, self.dof):
            raise ValueError(f"reference must have shape (1, {self.dof}), got {reference_values.shape}")
        for index in np.flatnonzero(self.continuous):
            aligned[:, index] = np.unwrap(aligned[:, index])
            aligned[:, index] += 2.0 * np.pi * np.round(
                (reference_values[0, index] - aligned[0, index]) / (2.0 * np.pi)
            )
        return aligned

    # 连续关节折回 [-pi, pi), 有限位关节不变。
    def wrap(self, joints: np.ndarray) -> np.ndarray:
        wrapped = self._validate(joints, "joints").copy()
        wrapped[..., self.continuous] = _wrap_pi(wrapped[..., self.continuous])
        return wrapped

    # 有限位关节: 加 2pi 整数倍把角度移入 [lower, upper], 多个可选时取最靠近区间中点的;
    # 返回 (规范化后的关节, valid 掩码 (..., B): 该点所有有限位关节均能落入限位)。
    def normalize(self, joints: np.ndarray, *, tolerance: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
        values = self._validate(joints, "joints")
        # 展平成 (N, dof) 逐关节处理; valid 初始全 True。
        flat = values.reshape(-1, self.dof)
        normalized = flat.copy()
        valid = np.ones(len(flat), dtype=bool)
        period = 2.0 * np.pi
        for index in np.flatnonzero(self.bounded):
            current = normalized[:, index]
            lower = self.lower[index]
            upper = self.upper[index]
            # 可行圈数范围 [minimum_turns, maximum_turns] (含 tolerance); 范围为空说明该关节无法落入限位。
            minimum_turns = np.ceil((lower - current - tolerance) / period).astype(np.int64)
            maximum_turns = np.floor((upper - current + tolerance) / period).astype(np.int64)
            feasible = minimum_turns <= maximum_turns
            valid &= feasible
            # 取最接近区间中点的圈数并夹到可行范围; 不可行的点保留原值 (由 valid 标记)。
            center_turns = np.rint((0.5 * (lower + upper) - current) / period).astype(np.int64)
            turns = np.clip(center_turns, minimum_turns, maximum_turns)
            normalized[:, index] = np.where(feasible, current + period * turns, current)
        return normalized.reshape(values.shape), valid.reshape(values.shape[:-1])

    # 软限位惩罚: 距最近限位小于 margin (rad) 时计 weight * Σ(margin - distance)^2, 返回形状 (..., B)。
    def limit_penalty(self, joints: np.ndarray, margin: float, weight: float) -> np.ndarray:
        values = self._validate(joints, "joints")
        # margin/weight 非正或没有有限位关节时惩罚恒为 0。
        if margin <= 0.0 or weight <= 0.0 or not np.any(self.bounded):
            return np.zeros(values.shape[:-1], dtype=float)
        # 到上下限的最小距离 (越限时为负, 惩罚随之增大)。
        distance = np.minimum(
            values[..., self.bounded] - self.lower[self.bounded],
            self.upper[self.bounded] - values[..., self.bounded],
        )
        violation = np.maximum(0.0, float(margin) - distance)
        return float(weight) * np.sum(violation * violation, axis=-1)
