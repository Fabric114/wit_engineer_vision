# viterbi.py — 分层图最短路 (Viterbi 动态规划, 给 type3 选解用)。
"""viterbi.py — 分层图最短路 (动态规划, 给 type3 选最优姿态序列用)。

【这个文件干什么】
type3 规划时, 路径上的每一个采样点都有一批候选姿态 (同一个末端位置, 绕
插入轴转不同角度 roll, 各自解出不同的关节配置)。 需要"每层挑一个候选、
让相邻层之间过渡尽量平滑、整条序列总代价最小"。 这正是经典的
"分层图最短路", 用 Viterbi 动态规划求解 —— 一层层往前推, 记录到每个候选
的最小累计代价和来路, 最后回溯出最优路径。

【要懂的概念】
- 想象一张从左到右的网格: 每一列 (层) 是一个路径采样点, 列里的每个格子
  (节点) 是一个候选姿态。 目标: 从左走到右, 每列过一个格子, 走一条总花费
  最小的路。 相邻格子换得越猛 (关节动得越多) 代价越高。
- 这是纯算法, 只依赖 numpy, 不含任何机器人参数, 换臂不用改。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


# 层间转移回调: (层号, 上一层累计代价 (N,)) -> (本层各节点最小累计代价 (N,), 对应的上一层前驱索引 (N,));
# 不可达节点代价为 inf。
TransitionStep = Callable[[int, np.ndarray], tuple[np.ndarray, np.ndarray]]


# Viterbi 结果: 每层选中的节点索引 (失败时为空)、最优总代价、每层可达节点数 (诊断在哪一层断开)。
@dataclass(frozen=True, slots=True)
class ViterbiResult:
    path_indices: tuple[int, ...]
    cost: float
    reachable_counts: tuple[int, ...]

    # 有路径即成功。
    @property
    def success(self) -> bool:
        return bool(self.path_indices)


def solve_viterbi(
    initial_costs: np.ndarray,
    num_layers: int,
    transition_step: TransitionStep,
) -> ViterbiResult:
    """Find the minimum-cost path through an equal-width layered graph."""
    # 输入校验: 层数为正, 初始代价为非空一维向量 (inf 表示不可达)。
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    previous_costs = np.asarray(initial_costs, dtype=float)
    if previous_costs.ndim != 1 or previous_costs.size == 0:
        raise ValueError("initial_costs must be a non-empty vector")

    # 每层节点数固定为 state_count; 第 0 层无可达节点则直接失败。
    state_count = previous_costs.size
    reachable_counts = [int(np.count_nonzero(np.isfinite(previous_costs)))]
    if reachable_counts[0] == 0:
        return ViterbiResult((), float("inf"), tuple(reachable_counts))

    # 前向递推: 逐层调用 transition_step, 保存每层的前驱表。
    predecessors: list[np.ndarray] = []
    for layer_index in range(1, num_layers):
        # 由回调算出本层代价与前驱, 并校验形状为 (state_count,)。
        next_costs, previous_indices = transition_step(layer_index, previous_costs)
        next_costs = np.asarray(next_costs, dtype=float)
        previous_indices = np.asarray(previous_indices, dtype=np.int64)
        if next_costs.shape != (state_count,) or previous_indices.shape != (state_count,):
            raise ValueError(
                f"layer {layer_index} must return two ({state_count},) vectors"
            )

        # 可达节点的前驱索引必须落在 [0, state_count)。
        reachable = np.isfinite(next_costs)
        invalid = reachable & (
            (previous_indices < 0) | (previous_indices >= state_count)
        )
        if np.any(invalid):
            raise ValueError(
                f"layer {layer_index} contains a reachable state with an invalid predecessor"
            )

        # 不可达节点前驱记为 -1 后保存; 本层全不可达则提前失败。
        previous_indices = previous_indices.copy()
        previous_indices[~reachable] = -1
        predecessors.append(previous_indices)
        previous_costs = next_costs
        reachable_counts.append(int(np.count_nonzero(reachable)))
        if reachable_counts[-1] == 0:
            return ViterbiResult((), float("inf"), tuple(reachable_counts))

    # 回溯: 从最后一层代价最小的节点沿前驱表倒推, 再反转得到正序路径。
    current = int(np.argmin(previous_costs))
    path = [current]
    for previous_indices in reversed(predecessors):
        current = int(previous_indices[current])
        path.append(current)
    path.reverse()
    return ViterbiResult(
        tuple(path),
        float(np.min(previous_costs)),
        tuple(reachable_counts),
    )
