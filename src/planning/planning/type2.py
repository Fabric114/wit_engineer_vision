"""type2.py — 接近段规划: 点到点无碰撞关节路径 (OMPL BIT*)。

【这个文件干什么】
给"起始关节角"和"目标关节角", 在 6 维关节空间里搜索一条从起点到终点、
不越限、不碰撞的路径, 然后简化成少量稀疏路点返回。 用在"把臂从当前任意
位置移动到兑换站前方的准备位姿"这一大段自由移动。

【要懂的概念】
- 这是"采样式运动规划": 在关节空间里随机撒点、连边、找连通路径。
  BIT* (Batch Informed Trees) 是 OMPL 库里一种又快又优的这类算法。
- 和 type3 的区别: type2 是"从 A 自由地走到 B"(接近段); type3 是"贴着
  装配约束一点点插进去"(装配段)。 两段用不同算法。
- validity_fn 是外部传入的"这个关节配置合不合法"判据 (通常接碰撞检测);
  不想开碰撞就传一个恒返回 True 的函数。

【依赖】 OMPL (系统级可选后端)。 没装 OMPL 时本文件不可用, 其余文件不受影响。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def plan_joint_path_bitstar(
    start_q,
    goal_q,
    *,
    validity_fn: Callable[[np.ndarray], bool],
    joint_lower: Sequence[float],
    joint_upper: Sequence[float],
    timeout_s: float,
) -> np.ndarray:
    """
    Minimal OMPL BIT* point-to-point planner in 6D joint space.

    Returns sparse joint waypoints including start and goal. The caller owns all
    task semantics, smoothing, time parameterization, and diagnostics.
    """

    import ompl.base as ob
    import ompl.geometric as og

    start_q = np.asarray(start_q, dtype=float)
    goal_q = np.asarray(goal_q, dtype=float)
    lower = np.asarray(joint_lower, dtype=float)
    upper = np.asarray(joint_upper, dtype=float)
    if start_q.shape != goal_q.shape:
        raise ValueError(f"start_q and goal_q shape mismatch: {start_q.shape} vs {goal_q.shape}")
    if lower.shape != start_q.shape or upper.shape != start_q.shape:
        raise ValueError("joint bounds must match joint vector shape")
    if timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive")

    dof = int(start_q.shape[0])
    space = ob.RealVectorStateSpace(dof)
    bounds = ob.RealVectorBounds(dof)
    for i in range(dof):
        bounds.setLow(i, float(lower[i]))
        bounds.setHigh(i, float(upper[i]))
    space.setBounds(bounds)

    si = ob.SpaceInformation(space)

    class ValidityChecker(ob.StateValidityChecker):
        def __init__(self, space_information):
            super().__init__(space_information)

        def isValid(self, state) -> bool:
            q = np.asarray([state[i] for i in range(dof)], dtype=float)
            return bool(validity_fn(q))

    si.setStateValidityChecker(ValidityChecker(si))
    si.setStateValidityCheckingResolution(0.003)
    si.setup()

    start_state = space.allocState()
    goal_state = space.allocState()
    for i in range(dof):
        start_state[i] = float(start_q[i])
        goal_state[i] = float(goal_q[i])

    problem = ob.ProblemDefinition(si)
    problem.setStartAndGoalStates(start_state, goal_state)
    problem.setOptimizationObjective(ob.PathLengthOptimizationObjective(si))

    planner = og.BITstar(si)
    planner.setProblemDefinition(problem)
    planner.setup()

    solved = planner.solve(float(timeout_s))
    if not solved:
        raise RuntimeError("OMPL BIT* failed to find a Type II path")

    path = problem.getSolutionPath()
    simplifier = og.PathSimplifier(si)
    simplifier.reduceVertices(path, maxSteps=200)
    simplifier.collapseCloseVertices(path)

    waypoints = np.asarray(
        [[path.getState(i)[j] for j in range(dof)] for i in range(path.getStateCount())],
        dtype=float,
    )
    if waypoints.ndim != 2 or waypoints.shape[0] < 2:
        raise RuntimeError(f"OMPL returned invalid path shape {waypoints.shape}")
    return waypoints
