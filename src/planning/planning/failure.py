# failure.py — 规划失败原因与 error_code 映射。
"""规划失败原因: 规划器在失败处直接给出, planning_node 按表映射到结果里的 error_code。

两层:
  - 原因 (FailureReason): 规划器内部的细分, 进 Type3PlanResult.failure_reason / PlanningFailure.reason,
    也写进日志, 给人看具体是哪种失败;
  - error_code: 结果消息里的稳定分类 (6 个值, 见 ApproachResult.msg / AssemblyResult.msg), 给程序用。
message 只给人看, 不参与分类。
"""

from __future__ import annotations

from enum import Enum

# ---- 失败原因 ----
class FailureReason(str, Enum):
    """规划失败原因。 str 混入: 比较 / 格式化时就是右边的值, 日志里可读。"""

    IK_UNREACHABLE = "ik_unreachable"  # 目标位姿无 IK 解 (装配段: 某层所有 roll 都无解)
    REPAIR_REJECTED = "repair_rejected"  # best_ik 近似解超残差 / 限位门槛, 等于够不到
    COLLISION = "collision"  # 有解但全部候选碰撞; 起点连接段 / 稠密轨迹碰撞
    JOINT_JUMP = "joint_jump"  # 修复后相邻路点单轴跳变超 max_joint_step
    NO_PATH = "no_path"  # 每层都有候选, 被带宽 / max_joint_step 挡住, 连不成路
    NO_CONTINUATION = "no_continuation"  # 末端走不完后续分支 (P 段 Q ±90°)
    TIMEOUT = "timeout"  # BIT* 在 bit_timeout_s 内没找到路径, 或超 plan_timeout_s
    TRAJECTORY_INVALID = "trajectory_invalid"  # 最终轨迹校验不过 (非有限 / 时间 / 首末点 / 限位 / 速度)
    START_MISMATCH = "start_mismatch"  # 实测起点与首路点相差超 start_joint_tolerance_rad

    # 3.10 的 (str, Enum) 的 str() 是 "FailureReason.X"; 统一成值, 与 format() 一致
    def __str__(self) -> str:
        return self.value


# ---- error_code (与 msg 里的取值一致; 空串 = 成功) ----
ERROR_INVALID_REQUEST = "INVALID_REQUEST"
ERROR_IK_UNREACHABLE = "IK_UNREACHABLE"
ERROR_COLLISION = "COLLISION"
ERROR_TRAJECTORY_INVALID = "TRAJECTORY_INVALID"
ERROR_PLANNER_FAILED = "PLANNER_FAILED"
ERROR_INTERNAL = "INTERNAL"

#: 原因 -> error_code, 必须覆盖全部 FailureReason (测试守着)。
ERROR_CODE_BY_REASON = {
    FailureReason.IK_UNREACHABLE: ERROR_IK_UNREACHABLE,
    FailureReason.REPAIR_REJECTED: ERROR_IK_UNREACHABLE,
    FailureReason.COLLISION: ERROR_COLLISION,
    FailureReason.JOINT_JUMP: ERROR_PLANNER_FAILED,
    FailureReason.NO_PATH: ERROR_PLANNER_FAILED,
    FailureReason.NO_CONTINUATION: ERROR_PLANNER_FAILED,
    FailureReason.TIMEOUT: ERROR_PLANNER_FAILED,
    FailureReason.TRAJECTORY_INVALID: ERROR_TRAJECTORY_INVALID,
    FailureReason.START_MISMATCH: ERROR_INVALID_REQUEST,
}


class PlanningFailure(ValueError):
    """带失败原因的规划异常。

    继承 ValueError: validate_trajectory 以前就抛 ValueError, 调用方的 except ValueError 照常生效。
    """

    def __init__(self, reason: FailureReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def error_code_for_reason(reason: FailureReason | None) -> str:
    """失败原因 -> error_code; 没带原因 (None) 的失败是代码 bug, 归 INTERNAL。"""
    if reason is None:
        return ERROR_INTERNAL
    return ERROR_CODE_BY_REASON[reason]


def error_code_for_exception(exc: Exception) -> str:
    """规划异常 -> error_code。

    PlanningFailure 按原因映射; 其余 ValueError 是请求 / 输入校验失败 (frame_id、关节名、四元数、
    path_name、超限关节角), 归 INVALID_REQUEST; 其它异常都是代码 bug, 归 INTERNAL。
    """
    if isinstance(exc, PlanningFailure):
        return error_code_for_reason(exc.reason)
    if isinstance(exc, ValueError):
        return ERROR_INVALID_REQUEST
    return ERROR_INTERNAL
