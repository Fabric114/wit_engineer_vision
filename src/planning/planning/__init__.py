# __init__.py — planning 包入口: 导出公共 API, 提供规划配置加载。
from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

# 规划失败原因与 error_code 映射
from . import failure
# 运动学模型 (DH / FK / IK)
from .arm_model import ArmModel
# 关节空间 (限位与角度处理)
from .joint_space import JointSpace
# 轨迹时间参数化与发布前校验
from .trajectory import (
    FixedDurationParameterizer,
    JointTrajectory,
    sample_quintic_trajectory,
    validate_trajectory,
)
# 齐次变换 / 四元数工具
from .transform import (
    quaternions_from_rotations,
    rotations_from_quaternions,
    validate_transforms,
)
# type3 装配段规划
from .type3 import (
    AssemblyPath,
    AssemblyState,
    Type3PlanResult,
    Type3Planner,
)
# 分层图 Viterbi 选解
from .viterbi import ViterbiResult, solve_viterbi

# 公共导出列表; collision (hpp-fcl) 与 type2 (OMPL) 依赖可选系统库, 不在包入口导入。
__all__ = [
    "load_config",
    "failure",
    "ArmModel",
    "JointSpace",
    "FixedDurationParameterizer",
    "JointTrajectory",
    "sample_quintic_trajectory",
    "validate_trajectory",
    "validate_transforms",
    "rotations_from_quaternions",
    "quaternions_from_rotations",
    "AssemblyPath",
    "AssemblyState",
    "Type3PlanResult",
    "Type3Planner",
    "ViterbiResult",
    "solve_viterbi",
]


# 进程内工作区根不变, 缓存结果只查一次。
@lru_cache(maxsize=1)
def _workspace_root() -> Path | None:
    """向上查找同时含 ``config/`` 与 ``src/`` 的工作区根; 找不到返回 None。"""
    here = Path(__file__).resolve()
    return next(
        (p for p in here.parents if (p / "config").is_dir() and (p / "src").is_dir()),
        None,
    )


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取规划配置。

    默认读**工作区根** ``config/planning.yaml`` (即 wit_engineer_vision/config/),
    与 camera/detector 等包一致 —— 全仓库配置集中一处, 一处一个真值。 向上查找同时含
    ``config/`` 与 ``src/`` 的工作区根 (源码运行/拷贝安装都能命中)。 包内没有兜底副本
    (``planning/config/planning.yaml`` 不存在): 脱离源码树时必须显式传 ``path``,
    否则在下面显式 ``FileNotFoundError``，不会静默用旧配置。
    """
    # 显式路径优先; 否则用工作区根 config/planning.yaml; 包内没有兜底副本, 缺失即显式报错。
    if path is not None:
        config_path = Path(path)
    else:
        root = _workspace_root()
        workspace_config = root / "config" / "planning.yaml" if root is not None else None
        config_path = (
            workspace_config
            if workspace_config is not None and workspace_config.is_file()
            else Path(str(files(__name__) / "config" / "planning.yaml"))
        )
    # 校验文件存在, safe_load 读取, 顶层必须是映射。
    if not config_path.is_file():
        raise FileNotFoundError(f"planning 配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"planning 配置必须是映射: {config_path}")
    return config
