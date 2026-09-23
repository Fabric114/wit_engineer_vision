"""planning: 运动规划纯算法库 (零 ROS 依赖)。

移植自 RM2026-Engineer-Assembly-Algorithm 的 arm_exchange_core:
  运动学   transform / joint_space / arm_model / trajectory
  规划     viterbi / collision / type2 (OMPL BIT*) / type3 (装配流形)

设计上运动学本应落在 task 包 (见 README_PRE.md), 当前仓库暂无 task 包,
为让 planning 自洽可跑, 先随规划一并落在这里, 后续可整体切出到 task。

用法:
    from planning import load_config, ArmModel, Type3Planner
    cfg = load_config()                 # 读包内 config/planning.example.yaml
    arm = ArmModel.from_config(cfg["arm"])
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from .arm_model import ArmModel
from .joint_space import JointSpace
from .trajectory import (
    FixedDurationParameterizer,
    JointTrajectory,
    sample_quintic_trajectory,
)
from .transform import (
    quaternions_from_rotations,
    rotations_from_quaternions,
    validate_transforms,
)
from .type3 import (
    AssemblyPath,
    AssemblyState,
    Type3PlanResult,
    Type3Planner,
)
from .viterbi import ViterbiResult, solve_viterbi

__all__ = [
    "load_config",
    "ArmModel",
    "JointSpace",
    "FixedDurationParameterizer",
    "JointTrajectory",
    "sample_quintic_trajectory",
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


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读取规划配置。默认用包内 config/planning.example.yaml, 可传入外部 YAML 覆盖。"""
    config_path = (
        Path(path)
        if path is not None
        else Path(str(files(__name__) / "config" / "planning.example.yaml"))
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"planning 配置文件不存在: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"planning 配置必须是映射: {config_path}")
    return config
