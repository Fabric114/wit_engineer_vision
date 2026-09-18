"""读工作区根 config/shm.yaml, 缺失时回退到内置默认 —— 让区域名/最大分辨率/槽数走配置。

配置统一集中在工作区根 config/ (wit_engineer_vision/config/), 各包不再各装一份到 share。
image_transport 本身不依赖它, 这里只是给节点一个统一的配置入口, 保持仓库"数据驱动"的约定。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import layout


@dataclass
class ShmConfig:
    region: str = layout.DEFAULT_REGION
    max_height: int = 720
    max_width: int = 960
    max_channels: int = 3
    n_slots: int = 3
    dtype: str = "uint8"

    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)


def _config_dir() -> Path:
    """统一读工作区根 config/ (wit_engineer_vision/config/)。

    向上查找同时含 config/ 与 src/ 的工作区根; 源码运行与拷贝安装下都能命中。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent / "config"
    return here.parents[3] / "config"


def load_shm_config(path: Path | None = None) -> ShmConfig:
    """加载配置; 任何缺失字段都回退到 ShmConfig 的默认值。"""
    cfg = ShmConfig()
    path = path or (_config_dir() / "shm.yaml")
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = (yaml.safe_load(fh) or {}).get("image", {})
        cfg.region = str(data.get("region", cfg.region))
        cfg.max_height = int(data.get("max_height", cfg.max_height))
        cfg.max_width = int(data.get("max_width", cfg.max_width))
        cfg.max_channels = int(data.get("max_channels", cfg.max_channels))
        cfg.n_slots = int(data.get("n_slots", cfg.n_slots))
        cfg.dtype = str(data.get("dtype", cfg.dtype))
    except FileNotFoundError:
        pass
    return cfg
