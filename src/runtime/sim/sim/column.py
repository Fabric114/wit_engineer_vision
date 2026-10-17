"""科技核心 (exchange_column) 跟随 TCP 的纯逻辑 (零 ROS / 零 MuJoCo, 可单测)。

sim 里站体与科技核心都不参与接触, 矿石推不动科技核心。 这里用运动学近似代替接触:
矿石插在科技核心里 (TCP 在站体系下离插入轴足够近) 时, 科技核心的 step3_slide
直接取 TCP 的站体 z (= planning type3 的 slide 状态), 拔出来以后停在原位。

带滞回: 进入用 engage_radius_m, 脱开用更大的 release_radius_m, 回放时几度的
跟踪误差不会让科技核心半路脱开。 以后改成 RM2026 那种接触推动时, 这个模块整体替换掉。
"""

from __future__ import annotations

import numpy as np


class ColumnFollower:
    def __init__(self, *, engage_radius_m: float, release_radius_m: float, slide_range: tuple[float, float]) -> None:
        if not 0.0 < engage_radius_m <= release_radius_m:
            raise ValueError("need 0 < engage_radius_m <= release_radius_m")
        self.engage_radius = float(engage_radius_m)
        self.release_radius = float(release_radius_m)
        self.lower, self.upper = (float(v) for v in slide_range)
        self.engaged = False
        self.slide = 0.0

    def update(self, tcp_in_station: np.ndarray) -> float:
        """输入 TCP 在站体系下的位置 (3,), 返回 step3_slide 目标 (m)。

        "离插入轴多远" 用站体系 (x, y): x 是轴向退开量, y 是侧偏; z 是滑动方向, 不参与判断。
        """
        x, y, z = (float(v) for v in tcp_in_station)
        offset = float(np.hypot(x, y))
        self.engaged = offset <= (self.release_radius if self.engaged else self.engage_radius)
        if self.engaged:
            self.slide = float(np.clip(z, self.lower, self.upper))
        return self.slide
