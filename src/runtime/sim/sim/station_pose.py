"""RMUC2026 规则手册 5.3.3 的科技核心位姿 -> sim 站体位姿 (纯函数, 零 ROS / 零 MuJoCo)。

规则的装配坐标系 OXYZ: 原点 O = 底座正面与地面交线的中点, 底座正面法线指向科技核心为 X 负向,
Z 竖直向上。 科技核心位姿 (x, y, z, θ, φ, α): E = 装配柱前表面中心在 OXYZ 下的坐标 (mm);
e = 前表面单位法向, θ = e 在水平面投影与 +X 的夹角, φ = e 与 +Z 的夹角, α = 绕 e 逆时针的转角 (°)。
初始位姿 θ=0, φ=90, α=0: 科技核心立着, 装配柱水平朝外 (+X, 朝机器人)。

本仓库站体系 (planning 的 station pose): 原点 = E, x 轴 = e (插入轴, 待插位在 +x 一侧),
z 轴 = 平移方向 (初始位姿下竖直向上, 与 RM2026 的 E 系一致)。 所以
    O_R_station = Rz(θ) · Ry(-(90° - φ)) · Rx(α)
x 轴列 = (sinφ cosθ, sinφ sinθ, cosφ) = e。

底盘在 OXYZ 下的摆放 chassis_in_O = (x, y, yaw): base_link 原点的水平位置 (m) 与 base_link +x 的朝向 (°);
yaw = 180 表示车头 (+x) 正对科技核心。 base_link 原点离地高度由调用方给 (sim 里读 MJCF)。
"""

from __future__ import annotations

import numpy as np

#: 规则表 5-10 三级难度的位姿范围 (mm / °)
LEVEL3_RANGES = {
    "x": (-100.0, 0.0),
    "y": (100.0, 300.0),
    "z": (500.0, 700.0),
    "theta": (-90.0, 90.0),
    "phi": (0.0, 90.0),
    "alpha": (-45.0, 45.0),
}


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def check_level3(pose: dict) -> list[str]:
    """返回超出规则三级范围的字段说明 (空列表 = 合规)。"""
    problems = []
    for key, (low, high) in LEVEL3_RANGES.items():
        value = float(pose[key])
        if not low - 1e-9 <= value <= high + 1e-9:
            problems.append(f"{key}={value:g} 不在三级范围 [{low:g}, {high:g}]")
    return problems


def station_in_O(pose: dict) -> np.ndarray:
    """规则位姿 {x,y,z (mm), theta,phi,alpha (°)} -> OXYZ 下的站体位姿 4x4 (m)。"""
    theta, phi, alpha = (np.deg2rad(float(pose[k])) for k in ("theta", "phi", "alpha"))
    transform = np.eye(4)
    transform[:3, :3] = _rz(theta) @ _ry(-(np.pi / 2.0 - phi)) @ _rx(alpha)
    transform[:3, 3] = [float(pose[k]) / 1000.0 for k in ("x", "y", "z")]
    return transform


def base_in_O(chassis_in_O, base_height_m: float) -> np.ndarray:
    """底盘 base_link 在 OXYZ 下的位姿 (base_link z 轴竖直, 原点离地 base_height_m)。"""
    x, y, yaw_deg = (float(v) for v in chassis_in_O)
    transform = np.eye(4)
    transform[:3, :3] = _rz(np.deg2rad(yaw_deg))
    transform[:3, 3] = [x, y, float(base_height_m)]
    return transform


def base_T_station(pose: dict, chassis_in_O, base_height_m: float) -> np.ndarray:
    """站体在 base_link 系下的位姿: inv(O_T_base) @ O_T_station。"""
    return np.linalg.inv(base_in_O(chassis_in_O, base_height_m)) @ station_in_O(pose)
