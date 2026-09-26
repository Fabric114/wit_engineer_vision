"""test_arm_capsules.py — 用连杆 mesh 验证 config 里的碰撞胶囊 (黄金检查)。

思路 (为什么这样验)
-------------------
- ``collision.arm.capsules`` 的每根胶囊都写在**某个 DH 系**里 (``frame: i``),
  运行时 ``CollisionChecker`` 用 ``ArmModel`` 的 FK 把它摆到 arm_base 系。
  所以"胶囊标得对不对"= "把 link_i 的 mesh 拉到同一个 DH 系后, 是否落在胶囊里"。
- 标准答案取 ``rm26_engineer_description/meshes/link_i.obj`` —— MJCF 里每个
  link 的 geom 都没有 pos/quat 偏置, URDF 的 visual/collision origin 也全是 0,
  所以 obj 顶点就是 link 系坐标, 不需要再对齐。
- 两个坐标系的桥: ``F0`` (line_link->arm_base) 与 URDF 关节 origin 链 ``A0``,
  都从 test_dh_mujoco 借用 (那边只用 URDF 几何算, 不读 config)。
    link_i 系 -> arm_base:  inv(F0) @ A0[i]
    DH_i 系   -> arm_base:  FK(q=0).link_transforms[i]
  q=0 一组就够: 胶囊和 mesh 都刚性固连在同一个 DH 系上, 一起转。
- **只查顶点就够**: 胶囊是凸的, 三顶点都在胶囊内 -> 整个三角面都在胶囊内。
  注意这个论证只对"单根胶囊覆盖单个连杆"成立; 哪天把一个连杆拆成多根胶囊,
  就必须改成在三角面上采样再查 (两根胶囊之间可能漏掉中间的连接腹板)。

怎么跑
------
  重新拟合并打印 YAML:  ``PYTHONPATH=. python3 test/test_arm_capsules.py``
  随 pytest:            ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest test/test_arm_capsules.py -v``
找不到 mesh 时自动 skip (可用环境变量 RM26_MESHES 指路)。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from planning import ArmModel, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 直接 python3 跑时也能 import 同目录
from test_dh_mujoco import _build_mount_constants, _urdf_frames  # noqa: E402

_HERE = Path(__file__).resolve().parent
_MESH_CANDIDATES = [
    os.environ.get("RM26_MESHES"),
    _HERE / ".." / ".." / "runtime" / "sim" / "model" / "rm26_engineer_description" / "meshes",
]
_LINKS = range(1, 7)


def _find_meshes() -> Path | None:
    for cand in _MESH_CANDIDATES:
        if cand and Path(cand).is_dir() and (Path(cand) / "link_1.obj").is_file():
            return Path(cand).resolve()
    return None


def _obj_vertices(path: Path) -> np.ndarray:
    vertices = [
        [float(value) for value in tokens[1:4]]
        for tokens in (line.split() for line in path.read_text().splitlines())
        if tokens and tokens[0] == "v"
    ]
    if not vertices:
        raise ValueError(f"mesh 里没有顶点: {path}")
    return np.asarray(vertices, dtype=float)


def _link_vertices_in_dh_frames(mesh_dir: Path, arm: ArmModel) -> dict[int, np.ndarray]:
    """返回 {i: link_i 的顶点在 DH 第 i 系的坐标 (N, 3)}。"""
    mount, _ = _build_mount_constants()                  # line_link -> arm_base
    urdf = _urdf_frames(np.zeros(6))                     # line_link -> URDF link_i (q=0)
    dh = arm.forward_kinematics(np.zeros((1, 6))).link_transforms[0]  # arm_base -> DH_i
    inverse_mount = np.linalg.inv(mount)
    out: dict[int, np.ndarray] = {}
    for index in _LINKS:
        transform = np.linalg.inv(dh[index]) @ inverse_mount @ urdf[index]  # link_i -> DH_i
        vertices = _obj_vertices(mesh_dir / f"link_{index}.obj")
        out[index] = vertices @ transform[:3, :3].T + transform[:3, 3]
    return out


def _distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(segment))
    offset = np.asarray(points, dtype=float) - np.asarray(start, dtype=float)
    if length < 1e-12:
        return np.linalg.norm(offset, axis=1)
    direction = segment / length
    projection = np.clip(offset @ direction, 0.0, length)
    return np.linalg.norm(offset - projection[:, None] * direction, axis=1)


def fit_capsule(points: np.ndarray, axis=None) -> tuple[np.ndarray, np.ndarray, float]:
    """拟合包住 ``points`` 的最小胶囊 (给定轴向; axis=None 用 PCA 主轴)。

    半径 = 点到轴线的最大垂距; 两端再按球帽能覆盖的程度收紧, 所以结果是
    "该轴向下的最紧胶囊"。
    """
    center = points.mean(axis=0)
    centered = points - center
    direction = (
        np.linalg.svd(centered, full_matrices=False)[2][0]
        if axis is None
        else np.asarray(axis, dtype=float)
    )
    direction = direction / np.linalg.norm(direction)
    along = centered @ direction
    perpendicular = np.linalg.norm(centered - along[:, None] * direction, axis=1)
    radius = float(perpendicular.max())
    cap = np.sqrt(np.maximum(0.0, radius**2 - perpendicular**2))
    low, high = float((along + cap).min()), float((along - cap).max())
    if high < low:                                       # 整团点都在一个球里
        low = high = 0.5 * (low + high)
    return center + low * direction, center + high * direction, radius


def _best_fit(points: np.ndarray) -> tuple[str, np.ndarray, np.ndarray, float]:
    """在 x/y/z/PCA 四个轴向里挑半径最小的; 轴对齐只要不差过 1 mm 就优先 (配置好读)。"""
    candidates = {
        "x": fit_capsule(points, (1.0, 0.0, 0.0)),
        "y": fit_capsule(points, (0.0, 1.0, 0.0)),
        "z": fit_capsule(points, (0.0, 0.0, 1.0)),
        "pca": fit_capsule(points, None),
    }
    smallest = min(value[2] for value in candidates.values())
    tag = next((t for t in ("x", "y", "z") if candidates[t][2] <= smallest + 1e-3), "pca")
    return tag, *candidates[tag]


def _config_capsules() -> dict[int, list[dict]]:
    """把 config 的胶囊按 frame 分组 (名字带 placeholder 的跳过: 那是待实测的夹爪)。"""
    grouped: dict[int, list[dict]] = {index: [] for index in _LINKS}
    for entry in load_config()["collision"]["arm"]["capsules"]:
        if "placeholder" in str(entry["name"]):
            continue
        grouped.setdefault(int(entry["frame"]), []).append(entry)
    return grouped


# ===================== pytest 用例 =====================

@pytest.fixture(scope="module")
def _vertices():
    mesh_dir = _find_meshes()
    if mesh_dir is None:
        pytest.skip("找不到 link_*.obj, 设 RM26_MESHES 后再跑")
    return _link_vertices_in_dh_frames(mesh_dir, ArmModel.from_config(load_config()["arm"]))


def test_every_link_has_a_capsule(_vertices):
    """6 根连杆每根都要有胶囊, 漏一根就是那段臂对碰撞检查隐身。"""
    grouped = _config_capsules()
    missing = [index for index in _LINKS if not grouped.get(index)]
    assert not missing, f"这些 DH 系没有胶囊 (对应连杆不参与碰撞检查): {missing}"


def test_capsules_enclose_link_meshes(_vertices):
    """config 的胶囊必须把对应连杆的 mesh 整个包住 (胶囊凸 -> 顶点包住即整面包住)。"""
    grouped = _config_capsules()
    for index in _LINKS:
        points = _vertices[index]
        distances = np.full(len(points), np.inf)
        for entry in grouped[index]:
            inside = _distance_to_segment(points, entry["from"], entry["to"]) - float(entry["radius"])
            distances = np.minimum(distances, inside)
        worst = float(distances.max())
        assert worst <= 1e-9, (
            f"link_{index} 有顶点露在胶囊外 {worst*1000:.2f} mm "
            f"-> 胶囊不是本臂的 (重新跑 `python3 test/test_arm_capsules.py` 生成)"
        )


def test_capsules_are_not_over_inflated(_vertices):
    """胶囊也不能明显比 mesh 胖: 虚胖会白丢可行解 (要安全余量请用 extra_radius)。"""
    grouped = _config_capsules()
    for index in _LINKS:
        if len(grouped[index]) != 1:                      # 一根连杆拆成多根时不适用
            continue
        _, _, _, tight = _best_fit(_vertices[index])
        radius = float(grouped[index][0]["radius"])
        assert radius <= tight + 0.02, (
            f"link_{index} 的胶囊半径 {radius*1000:.1f} mm 比最紧拟合 "
            f"{tight*1000:.1f} mm 胖了 20 mm 以上"
        )


if __name__ == "__main__":
    mesh_dir = _find_meshes()
    if mesh_dir is None:
        raise SystemExit("找不到 link_*.obj, 设 RM26_MESHES 后再跑")
    vertices = _link_vertices_in_dh_frames(mesh_dir, ArmModel.from_config(load_config()["arm"]))
    print(f"meshes: {mesh_dir}\n")
    print("      capsules:")
    for index in _LINKS:
        tag, start, end, radius = _best_fit(vertices[index])
        start, end = np.round(start, 4) + 0.0, np.round(end, 4) + 0.0   # +0.0: 把 -0.0 归一成 0.0
        radius_cfg = float(np.ceil(radius * 1000.0) / 1000.0)      # 向上取到 mm, 保守
        slack = radius_cfg - float(_distance_to_segment(vertices[index], start, end).max())
        print(f"      - name: cc_link{index}")
        print(f"        frame: {index}")
        print(f"        from: [{start[0]:g}, {start[1]:g}, {start[2]:g}]")
        print(f"        to: [{end[0]:g}, {end[1]:g}, {end[2]:g}]")
        print(f"        radius: {radius_cfg:g}"
              f"    # 轴向 {tag}, 最紧 {radius*1000:.2f} mm, 取整后余量 +{slack*1000:.2f} mm")
