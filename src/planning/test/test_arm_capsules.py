# test_arm_capsules.py — 用连杆 mesh 校验 config 碰撞胶囊: 必须包住 mesh, 且不过胖。
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

# mesh 目录候选: 环境变量 RM26_MESHES 优先, 其次仓库内相对路径。
_HERE = Path(__file__).resolve().parent
_MESH_CANDIDATES = [
    os.environ.get("RM26_MESHES"),
    _HERE / ".." / ".." / "runtime" / "sim" / "model" / "rm26_engineer_description" / "meshes",
]
# 参与校验的连杆 / DH 系编号 1..6。
_LINKS = range(1, 7)


# helper: 返回第一个含 link_1.obj 的 mesh 目录, 都找不到返回 None。
def _find_meshes() -> Path | None:
    for cand in _MESH_CANDIDATES:
        if cand and Path(cand).is_dir() and (Path(cand) / "link_1.obj").is_file():
            return Path(cand).resolve()
    return None


# helper: 只解析 OBJ 的 "v x y z" 顶点行; 没有顶点则报错。
def _obj_vertices(path: Path) -> np.ndarray:
    vertices = [
        [float(value) for value in tokens[1:4]]
        for tokens in (line.split() for line in path.read_text().splitlines())
        if tokens and tokens[0] == "v"
    ]
    if not vertices:
        raise ValueError(f"mesh 里没有顶点: {path}")
    return np.asarray(vertices, dtype=float)


# helper: 用 inv(DH_i) @ inv(F0) @ A0[i] (q=0) 把 link_i 顶点从 link 系换到 DH 第 i 系。
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


# helper: 点集到线段 [start, end] 的最短距离; 线段退化成点时即点距。
def _distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(segment))
    offset = np.asarray(points, dtype=float) - np.asarray(start, dtype=float)
    if length < 1e-12:
        return np.linalg.norm(offset, axis=1)
    direction = segment / length
    projection = np.clip(offset @ direction, 0.0, length)
    return np.linalg.norm(offset - projection[:, None] * direction, axis=1)


# helper (__main__ 也用它生成 YAML): 给定轴向下的最紧包围胶囊, 返回 (from, to, radius)。
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
    # 点在轴向上的投影 along 与到轴线的垂距 perpendicular; 最大垂距即半径
    along = centered @ direction
    perpendicular = np.linalg.norm(centered - along[:, None] * direction, axis=1)
    radius = float(perpendicular.max())
    # cap: 每点在半径球帽内可让出的轴向余量, 据此把两端点向内收紧
    cap = np.sqrt(np.maximum(0.0, radius**2 - perpendicular**2))
    low, high = float((along + cap).min()), float((along - cap).max())
    if high < low:                                       # 整团点都在一个球里
        low = high = 0.5 * (low + high)
    return center + low * direction, center + high * direction, radius


# helper: x/y/z/PCA 四种轴向拟合择优, 返回 (轴向标签, from, to, radius)。
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


# helper: 读 config 胶囊并按 DH frame 分组, 跳过 placeholder 条目。
def _config_capsules() -> dict[int, list[dict]]:
    """把 config 的胶囊按 frame 分组。

    跳过: 名字带 placeholder 的 (没有网格可验); payload: true 的 (被夹持的矿石, 不是连杆网格,
    由 test_payload_capsule_encloses_ore 单独验)。
    """
    grouped: dict[int, list[dict]] = {index: [] for index in _LINKS}
    for entry in load_config()["collision"]["arm"]["capsules"]:
        if "placeholder" in str(entry["name"]) or entry.get("payload"):
            continue
        grouped.setdefault(int(entry["frame"]), []).append(entry)
    return grouped


# ===================== pytest 用例 =====================

# fixture: 各连杆顶点在对应 DH 系下的坐标 {i: (N, 3)}; 找不到 mesh 则 skip。
@pytest.fixture(scope="module")
def _vertices():
    mesh_dir = _find_meshes()
    if mesh_dir is None:
        pytest.skip("找不到 link_*.obj, 设 RM26_MESHES 后再跑")
    return _link_vertices_in_dh_frames(mesh_dir, ArmModel.from_config(load_config()["arm"]))


# 期望: frame 1..6 每个都至少配有一根胶囊。
# TODO(注释): 本用例没用到 _vertices 的值, 推测只是借它在缺 mesh 时一并 skip, 待确认。
def test_every_link_has_a_capsule(_vertices):
    """6 根连杆每根都要有胶囊, 漏一根就是那段臂对碰撞检查隐身。"""
    grouped = _config_capsules()
    missing = [index for index in _LINKS if not grouped.get(index)]
    assert not missing, f"这些 DH 系没有胶囊 (对应连杆不参与碰撞检查): {missing}"


# 期望: 每个 link 的全部顶点到胶囊并集的有符号距离 <= 1e-9 (即都在胶囊内)。
def test_capsules_enclose_link_meshes(_vertices):
    """config 的胶囊必须把对应连杆的 mesh 整个包住 (胶囊凸 -> 顶点包住即整面包住)。"""
    grouped = _config_capsules()
    for index in _LINKS:
        points = _vertices[index]
        # 多根胶囊取并集: 每个顶点取对各胶囊有符号距离的最小值 (负 = 在内)
        distances = np.full(len(points), np.inf)
        for entry in grouped[index]:
            inside = _distance_to_segment(points, entry["from"], entry["to"]) - float(entry["radius"])
            distances = np.minimum(distances, inside)
        worst = float(distances.max())
        assert worst <= 1e-9, (
            f"link_{index} 有顶点露在胶囊外 {worst*1000:.2f} mm "
            f"-> 胶囊不是本臂的 (重新跑 `python3 test/test_arm_capsules.py` 生成)"
        )


# 期望: 单胶囊连杆的半径不超过该 mesh 最紧拟合半径 + 20 mm。
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


# helper: MJCF 里 energy_unit geom (挂在 link_6 上的矿石) 的顶点, 换到 DH 第 6 系。
def _ore_vertices_in_dh6() -> np.ndarray:
    mujoco = pytest.importorskip("mujoco")
    from planning import rotations_from_quaternions

    mjcf = Path(__file__).resolve().parents[2] / "runtime/sim/model/rm26_engineer_description/rm26_arm.xml"
    if not mjcf.is_file():
        pytest.skip(f"找不到 {mjcf}")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    geom = model.geom("energy_unit")
    assert model.geom_bodyid[geom.id] == model.body("link_6").id
    mesh = model.geom_dataid[geom.id]
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    # MuJoCo 的 geom_quat 就是 [w, x, y, z], 和全库约定一致, 直接喂统一入口, 不手动重排
    rotation = rotations_from_quaternions(np.asarray(model.geom_quat[geom.id])[None, :])[0]
    in_link6 = model.mesh_vert[start:start + count] @ rotation.T + model.geom_pos[geom.id]
    _, frame6_T_link6 = _build_mount_constants()
    return in_link6 @ frame6_T_link6[:3, :3].T + frame6_T_link6[:3, 3]


# 期望: payload 胶囊 cc_ore 包住 MJCF 里矿石的全部顶点, 且不比最紧拟合胖 5 mm 以上。
def test_payload_capsule_encloses_ore():
    """矿石胶囊必须和仿真里挂的矿石同一个位姿 (MJCF 改了位姿就要重拟合)。"""
    entries = [e for e in load_config()["collision"]["arm"]["capsules"] if e.get("payload")]
    assert [e["name"] for e in entries] == ["cc_ore"]
    entry = entries[0]
    assert int(entry["frame"]) == 6
    points = _ore_vertices_in_dh6()
    worst = float((_distance_to_segment(points, entry["from"], entry["to"]) - float(entry["radius"])).max())
    assert worst <= 1e-9, f"矿石有顶点露在 cc_ore 外 {worst * 1000:.2f} mm"
    _, _, tight = fit_capsule(points, (1.0, 0.0, 0.0))
    assert float(entry["radius"]) <= tight + 0.005


# 直接运行: 按 mesh 重新拟合每根连杆的胶囊, 打印可粘贴进 config 的 YAML 片段。
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
