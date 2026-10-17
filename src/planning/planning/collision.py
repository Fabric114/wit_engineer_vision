# collision.py — 碰撞检测: 机械臂胶囊体 vs 兑换站网格 (hpp-fcl)。
# 碰撞检测
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Sequence

import hppfcl
import numpy as np

from .arm_model import ArmModel
from .transform import validate_transforms


# Collision meshes use the modeling-tool axes; planning uses the station frame.
# 网格建模轴 -> 站体系的固定轴置换: x/y 互换, z 取反 (det=+1, 是合法旋转)。
_MESH_TO_STATION_ROTATION = np.asarray(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=float,
)

# 网格组: (一组 OBJ 路径, 该组 4x4 位姿[, 允许与该组接触的胶囊名]); 同组网格共享同一位姿。
# 第三项可省; 写了就跳过这些胶囊与本组的检测 (允许接触规则, 如装配时夹爪 vs 桩体)。
MeshGroup = tuple  # (Sequence[str | Path], np.ndarray) | (Sequence[str | Path], np.ndarray, Collection[str])


# 单个胶囊体规格: 挂在 FK 的第 frame 个连杆系 (0=基座, 1..6=各关节 DH 系, 6 为法兰),
# 端点为该系下的局部坐标, 端点与半径单位 m。
@dataclass(frozen=True, slots=True)
class CapsuleSpec:
    name: str
    frame: int
    point_from: np.ndarray
    point_to: np.ndarray
    radius: float

    # 规范化端点为 (3,) float 数组, 并校验 frame 范围与尺寸为正。
    def __post_init__(self) -> None:
        point_from = np.asarray(self.point_from, dtype=float)
        point_to = np.asarray(self.point_to, dtype=float)
        if point_from.shape != (3,) or point_to.shape != (3,):
            raise ValueError(f"collision capsule {self.name} endpoints must have shape (3,)")
        if self.frame not in range(7):
            raise ValueError(f"collision capsule {self.name} frame must be in [0, 6]")
        if self.radius <= 0.0 or np.linalg.norm(point_to - point_from) <= 0.0:
            raise ValueError(f"collision capsule {self.name} must have positive size")
        # frozen dataclass 回写规范化后的端点。
        object.__setattr__(self, "point_from", point_from)
        object.__setattr__(self, "point_to", point_to)

    # 从配置项构造 (键: name/frame/from/to/radius)。
    @classmethod
    def from_config(cls, config: dict) -> "CapsuleSpec":
        return cls(
            name=str(config["name"]),
            frame=int(config["frame"]),
            point_from=config["from"],
            point_to=config["to"],
            radius=float(config["radius"]),
        )

    # 胶囊轴线段长度 (两端点距离, 不含半球端帽), m。
    @property
    def segment_length(self) -> float:
        return float(np.linalg.norm(self.point_to - self.point_from))


# 碰撞模型配置: 站体网格路径、局部兑换桩网格路径 (可选)、机械臂胶囊体列表,
# 以及固定在 arm_base 系的车体 / 地面 (body_groups, 已是可直接交给 CollisionChecker 的网格组)。
@dataclass(frozen=True, slots=True)
class CollisionModel:
    station_meshes: tuple[Path, ...]
    local_exchange_meshes: tuple[Path, ...]
    arm_capsules: tuple[CapsuleSpec, ...]
    body_groups: tuple = ()
    # 装配段允许与局部桩体接触的胶囊名 (允许接触规则)
    assembly_allowed_contact: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, config: dict) -> "CollisionModel":
        # 解析胶囊体列表 (至少一个), 再解析网格目录与各网格路径 (站体网格必需, 局部网格可选)。
        station = config["station"]
        capsules = tuple(
            CapsuleSpec.from_config(entry)
            for entry in config["arm"]["capsules"]
        )
        if not capsules:
            raise ValueError("collision model requires at least one arm capsule")
        asset_dir = _asset_dir(station.get("asset_dir"))
        names = {capsule.name for capsule in capsules}
        allowed = frozenset(station.get("assembly_allowed_contact", ()) or ())
        _check_capsule_names(allowed, names, "collision.station.assembly_allowed_contact")
        return cls(
            station_meshes=_asset_paths(station["meshes"], asset_dir, required=True),
            local_exchange_meshes=_asset_paths(
                station.get("local_exchange_meshes", ()), asset_dir, required=False
            ),
            arm_capsules=capsules,
            body_groups=_body_groups(
                config.get("robot_body"), config.get("ground"), names, float(config.get("line_m", 0.0) or 0.0)
            ),
            assembly_allowed_contact=allowed,
        )

    def scene_groups(self, station_transform: np.ndarray, *, stage: str) -> list:
        """按阶段组装碰撞场景 (交给 CollisionChecker 的网格组)。

        - ``approach``: 站体 + 局部桩体 (全部胶囊都查) + 车体 / 地面;
        - ``assembly``: 同上, 但桩体对 ``assembly_allowed_contact`` 里的胶囊豁免 —— 装配本来就要
          贴着桩体, 这是"允许接触规则"。 豁免意味着这些胶囊与桩体之间**不再检查**, 不是"保证不撞"。
        """
        if stage not in ("approach", "assembly"):
            raise ValueError(f"unknown collision scene stage {stage!r}")
        groups = [(self.station_meshes, station_transform)]
        if self.local_exchange_meshes:
            allowed = self.assembly_allowed_contact if stage == "assembly" else frozenset()
            groups.append((self.local_exchange_meshes, station_transform, allowed))
        groups.extend(self.body_groups)
        return groups


def _check_capsule_names(requested, names: set[str], where: str) -> None:
    unknown = set(requested) - names
    if unknown:
        raise ValueError(f"{where} names unknown capsules: {sorted(unknown)}")


def _native_frame(arm_base_T_mesh: np.ndarray) -> np.ndarray:
    """车体网格本来就在自己的 body 系 (不是站体建模轴), 抵消 _build_station_bvh 里的轴置换。"""
    transform = np.asarray(arm_base_T_mesh, dtype=float).reshape(4, 4).copy()
    transform[:3, :3] = transform[:3, :3] @ _MESH_TO_STATION_ROTATION.T
    return transform


def _ground_box(z_m: float, half_extent_m: float, thickness_m: float) -> tuple[np.ndarray, np.ndarray]:
    """地面: 顶面在 arm_base 系 z = z_m 的薄盒 (12 个三角形)。 用实心盒而不是单个平面,
    胶囊整段沉到地下也能判碰撞。"""
    x = (-half_extent_m, half_extent_m)
    z = (z_m - thickness_m, z_m)
    vertices = np.asarray([(a, b, c) for a in x for b in x for c in z], dtype=float)
    # 顶点下标 = 4*ix + 2*iy + iz
    faces = [
        (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),  # z- / z+
        (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),  # y- / y+
        (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),  # x- / x+
    ]
    return vertices, np.asarray(faces, dtype=np.int64)


def _body_groups(body_cfg: dict | None, ground_cfg: dict | None, names: set[str], line_m: float = 0.0) -> tuple:
    """车体 (底盘 / 升降 / 轮子) 与地面网格组, 都在 arm_base 系。

    config 里的位姿按升降轴 line = 0 写; line 升 line_m 后 arm_base 随臂上移,
    在 arm_base 系里看底盘 / 轮子 / 地面整体下移 line_m。 标了 ``follows_line: true`` 的网格
    (升降滑台 line_link, 跟着臂一起升) 相对 arm_base 不动, 不下移。
    """
    if line_m < 0.0:
        raise ValueError("collision.line_m must be non-negative")
    lowered = np.eye(4)
    lowered[2, 3] = -float(line_m)
    groups = []
    if body_cfg:
        asset_dir = _asset_dir(body_cfg.get("asset_dir"))
        for entry in body_cfg.get("meshes", ()):
            allowed = frozenset(entry.get("allowed_contact", ()) or ())
            _check_capsule_names(allowed, names, f"collision.robot_body {entry['asset']}")
            groups.append(
                (
                    (_collision_asset(asset_dir, entry["asset"]),),
                    (np.eye(4) if entry.get("follows_line") else lowered) @ _native_frame(entry["arm_base_T_mesh"]),
                    allowed,
                )
            )
    if ground_cfg:
        box = _ground_box(
            float(ground_cfg["z_m"]),
            float(ground_cfg.get("half_extent_m", 3.0)),
            float(ground_cfg.get("thickness_m", 0.2)),
        )
        groups.append(((box,), lowered @ _native_frame(np.eye(4)), frozenset()))
    return tuple(groups)


def _asset_dir(configured: str | None) -> Path:
    """站体网格目录: config 的 ``collision.station.asset_dir`` (相对路径按工作区根解析),
    没写就用包内 ``assets/collision/``。"""
    # 未配置: 用包内自带资产目录。
    if not configured:
        return Path(str(files("planning") / "assets" / "collision"))
    # 相对路径按工作区根解析; 函数内导入避免与包 __init__ 循环导入。
    path = Path(configured)
    if not path.is_absolute():
        from . import _workspace_root

        root = _workspace_root()
        if root is None:
            raise FileNotFoundError(
                f"collision.station.asset_dir 是相对路径 {configured!r}, 但找不到工作区根"
            )
        path = root / path
    return path


# 把配置里的网格条目 (含 asset 文件名) 解析成路径元组; required 时不允许为空。
def _asset_paths(
    entries: Sequence[dict], asset_dir: Path, *, required: bool
) -> tuple[Path, ...]:
    paths = tuple(_collision_asset(asset_dir, entry["asset"]) for entry in entries)
    if required and not paths:
        raise ValueError("collision model requires at least one station mesh")
    return paths


# 拼出网格资产路径并确认文件存在。
def _collision_asset(asset_dir: Path, filename: str) -> Path:
    path = asset_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"collision asset does not exist: {path}")
    return path


# 极简 OBJ 解析: 只读顶点 v 和面 f (只取顶点索引, 转 0 基); 返回 (V, 3) float64 与 (F, 3) int64。
# TODO(注释): 多边形面只取前 3 个顶点、不支持负索引, 是否默认资产已三角化?
def _load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "v":
                vertices.append([float(value) for value in tokens[1:4]])
            elif tokens[0] == "f" and len(tokens) >= 4:
                triangles.append(
                    [int(token.split("/")[0]) - 1 for token in tokens[1:4]]
                )
    # 空网格视为资产错误。
    if not vertices or not triangles:
        raise ValueError(f"collision OBJ must contain vertices and triangles: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


# 把所有网格组合并成一个静态 BVH (OBBRSS), 顶点统一变换到规划坐标系 (与 FK 输出同系)。
def _build_station_bvh(mesh_groups: Sequence[MeshGroup]):
    # vertex_offset 把各网格三角形索引平移到合并后的全局顶点编号。
    transformed_vertices: list[np.ndarray] = []
    indexed_triangles: list[np.ndarray] = []
    vertex_offset = 0

    # 每组: 校验 4x4 位姿; 旋转 = 组位姿旋转 @ 网格轴置换, 即先轴置换再施加组位姿。
    for index, (mesh_paths, transform) in enumerate(mesh_groups):
        transform = validate_transforms(
            np.asarray(transform, dtype=float)[None],
            name=f"mesh_groups[{index}].transform",
        )[0]
        rotation = transform[:3, :3] @ _MESH_TO_STATION_ROTATION
        translation = transform[:3, 3]
        # 加载每个 OBJ (或直接给的 (顶点, 三角形) 数组), 顶点做刚体变换 (行向量形式 v @ R^T + t)。
        for path in mesh_paths:
            vertices, triangles = path if isinstance(path, tuple) else _load_obj(path)
            transformed_vertices.append(vertices @ rotation.T + translation)
            indexed_triangles.append(triangles + vertex_offset)
            vertex_offset += len(vertices)

    # 没有任何网格时无法建场景。
    if not transformed_vertices:
        raise ValueError("collision checker requires at least one mesh")

    # 拼接后构建 hpp-fcl BVH 模型。
    vertices = np.concatenate(transformed_vertices).astype(np.float64, copy=False)
    triangles = np.concatenate(indexed_triangles).astype(np.int64, copy=False)
    bvh = hppfcl.BVHModelOBBRSS()
    bvh.beginModel(len(triangles), len(vertices))
    bvh.addVertices(vertices)
    bvh.addTriangles(triangles)
    bvh.endModel()
    return bvh


def _rotations_from_z_directions(directions: np.ndarray) -> np.ndarray:
    """Return rotations whose local z axes follow ``(B, 3)`` directions."""
    # 形状校验: (B, 3)。
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"directions must have shape (B, 3), got {directions.shape}")

    # Rodrigues 公式求把 z 轴转到 unit 的旋转: 旋转轴 k = z × unit = (-uy, ux, 0),
    # sin^2 = |k|^2, cos = uz。
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    cross_x = -unit[:, 1]
    cross_y = unit[:, 0]
    sine_squared = cross_x**2 + cross_y**2
    cosine = unit[:, 2]

    # 反对称矩阵 K = [k]x, R = I + K + K^2 * (1 - cos) / sin^2。
    skew = np.zeros((len(unit), 3, 3))
    skew[:, 0, 2] = cross_y
    skew[:, 1, 2] = -cross_x
    skew[:, 2, 0] = -cross_y
    skew[:, 2, 1] = cross_x
    skew_squared = skew @ skew
    # 与 z 轴平行 (sin^2≈0) 时 factor 置 0, 得 R = I。
    factor = np.zeros_like(sine_squared)
    np.divide(
        1.0 - cosine,
        sine_squared,
        out=factor,
        where=sine_squared > 1e-12,
    )
    rotations = np.eye(3)[None] + skew + skew_squared * factor[:, None, None]
    # 反向平行 (方向≈-z) 单独处理: 绕 x 轴转 180°。
    antiparallel = (sine_squared <= 1e-12) & (cosine < 0.0)
    rotations[antiparallel] = np.diag([1.0, -1.0, -1.0])
    return rotations


# 胶囊体 (经 FK 变到机械臂基座系) 与站体 BVH (经 mesh_groups 位姿变到同一坐标系) 做离散碰撞检测。
class CollisionChecker:
    """Check arm capsules against station meshes using hpp-fcl."""

    def __init__(
        self,
        arm: ArmModel,
        mesh_groups: Sequence[MeshGroup],
        arm_capsules: Sequence[CapsuleSpec],
    ) -> None:
        # 校验胶囊非空; 预建站体 BVH 与各胶囊几何, 站体固定在单位变换。
        if not arm_capsules:
            raise ValueError("collision checker requires at least one arm capsule")
        self.arm = arm
        self._capsule_specs = tuple(arm_capsules)
        # 按"允许接触的胶囊集合"把网格组分桶, 每桶一个 BVH; 检测时跳过桶里豁免的胶囊。
        buckets: dict[frozenset[str], list] = {}
        for group in mesh_groups:
            if len(group) not in (2, 3):
                raise ValueError("mesh group must be (paths, transform) or (paths, transform, allowed_capsules)")
            # 名字只在 CollisionModel.from_config 里校验 (防 config 笔误); 这里允许列出本检查器
            # 没带的胶囊 (只用部分胶囊建检查器时很常见), 不在场的名字自然不起作用。
            allowed = frozenset(group[2]) if len(group) == 3 else frozenset()
            buckets.setdefault(allowed, []).append((group[0], group[1]))
        if not buckets:
            raise ValueError("collision checker requires at least one mesh")
        self._scenes = tuple(
            (_build_station_bvh(groups), allowed) for allowed, groups in buckets.items()
        )
        self._station_transform = hppfcl.Transform3f()
        # 只有一个无豁免场景时保留 _bvh (test_collision_assets 的精确距离对照直接用它)
        self._bvh = self._scenes[0][0] if len(self._scenes) == 1 else None
        self._capsules = tuple(self._make_capsule(spec) for spec in self._capsule_specs)
        # 开启 distance_lower_bound: 无碰撞时也能拿到距离下界, 用于软惩罚。
        self._collision_request = hppfcl.CollisionRequest()
        self._collision_request.enable_distance_lower_bound = True

    # 按规格创建 hpp-fcl 胶囊 (可选外扩半径 extra_radius, m); hpp-fcl 胶囊中心在原点、轴沿局部 z。
    @staticmethod
    def _make_capsule(spec: CapsuleSpec, extra_radius: float = 0.0):
        radius = spec.radius + float(extra_radius)
        if radius <= 0.0:
            raise ValueError("inflated capsule radius must be positive")
        return hppfcl.Capsule(radius, spec.segment_length)

    # 生成器: 对每个胶囊, 用 FK 把端点变到基座系, 产出 (胶囊几何, 中点 (B, 3), 姿态 (B, 3, 3))。
    def _capsule_geometry(self, joints: np.ndarray, extra_radius: float):
        # 批量 FK; link_rotations (B, 7, 3, 3), link_origins (B, 7, 3)。
        kinematics = self.arm.forward_kinematics(joints)
        for index, spec in enumerate(self._capsule_specs):
            frame_rotation = kinematics.link_rotations[:, spec.frame]
            frame_origin = kinematics.link_origins[:, spec.frame]
            # 局部端点 -> 基座系: p = o + R @ p_local。
            point_from = frame_origin + np.einsum(
                "bij,j->bi", frame_rotation, spec.point_from
            )
            point_to = frame_origin + np.einsum(
                "bij,j->bi", frame_rotation, spec.point_to
            )
            # 无外扩时复用预建胶囊, 否则按外扩半径临时新建。
            capsule = (
                self._capsules[index]
                if extra_radius == 0.0
                else self._make_capsule(spec, extra_radius)
            )
            # hpp-fcl 胶囊位姿: 中心放在线段中点, 局部 z 轴沿 from -> to。
            yield (
                capsule,
                0.5 * (point_from + point_to),
                _rotations_from_z_directions(point_to - point_from),
            )

    # 核心检测: 返回 (每个构型是否碰撞 (B,), 到站体的距离下界 (B,), 碰撞为 0)。
    def _evaluate(
        self,
        joints: np.ndarray,
        *,
        extra_radius: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 输入校验: joints 为 (B, 6), 外扩半径非负。
        joints = np.asarray(joints, dtype=float)
        if joints.ndim != 2 or joints.shape[1] != 6:
            raise ValueError(f"joints must have shape (B, 6), got {joints.shape}")
        if extra_radius < 0.0:
            raise ValueError("extra_radius must be non-negative")

        # 初始化结果; 空批次直接返回。
        collides = np.zeros(len(joints), dtype=bool)
        distance_lower_bounds = np.full(len(joints), np.inf)
        if not len(joints):
            return collides, distance_lower_bounds

        # 外层遍历胶囊, 内层只检查尚未判定碰撞的构型 (已碰撞的提前剪枝)。
        result = hppfcl.CollisionResult()
        capsule_transform = hppfcl.Transform3f()
        for spec, (capsule, midpoints, rotations) in zip(
            self._capsule_specs, self._capsule_geometry(joints, extra_radius)
        ):
            for bvh, allowed in self._scenes:
                if spec.name in allowed:
                    continue
                for index in np.flatnonzero(~collides):
                    capsule_transform.setTransform(rotations[index], midpoints[index])
                    result.clear()
                    # 窄相检测: 胶囊 vs 站体 BVH。
                    contacts = hppfcl.collide(
                        capsule,
                        capsule_transform,
                        bvh,
                        self._station_transform,
                        self._collision_request,
                        result,
                    )
                    # 有接触: 标记碰撞且距离置 0; 否则取所有胶囊中的最小距离下界。
                    # TODO(注释): distance_lower_bound 是保守下界, 可能明显小于真实距离, 软惩罚会偏大, 是否可接受待确认。
                    if contacts > 0:
                        collides[index] = True
                        distance_lower_bounds[index] = 0.0
                    else:
                        distance_lower_bounds[index] = min(
                            distance_lower_bounds[index],
                            float(result.distance_lower_bound),
                        )
        return collides, distance_lower_bounds

    def check_configs(
        self,
        joints: np.ndarray,
        extra_radius: float = 0.0,
    ) -> np.ndarray:
        """Return collision flags for a ``(B, 6)`` joint batch."""
        return self._evaluate(joints, extra_radius=extra_radius)[0]

    def check_configs_with_penalty(
        self,
        joints: np.ndarray,
        soft_margin: float,
        soft_weight: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return collision flags and conservative proximity penalties."""
        # 参数校验: soft_margin (m) 与 soft_weight 非负。
        if soft_margin < 0.0 or soft_weight < 0.0:
            raise ValueError("soft_margin and soft_weight must be non-negative")
        # 惩罚 = soft_weight * max(0, soft_margin - 距离下界); 已碰撞构型惩罚为 0 (由碰撞标志另行处理)。
        collides, distance_lower_bounds = self._evaluate(joints)
        penalties = np.where(
            collides,
            0.0,
            np.maximum(0.0, soft_margin - distance_lower_bounds) * soft_weight,
        )
        return collides, penalties
