"""collision.py — 碰撞检测 (可选, 依赖 hpp-fcl + 站体网格)。

【这个文件干什么】
判断某个关节配置下, 机械臂会不会撞到兑换站/自身。 做法: 把臂的每一段近似
成一根"胶囊" (一条线段 + 一个半径, 像加粗的骨架), 把兑换站/场地用三角网格
表示, 再用 hpp-fcl 几何库算它们之间的最近距离 / 是否相交。 规划器用它来
过滤掉危险配置 (check_configs), 或给"离得太近"加惩罚 (check_configs_with_penalty)。

【要懂的概念】
- 规划不光要能到达目标, 还要一路不撞东西。 碰撞检测就是给规划器一个
  "这个姿势安不安全"的判据。
- 胶囊是碰撞检测里最常用的臂几何近似: 算距离快、比包围盒贴身。

【本项目现状 —— 重要】
- 碰撞是可选的: type2 和 type3 都接受 collision_checker=None, 先不开也能跑通。
- 要开碰撞, 需要两样本仓库暂时没有的东西:
    1) 兑换站的碰撞网格 .obj 资产 (放到 planning/assets/collision/);
    2) 臂各段胶囊参数 (config 里的 collision.arm.capsules), 也要按你这台臂改。
- 依赖 hpp-fcl (系统级)。 本机环境已统一到 numpy 1.24.4, hpp-fcl 可正常导入使用
  (别把 numpy 升到 2.x, 否则会段错误; 详见 README 环境说明)。
"""

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
_MESH_TO_STATION_ROTATION = np.asarray(
    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=float,
)

MeshGroup = tuple[Sequence[str | Path], np.ndarray]


@dataclass(frozen=True, slots=True)
class CapsuleSpec:
    name: str
    frame: int
    point_from: np.ndarray
    point_to: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        point_from = np.asarray(self.point_from, dtype=float)
        point_to = np.asarray(self.point_to, dtype=float)
        if point_from.shape != (3,) or point_to.shape != (3,):
            raise ValueError(f"collision capsule {self.name} endpoints must have shape (3,)")
        if self.frame not in range(7):
            raise ValueError(f"collision capsule {self.name} frame must be in [0, 6]")
        if self.radius <= 0.0 or np.linalg.norm(point_to - point_from) <= 0.0:
            raise ValueError(f"collision capsule {self.name} must have positive size")
        object.__setattr__(self, "point_from", point_from)
        object.__setattr__(self, "point_to", point_to)

    @classmethod
    def from_config(cls, config: dict) -> "CapsuleSpec":
        return cls(
            name=str(config["name"]),
            frame=int(config["frame"]),
            point_from=config["from"],
            point_to=config["to"],
            radius=float(config["radius"]),
        )

    @property
    def segment_length(self) -> float:
        return float(np.linalg.norm(self.point_to - self.point_from))


@dataclass(frozen=True, slots=True)
class CollisionModel:
    station_meshes: tuple[Path, ...]
    local_exchange_meshes: tuple[Path, ...]
    arm_capsules: tuple[CapsuleSpec, ...]

    @classmethod
    def from_config(cls, config: dict) -> "CollisionModel":
        station = config["station"]
        capsules = tuple(
            CapsuleSpec.from_config(entry)
            for entry in config["arm"]["capsules"]
        )
        if not capsules:
            raise ValueError("collision model requires at least one arm capsule")
        return cls(
            station_meshes=_asset_paths(station["meshes"], required=True),
            local_exchange_meshes=_asset_paths(
                station.get("local_exchange_meshes", ()), required=False
            ),
            arm_capsules=capsules,
        )


def _asset_paths(entries: Sequence[dict], *, required: bool) -> tuple[Path, ...]:
    paths = tuple(_collision_asset(entry["asset"]) for entry in entries)
    if required and not paths:
        raise ValueError("collision model requires at least one station mesh")
    return paths


def _collision_asset(filename: str) -> Path:
    path = Path(str(files("planning") / "assets" / "collision" / filename))
    if not path.is_file():
        raise FileNotFoundError(f"collision asset does not exist: {path}")
    return path


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
    if not vertices or not triangles:
        raise ValueError(f"collision OBJ must contain vertices and triangles: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def _build_station_bvh(mesh_groups: Sequence[MeshGroup]):
    transformed_vertices: list[np.ndarray] = []
    indexed_triangles: list[np.ndarray] = []
    vertex_offset = 0

    for index, (mesh_paths, transform) in enumerate(mesh_groups):
        transform = validate_transforms(
            np.asarray(transform, dtype=float)[None],
            name=f"mesh_groups[{index}].transform",
        )[0]
        rotation = transform[:3, :3] @ _MESH_TO_STATION_ROTATION
        translation = transform[:3, 3]
        for path in mesh_paths:
            vertices, triangles = _load_obj(path)
            transformed_vertices.append(vertices @ rotation.T + translation)
            indexed_triangles.append(triangles + vertex_offset)
            vertex_offset += len(vertices)

    if not transformed_vertices:
        raise ValueError("collision checker requires at least one mesh")

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
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"directions must have shape (B, 3), got {directions.shape}")

    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    cross_x = -unit[:, 1]
    cross_y = unit[:, 0]
    sine_squared = cross_x**2 + cross_y**2
    cosine = unit[:, 2]

    skew = np.zeros((len(unit), 3, 3))
    skew[:, 0, 2] = cross_y
    skew[:, 1, 2] = -cross_x
    skew[:, 2, 0] = -cross_y
    skew[:, 2, 1] = cross_x
    skew_squared = skew @ skew
    factor = np.zeros_like(sine_squared)
    np.divide(
        1.0 - cosine,
        sine_squared,
        out=factor,
        where=sine_squared > 1e-12,
    )
    rotations = np.eye(3)[None] + skew + skew_squared * factor[:, None, None]
    antiparallel = (sine_squared <= 1e-12) & (cosine < 0.0)
    rotations[antiparallel] = np.diag([1.0, -1.0, -1.0])
    return rotations


class CollisionChecker:
    """Check arm capsules against station meshes using hpp-fcl."""

    def __init__(
        self,
        arm: ArmModel,
        mesh_groups: Sequence[MeshGroup],
        arm_capsules: Sequence[CapsuleSpec],
    ) -> None:
        if not arm_capsules:
            raise ValueError("collision checker requires at least one arm capsule")
        self.arm = arm
        self._bvh = _build_station_bvh(mesh_groups)
        self._station_transform = hppfcl.Transform3f()
        self._capsule_specs = tuple(arm_capsules)
        self._capsules = tuple(self._make_capsule(spec) for spec in self._capsule_specs)
        self._collision_request = hppfcl.CollisionRequest()
        self._collision_request.enable_distance_lower_bound = True

    @staticmethod
    def _make_capsule(spec: CapsuleSpec, extra_radius: float = 0.0):
        radius = spec.radius + float(extra_radius)
        if radius <= 0.0:
            raise ValueError("inflated capsule radius must be positive")
        return hppfcl.Capsule(radius, spec.segment_length)

    def _capsule_geometry(self, joints: np.ndarray, extra_radius: float):
        kinematics = self.arm.forward_kinematics(joints)
        for index, spec in enumerate(self._capsule_specs):
            frame_rotation = kinematics.link_rotations[:, spec.frame]
            frame_origin = kinematics.link_origins[:, spec.frame]
            point_from = frame_origin + np.einsum(
                "bij,j->bi", frame_rotation, spec.point_from
            )
            point_to = frame_origin + np.einsum(
                "bij,j->bi", frame_rotation, spec.point_to
            )
            capsule = (
                self._capsules[index]
                if extra_radius == 0.0
                else self._make_capsule(spec, extra_radius)
            )
            yield (
                capsule,
                0.5 * (point_from + point_to),
                _rotations_from_z_directions(point_to - point_from),
            )

    def _evaluate(
        self,
        joints: np.ndarray,
        *,
        extra_radius: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joints, dtype=float)
        if joints.ndim != 2 or joints.shape[1] != 6:
            raise ValueError(f"joints must have shape (B, 6), got {joints.shape}")
        if extra_radius < 0.0:
            raise ValueError("extra_radius must be non-negative")

        collides = np.zeros(len(joints), dtype=bool)
        distance_lower_bounds = np.full(len(joints), np.inf)
        if not len(joints):
            return collides, distance_lower_bounds

        result = hppfcl.CollisionResult()
        capsule_transform = hppfcl.Transform3f()
        for capsule, midpoints, rotations in self._capsule_geometry(
            joints, extra_radius
        ):
            for index in np.flatnonzero(~collides):
                capsule_transform.setTransform(rotations[index], midpoints[index])
                result.clear()
                contacts = hppfcl.collide(
                    capsule,
                    capsule_transform,
                    self._bvh,
                    self._station_transform,
                    self._collision_request,
                    result,
                )
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
        if soft_margin < 0.0 or soft_weight < 0.0:
            raise ValueError("soft_margin and soft_weight must be non-negative")
        collides, distance_lower_bounds = self._evaluate(joints)
        penalties = np.where(
            collides,
            0.0,
            np.maximum(0.0, soft_margin - distance_lower_bounds) * soft_weight,
        )
        return collides, penalties
