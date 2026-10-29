"""数据驱动的目标关键点 schema。"""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np
import yaml


@dataclass(frozen=True)
class KeypointSchema:
    """一个目标的完整关键点表。

    ``parts`` 把 detector 的 class_id 映射到它负责的槽位下标: 单类别 schema 由该类别
    拥有全部槽位; 组合 schema (如 pillar + exchange) 由多个类别实例拼成完整点集。
    """

    name: str
    class_id: Optional[int]
    names: tuple[str, ...]
    object_points: np.ndarray
    parts: Optional[Mapping[int, tuple[int, ...]]] = None

    def __post_init__(self) -> None:
        points = np.asarray(self.object_points, dtype=np.float64)
        if points.ndim != 2 or points.shape != (len(self.names), 3):
            raise ValueError("object_points must have shape (len(names), 3)")
        if not np.isfinite(points).all():
            raise ValueError("object points must be finite")
        points = np.ascontiguousarray(points)
        points.setflags(write=False)
        object.__setattr__(self, "object_points", points)

        if self.parts is None:
            if self.class_id is None:
                raise ValueError("schema needs class_id or parts")
            parts = {int(self.class_id): tuple(range(len(self.names)))}
        else:
            parts = {int(cid): tuple(int(i) for i in slots) for cid, slots in self.parts.items()}
        used: set[int] = set()
        for cid, slots in parts.items():
            if not slots or any(i < 0 or i >= len(self.names) for i in slots):
                raise ValueError(f"part for class_id {cid} has invalid slots")
            if used & set(slots):
                raise ValueError("a keypoint belongs to more than one part")
            used |= set(slots)
        object.__setattr__(self, "parts", MappingProxyType(parts))

    def points_for_observation(self, count: int) -> np.ndarray:
        if count != len(self.names):
            raise ValueError(f"observation count {count} does not match schema count {len(self.names)}")
        return self.object_points


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _read_root(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return _as_mapping(document, "schema")


def load_schema(path: str | Path, name: str) -> KeypointSchema:
    root = _read_root(path)
    if name not in root:
        raise KeyError(f"schema {name!r} missing")
    return _parse_schema(name, root[name])


def load_schemas(path: str | Path) -> dict[str, KeypointSchema]:
    """读取并校验文件里的全部 schema; 节点启动时调用, 有错直接起不来。"""
    root = _read_root(path)
    if not root:
        raise ValueError(f"{path}: no schema entries")
    return {str(name): _parse_schema(str(name), entry) for name, entry in root.items()}


def _parse_schema(name: str, entry: Any) -> KeypointSchema:
    entry = _as_mapping(entry, f"schema {name}")
    keypoints = entry.get("keypoints")
    if not isinstance(keypoints, dict) or not keypoints:
        raise ValueError(f"schema {name} keypoints must be a non-empty mapping")
    names = tuple(str(point_name) for point_name in keypoints)

    class_id = None
    if "class_id" in entry or "parts" not in entry:
        class_id = _as_int(entry.get("class_id"), f"schema {name} class_id")
    parts = None
    if "parts" in entry:
        parts = {}
        for part_name, part in _as_mapping(entry["parts"], f"schema {name} parts").items():
            part = _as_mapping(part, f"part {part_name}")
            part_class = _as_int(part.get("class_id"), f"part {part_name} class_id")
            if part_class in parts:
                raise ValueError(f"part {part_name} reuses class_id {part_class}")
            members = part.get("keypoints")
            if not isinstance(members, list) or not members:
                raise ValueError(f"part {part_name} keypoints must be a non-empty list")
            unknown = [m for m in members if str(m) not in names]
            if unknown:
                raise ValueError(f"part {part_name} has unknown keypoint {unknown}")
            parts[part_class] = tuple(names.index(str(m)) for m in members)

    rows = []
    for point_name, value in keypoints.items():
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"keypoint {point_name} must be a length-3 sequence")
        try:
            point = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"keypoint {point_name} must be numeric") from exc
        if not np.isfinite(point).all():
            raise ValueError(f"keypoint {point_name} must be finite")
        rows.append(point)
    return KeypointSchema(name, class_id, names, np.asarray(rows, dtype=np.float64), parts)
