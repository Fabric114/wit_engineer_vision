# test_compare_urdf.py — tools/compare_urdf.py 的用例: 新旧 URDF 对比的结构/几何/零位/网格报告与退出码。
"""test_compare_urdf.py — 用改过的临时 URDF 喂 compare_urdf, 锁定退出码与零位换算符号。

换算约定 (本文件锁定): ``zero_delta = theta_offset_new - theta_offset_old``,
同一物理姿态 ``q_new = q_old - zero_delta``。
"""

from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planning import ArmModel, load_config  # noqa: E402
from tools.compare_urdf import compare, exit_code, main  # noqa: E402
from tools.urdf_model import ARM_JOINTS, DEFAULT_URDF, chain_frames, derive_arm_geometry, load_urdf  # noqa: E402


# helper: 读默认 URDF, 交给 edit(root) 改, 写到 tmp_path 下并返回路径。
def _edited_urdf(tmp_path: Path, edit) -> Path:
    tree = ET.parse(str(DEFAULT_URDF))
    edit(tree.getroot())
    out = tmp_path / "new.urdf"
    tree.write(str(out), encoding="utf-8", xml_declaration=True)
    return out


# helper: 按名字找 joint / link 元素。
def _find(root, tag: str, name: str):
    return next(e for e in root.findall(tag) if e.get("name") == name)


# helper: 用派生的 DH 构造 ArmModel (其余字段沿用 config)。
def _arm(geometry) -> ArmModel:
    arm_cfg = copy.deepcopy(load_config()["arm"])
    for key in ("a", "alpha", "d", "theta_offset"):
        arm_cfg["dh"][key] = getattr(geometry, key).tolist()
    return ArmModel.from_config(arm_cfg)


# 期望: URDF 对自身 -> 退出码 0, zero_delta 全 0, 无结构错误/网格变化。
def test_identical_urdf_exit_0():
    model = load_urdf()
    report = compare(model, model)
    assert report.structure_errors == [] and report.mesh_changes == []
    assert np.allclose(report.zero_delta, 0.0)
    assert exit_code(report) == 0
    assert main([str(DEFAULT_URDF), str(DEFAULT_URDF)]) == 0


# 期望: J1 origin yaw 加 0.3 -> a/alpha/d 不变, zero_delta[0] = ±0.3, FK_new(q_old - Δ) == FK_old(q_old), 退出码 1。
def test_rotated_j1_origin_reports_zero_delta(tmp_path):
    def edit(root):
        origin = _find(root, "joint", "J1").find("origin")
        r, p, y = (float(v) for v in origin.get("rpy").split())
        origin.set("rpy", f"{r} {p} {y + 0.3}")

    old, new = load_urdf(), load_urdf(_edited_urdf(tmp_path, edit))
    report = compare(old, new)
    for key in ("a", "alpha", "d"):
        assert np.allclose(report.dh_shape_delta[key], 0.0, atol=1e-9), key
    # J1 轴向是 (0, 0, -1): origin 绕 +z 转 0.3 = 绕关节轴转 -0.3
    assert abs(report.zero_delta[0] - (-0.3)) < 1e-9, report.zero_delta
    assert np.allclose(report.zero_delta[1:], 0.0, atol=1e-9)

    # 物理对照 (只用 URDF 链式 FK, 不经派生 DH): 新 URDF 在 q_old - Δ 的 link_6 == 旧 URDF 在 q_old
    rng = np.random.default_rng(0)
    for q_old in rng.uniform(-1.0, 1.0, size=(20, 6)):
        f_old = chain_frames(old, dict(zip(ARM_JOINTS, q_old)))["link_6"]
        f_new = chain_frames(new, dict(zip(ARM_JOINTS, q_old - report.zero_delta)))["link_6"]
        assert np.allclose(f_new, f_old, atol=1e-9)

    arm_old, arm_new = _arm(derive_arm_geometry(old)), _arm(derive_arm_geometry(new))
    q_old = rng.uniform(-1.0, 1.0, size=(20, 6))
    fk_old = arm_old.forward_kinematics(q_old).tcp_transforms
    fk_new = arm_new.forward_kinematics(q_old - report.zero_delta).tcp_transforms
    assert np.allclose(fk_new, fk_old, atol=1e-9)
    assert exit_code(report) == 1


# 期望: J3 origin 平移 5 mm -> DH 形状变了, 退出码 2。
def test_moved_j3_origin_exit_2(tmp_path):
    def edit(root):
        origin = _find(root, "joint", "J3").find("origin")
        x, y, z = (float(v) for v in origin.get("xyz").split())
        origin.set("xyz", f"{x} {y} {z + 0.005}")

    report = compare(load_urdf(), load_urdf(_edited_urdf(tmp_path, edit)))
    assert max(float(np.max(np.abs(v))) for v in report.dh_shape_delta.values()) > 1e-3
    assert exit_code(report) == 2


# 期望: 删掉 J5 -> structure_errors 提到 J5, main 返回 2 且不抛异常。
def test_compare_urdf_reports_missing_joint(tmp_path, capsys):
    def edit(root):
        root.remove(_find(root, "joint", "J5"))

    new = _edited_urdf(tmp_path, edit)
    report = compare(load_urdf(), load_urdf(new))
    assert any("J5" in e for e in report.structure_errors), report.structure_errors
    assert main([str(DEFAULT_URDF), str(new)]) == 2
    assert "J5" in capsys.readouterr().out


# 期望: 只改 link_6 的网格文件名 -> 退出码 1, mesh_changes 提到 link_6。
def test_mesh_change_exit_1(tmp_path):
    def edit(root):
        for mesh in _find(root, "link", "link_6").iter("mesh"):
            mesh.set("filename", "package://GC3.1/meshes/link_6_v2.STL")

    report = compare(load_urdf(), load_urdf(_edited_urdf(tmp_path, edit)))
    assert any("link_6" in m for m in report.mesh_changes), report.mesh_changes
    assert report.structure_errors == []
    assert exit_code(report) == 1


# 期望: 文件不存在 -> main 返回 2, 不抛 traceback。
def test_main_missing_file_exit_2(tmp_path):
    assert main([str(DEFAULT_URDF), str(tmp_path / "nope.urdf")]) == 2


# 期望: 只挪车体 (轮子 / line origin) -> 臂 DH 不变, 但 robot_body / mount 变了, 报出来且退出码 1。
@pytest.mark.parametrize("joint, field_hint", [("wheel_1", "wheel_link_1"), ("line", "base_link")])
def test_chassis_change_exit_1(tmp_path, joint, field_hint):
    def edit(root):
        origin = _find(root, "joint", joint).find("origin")
        x, y, z = (float(v) for v in origin.get("xyz").split())
        origin.set("xyz", f"{x + 0.01} {y} {z}")

    report = compare(load_urdf(), load_urdf(_edited_urdf(tmp_path, edit)))
    assert any(field_hint in m for m in report.body_changes), report.body_changes
    assert exit_code(report) == 1


# 期望: 结构错误 (J1 改 continuous、删 limit) 再叠加零位变化, main 返回 2 且不抛异常。
def test_structure_error_with_zero_delta_no_traceback(tmp_path):
    def edit(root):
        j1 = _find(root, "joint", "J1")
        j1.set("type", "continuous")
        j1.remove(j1.find("limit"))
        j1.find("origin").set("rpy", "0 0 0.3")

    assert main([str(DEFAULT_URDF), str(_edited_urdf(tmp_path, edit))]) == 2


# 期望: 几何里出现非有限值时判 2, 不能因 NaN 比较为 False 而当成一致。
def test_nonfinite_geometry_exit_2():
    from tools.compare_urdf import Report

    report = Report(dh_shape_delta={"a": np.array([np.nan, 0, 0, 0, 0, 0])})
    assert exit_code(report) == 2
