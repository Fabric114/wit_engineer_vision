# test_urdf_model.py — 校验 tools.urdf_model 的 URDF 解析与 arm_base 新定义 (x 朝车头)。
"""test_urdf_model.py — tools/urdf_model.py 的单元测试。

- 解析: J1..J6 全在、类型 revolute、axis = (0, 0, -1)、限位与 URDF 一致;
- 链式 FK: q = 0 时各 link 位姿与 MuJoCo (MJCF, 独立的另一份模型) 一致;
- arm_base: 正交、z 朝上、x = base_link +x 投影、原点不变;
- DH: a / alpha / d / theta_offset[1:] 与当前 config 一致 (本次不改), theta_offset[0] 约 -0.8698 rad。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planning import ArmModel, load_config  # noqa: E402
from tools.urdf_model import (  # noqa: E402
    ARM_JOINTS,
    UrdfModel,
    DEFAULT_URDF,
    chain_frames,
    derive_arm_geometry,
    load_urdf,
    rpy_to_matrix,
)

_MJCF = DEFAULT_URDF.with_name("rm26_arm.xml")

# alpha 的容差放宽到 1e-5: URDF 里 J2 的 rpy roll 写成 3.1416 (π 截断, 多 7.3e-6),
# 投影到公垂线后 alpha[1] 比 config 的 -π 偏 4.7e-6; config 保持理想值 -π。
_ALPHA_TOL = 1e-5


# fixture: 解析一次默认 URDF 并推导几何, 供本文件各用例共用。
@pytest.fixture(scope="module")
def model():
    return load_urdf()


@pytest.fixture(scope="module")
def geometry(model):
    return derive_arm_geometry(model)


# helper: 用派生的 DH 替换 config 里的 dh, 构造 ArmModel。
def _arm_with_dh(geometry) -> ArmModel:
    arm_cfg = copy.deepcopy(load_config()["arm"])
    for key in ("a", "alpha", "d", "theta_offset"):
        arm_cfg["dh"][key] = getattr(geometry, key).tolist()
    return ArmModel.from_config(arm_cfg)


# 期望: 6 个臂关节都在, revolute, 轴向 (0, 0, -1)。
def test_load_urdf_has_arm_joints(model):
    for name in ARM_JOINTS:
        assert name in model.joints, f"URDF 缺关节 {name}"
        joint = model.joints[name]
        assert joint.type == "revolute"
        assert np.allclose(joint.axis, [0.0, 0.0, -1.0])


# 期望: rpy 约定 R = Rz(yaw)·Ry(pitch)·Rx(roll)。
def test_rpy_to_matrix_convention():
    roll, pitch, yaw = 0.3, -0.2, 1.1
    rx = rpy_to_matrix(roll, 0.0, 0.0)
    ry = rpy_to_matrix(0.0, pitch, 0.0)
    rz = rpy_to_matrix(0.0, 0.0, yaw)
    assert np.allclose(rpy_to_matrix(roll, pitch, yaw), rz @ ry @ rx, atol=1e-12)
    assert np.allclose(rx[1:, 1:], [[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])


# 期望: q = 0 时 link_1..link_6、line_link 相对 base_link 的位姿与 MuJoCo 同名 body 相对 root 一致。
def test_chain_frames_match_mujoco(model):
    mujoco = pytest.importorskip("mujoco")
    mj_model = mujoco.MjModel.from_xml_path(str(_MJCF))
    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)
    mujoco.mj_forward(mj_model, data)

    def body_T(name):
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        T = np.eye(4)
        T[:3, :3] = data.xmat[bid].reshape(3, 3)
        T[:3, 3] = data.xpos[bid]
        return T

    frames = chain_frames(model)
    root = np.linalg.inv(body_T("root"))
    for name in ("line_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"):
        meas = root @ body_T(name)
        assert np.allclose(frames[name], meas, atol=1e-6), f"{name} 与 MuJoCo 不一致:\n{frames[name]}\n{meas}"


# 期望: arm_base 正交、右手; z 朝上; x = base_link +x; 原点约 (0.0525, 0.106, 0.323)。
def test_derived_arm_base_is_orthonormal(geometry):
    R = geometry.base_T_arm_base[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(R) - 1.0) < 1e-9
    assert np.arccos(np.clip(R[:, 2] @ [0.0, 0.0, 1.0], -1.0, 1.0)) < 1e-3
    assert np.arccos(np.clip(R[:, 0] @ [1.0, 0.0, 0.0], -1.0, 1.0)) < 1e-6
    assert np.allclose(geometry.base_T_arm_base[:3, 3], [0.0525, 0.106, 0.323], atol=1e-4)


# 期望: a / alpha / d / theta_offset[1:] 与当前 config 一致 (本次不改)。
def test_dh_shape_matches_config(geometry):
    dh = load_config()["arm"]["dh"]
    assert np.allclose(geometry.a, dh["a"], atol=1e-6, rtol=0), f"a: {geometry.a} vs {dh['a']}"
    assert np.allclose(geometry.alpha, dh["alpha"], atol=_ALPHA_TOL, rtol=0), (
        f"alpha: {geometry.alpha} vs {dh['alpha']}"
    )
    assert np.allclose(geometry.d, dh["d"], atol=1e-6, rtol=0), f"d: {geometry.d} vs {dh['d']}"
    assert np.allclose(geometry.theta_offset[1:], dh["theta_offset"][1:], atol=1e-6, rtol=0), (
        f"theta_offset[1:]: {geometry.theta_offset[1:]} vs {dh['theta_offset'][1:]}"
    )


# 期望: theta_offset[0] ∈ (-π, π], 约 -0.8698; Rz(theta_offset[0]) 把 arm_base x 转到 DH 第 1 系 x。
def test_theta_offset0_in_range_and_matches_frame1(geometry):
    t0 = float(geometry.theta_offset[0])
    assert -np.pi < t0 <= np.pi
    assert abs(t0 - (-0.8698)) < 1e-3, f"theta_offset[0] = {t0}"

    frame1 = _arm_with_dh(geometry).forward_kinematics(np.zeros((1, 6))).link_transforms[0, 1]
    c, s = np.cos(t0), np.sin(t0)
    assert np.allclose(np.array([c, s, 0.0]), frame1[:3, 0], atol=1e-9)
    # 第 1 系在 base_link 下的 x 与按关节轴重建的 DH 第 1 系 x 一致
    x1_base = geometry.base_T_arm_base[:3, :3] @ frame1[:3, 0]
    assert np.allclose(x1_base, geometry.base_T_dh[1][:3, 0], atol=1e-9)


# 期望: 派生的关节限位 == URDF <limit>。
def test_limits_match_urdf(model, geometry):
    lower = [model.joints[n].lower for n in ARM_JOINTS]
    upper = [model.joints[n].upper for n in ARM_JOINTS]
    assert np.allclose(geometry.lower, lower) and np.allclose(geometry.upper, upper)
    assert np.allclose(geometry.lower, [-0.52, -0.52, -3.14, 0.0, -1.39, -1.57])


# 期望: J2 roll 写成 3.1417 / 3.142 (π 截断到 3~4 位, CAD 导出常见) 时仍按 J1∥J2 处理, 几何不发散。
@pytest.mark.parametrize("roll", [3.1417, 3.142])
def test_near_parallel_j1_j2_stays_finite(model, roll):
    joints = dict(model.joints)
    j2 = joints["J2"]
    origin = j2.origin.copy()
    origin[:3, :3] = rpy_to_matrix(roll, 0.0, 0.0)
    joints["J2"] = type(j2)(**{**j2.__dict__, "origin": origin})
    g = derive_arm_geometry(UrdfModel(joints=joints, links=model.links))
    assert np.allclose(g.base_T_arm_base[:3, 3], [0.0525, 0.106, 0.323], atol=1e-4)
    assert np.max(np.abs(g.d)) < 0.2 and np.max(np.abs(g.a)) < 0.3, (g.a, g.d)
