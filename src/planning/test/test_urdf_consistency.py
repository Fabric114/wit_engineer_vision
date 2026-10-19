# test_urdf_consistency.py — yaml 里从 URDF 派生的常量必须与 URDF 当场推导的值一致, 不同步即失败。
"""test_urdf_consistency.py — 守着 config/planning.yaml、config/sim.yaml 与 rm26_arm.urdf 同步。

被比对的量 (数值由 ``src/planning/tools/print_config.py`` 生成):
  - planning.yaml ``arm.dh`` 四组参数、``arm.joint_limits``;
  - planning.yaml ``collision.robot_body.meshes[*].arm_base_T_mesh`` (按 asset 文件名 = link 名比);
  - sim.yaml ``mount.line_link_T_arm_base`` / ``mount.frame6_T_link6``;
  - arm_base 约定本身: x 朝车头 (base_link +x), z 朝上。
``collision.ground.z_m`` / ``collision.line_m`` 不是 URDF 能推出的量, 不比对。

容差: 长度 / 矩阵元素 1e-6, 角度 1e-6 rad; alpha 例外放宽到 1e-5 rad —— URDF 里 J2 的
rpy roll 写成 3.1416 (π 截断, 多 7.3e-6), 推导出的 alpha[1] 比 config 的理想值 -π 偏 4.7e-6。
重导 URDF 后这里失败: 跑 print_config.py, 把输出替换进 yaml (alpha 保持理想值)。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planning import load_config  # noqa: E402
from tools.urdf_model import derive_arm_geometry, load_urdf  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
_TOL = 1e-6
_ALPHA_TOL = 1e-5


@pytest.fixture(scope="module")
def geometry():
    return derive_arm_geometry(load_urdf())


@pytest.fixture(scope="module")
def configs():
    with (REPO_ROOT / "config" / "sim.yaml").open(encoding="utf-8") as stream:
        sim_config = yaml.safe_load(stream)
    return load_config(), sim_config


# helper: 数组比对, 不一致时返回一条 "名字: 实际 vs 期望 (最大差)" 描述。
def _compare(name: str, actual, expected, tol: float) -> list[str]:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        return [f"{name}: 形状 {actual.shape} vs URDF {expected.shape}"]
    err = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if err > tol:
        return [f"{name}: yaml {np.round(actual, 6).tolist()} vs URDF {np.round(expected, 6).tolist()} (最大差 {err:.2e})"]
    return []


# 比较 yaml 与 URDF 派生值, 返回全部不一致项 (空列表 = 同步); 每项以所属区段名开头。
def _mismatches(config: dict, sim_config: dict, geometry) -> list[str]:
    out: list[str] = []
    dh = config["arm"]["dh"]
    for key in ("a", "alpha", "d", "theta_offset"):
        tol = _ALPHA_TOL if key == "alpha" else _TOL
        out += _compare(f"arm.dh.{key}", dh[key], getattr(geometry, key), tol)

    limits = config["arm"]["joint_limits"]
    out += _compare("arm.joint_limits.lower", limits["lower"], geometry.lower, _TOL)
    out += _compare("arm.joint_limits.upper", limits["upper"], geometry.upper, _TOL)

    mount = sim_config["mount"]
    out += _compare("mount.line_link_T_arm_base", np.reshape(mount["line_link_T_arm_base"], (4, 4)),
                    geometry.line_T_arm_base, _TOL)
    out += _compare("mount.frame6_T_link6", np.reshape(mount["frame6_T_link6"], (4, 4)),
                    geometry.frame6_T_link6, _TOL)

    for entry in config["collision"]["robot_body"]["meshes"]:
        link = Path(entry["asset"]).stem
        name = f"robot_body.{entry['asset']}"
        if link not in geometry.arm_base_T_link:
            out.append(f"{name}: URDF 里没有 link '{link}' (asset 文件名去掉扩展名必须是 link 名)")
            continue
        out += _compare(name, np.reshape(entry["arm_base_T_mesh"], (4, 4)), geometry.arm_base_T_link[link], _TOL)

    R = geometry.base_T_arm_base[:3, :3]
    if not R[:, 0] @ [1.0, 0.0, 0.0] > 1.0 - 1e-12:
        out.append(f"arm_base: x 轴不朝车头 (base_link +x), x = {R[:, 0]}")
    if not R[:, 2] @ [0.0, 0.0, 1.0] > 1.0 - 1e-6:
        out.append(f"arm_base: z 轴不朝上, z = {R[:, 2]}")
    return out


# helper: 取以 prefix 开头的不一致项。
def _section(configs, geometry, *prefixes: str) -> list[str]:
    return [m for m in _mismatches(*configs, geometry) if m.startswith(prefixes)]


# 期望: planning.yaml 的 arm.dh 四组参数 == URDF 派生值。
def test_planning_dh_matches_urdf(configs, geometry):
    assert not _section(configs, geometry, "arm.dh."), "\n".join(_section(configs, geometry, "arm.dh."))


# 期望: arm.joint_limits == URDF <limit>。
def test_joint_limits_match_urdf(configs, geometry):
    bad = _section(configs, geometry, "arm.joint_limits.")
    assert not bad, "\n".join(bad)


# 期望: sim.yaml 的 mount 两个安装常量 == URDF 派生值。
def test_sim_mount_matches_urdf(configs, geometry):
    bad = _section(configs, geometry, "mount.")
    assert not bad, "\n".join(bad)


# 期望: 每条 robot_body 网格都能按名字对上 URDF link, 位姿一致。
def test_robot_body_matches_urdf(configs, geometry):
    bad = _section(configs, geometry, "robot_body.")
    assert not bad, "\n".join(bad)


# 期望: arm_base x 朝车头、z 朝上。
def test_arm_base_faces_chassis_forward(configs, geometry):
    bad = _section(configs, geometry, "arm_base:")
    assert not bad, "\n".join(bad)


# 期望: 比较函数不是恒真 —— theta_offset[0] 改回旧值 0、网格名写错都会被报出。
def test_consistency_detects_stale_yaml(configs, geometry):
    config, sim_config = configs
    stale = copy.deepcopy(config)
    stale["arm"]["dh"]["theta_offset"][0] = 0.0
    stale["collision"]["robot_body"]["meshes"][0]["asset"] = "no_such_link.obj"
    found = _mismatches(stale, sim_config, geometry)
    assert any(m.startswith("arm.dh.theta_offset") for m in found), found
    assert any("no_such_link" in m for m in found), found


# 期望: URDF 里的虚拟系 arm_base / tcp 与规划 FK 一致 —— 任意 q 下
# arm_base_T_tcp (URDF 链) == ArmModel.forward_kinematics(q).tcp_transforms, 且 line 升降不影响。
# 容差用 _ALPHA_TOL: yaml DH 保留 6 位小数、alpha 取理想 π, 与 URDF 链实测差约 6e-6。
def test_urdf_virtual_frames_match_planning_fk(configs):
    from planning import ArmModel
    from tools.urdf_model import chain_frames

    config, _ = configs
    model = load_urdf()
    arm = ArmModel.from_config(config["arm"])
    names = ["J1", "J2", "J3", "J4", "J5", "J6"]
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = np.array([rng.uniform(model.joints[n].lower, model.joints[n].upper) for n in names])
        line = rng.uniform(model.joints["line"].lower, model.joints["line"].upper)
        frames = chain_frames(model, {**dict(zip(names, q)), "line": line})
        urdf_tcp = np.linalg.inv(frames["arm_base"]) @ frames["tcp"]
        fk_tcp = arm.forward_kinematics(q[None, :]).tcp_transforms[0]
        assert np.max(np.abs(urdf_tcp - fk_tcp)) < _ALPHA_TOL, (q, line, urdf_tcp, fk_tcp)
