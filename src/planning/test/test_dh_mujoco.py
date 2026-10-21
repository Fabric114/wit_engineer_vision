# test_dh_mujoco.py — 以 MuJoCo (MJCF) 为真值, 校验 config 的 DH 参数与关节限位。
"""test_dh_mujoco.py — 用 MuJoCo 把 config 里的 rm26_arm DH 表验证一遍 (黄金检查)。

思路 (为什么这样验)
-------------------
- MuJoCo 从 ``rm26_arm.xml`` (MJCF) 加载, 它是与 URDF 独立的另一份模型文件, 拿它
  当"标准答案": 给一组关节角, 让 MuJoCo 正向求解, 读 ``link_6`` 相对 ``line_link``
  的真实位姿。
- 被测对象: 仓库根 ``config/planning.yaml`` 里的 ``arm.dh`` (经 ``load_config()`` 读取), 经
  ``ArmModel.forward_kinematics`` 算出的位姿。 arm_model 的 FK 在 arm_base 系;
  要和 MuJoCo 的 link_6 对上, 需要两个与"DH 参数"无关、只描述"DH 链怎么挂到机器人上"
  的安装常量:
    * ``F0`` = line_link -> arm_base  (base 装在 J1 轴上, 常量)
    * ``C``  = DH 第 6 系 -> link_6 法兰 (差一个绕 x 180°, 常量)
  这两个常量从 URDF 几何算出 (不读 config), 所以它们不会掩盖 config 里 DH 的错误。
- 校验等式:   ``line_link_T_link6(q)  ==  F0 @ ArmModelFK_frame6(q) @ C``
  左边来自 MuJoCo (独立真值), 右边的 ``ArmModelFK_frame6`` 来自 config 的 DH。
  config 的 DH 一旦填错, 右边就对不上 -> 断言失败。

怎么跑
------
  单跑详细报告:  ``PYTHONPATH=. python3 test/test_dh_mujoco.py``
  随 pytest:     ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest test/test_dh_mujoco.py -v -s``
没装 mujoco 或找不到 MJCF 时, 该用例自动 skip。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from planning import ArmModel, load_config

# --- 定位 MJCF: 环境变量优先, 否则按相对路径找, 找不到就 skip ---
_HERE = Path(__file__).resolve().parent
_MJCF_CANDIDATES = [
    os.environ.get("RM26_MJCF"),
    _HERE / ".." / ".." / "runtime" / "sim" / "model" / "rm26_engineer_description" / "rm26_arm.xml",
]


# helper: 返回第一个存在的 MJCF 路径, 都找不到返回 None。
def _find_mjcf() -> Path | None:
    for cand in _MJCF_CANDIDATES:
        if cand and Path(cand).is_file():
            return Path(cand).resolve()
    return None


# --- URDF 几何: 全部委托 tools.urdf_model (直接解析 rm26_arm.urdf, 不再手抄关节 origin) ---
# 只用于算 F0 / C 两个安装常量, 不读 config, 因此不会掩盖 config 里 DH 的错误。
sys.path.insert(0, str(_HERE.parent))
from tools.urdf_model import ARM_JOINTS, chain_frames, derive_arm_geometry, load_urdf  # noqa: E402


# helper: 沿 URDF 关节链做 FK, 得各 link 在 line_link 下的位姿。
def _urdf_frames(q: np.ndarray) -> list[np.ndarray]:
    """返回 A[0..6]: A[0]=line_link(=I), A[i]=line_link->link_i (含转角 q_i, line = 0)。"""
    frames = chain_frames(load_urdf(), dict(zip(ARM_JOINTS, map(float, q))))
    inv_line = np.linalg.inv(frames["line_link"])
    return [np.eye(4)] + [inv_line @ frames[f"link_{i}"] for i in range(1, 7)]


# helper: 从 URDF 几何算出安装常量 F0 (line_link->arm_base) 与 C (DH第6系->link_6)。
def _build_mount_constants() -> tuple[np.ndarray, np.ndarray]:
    """arm_base 的定义 (x 朝车头) 与推导见 tools/urdf_model.py::derive_arm_geometry。"""
    geometry = derive_arm_geometry(load_urdf())
    return geometry.line_T_arm_base, geometry.frame6_T_link6


# helper: 读 MuJoCo body 的世界位姿 (xmat/xpos), 拼成 4x4。
def _body_T(model, data, name: str) -> np.ndarray:
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"MJCF 里找不到 body: {name}")
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3, 3] = data.xpos[bid]
    return T


# helper: 两旋转矩阵之间的夹角 (度)。
def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra.T @ Rb
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


# helper: 在 MJCF 限位内随机采样 q, 比较 MuJoCo 实测 line_link->link_6 与 F0 @ FK_6(q) @ C。
def _run_check(n_samples: int = 5000, seed: int = 0, verbose: bool = False):
    """核心校验: 返回 (max_pos_err_m, max_rot_err_deg, F0, C)。"""
    import mujoco

    mjcf = _find_mjcf()
    assert mjcf is not None, "找不到 rm26_arm.xml (可设环境变量 RM26_MJCF 指向它)"

    # 加载 MuJoCo 模型, 准备安装常量与被测 ArmModel
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)

    F0, C = _build_mount_constants()
    arm = ArmModel.from_config(load_config()["arm"])

    # J1..J6 的 qpos 地址与限位, 直接从 MJCF 读 (不依赖 config 限位是否已改)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"J{k}") for k in range(1, 7)]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    lo = np.array([model.jnt_range[j, 0] for j in jids])
    hi = np.array([model.jnt_range[j, 1] for j in jids])
    line_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "line")
    line_adr = model.jnt_qposadr[line_id]

    # 逐样本: MuJoCo 正解得实测位姿, 与 DH 预测比较, 记录最大位置/姿态误差
    rng = np.random.default_rng(seed)
    max_pos = 0.0
    max_rot = 0.0
    worst_q = None
    for _ in range(n_samples):
        q = rng.uniform(lo, hi)
        mujoco.mj_resetData(model, data)      # 恢复合法默认 qpos (含 free joint)
        data.qpos[qadr] = q
        data.qpos[line_adr] = 0.0             # line 在 line_link 上游, 不影响相对位姿
        mujoco.mj_forward(model, data)

        meas = np.linalg.inv(_body_T(model, data, "line_link")) @ _body_T(model, data, "link_6")

        frame6 = arm.forward_kinematics(q[None, :]).link_transforms[0, 6]  # arm_base -> 第6系
        pred = F0 @ frame6 @ C

        pos = float(np.linalg.norm(pred[:3, 3] - meas[:3, 3]))
        rot = _rot_angle_deg(pred[:3, :3], meas[:3, :3])
        if pos > max_pos:
            max_pos = pos; worst_q = q
        max_rot = max(max_rot, rot)

    # verbose: 打印安装常量与误差报告 (单跑模式用)
    if verbose:
        ang = _rot_angle_deg(np.eye(3), C[:3, :3])
        print(f"\nMJCF          : {mjcf}")
        print(f"安装常量 F0 (line_link->arm_base):\n{np.round(F0, 6)}")
        print(f"安装常量 C  (DH第6系->link_6): 旋转 {ang:.3f}°, 平移(mm) {np.round(C[:3,3]*1000,3)}")
        print(f"\n随机 {n_samples} 组关节角 (取自 MJCF 限位) 回代:")
        print(f"  最大位置误差: {max_pos*1000:.4f} mm")
        print(f"  最大姿态误差: {max_rot:.5f} °")
        print(f"  最差位置对应 q = {np.round(worst_q, 4)}")
    return max_pos, max_rot, F0, C


# ===================== pytest 用例 =====================

# fixture: 没装 mujoco 则 skip 依赖它的用例, 否则返回 mujoco 模块。
@pytest.fixture(scope="module")
def _mujoco():
    return pytest.importorskip("mujoco")


# 期望: MJCF 可定位且能被 MuJoCo 加载。
def test_mjcf_available(_mujoco):
    """能定位到 MJCF 且 MuJoCo 能加载它 (否则本文件其余用例无意义)。"""
    mjcf = _find_mjcf()
    if mjcf is None:
        pytest.skip("找不到 rm26_arm.xml, 设 RM26_MJCF 后再跑")
    _mujoco.MjModel.from_xml_path(str(mjcf))


# 期望: 2000 组随机关节角下, 位置误差 < 1 mm, 姿态误差 < 0.1°。
def test_config_dh_matches_mujoco(_mujoco):
    """config 的 DH 经 ArmModel FK, 应复现 MuJoCo 的 link_6 位姿 (mm / 毫度级)。"""
    if _find_mjcf() is None:
        pytest.skip("找不到 rm26_arm.xml, 设 RM26_MJCF 后再跑")
    max_pos, max_rot, _, _ = _run_check(n_samples=2000, seed=0, verbose=False)
    assert max_pos < 1e-3, f"最大位置误差 {max_pos*1000:.4f} mm 过大 -> config 的 DH 可能填错"
    assert max_rot < 0.1, f"最大姿态误差 {max_rot:.5f}° 过大 -> config 的 DH 可能填错"


# 期望: 6 轴、无连续关节, lower/upper 与 MJCF jnt_range 逐轴一致 (atol 1e-9)。
def test_config_joint_limits_match_mujoco(_mujoco):
    """config 的 arm.joint_limits 应逐轴等于 MJCF 的 jnt_range, 且不含连续关节。

    为什么要卡死这一条:
      - 配置比 MJCF 宽 -> 规划出的点会被 arm_plugin 按模型悄悄夹掉, 轨迹跟不上;
      - 配置比 MJCF 窄 -> 白丢可行解;
      - 写成 null (连续关节) -> joint_space 会把该轴折到 [-pi, pi] 并允许整圈
        unwrap 衔接, 规划器以为能绕过去, 实际会撞机械死点。 本臂 6 轴在 URDF 里
        都是 revolute、在 MJCF 里都是 limited="true", 一个连续关节都没有。
    """
    mjcf = _find_mjcf()
    if mjcf is None:
        pytest.skip("找不到 rm26_arm.xml, 设 RM26_MJCF 后再跑")

    # 真值: 从 MJCF 读 J1..J6 的 jnt_range
    model = _mujoco.MjModel.from_xml_path(str(mjcf))
    jids = [_mujoco.mj_name2id(model, _mujoco.mjtObj.mjOBJ_JOINT, f"J{k}") for k in range(1, 7)]
    lo = np.array([model.jnt_range[j, 0] for j in jids])
    hi = np.array([model.jnt_range[j, 1] for j in jids])

    # 被测: config 解析后的 joint_space
    space = ArmModel.from_config(load_config()["arm"]).joint_space
    assert space.dof == 6, f"joint_limits 长度应为 6, 实际 {space.dof}"
    assert not space.continuous.any(), (
        f"本臂没有连续关节, joint_limits 里不该出现 null: continuous={space.continuous}"
    )
    assert np.allclose(space.lower, lo, atol=1e-9), f"lower 与 MJCF 不一致: {space.lower} vs {lo}"
    assert np.allclose(space.upper, hi, atol=1e-9), f"upper 与 MJCF 不一致: {space.upper} vs {hi}"


# 单跑: 5000 组样本的详细报告并打印结论。
if __name__ == "__main__":
    mp, mr, _, _ = _run_check(n_samples=5000, seed=0, verbose=True)
    ok = mp < 1e-3 and mr < 0.1
    print("\n结论:", "✅ DH 表与 MuJoCo 一致" if ok else "❌ 对不上, 检查 config 的 arm.dh")

