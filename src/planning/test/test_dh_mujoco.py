"""test_dh_mujoco.py — 用 MuJoCo 把 config 里的 rm26_arm DH 表验证一遍 (黄金检查)。

思路 (为什么这样验)
-------------------
- MuJoCo 从 ``rm26_arm.xml`` (MJCF) 加载, 它是与 URDF 独立的另一份模型文件, 拿它
  当"标准答案": 给一组关节角, 让 MuJoCo 正向求解, 读 ``link_6`` 相对 ``line_link``
  的真实位姿。
- 被测对象: ``config/planning.example.yaml`` 里的 ``arm.dh``, 经
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


def _find_mjcf() -> Path | None:
    for cand in _MJCF_CANDIDATES:
        if cand and Path(cand).is_file():
            return Path(cand).resolve()
    return None


# --- URDF 关节 origin (相对父连杆): (xyz, rpy), 与 rm26_arm.urdf 一致 ---
# 只用于算 F0 / C 两个安装常量, 不参与被测的 DH 参数。
_JOINTS = [
    ((0.0, 0.0, -0.106999999999882), (0.0, 0.0, 0.0)),
    ((0.1232, 0.14596, -0.0545), (3.1416, 0.0, 0.0)),
    ((0.000655997394275049, 0.0989978265792674, 0.0), (1.5707963267949, 0.0, -0.00662628479646784)),
    ((-0.000499676236049573, -1.79905288163473e-05, -0.0465000000000025),
     (-1.5707963267949, 0.182122404429995, -1.53480750090966)),
    ((0.0145369266962608, 0.0817170591873439, 0.0), (1.57079632679489, -1.53476796314872, -0.176051717388886)),
    ((-0.0429147465036577, -0.00270638735710615, -0.040500000000001),
     (-1.5707963267949, 0.101460063365565, -1.50781545764258)),
]
_AXIS_LOCAL = np.array([0.0, 0.0, -1.0])  # 各关节正转方向 (URDF 里 axis 全是 0 0 -1)

def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    """URDF 约定: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _T(xyz, rpy) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = _rpy_to_R(*rpy)
    M[:3, 3] = xyz
    return M


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4)
    M[0, 0] = c; M[0, 1] = -s; M[1, 0] = s; M[1, 1] = c
    return M


def _urdf_frames(q: np.ndarray) -> list[np.ndarray]:
    """返回 A[0..6]: A[0]=line_link(=I), A[i]=line_link->关节 i 坐标系 (含转角 q_i)。"""
    A = [np.eye(4)]
    cur = np.eye(4)
    for i in range(6):
        cur = cur @ _T(_JOINTS[i][0], _JOINTS[i][1]) @ _rotz(-q[i])  # 正转方向为 (0,0,-1)
        A.append(cur.copy())
    return A


def _closest_pts(p1, z1, p2, z2):
    """两条空间直线的最近点对; 平行时返回 None。"""
    w0 = p1 - p2
    b = float(np.dot(z1, z2)); d = float(np.dot(z1, w0)); e = float(np.dot(z2, w0))
    den = 1.0 - b * b
    if abs(den) < 1e-9:
        return None
    s = (b * e - d) / den
    t = (e - b * d) / den
    return p1 + s * z1, p2 + t * z2


def _build_mount_constants() -> tuple[np.ndarray, np.ndarray]:
    """从 URDF 几何算出安装常量 F0 (line_link->arm_base) 与 C (DH第6系->link_6)。

    只用 URDF 关节 origin, 不读 config, 因此不会掩盖 config 里 DH 的错误。
    """
    A0 = _urdf_frames(np.zeros(6))
    P = [A0[i][:3, 3].copy() for i in range(1, 7)]
    Z = [A0[i][:3, :3] @ _AXIS_LOCAL for i in range(1, 7)]
    Z = [z / np.linalg.norm(z) for z in Z]

    # 逐关节按公垂线定 x 轴、原点, 建 DH 坐标系 frames[1..6]
    X = [None] * 7
    O = [None] * 7
    for i in range(5):  # frame 1..5
        cp = _closest_pts(P[i], Z[i], P[i + 1], Z[i + 1])
        if cp is None:  # 平行: x 沿公共法向
            diff = P[i + 1] - P[i]
            diff = diff - np.dot(diff, Z[i]) * Z[i]
            X[i + 1] = diff / np.linalg.norm(diff)
            O[i + 1] = P[i].copy()
        else:
            ci, cip1 = cp
            diff = cip1 - ci
            if np.linalg.norm(diff) > 1e-7:
                X[i + 1] = diff / np.linalg.norm(diff)
            else:  # 相交: x = z_i × z_{i+1}
                x = np.cross(Z[i], Z[i + 1])
                X[i + 1] = x / np.linalg.norm(x)
            O[i + 1] = ci
    # frame 6: x 轴取 URDF link_6 的 x 投影到垂直 z6 的平面
    u = A0[6][:3, 0]
    x6 = u - np.dot(u, Z[5]) * Z[5]
    X[6] = x6 / np.linalg.norm(x6)
    O[6] = P[5].copy()

    frames = [None] * 7
    for i in range(1, 7):
        zi = Z[i - 1]; xi = X[i]; yi = np.cross(zi, xi)
        F = np.eye(4)
        F[:3, 0] = xi; F[:3, 1] = yi; F[:3, 2] = zi; F[:3, 3] = O[i]
        frames[i] = F

    F0 = frames[1].copy()                       # line_link -> arm_base
    C = np.linalg.inv(frames[6]) @ A0[6]        # DH 第 6 系 -> URDF link_6 法兰
    return F0, C


def _body_T(model, data, name: str) -> np.ndarray:
    import mujoco
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"MJCF 里找不到 body: {name}")
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3, 3] = data.xpos[bid]
    return T


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra.T @ Rb
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def _run_check(n_samples: int = 5000, seed: int = 0, verbose: bool = False):
    """核心校验: 返回 (max_pos_err_m, max_rot_err_deg, F0, C)。"""
    import mujoco

    mjcf = _find_mjcf()
    assert mjcf is not None, "找不到 rm26_arm.xml (可设环境变量 RM26_MJCF 指向它)"

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

@pytest.fixture(scope="module")
def _mujoco():
    return pytest.importorskip("mujoco")


def test_mjcf_available(_mujoco):
    """能定位到 MJCF 且 MuJoCo 能加载它 (否则本文件其余用例无意义)。"""
    mjcf = _find_mjcf()
    if mjcf is None:
        pytest.skip("找不到 rm26_arm.xml, 设 RM26_MJCF 后再跑")
    _mujoco.MjModel.from_xml_path(str(mjcf))


def test_config_dh_matches_mujoco(_mujoco):
    """config 的 DH 经 ArmModel FK, 应复现 MuJoCo 的 link_6 位姿 (mm / 毫度级)。"""
    if _find_mjcf() is None:
        pytest.skip("找不到 rm26_arm.xml, 设 RM26_MJCF 后再跑")
    max_pos, max_rot, _, _ = _run_check(n_samples=2000, seed=0, verbose=False)
    assert max_pos < 1e-3, f"最大位置误差 {max_pos*1000:.4f} mm 过大 -> config 的 DH 可能填错"
    assert max_rot < 0.1, f"最大姿态误差 {max_rot:.5f}° 过大 -> config 的 DH 可能填错"


if __name__ == "__main__":
    mp, mr, _, _ = _run_check(n_samples=5000, seed=0, verbose=True)
    ok = mp < 1e-3 and mr < 0.1
    print("\n结论:", "✅ DH 表与 MuJoCo 一致" if ok else "❌ 对不上, 检查 config 的 arm.dh")

