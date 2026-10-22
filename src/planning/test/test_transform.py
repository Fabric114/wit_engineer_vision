# test_transform.py — 锁死四元数顺序: 全库算法层统一 [w, x, y, z]。
"""test_transform.py — 四元数 <-> 旋转矩阵的黄金检查 (顺序 / 往返 / 校验)。

为什么单独测顺序
----------------
同一组四个数, 按 ``[w, x, y, z]`` 还是 ``[x, y, z, w]`` 读, 两种解释都是合法的
单位四元数: 不报错、形状对、转出来的矩阵也正交, 只是姿态悄悄错了。
所以期望值必须来自**手算的旋转矩阵**, 不能用被测函数转过去再转回来 ——
顺序写错时往返依然自洽, 测试会全部通过。

数字钉死两条:
- 绕 z 转 +90° -> ``[w, x, y, z] = [√½, 0, 0, √½]``;
- 绕 x 转 +90° -> ``[w, x, y, z] = [√½, √½, 0, 0]``。
把下标 0 当成 x 的实现会在这两条上得到绕另一根轴的旋转, 直接失败。

跑法::

    PYTHONPATH=src/planning PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        python3 -m pytest src/planning/test/test_transform.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from planning import quaternions_from_rotations, rotations_from_quaternions

_SQRT_HALF = float(np.sqrt(0.5))


def _rotation_about(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    """用 Rodrigues 公式手算旋转矩阵 (不经过 scipy, 保证期望值独立于被测实现)。"""
    axis_array = np.asarray(axis, dtype=float)
    axis_array = axis_array / np.linalg.norm(axis_array)
    skew = np.array([
        [0.0, -axis_array[2], axis_array[1]],
        [axis_array[2], 0.0, -axis_array[0]],
        [-axis_array[1], axis_array[0], 0.0],
    ])
    return np.eye(3) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)


def test_identity_quaternion_is_identity_rotation():
    """``[1, 0, 0, 0]`` 必须是单位旋转。"""
    quaternion = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert np.allclose(rotations_from_quaternions(quaternion)[0], np.eye(3), atol=1e-12)


def test_w_sits_at_index_zero():
    """绕 z 转 90° 的 ``[w, x, y, z] = [√½, 0, 0, √½]``; 按下标 0 当 x 读会变成绕 x 转。"""
    quaternion = np.array([[_SQRT_HALF, 0.0, 0.0, _SQRT_HALF]])
    expected = _rotation_about((0.0, 0.0, 1.0), np.pi / 2.0)
    assert np.allclose(rotations_from_quaternions(quaternion)[0], expected, atol=1e-12)


def test_x_sits_at_index_one():
    """绕 x 转 90° 的 ``[w, x, y, z] = [√½, √½, 0, 0]``。"""
    quaternion = np.array([[_SQRT_HALF, _SQRT_HALF, 0.0, 0.0]])
    expected = _rotation_about((1.0, 0.0, 0.0), np.pi / 2.0)
    assert np.allclose(rotations_from_quaternions(quaternion)[0], expected, atol=1e-12)


def test_roundtrip_preserves_rotation():
    """旋转 -> 四元数 -> 旋转 必须回到原矩阵, 且输出是 ``(B, 4)``。"""
    rotations = np.stack([
        _rotation_about((0.0, 0.0, 1.0), np.pi / 2.0),
        _rotation_about((1.0, 0.0, 0.0), -np.pi / 3.0),
        _rotation_about((1.0, 2.0, 3.0), 0.7),
    ])
    quaternions = quaternions_from_rotations(rotations)
    assert quaternions.shape == (3, 4)
    assert np.allclose(rotations_from_quaternions(quaternions), rotations, atol=1e-12)


def test_antipodal_quaternions_are_the_same_rotation():
    """``q`` 和 ``-q`` 描述同一个姿态 (四元数双覆盖)。"""
    quaternion = np.array([[_SQRT_HALF, 0.0, 0.0, _SQRT_HALF]])
    assert np.allclose(
        rotations_from_quaternions(quaternion),
        rotations_from_quaternions(-quaternion),
        atol=1e-12,
    )


def test_rejects_malformed_quaternions():
    """单个 ``(4,)`` 不带批量维、零四元数都必须在入口报错, 不许静默归一化。"""
    with pytest.raises(ValueError, match=r"shape \(B, 4\)"):
        rotations_from_quaternions(np.array([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="non-zero"):
        rotations_from_quaternions(np.zeros((1, 4)))
