import numpy as np
import pytest

from types import SimpleNamespace

from solver.extrinsic import matrix_from_pose
from solver.pnp_node import (
    _FrameBuffer,
    _camera_info_to_arrays,
    _config_dir,
    _fill_pose,
    _fill_transform,
    _load_extrinsic_config,
    _matrix_from_transform_msg,
    _observation_to_arrays,
)


class Header:
    frame_id = "camera"


class CameraInfo:
    k = list(range(9))
    d = [0.1, 0.2, 0.3, 0.4, 0.5]
    header = Header()


class Observation:
    schema = "exchange"
    class_id = 3
    u = [10.0, 20.0]
    v = [11.0, 21.0]
    confidence = [0.9, 0.8]


def test_camera_info_conversion():
    K, D = _camera_info_to_arrays(CameraInfo())
    assert K.shape == (3, 3)
    assert D.shape == (5,)
    np.testing.assert_allclose(K[0], [0, 1, 2])


def test_observation_conversion_preserves_parallel_arrays():
    schema, class_id, points, confidence = _observation_to_arrays(Observation())
    assert schema == "exchange"
    assert class_id == 3
    assert points.shape == (2, 2)
    assert confidence.shape == (2,)
    np.testing.assert_allclose(points[:, 0], [10.0, 20.0])


@pytest.mark.parametrize("field", ["v", "confidence"])
def test_observation_conversion_rejects_mismatched_arrays(field):
    message = Observation()
    setattr(message, field, [1.0])
    with pytest.raises(ValueError, match="parallel"):
        _observation_to_arrays(message)


def test_frame_buffer_releases_frame_when_newer_stamp_arrives():
    buffer = _FrameBuffer(timeout_s=0.05)
    assert buffer.add((1, 0), "a", now=0.00) == []
    assert buffer.add((1, 0), "b", now=0.01) == []
    assert buffer.add((2, 0), "c", now=0.02) == [((1, 0), ["a", "b"])]
    assert buffer.flush(now=0.03) == []
    assert buffer.flush(now=0.08) == [((2, 0), ["c"])]


def test_frame_buffer_drops_late_message_of_released_frame():
    buffer = _FrameBuffer(timeout_s=0.05)
    buffer.add((2, 0), "new", now=0.0)
    assert buffer.add((1, 0), "late", now=0.01) == []
    assert buffer.flush(now=1.0) == [((2, 0), ["new"])]


def test_camera_info_conversion_rejects_bad_intrinsics():
    message = CameraInfo()
    message.k = [1.0] * 8
    with pytest.raises(ValueError, match="K"):
        _camera_info_to_arrays(message)


def _vec():
    return SimpleNamespace(x=0.0, y=0.0, z=0.0)


def _quat():
    return SimpleNamespace(w=0.0, x=0.0, y=0.0, z=0.0)


def test_pose_and_transform_fill_by_field_name_round_trip():
    q = np.array([0.7, -0.1, 0.5, 0.3])
    T = matrix_from_pose([0.4, -0.1, 0.25], q / np.linalg.norm(q))

    pose = SimpleNamespace(position=_vec(), orientation=_quat())
    _fill_pose(pose, T)
    back = matrix_from_pose((pose.position.x, pose.position.y, pose.position.z),
                            (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z))
    np.testing.assert_allclose(back, T, atol=1e-9)

    transform = SimpleNamespace(translation=_vec(), rotation=_quat())
    _fill_transform(transform, T)
    np.testing.assert_allclose(_matrix_from_transform_msg(transform), T, atol=1e-9)


def test_repo_extrinsic_config_has_frames():
    cfg, T = _load_extrinsic_config(_config_dir() / "camera_extrinsic.yaml")
    assert (cfg["camera_frame"], cfg["base_frame"], cfg["station_frame"]) == ("camera", "arm_base", "exchange_station")
    assert T.shape == (4, 4)


def test_extrinsic_config_requires_frames(tmp_path):
    path = tmp_path / "extrinsic.yaml"
    path.write_text("arm_base_from_camera: {rows: 4, cols: 4, data: [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]}\n"
                    "camera_frame: camera\nbase_frame: arm_base\n", encoding="utf-8")
    with pytest.raises(ValueError, match="station_frame"):
        _load_extrinsic_config(path)
