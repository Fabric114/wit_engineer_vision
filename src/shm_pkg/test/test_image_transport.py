"""shm_pkg 纯算法测试: 同进程内开一个 Publisher + 一个 Subscriber, 映射同一块
/dev/shm 区域, 验证收发一致性、三缓冲零拷贝语义、心跳判活与魔数校验。"""

import numpy as np
import pytest

from shm_pkg import ImagePublisher, ImageSubscriber, ShmRegion
from shm_pkg import layout


REGION = "wit_engineer_test_region"  # 测试专用名, 避免撞到真实相机区域


@pytest.fixture
def region_name():
    """每个测试前后清掉残留的 /dev/shm 文件, 保证干净。"""
    ShmRegion.path_for(REGION).unlink(missing_ok=True)
    yield REGION
    ShmRegion.path_for(REGION).unlink(missing_ok=True)


def _make_frame(h, w, val):
    img = np.full((h, w, 3), val, dtype=np.uint8)
    img[0, 0] = (val, val + 1, val + 2)  # 一个可辨识的角点, 防止全同值蒙混
    return img


def test_roundtrip_single_frame(region_name):
    with ImagePublisher(region_name, 720, 960, 3) as pub:
        with ImageSubscriber(region_name) as sub:
            assert sub.try_recv() is None  # 还没发布
            sent = _make_frame(720, 960, 42)
            seq = pub.publish(sent)
            assert seq == 1

            frame = sub.try_recv()
            assert frame is not None
            assert frame.seq == 1
            assert frame.image.shape == (720, 960, 3)
            np.testing.assert_array_equal(frame.image, sent)


def test_no_new_frame_returns_none(region_name):
    with ImagePublisher(region_name) as pub, ImageSubscriber(region_name) as sub:
        pub.publish(_make_frame(100, 100, 7))
        assert sub.try_recv() is not None
        assert sub.try_recv() is None  # 同一帧不会被读第二次


def test_smaller_frame_than_max(region_name):
    """实际帧比槽位上限小时应正常收发, 用 cur_* 记录真实尺寸。"""
    with ImagePublisher(region_name, 720, 960, 3) as pub, ImageSubscriber(region_name) as sub:
        sent = _make_frame(480, 640, 200)
        pub.publish(sent)
        frame = sub.try_recv()
        assert frame is not None
        assert frame.image.shape == (480, 640, 3)
        np.testing.assert_array_equal(frame.image, sent)


def test_latest_wins_across_slots(region_name):
    """连发多帧, 读者拿到的应是最新一帧 (三缓冲轮转后仍正确)。"""
    with ImagePublisher(region_name, 100, 100, 3) as pub, ImageSubscriber(region_name) as sub:
        for i in range(1, 6):  # 超过 n_slots=3, 触发轮转
            pub.publish(_make_frame(100, 100, i * 10))
        frame = sub.try_recv()
        assert frame is not None
        assert frame.seq == 5
        assert int(frame.image[10, 10, 0]) == 50


def test_zero_copy_view_vs_copy(region_name):
    """默认零拷贝: 视图会随槽被覆盖而变; copy=True 拿到独立快照。"""
    with ImagePublisher(region_name, 100, 100, 3) as pub, ImageSubscriber(region_name) as sub:
        pub.publish(_make_frame(100, 100, 11))
        view = sub.try_recv()          # 零拷贝
        snapshot = _make_frame(100, 100, 11)

        # 绕回覆盖同一个槽 (slot0: seq=1 与 seq=4)
        for i in range(2, 5):
            pub.publish(_make_frame(100, 100, i * 10))
        # 视图已被后续写入覆盖, 不再等于最初内容
        assert not np.array_equal(view.image, snapshot)

        pub.publish(_make_frame(100, 100, 99))
        kept = sub.try_recv(copy=True)  # 快照
        pub.publish(_make_frame(100, 100, 1))
        pub.publish(_make_frame(100, 100, 2))
        pub.publish(_make_frame(100, 100, 3))
        assert int(kept.image[10, 10, 0]) == 99  # 拷贝不受后续覆盖影响


def test_dtype_mismatch_rejected(region_name):
    with ImagePublisher(region_name, 100, 100, 3, dtype=np.uint8) as pub:
        with pytest.raises(ValueError):
            pub.publish(np.zeros((100, 100, 3), dtype=np.uint16))


def test_oversized_frame_rejected(region_name):
    with ImagePublisher(region_name, 100, 100, 3) as pub:
        with pytest.raises(ValueError):
            pub.publish(_make_frame(200, 200, 1))


def test_heartbeat_alive(region_name):
    with ImagePublisher(region_name) as pub, ImageSubscriber(region_name) as sub:
        pub.heartbeat()
        assert sub.is_producer_alive()
        assert sub.wait_for_producer(timeout_s=0.5)


def test_bad_magic_rejected(region_name):
    """区域存在但魔数不对 (残留旧文件) 时, 订阅端应拒绝连接。"""
    region = ShmRegion.create(region_name, layout.HEADER_SIZE + 16)
    try:
        with pytest.raises(ValueError):
            ImageSubscriber(region_name)
    finally:
        region.owner = False  # 别让它 unlink, fixture 负责清理
        region.close()
