"""共享内存图像收发的高层 API —— 纯逻辑, 零 ROS 依赖, 可 pytest。

- ImagePublisher: 生产者 (ros2_camera_pkg 相机节点), 把每帧写进三缓冲的下一个槽,
  最后自增 seq 发布。写者只写一份到共享内存 (相机帧 -> shm 这次拷贝无法避免)。
- ImageSubscriber: 消费者 (detector / solver), 用 seqlock 读到一致的一帧。
  返回的 Frame.image 默认是**零拷贝** numpy 视图, 直接落在共享内存上。

seqlock 协议: 写者写完像素后, 先写 latest_slot/cur_*/timestamp, 最后自增 seq;
读者读 seq -> 读 slot/几何/取视图 -> 再读 seq, 两次一致才算读到未被打断的一帧。
配合三缓冲, 读者的槽要过 n_slots 帧才会被写者绕回覆盖, 有足够安全余量。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import layout
from .region import ShmRegion


@dataclass
class Frame:
    """读到的一帧。image 默认是共享内存上的零拷贝视图 —— 见 ImageSubscriber.try_recv。"""

    image: np.ndarray
    seq: int
    timestamp_ns: int

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


def _now_ns() -> int:
    """墙钟纳秒时间戳; 生产者/消费者用同一时钟, 心跳判活才有意义。"""
    return time.time_ns()


class ImagePublisher:
    """生产者: 创建并持有共享内存区域, 逐帧发布。用完必须 close()。"""

    def __init__(
        self,
        name: str = layout.DEFAULT_REGION,
        max_height: int = 720,
        max_width: int = 960,
        max_channels: int = 3,
        dtype: np.dtype = np.uint8,  # type: ignore[assignment]
        n_slots: int = 3,
    ):
        dt = np.dtype(dtype)
        code = layout.code_from_dtype(dt)
        slot_size = layout.slot_bytes(max_height, max_width, max_channels, dt.itemsize)
        total = layout.region_size(slot_size, n_slots)

        self.region = ShmRegion.create(name, total)
        self._dtype = dt
        self._slot_size = slot_size
        self._n_slots = int(n_slots)
        self._write_count = 0  # 已发布帧数, slot = write_count % n_slots
        self._hdr = np.ndarray((), dtype=layout.HEADER_DTYPE, buffer=self.region.mm, offset=0)

        # 写一次静态头部 (create 后内存已被 ftruncate 清零)
        self._hdr["magic"] = layout.MAGIC
        self._hdr["version"] = layout.VERSION
        self._hdr["max_height"] = max_height
        self._hdr["max_width"] = max_width
        self._hdr["max_channels"] = max_channels
        self._hdr["itemsize"] = dt.itemsize
        self._hdr["dtype_code"] = code
        self._hdr["n_slots"] = n_slots
        self._hdr["slot_bytes"] = slot_size
        self._hdr["created_ns"] = _now_ns()
        self._hdr["seq"] = 0  # 0 = 尚未发布任何帧
        self._hdr["heartbeat_ns"] = _now_ns()

    def publish(self, image: np.ndarray, timestamp_ns: Optional[int] = None) -> int:
        """把一帧写进下一个槽并发布, 返回新的帧序号 (从 1 开始)。

        image 可以是 (H, W) 或 (H, W, C), 尺寸/通道不超过创建时的最大值, dtype 需一致。
        """
        if image.ndim == 2:
            h, w = image.shape
            c = 1
        elif image.ndim == 3:
            h, w, c = image.shape
        else:
            raise ValueError(f"图像维度必须是 2 或 3, 收到 shape={image.shape}")

        if np.dtype(image.dtype) != self._dtype:
            raise ValueError(f"dtype 不匹配: 区域是 {self._dtype}, 帧是 {image.dtype}")
        if (h > int(self._hdr["max_height"]) or w > int(self._hdr["max_width"])
                or c > int(self._hdr["max_channels"])):
            raise ValueError(
                f"帧 {h}x{w}x{c} 超过区域上限 "
                f"{int(self._hdr['max_height'])}x{int(self._hdr['max_width'])}"
                f"x{int(self._hdr['max_channels'])}"
            )

        slot = self._write_count % self._n_slots
        off = layout.slot_offset(slot, self._slot_size)
        dst = np.ndarray((h, w, c), dtype=self._dtype, buffer=self.region.mm, offset=off)
        dst[:] = image.reshape(h, w, c)  # 唯一一次拷贝: 相机帧 -> 共享内存

        # 先更新元数据, 最后自增 seq (seqlock 发布屏障)
        self._hdr["cur_height"] = h
        self._hdr["cur_width"] = w
        self._hdr["cur_channels"] = c
        self._hdr["timestamp_ns"] = _now_ns() if timestamp_ns is None else int(timestamp_ns)
        self._hdr["latest_slot"] = slot
        self._hdr["heartbeat_ns"] = _now_ns()

        self._write_count += 1
        self._hdr["seq"] = self._write_count  # 发布: 让消费者可见
        return self._write_count

    def heartbeat(self) -> None:
        """没有新帧时也可周期性调用, 让消费者的 is_producer_alive 保持为真。"""
        self._hdr["heartbeat_ns"] = _now_ns()

    def close(self) -> None:
        self.region.close()

    def __enter__(self) -> "ImagePublisher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ImageSubscriber:
    """消费者: 映射已存在的区域, 用 seqlock 读到一致的一帧。用完必须 close()。"""

    _MAX_RETRY = 4  # seqlock 被写者打断时的重试次数

    def __init__(self, name: str = layout.DEFAULT_REGION):
        self.region = ShmRegion.open(name, size=None, writable=False)
        self._hdr = np.ndarray((), dtype=layout.HEADER_DTYPE, buffer=self.region.mm, offset=0)

        magic = int(self._hdr["magic"])
        version = int(self._hdr["version"])
        if magic != layout.MAGIC:
            self.region.close()
            raise ValueError(f"魔数不匹配 (期望 {layout.MAGIC:#x}, 实际 {magic:#x}), 可能是残留旧文件")
        if version != layout.VERSION:
            self.region.close()
            raise ValueError(f"协议版本不匹配 (期望 {layout.VERSION}, 实际 {version})")

        self._dtype = layout.dtype_from_code(int(self._hdr["dtype_code"]))
        self._slot_size = int(self._hdr["slot_bytes"])
        self._last_seq = 0  # 上次成功读到的帧序号, 用于判断"有没有新帧"

    def try_recv(self, copy: bool = False) -> Optional[Frame]:
        """尝试读最新的一帧。没有新帧 (或读被反复打断) 返回 None。

        copy=False (默认): Frame.image 是共享内存上的零拷贝视图, 高帧率下最省。
            注意视图只在写者绕回覆盖该槽前有效 (三缓冲 = 约 n_slots 帧余量);
            要跨多帧留用请传 copy=True 或自行 .copy()。
        copy=True: 立刻拷出一份独立数组, 之后随便持有。
        """
        for _ in range(self._MAX_RETRY):
            s1 = int(self._hdr["seq"])
            if s1 == 0 or s1 == self._last_seq:
                return None  # 还没发布过 / 没有新帧
            slot = int(self._hdr["latest_slot"])
            h = int(self._hdr["cur_height"])
            w = int(self._hdr["cur_width"])
            c = int(self._hdr["cur_channels"])
            ts = int(self._hdr["timestamp_ns"])

            off = layout.slot_offset(slot, self._slot_size)
            view = np.ndarray((h, w, c), dtype=self._dtype, buffer=self.region.mm, offset=off)

            # seqlock 校验: 取视图期间 seq 没变, 才认为这一帧一致
            if int(self._hdr["seq"]) != s1:
                continue
            self._last_seq = s1
            image = view.copy() if copy else view
            return Frame(image=image, seq=s1, timestamp_ns=ts)
        return None  # 写者太快, 连续 _MAX_RETRY 次都被打断

    def recv(self, timeout_s: float = 1.0, poll_s: float = 0.001) -> Optional[Frame]:
        """阻塞轮询直到有新帧或超时。低频消费或测试用; 高频链路建议自己的循环里 try_recv。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = self.try_recv()
            if frame is not None:
                return frame
            time.sleep(poll_s)
        return None

    def is_producer_alive(self, timeout_ns: int = 1_000_000_000) -> bool:
        """心跳在 timeout_ns 内更新过则认为生产者存活。"""
        return _now_ns() - int(self._hdr["heartbeat_ns"]) < timeout_ns

    def wait_for_producer(self, timeout_s: float = 5.0, poll_s: float = 0.05) -> bool:
        """等生产者心跳变活, 防止连到一个已经死掉的残留区域。"""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.is_producer_alive():
                return True
            time.sleep(poll_s)
        return False

    @property
    def latest_seq(self) -> int:
        """当前共享内存里最新帧的序号 (0 表示尚未发布)。"""
        return int(self._hdr["seq"])

    def close(self) -> None:
        self.region.close()

    def __enter__(self) -> "ImageSubscriber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
