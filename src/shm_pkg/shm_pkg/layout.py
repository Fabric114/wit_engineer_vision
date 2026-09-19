"""共享内存图像区域的二进制布局 —— 纯逻辑, 零 ROS / 零 mmap 依赖。

用固定字段的头部+ magic/version 把内存布局钉死, 这样同一个 /dev/shm 文件将来也能被 C++/Rust
按同样的偏移解读。区域整体布局:

    ┌──────────────────────────────┐ offset 0
    │ Header (HEADER_SIZE=256 字节) │  magic/version/几何 + seqlock 热字段
    ├──────────────────────────────┤ offset HEADER_SIZE
    │ slot 0  (slot_bytes 字节)     │  一帧图像的原始像素
    │ slot 1                        │  三缓冲轮转, 写者写 seq%N 槽
    │ slot 2                        │
    └──────────────────────────────┘

只有 seq / latest_slot / cur_* / *_ns 是运行期高频改写的「热字段」, 其余在
create() 时写一次就不动。seq 是 seqlock 的核心: 写者最后自增它, 读者前后各读
一次判断是否被打断。
"""

from __future__ import annotations

import numpy as np

# "WITE" (Wit engineer ImagE) 的小端表示, 换布局就改这个值让旧连接失效
MAGIC = 0x57495445
VERSION = 1

# 头部预留 256 字节 (缓存行整数倍), 像素数据从 HEADER_SIZE 开始, 保证对齐
HEADER_SIZE = 256

DEFAULT_REGION = "wit_engineer_image"
SHM_DIR = "/dev/shm"

# dtype_code <-> numpy dtype 的映射, 目前相机出 bgr8 用 uint8 就够
_CODE_TO_DTYPE = {0: np.uint8, 1: np.uint16, 2: np.float32}
_DTYPE_TO_CODE = {np.dtype(v): k for k, v in _CODE_TO_DTYPE.items()}

# 头部字段 (小端, packed)。前 8 个 u4 是静态几何, 之后是 seqlock 热字段。
# seq 落在 offset 48 (8 字节对齐), 保证单次写在 x86-64 上是原子的。
HEADER_DTYPE = np.dtype(
    [
        ("magic", "<u4"),        # 0  校验魔数
        ("version", "<u4"),      # 4  协议版本
        ("max_height", "<u4"),   # 8  槽位能容纳的最大高
        ("max_width", "<u4"),    # 12 最大宽
        ("max_channels", "<u4"), # 16 最大通道数
        ("itemsize", "<u4"),     # 20 每元素字节数 (uint8=1)
        ("dtype_code", "<u4"),   # 24 见 _CODE_TO_DTYPE
        ("n_slots", "<u4"),      # 28 缓冲槽数量 (三缓冲=3)
        ("slot_bytes", "<u8"),   # 32 单槽字节数
        ("created_ns", "<u8"),   # 40 创建时间戳
        ("seq", "<u8"),          # 48 单调递增帧序号 (seqlock 核心, 0=未发布)
        ("latest_slot", "<u4"),  # 56 最新一帧所在槽位
        ("cur_height", "<u4"),   # 60 当前帧实际高
        ("cur_width", "<u4"),    # 64 当前帧实际宽
        ("cur_channels", "<u4"), # 68 当前帧实际通道数
        ("timestamp_ns", "<u8"), # 72 当前帧时间戳
        ("heartbeat_ns", "<u8"), # 80 生产者心跳, 消费者据此判活
    ]
)
assert HEADER_DTYPE.itemsize <= HEADER_SIZE, "头部字段超过预留空间"


def dtype_from_code(code: int) -> np.dtype:
    """dtype_code -> numpy dtype, 未知码按 uint8 兜底。"""
    return np.dtype(_CODE_TO_DTYPE.get(int(code), np.uint8))


def code_from_dtype(dtype: np.dtype) -> int:
    """numpy dtype -> dtype_code, 不支持的类型直接报错 (别静默塞进共享内存)。"""
    dt = np.dtype(dtype)
    if dt not in _DTYPE_TO_CODE:
        raise ValueError(f"不支持的图像 dtype={dt}, 支持: {list(_CODE_TO_DTYPE.values())}")
    return _DTYPE_TO_CODE[dt]


def slot_bytes(max_height: int, max_width: int, max_channels: int, itemsize: int) -> int:
    """单个槽位需要的字节数 = 最大分辨率 × 通道 × 元素大小。"""
    return int(max_height) * int(max_width) * int(max_channels) * int(itemsize)


def region_size(slot_size: int, n_slots: int) -> int:
    """整个共享内存区域的总字节数 = 头部 + N 个槽。"""
    return HEADER_SIZE + int(slot_size) * int(n_slots)


def slot_offset(slot_index: int, slot_size: int) -> int:
    """第 slot_index 个槽在区域内的字节偏移。"""
    return HEADER_SIZE + int(slot_index) * int(slot_size)
