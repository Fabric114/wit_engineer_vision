"""shm_pkg —— /dev/shm mmap 三缓冲共享图像, 绕开 DDS 大图拷贝。

设计对标 ~/awakening/3rdparty/daedalus_interface (C++ 的 shm_region / shm_client),
用途: ros2_camera_pkg 相机节点做生产者, detector / solver 做消费者, 零拷贝取帧。

典型用法:
    # 生产者 (相机节点)
    from shm_pkg import ImagePublisher, load_shm_config
    cfg = load_shm_config()
    pub = ImagePublisher(cfg.region, cfg.max_height, cfg.max_width,
                         cfg.max_channels, cfg.numpy_dtype(), cfg.n_slots)
    pub.publish(bgr_frame)              # 每帧调用

    # 消费者 (detector / solver)
    from shm_pkg import ImageSubscriber, load_shm_config
    sub = ImageSubscriber(load_shm_config().region)
    frame = sub.try_recv()              # 零拷贝视图, 无新帧则 None
    if frame is not None:
        detect(frame.image)             # frame.image 直接指向共享内存
"""

from .config import ShmConfig, load_shm_config
from .image_transport import Frame, ImagePublisher, ImageSubscriber
from .region import ShmRegion

__all__ = [
    "Frame",
    "ImagePublisher",
    "ImageSubscriber",
    "ShmRegion",
    "ShmConfig",
    "load_shm_config",
]
