"""薄 ROS 节点: 按 config/camera.yaml 选相机后端 (hik | daheng), 取流后
发布 /camera/image_raw + /camera/camera_info。

后端由 camera.yaml 的 `type` 决定, 也可用 ROS 参数覆盖:
    ros2 run ros2_camera_pkg camera_node --ros-args -p type:=daheng

每个后端的参数放在同名子字典 (hik_camera / daheng_camera) 里, 与驱动
HikCamera / DahengCamera 的构造参数一一对应。内参从 config/camera_info.yaml
读取 (camera_calib 标定产物), 缺失时发空 CameraInfo 并告警。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

# 驱动构造参数: config 里的 key -> HikCamera/DahengCamera 的 __init__ 形参
DRIVER_KEYS = (
    "sn", "width", "height", "binning", "exposure_us", "gain",
    "bayer_mode", "trigger_enable", "trigger_source",
    "wb_mode", "wb_r", "wb_g", "wb_b",
)


def _config_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent / "config"
    return here.parents[3] / "config"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _build_driver(cam_type: str, params: dict, logger):
    """按 type 惰性导入对应驱动并构造。惰性导入: 只加载真正用到的 SDK,
    跑 hik 时不会因为缺 gxipy 而失败, 反之亦然。"""
    kwargs = {k: params[k] for k in DRIVER_KEYS if k in params}
    # config 用 fps 表示目标帧率, 驱动构造参数名是 frame_rate
    if "fps" in params:
        kwargs["frame_rate"] = params["fps"]

    if cam_type == "hik":
        from ros2_camera_pkg.hik_camera import HikCamera
        return HikCamera(**kwargs)
    if cam_type == "daheng":
        import sys
        pkg_root = str(Path(__file__).resolve().parents[1])
        if pkg_root not in sys.path:
            sys.path.append(pkg_root)
        from ros2_camera_pkg.daheng_camera import DahengCamera
        return DahengCamera(**kwargs)
    raise ValueError(f"未知相机类型 type={cam_type!r} (支持: hik | daheng)")


def _build_camera_info(frame_id: str, logger) -> CameraInfo:
    """从 config/camera_info.yaml 读标准 ROS 内参; 没标定就发空 CameraInfo 并告警。"""
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.distortion_model = "plumb_bob"
    path = _config_dir() / "camera_info.yaml"
    try:
        data = _load_yaml(path)
        info.width = int(data["image_width"])
        info.height = int(data["image_height"])
        info.distortion_model = str(data.get("distortion_model", "plumb_bob"))
        info.k = [float(x) for x in data["camera_matrix"]["data"]]
        info.d = [float(x) for x in data["distortion_coefficients"]["data"]]
        info.r = [float(x) for x in data["rectification_matrix"]["data"]]
        info.p = [float(x) for x in data["projection_matrix"]["data"]]
        logger.info(
            f"内参已加载: {info.width}x{info.height} "
            f"fx={info.k[0]:.2f} fy={info.k[4]:.2f}"
        )
    except Exception as exc:
        logger.warn(
            f"未加载到有效内参 ({exc}); /camera/camera_info 将为空, PnP 无法工作。"
            f" 先跑 camera_calib 标定, 把 camera_info.yaml 拷到 {path.parent}"
        )
    return info


def _build_shm_publisher(config: dict, logger):
    """按 camera.yaml 的 publish_shm 建共享内存生产者; 关闭或 shm_pkg 缺失时返回 None。

    区域名/最大分辨率/槽数从 shm_pkg 的 config/shm.yaml 读, 与消费者对齐。
    """
    if not bool(config.get("publish_shm", True)):
        logger.info("publish_shm=false, 只发 ROS topic, 不写共享内存")
        return None
    try:
        from shm_pkg import ImagePublisher, load_shm_config
    except Exception as exc:
        logger.warn(f"未找到 shm_pkg ({exc}), 跳过共享内存发布; 请先 build shm_pkg")
        return None
    cfg = load_shm_config()
    region = str(config.get("shm_region", "")).strip() or cfg.region
    pub = ImagePublisher(region, cfg.max_height, cfg.max_width,
                         cfg.max_channels, cfg.numpy_dtype(), cfg.n_slots)
    logger.info(
        f"共享内存就绪: /dev/shm/{region} "
        f"{cfg.max_height}x{cfg.max_width}x{cfg.max_channels} n_slots={cfg.n_slots}"
    )
    return pub


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        config = _load_yaml(_config_dir() / "camera.yaml")

        # ROS 参数可覆盖 config 的 type; 传空串 (launch 默认) 则沿用 config。
        config_type = str(config.get("type", "hik"))
        self.declare_parameter("type", "")
        cam_type = (str(self.get_parameter("type").value).strip() or config_type).lower()

        params = dict(config.get(f"{cam_type}_camera", {}))
        self._frame_id = str(params.get("frame_id", "camera"))
        self._publish_fps = float(params.get("fps", 30.0))
        self._retry_interval_s = float(params.get("retry_interval_s", 2.0))
        self._no_frame_timeout_s = float(params.get("no_frame_timeout_s", 3.0))

        self._bridge = CvBridge()
        self._image_pub = self.create_publisher(
            Image, str(params.get("image_topic", "/camera/image_raw")),
            qos_profile_sensor_data,
        )
        self._info_pub = self.create_publisher(
            CameraInfo, str(params.get("camera_info_topic", "/camera/camera_info")), 1
        )
        self._camera_info = _build_camera_info(self._frame_id, self.get_logger())

        self._driver = _build_driver(cam_type, params, self.get_logger())
        self._last_ts = None
        self._last_frame_monotonic = time.monotonic()
        self._last_retry_monotonic = 0.0

        # 共享内存生产者: 把每帧零拷贝写进 /dev/shm, 供 detector/solver 消费。
        # publish_shm 默认开; 关掉则只发 ROS topic (见 camera.yaml)。
        self._shm_pub = _build_shm_publisher(config, self.get_logger())

        self.get_logger().info(
            f"相机后端: {cam_type} sn={params.get('sn') or '<auto>'} "
            f"{int(params.get('width', 0))}x{int(params.get('height', 0))}@{self._publish_fps:g}fps"
        )
        self._try_open(initial=True)
        self._timer = self.create_timer(1.0 / max(self._publish_fps, 1.0), self._on_timer)

    # ---------- 取流与重连 ----------
    def _try_open(self, initial: bool = False) -> bool:
        now = time.monotonic()
        if not initial and now - self._last_retry_monotonic < self._retry_interval_s:
            return False
        self._last_retry_monotonic = now
        try:
            self._driver.close()
            if self._driver.start_streaming():
                self._last_frame_monotonic = time.monotonic()
                self._last_ts = None
                caps = self._driver.get_capabilities()
                if caps:
                    self.get_logger().info(f"相机就绪: {caps}")
                return True
        except Exception as exc:
            self.get_logger().error(f"相机打开失败: {exc}")
            return False
        self.get_logger().error(f"相机打开失败, {self._retry_interval_s:.0f}s 后重试")
        return False

    def _on_timer(self) -> None:
        if not self._driver.is_connected():
            self._try_open()
            return

        now_monotonic = time.monotonic()
        if now_monotonic - self._last_frame_monotonic >= self._no_frame_timeout_s:
            self.get_logger().warn(
                f"超过 {self._no_frame_timeout_s:.0f}s 无新帧, 重启相机"
            )
            self._try_open(initial=True)
            return

        frame, ts = self._driver.get_latest()
        if frame is None or ts == self._last_ts:
            return
        self._last_ts = ts
        self._last_frame_monotonic = now_monotonic

        frame = np.ascontiguousarray(frame)
        stamp = self.get_clock().now().to_msg()

        # 共享内存 (零拷贝主链路): detector/solver 从这里取帧
        if self._shm_pub is not None:
            try:
                self._shm_pub.publish(frame)
            except Exception as exc:  # 尺寸/类型不匹配等, 不应拖垮取流
                self.get_logger().warn(f"共享内存发布失败: {exc}", throttle_duration_sec=5.0)

        # ROS topic (兼容 rviz / 录包 / 未接入 shm 的消费者)
        image_msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self._frame_id
        self._image_pub.publish(image_msg)

        self._camera_info.header.stamp = stamp
        self._info_pub.publish(self._camera_info)

    def destroy_node(self):
        try:
            self._driver.close()
            if self._shm_pub is not None:
                self._shm_pub.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
