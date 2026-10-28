from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

# 驱动构造参数: config 里的 key -> HikCamera/DahengCamera 的 __init__ 形参
DRIVER_KEYS = (
    "sn", "width", "height", "binning", "exposure_us", "gain",
    "bayer_mode", "trigger_enable", "trigger_source",
    "wb_mode", "wb_r", "wb_g", "wb_b",
)
# video 后端的构造参数 (VideoCamera.__init__)
VIDEO_KEYS = ("path", "width", "height", "loop", "start_frame")
# realsense 后端的构造参数 (RealSenseCamera.__init__); enable_depth 由节点按 depth 设置算出
REALSENSE_KEYS = ("sn", "width", "height", "enable_depth", "timeout_ms")


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
    keys = {"video": VIDEO_KEYS, "realsense": REALSENSE_KEYS}.get(cam_type, DRIVER_KEYS)
    kwargs = {k: params[k] for k in keys if k in params}
    # config 用 fps 表示目标帧率, 驱动构造参数名是 frame_rate
    if "fps" in params:
        kwargs["frame_rate"] = params["fps"]

    if cam_type == "video":
        from camera.video_camera import VideoCamera
        return VideoCamera(**kwargs)
    if cam_type == "realsense":
        from camera.realsense_camera import RealSenseCamera
        return RealSenseCamera(**kwargs)
    if cam_type == "hik":
        from camera.hik_camera import HikCamera
        return HikCamera(**kwargs)
    if cam_type == "daheng":
        import sys
        pkg_root = str(Path(__file__).resolve().parents[1])
        if pkg_root not in sys.path:
            sys.path.append(pkg_root)
        from camera.daheng_camera import DahengCamera
        return DahengCamera(**kwargs)
    raise ValueError(f"未知相机类型 type={cam_type!r} (支持: hik | daheng | realsense | video)")


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


def _camera_info_from_intrinsics(frame_id: str, intr: dict) -> CameraInfo:
    """RealSense 彩色流出厂内参 -> CameraInfo (对齐深度与彩色共用)。"""
    info = CameraInfo()
    info.header.frame_id = frame_id
    info.width = int(intr["width"])
    info.height = int(intr["height"])
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    info.distortion_model = "plumb_bob"
    info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.d = [float(c) for c in intr["coeffs"]]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def _build_depth_shm_publisher(config: dict, height: int, width: int, logger):
    """深度共享内存: 单通道 uint16, 区域名 depth_shm_region; 与彩色区域同槽数。"""
    if not bool(config.get("publish_shm", True)):
        return None
    try:
        from shm import ImagePublisher, load_shm_config
    except Exception as exc:
        logger.warn(f"未找到 shm ({exc}), 跳过深度共享内存发布; 请先 build shm")
        return None
    region = str(config.get("depth_shm_region", "")).strip() or "wit_engineer_depth"
    pub = ImagePublisher(region, height, width, 1, np.uint16, load_shm_config().n_slots)
    logger.info(f"深度共享内存就绪: /dev/shm/{region} {height}x{width}x1 uint16")
    return pub


def _build_shm_publisher(config: dict, height: int, width: int, logger):
    """按 camera.yaml 的 publish_shm 建共享内存生产者; 关闭或 shm 缺失时返回 None。

    槽位分辨率用相机配置的 width/height (消费者从头部读尺寸);
    区域名/通道/槽数从 config/shm.yaml 读, 与消费者对齐。
    """
    if not bool(config.get("publish_shm", True)):
        logger.info("publish_shm=false, 只发 ROS topic, 不写共享内存")
        return None
    try:
        from shm import ImagePublisher, load_shm_config
    except Exception as exc:
        logger.warn(f"未找到 shm ({exc}), 跳过共享内存发布; 请先 build shm")
        return None
    cfg = load_shm_config()
    region = str(config.get("shm_region", "")).strip() or cfg.region
    pub = ImagePublisher(region, height, width,
                         cfg.max_channels, cfg.numpy_dtype(), cfg.n_slots)
    logger.info(
        f"共享内存就绪: /dev/shm/{region} "
        f"{height}x{width}x{cfg.max_channels} n_slots={cfg.n_slots}"
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
        # video 后端可用 ROS 参数直接指定文件, 不必改 yaml
        self.declare_parameter("video_path", "")
        video_path = str(self.get_parameter("video_path").value).strip()
        if cam_type == "video" and video_path:
            params["path"] = video_path
        # realsense: 是否开深度。ROS 参数 depth 可覆盖 (空串沿用 config);
        # auto 跟 config/detector.yaml 的 type 走 (eu 开, tc 关)
        self._depth_enabled = False
        if cam_type == "realsense":
            from camera.realsense_camera import resolve_enable_depth
            # dynamic_typing: 命令行 depth:=true 会被解析成 bool, depth:=auto 是 str, 两种都收
            self.declare_parameter("depth", "", ParameterDescriptor(dynamic_typing=True))
            depth_override = self.get_parameter("depth").value
            depth_setting = (
                depth_override if isinstance(depth_override, bool)
                else (str(depth_override).strip() or params.get("depth", "auto"))
            )
            detector_type = str(_load_yaml(_config_dir() / "detector.yaml").get("type", ""))
            self._depth_enabled = resolve_enable_depth(depth_setting, detector_type)
            params["enable_depth"] = self._depth_enabled
            self.get_logger().info(
                f"realsense depth={depth_setting} detector.type={detector_type or '<未设>'}"
                f" -> {'彩色+对齐深度' if self._depth_enabled else '仅彩色'}"
            )
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
        self._depth_pub = None
        if self._depth_enabled:
            self._depth_pub = self.create_publisher(
                Image, str(params.get("depth_topic", "/camera/depth/image_raw")),
                qos_profile_sensor_data,
            )
        self._driver = _build_driver(cam_type, params, self.get_logger())
        # 分辨率以驱动为准: video 的 width/height=0 表示用视频原生尺寸
        width, height = int(self._driver.width), int(self._driver.height)
        if cam_type == "video":
            self._publish_fps = float(self._driver.frame_rate)
        # realsense 用设备出厂内参 (open 后在 _try_open 里填), 不读 camera_info.yaml
        self._uses_device_intrinsics = hasattr(self._driver, "intrinsics")
        if self._uses_device_intrinsics:
            self._camera_info = CameraInfo()
            self._camera_info.header.frame_id = self._frame_id
        else:
            self._camera_info = _build_camera_info(self._frame_id, self.get_logger())
        if self._camera_info.width and (self._camera_info.width, self._camera_info.height) != (width, height):
            self.get_logger().warn(
                f"camera_info.yaml 是 {self._camera_info.width}x{self._camera_info.height}, "
                f"相机配置是 {width}x{height}; 内参与分辨率不匹配, 请按当前分辨率重新标定"
            )

        self._last_ts = None
        self._last_frame_monotonic = time.monotonic()
        self._last_retry_monotonic = 0.0

        # 共享内存生产者: 把每帧零拷贝写进 /dev/shm, 供 detector/solver 消费。
        # publish_shm 默认开; 关掉则只发 ROS topic (见 camera.yaml)。
        self._shm_pub = _build_shm_publisher(config, height, width, self.get_logger())
        self._depth_shm_pub = (
            _build_depth_shm_publisher(config, height, width, self.get_logger())
            if self._depth_enabled else None
        )

        self.get_logger().info(
            f"相机后端: {cam_type} "
            f"{('path=' + str(params.get('path'))) if cam_type == 'video' else ('sn=' + str(params.get('sn') or '<auto>'))} "
            f"{width}x{height}@{self._publish_fps:g}fps"
        )
        self._try_open(initial=True)
        # video: 解码线程已按帧率出帧, 定时器 2 倍频轮询, 避免同频拍频漏帧
        poll_hz = self._publish_fps * (2.0 if cam_type == "video" else 1.0)
        self._timer = self.create_timer(1.0 / max(poll_hz, 1.0), self._on_timer)

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
                if self._uses_device_intrinsics:
                    intr = self._driver.intrinsics()
                    self._camera_info = _camera_info_from_intrinsics(self._frame_id, intr)
                    self.get_logger().info(
                        f"内参取自设备出厂标定: {intr['width']}x{intr['height']} "
                        f"fx={intr['fx']:.2f} fy={intr['fy']:.2f}"
                    )
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

        # 视频不循环且已放完: 停发, 不当成掉线重启
        if getattr(self._driver, "is_finished", lambda: False)():
            return

        now_monotonic = time.monotonic()
        if now_monotonic - self._last_frame_monotonic >= self._no_frame_timeout_s:
            self.get_logger().warn(
                f"超过 {self._no_frame_timeout_s:.0f}s 无新帧, 重启相机"
            )
            self._try_open(initial=True)
            return

        depth = None
        if self._depth_enabled:
            frame, depth, ts = self._driver.get_latest_rgbd()
        else:
            frame, ts = self._driver.get_latest()
        if frame is None or ts == self._last_ts:
            return
        if self._depth_enabled and depth is None:
            # 要 RGB-D 却没有深度: 显式报错并丢帧, 不发只有彩色的"RGB-D"
            self.get_logger().error("开了深度但本帧无深度, 丢弃", throttle_duration_sec=2.0)
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

        if depth is not None:
            depth = np.ascontiguousarray(depth)
            if self._depth_shm_pub is not None:
                try:
                    self._depth_shm_pub.publish(depth)
                except Exception as exc:
                    self.get_logger().warn(f"深度共享内存发布失败: {exc}", throttle_duration_sec=5.0)
            # 已对齐到彩色: 与彩色同 frame_id / 同 camera_info, 单位 = 设备 depth_scale (D435 为 mm)
            depth_msg = self._bridge.cv2_to_imgmsg(depth, encoding="16UC1")
            depth_msg.header.stamp = stamp
            depth_msg.header.frame_id = self._frame_id
            self._depth_pub.publish(depth_msg)

        self._camera_info.header.stamp = stamp
        self._info_pub.publish(self._camera_info)

    def destroy_node(self):
        try:
            self._driver.close()
            if self._shm_pub is not None:
                self._shm_pub.close()
            if self._depth_shm_pub is not None:
                self._depth_shm_pub.close()
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
