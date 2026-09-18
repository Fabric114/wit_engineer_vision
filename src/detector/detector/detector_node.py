"""薄 ROS 节点: 按 config/detector.yaml 的 `type` 选检测器 (tc | eu), 从共享内存
(shm_pkg, 由 ros2_camera_pkg 相机节点写入) 零拷贝取帧, 推理后可视化 + 发布结果。

    type: tc  -> 兑换站 yolo-pose (OpenVINO), 出 KeypointObservation + 标注图
    type: eu  -> 能量单元/矿石 yolo detect (ultralytics), 出 标注图 (+ 日志)

也可用 ROS 参数覆盖 type:
    ros2 run detector detector_node --ros-args -p type:=eu

输入源默认 shm (input.source: shm); 设为 topic 时退回订阅 sensor_msgs/Image。
可视化: publish_annotated -> /detector/annotated; show_image -> 本地 cv2 窗口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from detector.types import DetectionResult
from detector.visualization import draw_detections

PACKAGE_NAME = "detector"


def _source_dir() -> Path:
    """源码树里的包目录 (src/detector)。symlink-install 下 __file__ 指向源码。"""
    return Path(__file__).resolve().parents[1]


def _config_dir() -> Path:
    """统一读工作区根 config/ (wit_engineer_vision/config/)。

    向上查找同时含 config/ 与 src/ 的工作区根; 源码运行与拷贝安装下都能命中。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent / "config"
    return here.parents[3] / "config"


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_model(rel_or_abs: str) -> str:
    """模型路径解析: 绝对路径原样用; 相对路径相对 detector/model 目录。

    权重体积大且 .gitignore 排除, 不进 install; 用 --symlink-install 时 __file__
    指向源码, model/ 就在源码树里。找不到时报清晰错误。
    """
    path = Path(rel_or_abs)
    if path.is_absolute():
        return str(path)
    candidates = [_source_dir() / "model" / rel_or_abs]
    try:
        candidates.append(Path(get_package_share_directory(PACKAGE_NAME)) / "model" / rel_or_abs)
    except Exception:
        pass
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"找不到模型 {rel_or_abs!r}; 已尝试: {[str(c) for c in candidates]}"
    )


def _build_detector(det_type: str, cfg: dict):
    if det_type == "tc":
        from detector.detector_tc import TcPoseDetector
        return TcPoseDetector(cfg)
    if det_type == "eu":
        from detector.detector_eu import EuDetector
        return EuDetector(cfg)
    raise ValueError(f"未知检测类型 type={det_type!r} (tc/eu)")


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")
        config = _load_yaml(_config_dir() / "detector.yaml")

        # ROS 参数可覆盖 config 的 type; 空串 (launch 默认) 沿用 config。
        self.declare_parameter("type", "")
        override = str(self.get_parameter("type").value).strip()
        self._type = (override or str(config.get("type", "tc"))).lower()

        det_cfg = dict(config.get(self._type, {}))
        det_cfg["model"] = _resolve_model(str(det_cfg["model"]))
        self._detector = _build_detector(self._type, det_cfg)
        self._detector.open()

        self._debug = bool(config.get("debug", True))
        self._show_image = bool(config.get("show_image", True))
        self._kp_threshold = float(det_cfg.get("keypoint_score_threshold", 0.0))

        # ---- 输出 ----
        from cv_bridge import CvBridge
        self._bridge = CvBridge()
        self._publish_annotated = bool(config.get("publish_annotated", True))
        self._annotated_pub = None
        if self._publish_annotated:
            from sensor_msgs.msg import Image
            self._annotated_pub = self.create_publisher(
                Image, str(config.get("annotated_topic", "/detector/annotated")),
                qos_profile_sensor_data,
            )
        self._obs_pub = None
        if self._type == "tc":
            from interfaces.msg import KeypointObservation
            self._KeypointObservation = KeypointObservation
            self._obs_pub = self.create_publisher(
                KeypointObservation, str(config.get("observation_topic", "/detector/keypoints")), 5
            )

        # ---- 输入源 ----
        in_cfg = dict(config.get("input", {}))
        self._source = str(in_cfg.get("source", "shm")).lower()
        self._frame_id = str(in_cfg.get("frame_id", "camera"))
        if self._source == "shm":
            from shm_pkg import ImageSubscriber, load_shm_config
            region = str(in_cfg.get("region", "")).strip() or load_shm_config().region
            self._sub = _open_shm(self, ImageSubscriber, region)
            self._region = region
            poll_hz = float(in_cfg.get("poll_hz", 200.0))
            self._timer = self.create_timer(1.0 / max(poll_hz, 1.0), self._on_shm_timer)
        else:
            from sensor_msgs.msg import Image
            self._image_sub = self.create_subscription(
                Image, str(in_cfg.get("image_topic", "/camera/image_raw")),
                self._on_image, qos_profile_sensor_data,
            )

        self.get_logger().info(
            f"detector 就绪: type={self._type} 源={self._source} {self._detector.description}"
        )

    # ---------- 输入回调 ----------
    def _on_shm_timer(self) -> None:
        if self._sub is None:
            from shm_pkg import ImageSubscriber
            self._sub = _open_shm(self, ImageSubscriber, self._region)
            return
        frame = self._sub.try_recv(copy=True)  # copy: 推理期间防止槽被写者覆盖
        if frame is None:
            return
        self._process(frame.image, self._stamp_from_ns(frame.timestamp_ns))

    def _on_image(self, msg) -> None:
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._process(np.ascontiguousarray(image), msg.header.stamp)

    def _stamp_from_ns(self, ts_ns: int):
        stamp = self.get_clock().now().to_msg()
        if ts_ns > 0:
            stamp.sec = int(ts_ns // 1_000_000_000)
            stamp.nanosec = int(ts_ns % 1_000_000_000)
        return stamp

    # ---------- 处理一帧 ----------
    def _process(self, image: np.ndarray, stamp) -> None:
        result: DetectionResult = self._detector.detect(image)

        if self._obs_pub is not None:
            self._publish_observations(result, stamp)

        if self._debug:
            self.get_logger().info(
                f"{self._type}: {len(result)} det, {result.inference_ms:.1f} ms"
            )

        if self._publish_annotated or self._show_image:
            annotated = draw_detections(image, result, kp_score_threshold=self._kp_threshold)
            if self._annotated_pub is not None:
                out = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                out.header.stamp = stamp
                out.header.frame_id = self._frame_id
                self._annotated_pub.publish(out)
            if self._show_image:
                import cv2
                cv2.imshow(f"detector [{self._type}]", annotated)
                cv2.waitKey(1)

    def _publish_observations(self, result: DetectionResult, stamp) -> None:
        for inst in result.instances:
            if not inst.has_keypoints():
                continue
            msg = self._KeypointObservation()
            msg.header.stamp = stamp
            msg.header.frame_id = self._frame_id
            msg.schema = self._detector.schema
            msg.class_id = int(inst.class_id)
            kp = np.asarray(inst.keypoints, dtype=np.float32)
            msg.u = kp[:, 0].tolist()
            msg.v = kp[:, 1].tolist()
            msg.confidence = kp[:, 2].tolist()
            self._obs_pub.publish(msg)

    def destroy_node(self):
        try:
            if getattr(self, "_sub", None) is not None:
                self._sub.close()
            if self._show_image:
                import cv2
                cv2.destroyAllWindows()
            self._detector.close()
        finally:
            super().destroy_node()


def _open_shm(node: Node, subscriber_cls, region: str):
    """连接共享内存区域; 相机还没起来时返回 None, 由定时器重试。"""
    try:
        sub = subscriber_cls(region)
        if not sub.wait_for_producer(timeout_s=0.0):
            node.get_logger().warn(f"共享内存 {region!r} 已连, 但相机心跳未活, 等待帧...")
        return sub
    except (FileNotFoundError, ValueError) as exc:
        node.get_logger().warn(f"共享内存 {region!r} 暂不可用 ({exc}); 等相机节点启动后重试")
        return None


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
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

