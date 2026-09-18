"""在线内参标定节点: 订阅 /camera/image_raw, 采样 -> 标定 -> 写 camera_info.yaml。

用法:
    ros2 run ros2_camera_pkg camera_node          # 先把图发出来
    ros2 run camera_calib intrinsic_calib_node
按键 (在预览窗口里按): 空格=手动采一张, c=开始标定, r=清空采样, q=退出。
auto_capture=true 时会自动采合格且视角新的图, 你只要慢慢挪板子。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from camera_calib.intrinsic import SampleCollector, calibrate, save_camera_info, save_report
from camera_calib.patterns import create_pattern


def _config_dir() -> Path:
    """工作区根 config/ (wit_engineer_vision/config/); 所有配置集中在此。

    向上查找同时含 config/ 与 src/ 的工作区根; 源码运行与拷贝安装下都能命中。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent / "config"
    return here.parents[3] / "config"


def _load_config(filename: str) -> dict:
    with (_config_dir() / filename).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


class IntrinsicCalibNode(Node):
    def __init__(self):
        super().__init__("intrinsic_calib_node")
        config = _load_config("intrinsic.yaml")
        self._config = config

        self._pattern = create_pattern(config.get("board_type", "chessboard"), config)
        self._collector = SampleCollector(
            self._pattern,
            min_points=config.get("min_points", 20),
            min_sharpness=config.get("min_sharpness", 80.0),
            min_coverage=config.get("min_coverage", 0.035),
            min_signature_distance=config.get("min_signature_distance", 0.05),
            max_samples=config.get("max_samples", 300),
        )
        self._min_samples = int(config.get("min_samples", 15))
        self._auto_capture = bool(config.get("auto_capture", True))
        self._auto_interval_s = float(config.get("auto_interval_s", 0.5))
        self._show_image = bool(config.get("show_image", True))
        self._image_size = None
        self._last_capture = 0.0
        self._last_reason = ""
        self._bridge = CvBridge()

        self.create_subscription(
            Image, str(config.get("image_topic", "/camera/image_raw")),
            self._on_image, qos_profile_sensor_data,
        )
        self.get_logger().info(f"标定板: {self._pattern.description}")
        self.get_logger().info(
            f"目标样本数 >= {self._min_samples}; auto_capture={self._auto_capture}; "
            f"窗口里按 c 开始标定"
        )

    # ---------- 回调 ----------
    def _on_image(self, message: Image) -> None:
        image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._image_size = (gray.shape[1], gray.shape[0])

        detection = self._pattern.detect(gray)
        if self._auto_capture and time.monotonic() - self._last_capture >= self._auto_interval_s:
            added, reason = self._collector.try_add(gray, label=f"auto_{len(self._collector):03d}")
            self._last_reason = reason
            if added:
                self._last_capture = time.monotonic()
                self.get_logger().info(f"采样 {len(self._collector)} 张")

        if not self._show_image:
            return
        preview = image.copy()
        self._pattern.draw(preview, detection)
        cv2.putText(
            preview,
            f"samples={len(self._collector)}/{self._min_samples}  {self._last_reason}",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        cv2.imshow("intrinsic_calib", preview)
        self._handle_key(cv2.waitKey(1) & 0xFF, gray)

    def _handle_key(self, key: int, gray) -> None:
        if key == ord(" "):
            added, reason = self._collector.try_add(
                gray, label=f"manual_{len(self._collector):03d}"
            )
            self.get_logger().info(f"手动采样: {'成功' if added else '拒绝'} ({reason})")
        elif key == ord("r"):
            self._collector.samples.clear()
            self.get_logger().info("采样已清空")
        elif key == ord("c"):
            self._run_calibration()
        elif key == ord("q"):
            raise KeyboardInterrupt

    # ---------- 标定 ----------
    def _run_calibration(self) -> None:
        if self._image_size is None:
            return
        try:
            result = calibrate(
                self._collector.samples,
                self._image_size,
                min_samples=self._min_samples,
                max_view_error_px=float(self._config.get("max_view_error_px", 0.8)),
            )
        except ValueError as exc:
            self.get_logger().warn(f"还不能标定: {exc}")
            return

        output_directory = Path(
            str(self._config.get("output_directory", "")).strip()
            or Path.cwd() / "calibration" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        camera_info_path = save_camera_info(
            result, self._image_size,
            str(self._config.get("camera_name", "camera")),
            output_directory / "camera_info.yaml",
        )
        save_report(
            result, self._collector.samples, self._image_size,
            self._pattern.description, output_directory / "calibration_report.json",
        )
        self.get_logger().info(
            f"标定完成: RMS={result.rms:.4f}px 用了 {len(result.kept)} 张 "
            f"(剔除 {len(result.rejected)} 张)"
        )
        self.get_logger().info(f"结果: {camera_info_path}")
        self.get_logger().info(
            f"确认无误后拷到统一 config: cp {camera_info_path} "
            f"{_config_dir() / 'camera_info.yaml'}"
        )
        if result.rms > float(self._config.get("warn_rms_px", 0.5)):
            self.get_logger().warn(
                f"RMS {result.rms:.3f}px 偏大: 多采不同角度/距离, 别让板子只在画面中央"
            )


def main(args=None):
    rclpy.init(args=args)
    node = IntrinsicCalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
