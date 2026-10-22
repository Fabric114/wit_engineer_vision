"""ROS 薄节点：detector 观测 -> PnP 站姿, 同时发相机系与 arm_base 系两份。

发布:
  /solver/exchange_station_pose  PoseStamped, frame = camera (PnP 原始结果, 台架调试主要看它)
  /vision/exchange_pose          ExchangeStationPose, frame = arm_base (× 外参, 给 planning)
  /tf_static  arm_base -> camera            (config/camera_extrinsic.yaml)
  /tf         camera -> exchange_station    (每帧 PnP 结果, rviz 用)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .extrinsic import compose, is_identity, load_extrinsic, matrix_from_pose, pose_from_matrix
from .pairing import PartObservation, solve_best_candidate
from .schema import load_schema
from .solver import PnPEstimator


def _camera_info_to_arrays(msg):
    K = np.asarray(msg.k, dtype=np.float64)
    if K.size != 9:
        raise ValueError("CameraInfo K must contain 9 values")
    K = K.reshape(3, 3)
    D = np.asarray(msg.d, dtype=np.float64).reshape(-1)
    if D.size not in (0, 4, 5, 8, 12, 14):
        raise ValueError("CameraInfo D has invalid length")
    if not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError("CameraInfo K/D must be finite")
    return K, D


def _observation_to_arrays(msg):
    u = np.asarray(msg.u, dtype=np.float64).reshape(-1)
    v = np.asarray(msg.v, dtype=np.float64).reshape(-1)
    confidence = np.asarray(msg.confidence, dtype=np.float64).reshape(-1)
    if not (len(u) == len(v) == len(confidence)):
        raise ValueError("observation parallel arrays must have equal length")
    points = np.column_stack((u, v))
    if not np.isfinite(points).all() or not np.isfinite(confidence).all():
        raise ValueError("observation arrays must be finite")
    return str(msg.schema), int(msg.class_id), points, confidence


def _config_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config").is_dir() and (parent / "src").is_dir():
            return parent / "config"
    return here.parents[3] / "config"


def _load_extrinsic_config(path) -> tuple[dict, np.ndarray]:
    import yaml

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream) or {}
    for key in ("camera_frame", "base_frame", "station_frame"):
        if not cfg.get(key):
            raise ValueError(f"{path}: missing {key}")
    return cfg, load_extrinsic(cfg, base_dir=path.parent)


def _matrix_from_transform_msg(transform) -> np.ndarray:
    t, q = transform.translation, transform.rotation
    return matrix_from_pose((t.x, t.y, t.z), (q.w, q.x, q.y, q.z))


def _fill_pose(pose, T) -> None:
    """``(4, 4)`` -> ``geometry_msgs/Pose``, 四元数按字段名赋值。"""
    xyz, (qw, qx, qy, qz) = pose_from_matrix(T)
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in xyz)
    pose.orientation.w, pose.orientation.x = float(qw), float(qx)
    pose.orientation.y, pose.orientation.z = float(qy), float(qz)


def _fill_transform(transform, T) -> None:
    """``(4, 4)`` -> ``geometry_msgs/Transform``, 四元数按字段名赋值。"""
    xyz, (qw, qx, qy, qz) = pose_from_matrix(T)
    transform.translation.x, transform.translation.y, transform.translation.z = (float(v) for v in xyz)
    transform.rotation.w, transform.rotation.x = float(qw), float(qx)
    transform.rotation.y, transform.rotation.z = float(qy), float(qz)


def _describe_transform(T) -> str:
    import cv2

    rotvec, _ = cv2.Rodrigues(np.asarray(T, dtype=np.float64)[:3, :3])
    xyz = ", ".join(f"{v:+.4f}" for v in T[:3, 3])
    return f"t = ({xyz}) m, |rotvec| = {np.degrees(np.linalg.norm(rotvec)):.2f} deg, rotvec = {rotvec[:, 0].round(4).tolist()}"


class _FrameBuffer:
    """把同一时间戳的逐实例观测攒成一帧。

    detector 每帧按实例连续发布多条消息且时间戳相同; 收到更新的时间戳, 或当前帧
    超过 timeout_s 没有新消息时, 释放该帧。早于已释放帧的迟到消息直接丢弃。
    """

    def __init__(self, timeout_s: float):
        self.timeout_s = float(timeout_s)
        self._key = None
        self._items: list = []
        self._last_arrival = 0.0
        self._released = None

    def add(self, key, item, now: float) -> list:
        if self._released is not None and key <= self._released:
            return []
        released = []
        if self._key is not None and key > self._key:
            released = self._release()
        elif self._key is not None and key < self._key:
            return []
        if self._key is None:
            self._key = key
        self._items.append(item)
        self._last_arrival = now
        return released

    def flush(self, now: float) -> list:
        if self._key is None or now - self._last_arrival < self.timeout_s:
            return []
        return self._release()

    def _release(self) -> list:
        frame = [(self._key, self._items)]
        self._released, self._key, self._items = self._key, None, []
        return frame


class PnpNode:
    def __init__(self):
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from interfaces.msg import ExchangeStationPose, KeypointObservation
        from rclpy.duration import Duration
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.time import Time
        from sensor_msgs.msg import CameraInfo
        from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformException
        from tf2_ros.transform_listener import TransformListener

        class _Node(Node):
            def __init__(self, outer):
                super().__init__("pnp_node")
                self.outer = outer
                self.declare_parameter("observation_topic", "/detector/keypoints")
                self.declare_parameter("camera_info_topic", "/camera/camera_info")
                self.declare_parameter("output_topic", "/solver/exchange_station_pose")
                self.declare_parameter("base_output_topic", "/vision/exchange_pose")
                self.declare_parameter("extrinsic_config", str(_config_dir() / "camera_extrinsic.yaml"))
                self.declare_parameter("tf_lookup_timeout_s", 0.05)
                self.declare_parameter("confidence_threshold", 0.2)
                self.declare_parameter("reprojection_error_threshold_px", 5.0)
                self.declare_parameter("frame_timeout_s", 0.03)
                self._camera_info = None
                self._schemas = {}
                self._frames = _FrameBuffer(self.get_parameter("frame_timeout_s").value)
                self._estimator = PnPEstimator(
                    self.get_parameter("confidence_threshold").value,
                    self.get_parameter("reprojection_error_threshold_px").value,
                )
                self._pub = self.create_publisher(
                    PoseStamped, self.get_parameter("output_topic").value, 10
                )
                self._base_pub = self.create_publisher(
                    ExchangeStationPose, self.get_parameter("base_output_topic").value, 10
                )
                self._lookup_timeout = Duration(seconds=float(self.get_parameter("tf_lookup_timeout_s").value))

                # 外参: 读不到或矩阵非法直接起不来; 单位阵只告警, 两份位姿照发。
                cfg, T_base_camera = _load_extrinsic_config(self.get_parameter("extrinsic_config").value)
                self._camera_frame = cfg["camera_frame"]
                self._base_frame = cfg["base_frame"]
                self._station_frame = cfg["station_frame"]
                if is_identity(T_base_camera):
                    self.get_logger().error(
                        "⚠️ 外参 T_arm_base_camera 是单位阵占位 (config/camera_extrinsic.yaml): "
                        "/vision/exchange_pose 照发但数值等于相机系, 不可信")
                self.get_logger().info(f"外参 {self._base_frame} <- {self._camera_frame}: "
                                       f"{_describe_transform(T_base_camera)}")
                static = TransformStamped()
                static.header.stamp = self.get_clock().now().to_msg()
                static.header.frame_id = self._base_frame
                static.child_frame_id = self._camera_frame
                _fill_transform(static.transform, T_base_camera)
                self._static_tf = StaticTransformBroadcaster(self)
                self._static_tf.sendTransform(static)
                self._tf = TransformBroadcaster(self)
                self._tf_buffer = Buffer()
                # 不开 spin_thread: Humble 下它的独立线程在 Ctrl-C 时会抛 ExternalShutdownException。
                self._tf_listener = TransformListener(self._tf_buffer, self)
                self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value,
                                         self._on_camera_info, qos_profile_sensor_data)
                self.create_subscription(KeypointObservation,
                                         self.get_parameter("observation_topic").value,
                                         self._on_observation, qos_profile_sensor_data)
                self.create_timer(0.01, self._on_timer)

            def _now(self):
                return self.get_clock().now().nanoseconds * 1e-9

            def _on_camera_info(self, msg):
                if msg.header.frame_id != self._camera_frame:
                    # frame 名对不上时外参 TF 也查不到, 宁可不解算也不冒充。
                    self.get_logger().error(
                        f"CameraInfo frame_id={msg.header.frame_id!r} != camera_frame={self._camera_frame!r}, 拒收",
                        throttle_duration_sec=2.0)
                    return
                self._camera_info = msg

            def _on_timer(self):
                for key, items in self._frames.flush(self._now()):
                    self._solve_frame(items)

            def _schema(self, name):
                schema = self._schemas.get(name)
                if schema is None:
                    schema = load_schema(_config_dir() / "keypoint_schema.yaml", name)
                    self._schemas[name] = schema
                return schema

            def _on_observation(self, msg):
                key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
                for _, items in self._frames.add(key, msg, self._now()):
                    self._solve_frame(items)

            def _solve_frame(self, messages):
                if self._camera_info is None:
                    self.get_logger().warning("等待 CameraInfo，跳过观测")
                    return
                camera_frame = self._camera_info.header.frame_id
                by_schema = {}
                for msg in messages:
                    if msg.header.frame_id and camera_frame and msg.header.frame_id != camera_frame:
                        self.get_logger().warning("观测与 CameraInfo frame_id 不一致")
                        continue
                    try:
                        schema_name, class_id, image_points, confidence = _observation_to_arrays(msg)
                    except ValueError as exc:
                        self.get_logger().warning(f"无效视觉观测: {exc}")
                        continue
                    by_schema.setdefault(schema_name, []).append(
                        (msg, PartObservation(class_id, image_points, confidence)))
                for schema_name, entries in by_schema.items():
                    try:
                        schema = self._schema(schema_name)
                        K, D = _camera_info_to_arrays(self._camera_info)
                        solution = solve_best_candidate(
                            schema, [part for _, part in entries], self._estimator, K, D)
                    except (KeyError, ValueError) as exc:
                        self.get_logger().warning(f"无效视觉观测: {exc}")
                        continue
                    if solution.estimate is None:
                        self.get_logger().warning(f"PnP 无效: {'; '.join(solution.reasons)}")
                        continue
                    self._publish(entries[0][0].header.stamp, solution.estimate)

            def _publish(self, stamp, estimate):
                T_camera_station = estimate.T_camera_exchange_station

                # 相机系原始结果 + TF camera -> exchange_station
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = self._camera_frame
                _fill_pose(pose.pose, T_camera_station)
                self._pub.publish(pose)
                station_tf = TransformStamped()
                station_tf.header = pose.header
                station_tf.child_frame_id = self._station_frame
                _fill_transform(station_tf.transform, T_camera_station)
                self._tf.sendTransform(station_tf)

                # arm_base 系 /vision/exchange_pose (给 planning)。
                try:
                    transform = self._tf_buffer.lookup_transform(
                        self._base_frame, self._camera_frame, Time.from_msg(stamp), timeout=self._lookup_timeout)
                except TransformException as exc:
                    self.get_logger().warning(f"查外参 TF 失败, 丢弃 arm_base 输出: {exc}",
                                              throttle_duration_sec=1.0)
                    return
                base_pose = ExchangeStationPose()
                base_pose.header.stamp = stamp
                base_pose.header.frame_id = self._base_frame
                _fill_pose(base_pose.pose, compose(_matrix_from_transform_msg(transform.transform), T_camera_station))
                base_pose.valid = True       # pnp_node 只发有效解
                base_pose.confidence = 1.0   # 占位: 重投影误差 / 内点数尚未接入
                base_pose.station_id = 0
                self._base_pub.publish(base_pose)

        self._node = _Node(self)

    def spin(self):
        import rclpy
        rclpy.spin(self._node)


def main(args=None):
    import rclpy
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = PnpNode()
    try:
        node.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node._node.destroy_node()
        rclpy.try_shutdown()
