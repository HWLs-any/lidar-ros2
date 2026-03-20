#!/usr/bin/env python3
import math
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclTime
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> List[List[float]]:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ]


def apply_transform(
    x: float, y: float, z: float,
    R: List[List[float]],
    t: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (
        R[0][0] * x + R[0][1] * y + R[0][2] * z + t[0],
        R[1][0] * x + R[1][1] * y + R[1][2] * z + t[1],
        R[2][0] * x + R[2][1] * y + R[2][2] * z + t[2],
    )


class CloudFusionNode(Node):
    """
    Fuses multiple PointCloud2 topics into a single cloud in target_frame.

    All tunable parameters respond to 'ros2 param set' at runtime —
    no restart needed. The parameter callback updates local variables
    immediately so the next timer cycle uses the new values.

    Tunable at runtime:
        voxel_size, max_age_sec, publish_rate_hz
    """

    def __init__(self):
        super().__init__("cloud_fusion")

        # ── Declare all parameters ────────────────────────────────────
        self.declare_parameter(
            "input_topics",
            ["/front/cloud", "/right/cloud", "/back/cloud", "/left/cloud"]
        )
        self.declare_parameter("target_frame",    "base_link")
        self.declare_parameter("voxel_size",      0.05)
        self.declare_parameter("max_age_sec",     2.0)
        self.declare_parameter("publish_topic",   "/nova/cloud_fused")
        self.declare_parameter("publish_rate_hz", 10.0)

        # ── Read initial values ───────────────────────────────────────
        self.input_topics: List[str] = [
            str(x) for x in self.get_parameter("input_topics").value
        ]
        self.target_frame: str      = str(self.get_parameter("target_frame").value)
        self.voxel_size: float      = float(self.get_parameter("voxel_size").value)
        self.max_age_sec: float     = float(self.get_parameter("max_age_sec").value)
        self.publish_topic: str     = str(self.get_parameter("publish_topic").value)
        self.publish_rate_hz: float = float(self.get_parameter("publish_rate_hz").value)

        # ── Register dynamic parameter callback ───────────────────────
        # This is what makes 'ros2 param set' and the web tuner work.
        # Every time a parameter changes, _on_param_change() is called
        # and updates the local variable immediately.
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── TF ────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_msgs: Dict[str, PointCloud2] = {}

        self.pub = self.create_publisher(PointCloud2, self.publish_topic, 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        for topic in self.input_topics:
            self.create_subscription(
                PointCloud2, topic,
                lambda msg, t=topic: self._on_cloud(msg, t),
                sensor_qos
            )
            self.get_logger().info(f"Subscribed: {topic}")

        period = 1.0 / max(0.1, self.publish_rate_hz)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"cloud_fusion target_frame={self.target_frame} voxel={self.voxel_size} "
            f"max_age={self.max_age_sec}s publish={self.publish_topic} "
            f"rate={self.publish_rate_hz}Hz"
        )

    # ── Dynamic parameter callback ────────────────────────────────────
    def _on_param_change(self, params) -> SetParametersResult:
        """
        Called automatically by ROS2 when 'ros2 param set' is used.
        Updates local variables immediately — no restart needed.
        """
        for p in params:
            if p.name == "voxel_size":
                self.voxel_size = float(p.value)
                self.get_logger().info(f"[PARAM] voxel_size → {self.voxel_size}")

            elif p.name == "max_age_sec":
                self.max_age_sec = float(p.value)
                self.get_logger().info(f"[PARAM] max_age_sec → {self.max_age_sec}")

            elif p.name == "publish_rate_hz":
                self.publish_rate_hz = float(p.value)
                # Recreate timer with new period
                self.timer.cancel()
                period = 1.0 / max(0.1, self.publish_rate_hz)
                self.timer = self.create_timer(period, self._on_timer)
                self.get_logger().info(f"[PARAM] publish_rate_hz → {self.publish_rate_hz}")

        return SetParametersResult(successful=True)

    # ── Cloud receive callback ────────────────────────────────────────
    def _on_cloud(self, msg: PointCloud2, topic: str):
        self.latest_msgs[topic] = msg

    # ── TF lookup ─────────────────────────────────────────────────────
    def _lookup_tf(
        self,
        target: str,
        source: str,
        stamp: RclTime
    ) -> Optional[Tuple[List[List[float]], Tuple[float, float, float]]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, stamp,
                rclpy.duration.Duration(seconds=0.2)
            )
        except (LookupException, ConnectivityException, ExtrapolationException):
            try:
                tf = self.tf_buffer.lookup_transform(
                    target, source,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.2)
                )
            except Exception as e:
                self.get_logger().warn(f"No TF {target} <- {source}: {e}")
                return None

        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
        return R, (t.x, t.y, t.z)

    # ── Timer callback — fuse and publish ─────────────────────────────
    def _on_timer(self):
        if not self.latest_msgs:
            return

        now = self.get_clock().now()

        all_pts: List[Tuple[float, float, float, float]] = []
        any_intensity  = False
        used_sources   = 0
        last_valid_msg: Optional[PointCloud2] = None

        for topic in self.input_topics:
            msg = self.latest_msgs.get(topic)
            if msg is None:
                continue

            last_valid_msg = msg

            msg_time = RclTime.from_msg(msg.header.stamp)
            age = (now - msg_time).nanoseconds * 1e-9
            if age > self.max_age_sec:
                continue

            src_frame = msg.header.frame_id
            if not src_frame:
                self.get_logger().warn(f"{topic}: empty frame_id, skipping")
                continue

            tf_rt = self._lookup_tf(self.target_frame, src_frame, msg_time)
            if tf_rt is None:
                continue
            R, tvec = tf_rt

            field_names   = [f.name for f in msg.fields]
            has_intensity = "intensity" in field_names
            any_intensity = any_intensity or has_intensity
            read_fields   = ["x", "y", "z"] + (["intensity"] if has_intensity else [])

            for pt in pc2.read_points(msg, field_names=read_fields, skip_nans=True):
                x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                X, Y, Z = apply_transform(x, y, z, R, tvec)
                I = float(pt[3]) if has_intensity else 0.0
                all_pts.append((X, Y, Z, I))

            used_sources += 1

        if not all_pts:
            return

        if last_valid_msg is None:
            return

        # Voxel deduplication
        res = self.voxel_size
        if res > 1e-6:
            vox: Dict[Tuple[int, int, int], Tuple[float, float, float, float, float]] = {}
            for (x, y, z, I) in all_pts:
                ix  = math.floor(x / res)
                iy  = math.floor(y / res)
                iz  = math.floor(z / res)
                key = (ix, iy, iz)
                r2  = x * x + y * y + z * z
                if key not in vox or r2 < vox[key][4]:
                    vox[key] = (x, y, z, I, r2)
            fused = [(v[0], v[1], v[2], v[3]) for v in vox.values()]
        else:
            fused = all_pts

        header          = last_valid_msg.header
        header.stamp    = now.to_msg()
        header.frame_id = self.target_frame

        fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]

        if any_intensity:
            fields.append(
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
            )
            out_points = fused
        else:
            out_points = [(x, y, z) for (x, y, z, _I) in fused]

        out = pc2.create_cloud(header, fields, out_points)
        self.pub.publish(out)

        self.get_logger().debug(
            f"Fused points={len(out_points)} sources_used={used_sources}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CloudFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()