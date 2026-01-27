#!/usr/bin/env python3
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


class CloudMonitorNode(Node):
    """
    Subscribes to a PointCloud2 and prints basic stats:
    - message rate
    - point count
    - min/avg/max range
    - intensity presence
    """

    def __init__(self):
        super().__init__("cloud_monitor")

        self.declare_parameter("cloud_topic", "/front/cloud")
        self.declare_parameter("log_hz", 1.0)
        self.declare_parameter("expected_frame_id", "")

        self.cloud_topic_ = str(self.get_parameter("cloud_topic").value)
        self.log_hz_ = float(self.get_parameter("log_hz").value)
        self.expected_frame_id_ = str(self.get_parameter("expected_frame_id").value)

        self.log_period_ = 1.0 / max(0.1, self.log_hz_)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub_ = self.create_subscription(PointCloud2, self.cloud_topic_, self.callback_on_cloud_, sensor_qos)

        self.msg_count_ = 0
        self.last_log_time_ = self.get_clock().now()
        self.last_stats_: Optional[dict] = None

        self.get_logger().info(f"cloud_monitor subscribing to {self.cloud_topic_}")

    def callback_on_cloud_(self, msg: PointCloud2):
        self.msg_count_ += 1

        if self.expected_frame_id_ and msg.header.frame_id != self.expected_frame_id_:
            self.get_logger().warn_once(
                f'Incoming frame_id "{msg.header.frame_id}" differs from expected "{self.expected_frame_id_}"'
            )

        field_names = [f.name for f in msg.fields]
        has_intensity = "intensity" in field_names
        read_fields = ["x", "y", "z"] + (["intensity"] if has_intensity else [])

        n = 0
        min_r = float("inf")
        max_r = 0.0
        sum_r = 0.0

        for pt in pc2.read_points(msg, field_names=read_fields, skip_nans=True):
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            r = math.sqrt(x * x + y * y + z * z)
            n += 1
            sum_r += r
            min_r = min(min_r, r)
            max_r = max(max_r, r)

        avg_r = (sum_r / n) if n > 0 else 0.0
        self.last_stats_ = {
            "points": n,
            "min_r": min_r if n > 0 else 0.0,
            "max_r": max_r,
            "avg_r": avg_r,
            "has_intensity": has_intensity,
            "frame": msg.header.frame_id,
        }

        now = self.get_clock().now()
        elapsed = (now - self.last_log_time_).nanoseconds * 1e-9
        if elapsed >= self.log_period_:
            rate = self.msg_count_ / max(1e-6, elapsed)
            s = self.last_stats_
            if s is not None:
                self.get_logger().info(
                    f'[{self.cloud_topic_}] rate: {rate:.2f} Hz, points: {s["points"]}, '
                    f'range min/avg/max: {s["min_r"]:.2f}/{s["avg_r"]:.2f}/{s["max_r"]:.2f} m, '
                    f'intensity: {s["has_intensity"]}, frame: {s["frame"]}'
                )
            else:
                self.get_logger().info(f"[{self.cloud_topic_}] rate: {rate:.2f} Hz (no stats yet)")
            self.last_log_time_ = now
            self.msg_count_ = 0


def main(args=None):
    rclpy.init(args=args)
    node = CloudMonitorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()