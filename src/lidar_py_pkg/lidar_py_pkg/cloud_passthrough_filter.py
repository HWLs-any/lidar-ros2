#!/usr/bin/env python3
import math
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


class CloudPassThroughFilter(Node):
    """
    Simple pass-through filter (range and XYZ bounds). Republish as PointCloud2.
    Keeps intensity if present.
    """

    def __init__(self):
        super().__init__("cloud_filter")

        self.declare_parameter("input_topic", "/front/cloud")
        self.declare_parameter("output_topic", "/front/cloud_filtered")
        self.declare_parameter("min_range", 0.2)
        self.declare_parameter("max_range", 80.0)
        self.declare_parameter("min_x", -math.inf)
        self.declare_parameter("max_x", math.inf)
        self.declare_parameter("min_y", -math.inf)
        self.declare_parameter("max_y", math.inf)
        self.declare_parameter("min_z", -math.inf)
        self.declare_parameter("max_z", math.inf)
        self.declare_parameter("output_frame_id", "")

        self.input_topic_ = str(self.get_parameter("input_topic").value)
        self.output_topic_ = str(self.get_parameter("output_topic").value)

        self.min_range_ = float(self.get_parameter("min_range").value)
        self.max_range_ = float(self.get_parameter("max_range").value)

        self.min_x_ = float(self.get_parameter("min_x").value)
        self.max_x_ = float(self.get_parameter("max_x").value)
        self.min_y_ = float(self.get_parameter("min_y").value)
        self.max_y_ = float(self.get_parameter("max_y").value)
        self.min_z_ = float(self.get_parameter("min_z").value)
        self.max_z_ = float(self.get_parameter("max_z").value)

        self.output_frame_id_ = str(self.get_parameter("output_frame_id").value)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.sub_ = self.create_subscription(PointCloud2, self.input_topic_, self.callback_on_cloud_, sensor_qos)
        self.pub_ = self.create_publisher(PointCloud2, self.output_topic_, 10)

        self.get_logger().info(
            f"cloud_filter: {self.input_topic_} -> {self.output_topic_}, "
            f"range [{self.min_range_}, {self.max_range_}]"
        )

    def callback_on_cloud_(self, msg: PointCloud2):
        field_names = [f.name for f in msg.fields]
        has_intensity = "intensity" in field_names
        read_fields = ["x", "y", "z"] + (["intensity"] if has_intensity else [])

        filtered_points: List[Tuple] = []
        for pt in pc2.read_points(msg, field_names=read_fields, skip_nans=True):
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            r = math.sqrt(x * x + y * y + z * z)

            if not (self.min_range_ <= r <= self.max_range_):
                continue
            if not (self.min_x_ <= x <= self.max_x_):
                continue
            if not (self.min_y_ <= y <= self.max_y_):
                continue
            if not (self.min_z_ <= z <= self.max_z_):
                continue

            if has_intensity:
                filtered_points.append((x, y, z, float(pt[3])))
            else:
                filtered_points.append((x, y, z))

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        if has_intensity:
            fields.append(PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1))

        header = msg.header
        if self.output_frame_id_:
            header.frame_id = self.output_frame_id_

        out_msg = pc2.create_cloud(header, fields, filtered_points)
        self.pub_.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CloudPassThroughFilter()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()