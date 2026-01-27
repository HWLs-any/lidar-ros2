#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2

import open3d as o3d


class PlaneRemoveO3D(Node):
    """
    Removes the dominant plane (floor) using Open3D RANSAC plane segmentation.
    Publishes remaining points as PointCloud2.
    """

    def __init__(self):
        super().__init__("plane_remove_o3d")

        self.declare_parameter("input_topic", "/nova/cloud_fused")
        self.declare_parameter("output_topic", "/nova/cloud_ground_removed")
        self.declare_parameter("distance_thresh", 0.04)
        self.declare_parameter("min_points", 200)

        self.in_topic = str(self.get_parameter("input_topic").value)
        self.out_topic = str(self.get_parameter("output_topic").value)
        self.dist = float(self.get_parameter("distance_thresh").value)
        self.min_points = int(self.get_parameter("min_points").value)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(PointCloud2, self.in_topic, self._cb, sensor_qos)
        self.pub = self.create_publisher(PointCloud2, self.out_topic, 10)

        self.get_logger().info(f"plane_remove_o3d {self.in_topic} -> {self.out_topic} dist={self.dist}")

    def _cb(self, msg: PointCloud2):
        pts = np.asarray(
            [p for p in pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)],
            dtype=np.float32,
        )
        if pts.shape[0] < self.min_points:
            return

        o3 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))

        model, inliers = o3.segment_plane(
            distance_threshold=self.dist,
            ransac_n=3,
            num_iterations=300,
        )

        if len(inliers) == 0:
            return

        mask = np.ones(pts.shape[0], dtype=bool)
        mask[np.array(inliers, dtype=int)] = False
        out_pts = pts[mask]

        header = msg.header
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        self.pub.publish(pc2.create_cloud(header, fields, out_pts))


def main(args=None):
    rclpy.init(args=args)
    node = PlaneRemoveO3D()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
