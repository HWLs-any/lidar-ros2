#!/usr/bin/env python3
import math
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

try:
    from sklearn.cluster import DBSCAN
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float32)


class CloudDetectorNode(Node):
    """
    Detects objects by:
    - optional TF transform to target_frame
    - passthrough filtering by range and z
    - optional voxel downsample
    - DBSCAN clustering in XY
    - publishes MarkerArray boxes in target_frame
    """

    def __init__(self):
        super().__init__("cloud_detector")

        self.declare_parameter("input_topic", "/nova/cloud_fused")
        self.declare_parameter("marker_topic", "/nova/detections")
        self.declare_parameter("target_frame", "base_link")

        self.declare_parameter("min_range", 0.5)
        self.declare_parameter("max_range", 80.0)
        self.declare_parameter("min_z", -0.5)
        self.declare_parameter("max_z", 3.0)

        self.declare_parameter("voxel_size", 0.05)

        self.declare_parameter("cluster_eps", 0.35)
        self.declare_parameter("cluster_min_pts", 15)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        self.min_range = float(self.get_parameter("min_range").value)
        self.max_range = float(self.get_parameter("max_range").value)
        self.min_z = float(self.get_parameter("min_z").value)
        self.max_z = float(self.get_parameter("max_z").value)

        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.cluster_eps = float(self.get_parameter("cluster_eps").value)
        self.cluster_min_pts = int(self.get_parameter("cluster_min_pts").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self._on_cloud, sensor_qos)
        self.pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.get_logger().info(
            f"cloud_detector input={self.input_topic} markers={self.marker_topic} "
            f"target_frame={self.target_frame} "
            f"filters: range[{self.min_range},{self.max_range}] z[{self.min_z},{self.max_z}] "
            f"voxel={self.voxel_size} dbscan eps={self.cluster_eps} min_pts={self.cluster_min_pts} "
            f"sklearn={'yes' if HAVE_SKLEARN else 'no'}"
        )

    def _lookup_tf(self, target: str, source: str, stamp: rclpy.time.Time) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            tf = self.tf_buffer.lookup_transform(target, source, stamp, rclpy.duration.Duration(seconds=0.2))
        except (LookupException, ConnectivityException, ExtrapolationException):
            try:
                tf = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.2))
            except Exception as e:
                self.get_logger().warn(f"No TF {target} <- {source}: {e}")
                return None

        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
        T = np.array([t.x, t.y, t.z], dtype=np.float32)
        return R, T

    def _voxel_downsample(self, P: np.ndarray, voxel: float) -> np.ndarray:
        if P.shape[0] == 0:
            return P
        ix = np.floor(P[:, 0] / voxel).astype(np.int64)
        iy = np.floor(P[:, 1] / voxel).astype(np.int64)
        iz = np.floor(P[:, 2] / voxel).astype(np.int64)
        keys = np.stack([ix, iy, iz], axis=1)

        r2 = np.sum(P * P, axis=1)
        vox = {}
        for k, d, p in zip(map(tuple, keys), r2, P):
            if (k not in vox) or (d < vox[k][0]):
                vox[k] = (d, p)
        return np.stack([v[1] for v in vox.values()], axis=0)

    def _cluster(self, P: np.ndarray) -> List[np.ndarray]:
        if P.shape[0] < self.cluster_min_pts:
            return []

        if not HAVE_SKLEARN:
            # fallback: one cluster
            return [P]

        X = P[:, :2]  # XY
        db = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_pts).fit(X)
        labels = db.labels_

        clusters: List[np.ndarray] = []
        for lab in set(labels):
            if lab == -1:
                continue
            idx = np.where(labels == lab)[0]
            if idx.size >= self.cluster_min_pts:
                clusters.append(P[idx])
        return clusters

    def _on_cloud(self, msg: PointCloud2):
        src_frame = msg.header.frame_id
        if not src_frame:
            return

        # Read points
        pts: List[Tuple[float, float, float]] = []
        for pt in pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True):
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            r = math.sqrt(x * x + y * y + z * z)
            if (self.min_range <= r <= self.max_range) and (self.min_z <= z <= self.max_z):
                pts.append((x, y, z))

        if not pts:
            self.pub.publish(MarkerArray())
            return

        P = np.asarray(pts, dtype=np.float32)

        # Transform into target_frame if needed
        if src_frame != self.target_frame:
            stamp = rclpy.time.Time.from_msg(msg.header.stamp)
            tf_rt = self._lookup_tf(self.target_frame, src_frame, stamp)
            if tf_rt is None:
                return
            R, T = tf_rt
            P = (P @ R.T) + T  # apply rotation+translation

        # Downsample
        if self.voxel_size > 1e-6:
            P = self._voxel_downsample(P, self.voxel_size)

        clusters = self._cluster(P)
        if not clusters:
            self.pub.publish(MarkerArray())
            return

        now = self.get_clock().now().to_msg()
        lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

        out = MarkerArray()
        for i, C in enumerate(clusters):
            mins = C.min(axis=0)
            maxs = C.max(axis=0)
            center = (mins + maxs) / 2.0
            size = (maxs - mins)

            m = Marker()
            m.header.frame_id = self.target_frame
            m.header.stamp = now
            m.ns = "detections"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(center[0])
            m.pose.position.y = float(center[1])
            m.pose.position.z = float(center[2])
            m.pose.orientation.w = 1.0
            m.scale.x = max(0.05, float(size[0]))
            m.scale.y = max(0.05, float(size[1]))
            m.scale.z = max(0.05, float(size[2]))

            # keep color simple; RViz can override if desired
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.2
            m.color.a = 0.6

            m.lifetime = lifetime
            out.markers.append(m)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CloudDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
