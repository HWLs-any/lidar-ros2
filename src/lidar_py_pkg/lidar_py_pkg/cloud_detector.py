#!/usr/bin/env python3
import math
import time
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int32, Float32
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
    DBSCAN-based object detector.

    All tunable parameters respond to 'ros2 param set' at runtime —
    no restart needed. The parameter callback updates local variables
    immediately so the next processing cycle uses the new values.

    Tunable at runtime:
        cluster_eps, cluster_min_pts, voxel_size,
        min_range, max_range, min_z, max_z
    """

    def __init__(self):
        super().__init__("cloud_detector")

        # ── Declare all parameters ────────────────────────────────────
        self.declare_parameter("input_topic",     "/nova/cloud_fused")
        self.declare_parameter("marker_topic",    "/nova/detections")
        self.declare_parameter("target_frame",    "base_link")
        self.declare_parameter("min_range",       0.5)
        self.declare_parameter("max_range",       80.0)
        self.declare_parameter("min_z",          -0.5)
        self.declare_parameter("max_z",           3.0)
        self.declare_parameter("voxel_size",      0.05)
        self.declare_parameter("cluster_eps",     0.35)
        self.declare_parameter("cluster_min_pts", 15)
        self.declare_parameter("metrics_prefix",  "")
        self.declare_parameter("publish_metrics", True)

        # ── Read initial values ───────────────────────────────────────
        self.input_topic     = str(self.get_parameter("input_topic").value)
        self.marker_topic    = str(self.get_parameter("marker_topic").value)
        self.target_frame    = str(self.get_parameter("target_frame").value)
        self.min_range       = float(self.get_parameter("min_range").value)
        self.max_range       = float(self.get_parameter("max_range").value)
        self.min_z           = float(self.get_parameter("min_z").value)
        self.max_z           = float(self.get_parameter("max_z").value)
        self.voxel_size      = float(self.get_parameter("voxel_size").value)
        self.cluster_eps     = float(self.get_parameter("cluster_eps").value)
        self.cluster_min_pts = int(self.get_parameter("cluster_min_pts").value)
        self.metrics_prefix  = str(self.get_parameter("metrics_prefix").value)
        self.metrics_enabled = bool(self.get_parameter("publish_metrics").value)

        # Auto-derive metrics prefix from input topic if not set
        if not self.metrics_prefix:
            topic_tag = self.input_topic.strip("/").replace("/", "_")
            self.metrics_prefix = f"/metrics/dbscan/{topic_tag}"

        # ── Register dynamic parameter callback ───────────────────────
        # This is what makes 'ros2 param set' work at runtime.
        # Every time a parameter changes, _on_param_change() is called
        # and updates the local variable immediately.
        self.add_on_set_parameters_callback(self._on_param_change)

        # ── TF ────────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.sub = self.create_subscription(
            PointCloud2, self.input_topic, self._on_cloud, sensor_qos
        )
        self.pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # ── Metrics publishers ────────────────────────────────────────
        if self.metrics_enabled:
            self.pub_count  = self.create_publisher(
                Int32,   self.metrics_prefix + "/detections_count", 10
            )
            self.pub_ms     = self.create_publisher(
                Float32, self.metrics_prefix + "/processing_ms", 10
            )
            self.pub_points = self.create_publisher(
                Int32,   self.metrics_prefix + "/points_in", 10
            )

        self.get_logger().info(
            f"cloud_detector input={self.input_topic} markers={self.marker_topic} "
            f"target_frame={self.target_frame} "
            f"filters: range[{self.min_range},{self.max_range}] z[{self.min_z},{self.max_z}] "
            f"voxel={self.voxel_size} dbscan eps={self.cluster_eps} "
            f"min_pts={self.cluster_min_pts} "
            f"sklearn={'yes' if HAVE_SKLEARN else 'no'} "
            f"metrics={'on → ' + self.metrics_prefix if self.metrics_enabled else 'off'}"
        )

    # ── Dynamic parameter callback ────────────────────────────────────
    def _on_param_change(self, params) -> SetParametersResult:
        """
        Called automatically by ROS2 when 'ros2 param set' is used.
        Updates local variables so the change takes effect immediately
        on the next processing cycle — no restart needed.
        """
        for p in params:
            if p.name == "cluster_eps":
                self.cluster_eps = float(p.value)
                self.get_logger().info(f"[PARAM] cluster_eps → {self.cluster_eps}")

            elif p.name == "cluster_min_pts":
                self.cluster_min_pts = int(p.value)
                self.get_logger().info(f"[PARAM] cluster_min_pts → {self.cluster_min_pts}")

            elif p.name == "voxel_size":
                self.voxel_size = float(p.value)
                self.get_logger().info(f"[PARAM] voxel_size → {self.voxel_size}")

            elif p.name == "min_range":
                self.min_range = float(p.value)
                self.get_logger().info(f"[PARAM] min_range → {self.min_range}")

            elif p.name == "max_range":
                self.max_range = float(p.value)
                self.get_logger().info(f"[PARAM] max_range → {self.max_range}")

            elif p.name == "min_z":
                self.min_z = float(p.value)
                self.get_logger().info(f"[PARAM] min_z → {self.min_z}")

            elif p.name == "max_z":
                self.max_z = float(p.value)
                self.get_logger().info(f"[PARAM] max_z → {self.max_z}")

        return SetParametersResult(successful=True)

    # ── Metrics helper ────────────────────────────────────────────────
    def _publish_metrics(self, points_in: int, detections: int, ms: float) -> None:
        if not self.metrics_enabled:
            return
        c = Int32();   c.data = detections
        t = Float32(); t.data = float(ms)
        p = Int32();   p.data = points_in
        self.pub_count.publish(c)
        self.pub_ms.publish(t)
        self.pub_points.publish(p)

    # ── TF lookup ─────────────────────────────────────────────────────
    def _lookup_tf(
        self, target: str, source: str, stamp: rclpy.time.Time
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
        T = np.array([t.x, t.y, t.z], dtype=np.float32)
        return R, T

    # ── Voxel downsample ──────────────────────────────────────────────
    def _voxel_downsample(self, P: np.ndarray, voxel: float) -> np.ndarray:
        if P.shape[0] == 0:
            return P
        ix   = np.floor(P[:, 0] / voxel).astype(np.int64)
        iy   = np.floor(P[:, 1] / voxel).astype(np.int64)
        iz   = np.floor(P[:, 2] / voxel).astype(np.int64)
        keys = np.stack([ix, iy, iz], axis=1)
        r2   = np.sum(P * P, axis=1)
        vox  = {}
        for k, d, p in zip(map(tuple, keys), r2, P):
            if (k not in vox) or (d < vox[k][0]):
                vox[k] = (d, p)
        return np.stack([v[1] for v in vox.values()], axis=0)

    # ── DBSCAN clustering ─────────────────────────────────────────────
    def _cluster(self, P: np.ndarray) -> List[np.ndarray]:
        if P.shape[0] < self.cluster_min_pts:
            return []
        if not HAVE_SKLEARN:
            return [P]

        X      = P[:, :2]
        db     = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_pts
        ).fit(X)
        labels = db.labels_

        clusters: List[np.ndarray] = []
        for lab in set(labels):
            if lab == -1:
                continue
            idx = np.where(labels == lab)[0]
            if idx.size >= self.cluster_min_pts:
                clusters.append(P[idx])
        return clusters

    # ── Main callback ─────────────────────────────────────────────────
    def _on_cloud(self, msg: PointCloud2):
        t0        = time.monotonic()
        src_frame = msg.header.frame_id
        if not src_frame:
            return

        pts: List[Tuple[float, float, float]] = []
        for pt in pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True):
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
            r = math.sqrt(x * x + y * y + z * z)
            if (self.min_range <= r <= self.max_range) and (self.min_z <= z <= self.max_z):
                pts.append((x, y, z))

        points_in = len(pts)

        if not pts:
            self.pub.publish(MarkerArray())
            self._publish_metrics(points_in, 0, (time.monotonic() - t0) * 1000.0)
            return

        P = np.asarray(pts, dtype=np.float32)

        if src_frame != self.target_frame:
            stamp  = rclpy.time.Time.from_msg(msg.header.stamp)
            tf_rt  = self._lookup_tf(self.target_frame, src_frame, stamp)
            if tf_rt is None:
                return
            R, T = tf_rt
            P = (P @ R.T) + T

        if self.voxel_size > 1e-6:
            P = self._voxel_downsample(P, self.voxel_size)

        clusters = self._cluster(P)
        if not clusters:
            self.pub.publish(MarkerArray())
            self._publish_metrics(points_in, 0, (time.monotonic() - t0) * 1000.0)
            return

        now      = self.get_clock().now().to_msg()
        lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

        out = MarkerArray()
        for i, C in enumerate(clusters):
            mins   = C.min(axis=0)
            maxs   = C.max(axis=0)
            center = (mins + maxs) / 2.0
            size   = (maxs - mins)

            m = Marker()
            m.header.frame_id    = self.target_frame
            m.header.stamp       = now
            m.ns                 = "detections"
            m.id                 = i
            m.type               = Marker.CUBE
            m.action             = Marker.ADD
            m.pose.position.x    = float(center[0])
            m.pose.position.y    = float(center[1])
            m.pose.position.z    = float(center[2])
            m.pose.orientation.w = 1.0
            m.scale.x            = max(0.05, float(size[0]))
            m.scale.y            = max(0.05, float(size[1]))
            m.scale.z            = max(0.05, float(size[2]))
            m.color.r            = 0.0
            m.color.g            = 1.0
            m.color.b            = 0.2
            m.color.a            = 0.6
            m.lifetime           = lifetime
            out.markers.append(m)

        self.pub.publish(out)

        ms = (time.monotonic() - t0) * 1000.0
        self._publish_metrics(points_in, len(out.markers), ms)

        self.get_logger().debug(
            f"DBSCAN ({self.input_topic}): points={points_in} "
            f"clusters={len(out.markers)} time={ms:.1f}ms"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CloudDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()