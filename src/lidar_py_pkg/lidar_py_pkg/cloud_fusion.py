#!/usr/bin/env python3
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time as RclTime
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx = qx * qx; yy = qy * qy; zz = qz * qz
    xy = qx * qy; xz = qx * qz; yz = qy * qz
    wx = qw * qx; wy = qw * qy; wz = qw * qz
    return np.array([
        [1.0 - 2.0*(yy+zz), 2.0*(xy-wz),       2.0*(xz+wy)],
        [2.0*(xy+wz),       1.0 - 2.0*(xx+zz), 2.0*(yz-wx)],
        [2.0*(xz-wy),       2.0*(yz+wx),       1.0 - 2.0*(xx+yy)],
    ], dtype=np.float64)


class CloudFusionNode(Node):
    """
    Fuses multiple PointCloud2 topics into a single cloud in target_frame.

    Fully numpy-vectorised pipeline:
      - read_points_numpy()  : reads entire cloud at once
      - matrix multiply      : transforms all points in one operation
      - np.unique + argsort  : fast voxel deduplication
      - direct .tobytes()    : fast binary PointCloud2 packing

    Age check removed — SICK MRS 6000 clock offset causes all messages
    to appear stale. TF uses RclTime() (latest available transform).
    """

    def __init__(self):
        super().__init__("cloud_fusion")

        self.declare_parameter(
            "input_topics",
            ["/front/cloud", "/right/cloud", "/back/cloud", "/left/cloud"]
        )
        self.declare_parameter("target_frame",    "base_link")
        self.declare_parameter("voxel_size",      0.05)
        self.declare_parameter("publish_topic",   "/nova/cloud_fused")
        self.declare_parameter("publish_rate_hz", 10.0)

        self.input_topics: List[str] = [
            str(x) for x in self.get_parameter("input_topics").value
        ]
        self.target_frame: str      = str(self.get_parameter("target_frame").value)
        self.voxel_size: float      = float(self.get_parameter("voxel_size").value)
        self.publish_topic: str     = str(self.get_parameter("publish_topic").value)
        self.publish_rate_hz: float = float(self.get_parameter("publish_rate_hz").value)

        self.add_on_set_parameters_callback(self._on_param_change)

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.latest_msgs: Dict[str, PointCloud2] = {}

        # Pre-cache TF transforms — static transforms never change
        # so we look them up once and reuse every cycle
        self._tf_cache: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}

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
            f"publish={self.publish_topic} rate={self.publish_rate_hz}Hz "
            f"[numpy fast mode + TF cache]"
        )

    def _on_param_change(self, params) -> SetParametersResult:
        for p in params:
            if p.name == "voxel_size":
                self.voxel_size = float(p.value)
                self.get_logger().info(f"[PARAM] voxel_size -> {self.voxel_size}")
            elif p.name == "publish_rate_hz":
                self.publish_rate_hz = float(p.value)
                self.timer.cancel()
                self.timer = self.create_timer(
                    1.0 / max(0.1, self.publish_rate_hz), self._on_timer
                )
                self.get_logger().info(f"[PARAM] publish_rate_hz -> {self.publish_rate_hz}")
        return SetParametersResult(successful=True)

    def _on_cloud(self, msg: PointCloud2, topic: str):
        self.latest_msgs[topic] = msg

    def _lookup_tf(self, target: str, source: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # Return cached transform if available
        cache_key = f"{target}<-{source}"
        if cache_key in self._tf_cache and self._tf_cache[cache_key] is not None:
            return self._tf_cache[cache_key]

        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, RclTime(),
                rclpy.duration.Duration(seconds=0.2)
            )
        except Exception as e:
            self.get_logger().warn(
                f"No TF {target} <- {source}: {e}",
                throttle_duration_sec=3.0
            )
            return None

        t = tf.transform.translation
        q = tf.transform.rotation
        R = quat_to_rot_matrix(q.x, q.y, q.z, q.w)
        T = np.array([t.x, t.y, t.z], dtype=np.float64)

        # Cache static transform for reuse every cycle
        self._tf_cache[cache_key] = (R, T)
        return R, T

    def _on_timer(self):
        if not self.latest_msgs:
            return

        now = self.get_clock().now()
        clouds: List[np.ndarray] = []
        any_intensity  = False
        last_valid_msg: Optional[PointCloud2] = None

        for topic in self.input_topics:
            msg = self.latest_msgs.get(topic)
            if msg is None:
                continue

            last_valid_msg = msg
            src_frame = msg.header.frame_id
            if not src_frame:
                continue

            tf_rt = self._lookup_tf(self.target_frame, src_frame)
            if tf_rt is None:
                continue
            R, T = tf_rt

            field_names   = [f.name for f in msg.fields]
            has_intensity = "intensity" in field_names
            any_intensity = any_intensity or has_intensity
            read_fields   = ["x", "y", "z"] + (["intensity"] if has_intensity else [])

            # ── Fast numpy read — entire cloud in one call ────────────
            try:
                pts = pc2.read_points_numpy(
                    msg, field_names=read_fields, skip_nans=True
                )
            except Exception:
                pts = np.array(
                    list(pc2.read_points(msg, field_names=read_fields, skip_nans=True)),
                    dtype=np.float32
                )

            if pts is None or len(pts) == 0:
                continue

            pts  = pts.astype(np.float64)
            xyz  = pts[:, :3]

            # ── Vectorised transform — matrix multiply all points ─────
            xyz_t = (xyz @ R.T) + T

            I = pts[:, 3:4] if has_intensity else np.zeros((len(xyz_t), 1), dtype=np.float64)
            clouds.append(np.hstack([xyz_t, I]))

        if not clouds or last_valid_msg is None:
            return

        # ── Merge all sensor clouds into one array ────────────────────
        all_pts = np.vstack(clouds)

        # ── Voxel deduplication — fully numpy ────────────────────────
        res = self.voxel_size
        if res > 1e-6:
            vox_idx = np.floor(all_pts[:, :3] / res).astype(np.int64)
            keys    = vox_idx[:, 0] * 1_000_003 + vox_idx[:, 1] * 1_009 + vox_idx[:, 2]
            r2      = np.sum(all_pts[:, :3] ** 2, axis=1)
            order   = np.argsort(r2)
            _, first_idx = np.unique(keys[order], return_index=True)
            fused = all_pts[order][first_idx]
        else:
            fused = all_pts

        # ── Build PointCloud2 — direct binary packing ─────────────────
        # Using .tobytes() is 10-50x faster than pc2.create_cloud()
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

        out              = PointCloud2()
        out.header       = header
        out.height       = 1
        out.width        = len(fused)
        out.fields       = fields
        out.is_bigendian = False
        out.is_dense     = True
        out.point_step   = 16 if any_intensity else 12
        out.row_step     = out.point_step * out.width
        out.data         = (fused[:, :4].astype(np.float32).tobytes() if any_intensity
                            else fused[:, :3].astype(np.float32).tobytes())

        self.pub.publish(out)

        self.get_logger().debug(
            f"Fused points={len(fused)} sources={len(clouds)}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CloudFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()