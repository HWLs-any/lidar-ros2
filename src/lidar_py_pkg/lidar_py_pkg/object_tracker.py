#!/usr/bin/env python3
"""
object_tracker.py
-----------------
Subscribes to a MarkerArray of detections (from PCL or DBSCAN detector),
associates them to persistent tracks using the Hungarian algorithm,
maintains a Kalman filter per track (constant-velocity model),
and publishes confirmed tracks as a MarkerArray with stable IDs.

Drop-in addition to the existing pipeline:
  /detections/pcl/fused  -->  [object_tracker]  -->  /tracks/markers
                                                  -->  /tracks/text
                                                  -->  /tracks/velocity

Fits directly after either PCL or DBSCAN detector — no other node changes needed.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

# Hungarian algorithm — best available, fallback to greedy
try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ─────────────────────────────────────────────────────────────────────────────
# Track state constants
# ─────────────────────────────────────────────────────────────────────────────
TENTATIVE = 0   # newly created — not shown in RViz yet
CONFIRMED = 1   # shown with stable colour
LOST      = 2   # missed recently — shown faded, still predicted


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter — Constant Velocity Model (6-state, 3-measurement)
# ─────────────────────────────────────────────────────────────────────────────
# State  : [x, y, z, vx, vy, vz]^T
# Measure: [x, y, z]^T
# ─────────────────────────────────────────────────────────────────────────────
class KalmanTrack:
    """
    One tracked object. Holds Kalman filter state, lifecycle counters,
    and a smoothed bounding box size.
    """

    _next_id: int = 0

    def __init__(
        self,
        cx: float, cy: float, cz: float,
        sx: float, sy: float, sz: float,
        dt: float,
        process_noise: float,
        measure_noise: float,
    ) -> None:
        self.id: int = KalmanTrack._next_id
        KalmanTrack._next_id += 1

        self.state: int = TENTATIVE
        self.hits: int = 1       # consecutive matched frames
        self.misses: int = 0     # consecutive unmatched frames
        self.age: int = 1        # total frames since creation
        self.dt: float = dt

        # Kalman state vector [x, y, z, vx, vy, vz]
        self.x = np.array([cx, cy, cz, 0.0, 0.0, 0.0], dtype=np.float64)

        # State transition matrix F (constant velocity)
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix H — we observe position only
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise covariance Q
        q = process_noise
        self.Q = np.diag([q, q, q, q * 2.0, q * 2.0, q * 2.0])

        # Measurement noise covariance R
        r = measure_noise
        self.R = np.diag([r, r, r * 2.0])   # z noisier than x/y

        # Initial state covariance P
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])

        # Smoothed bounding box size (exponential moving average)
        self.size = np.array([sx, sy, sz], dtype=np.float64)
        self._size_alpha: float = 0.3   # smoothing factor

    # ------------------------------------------------------------------
    # Kalman predict step — call once per timer tick before association
    # ------------------------------------------------------------------
    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.misses += 1   # will be reset to 0 if matched this cycle

    # ------------------------------------------------------------------
    # Kalman update step — call when a detection is matched
    # ------------------------------------------------------------------
    def update(self, cx: float, cy: float, cz: float,
               sx: float, sy: float, sz: float) -> None:
        z = np.array([cx, cy, cz], dtype=np.float64)
        y = z - self.H @ self.x                        # innovation
        S = self.H @ self.P @ self.H.T + self.R        # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        # Smooth bounding box size
        new_size = np.array([sx, sy, sz], dtype=np.float64)
        self.size = (1.0 - self._size_alpha) * self.size + self._size_alpha * new_size

        self.hits += 1
        self.misses = 0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def position(self) -> Tuple[float, float, float]:
        return float(self.x[0]), float(self.x[1]), float(self.x[2])

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return float(self.x[3]), float(self.x[4]), float(self.x[5])

    @property
    def speed(self) -> float:
        vx, vy, vz = self.velocity
        return math.sqrt(vx * vx + vy * vy + vz * vz)


# ─────────────────────────────────────────────────────────────────────────────
# Association helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_cost_matrix(
    tracks: List[KalmanTrack],
    detections: List[Tuple[float, float, float]],
) -> np.ndarray:
    """Euclidean distance between each predicted track position and each detection."""
    n, m = len(tracks), len(detections)
    C = np.full((n, m), fill_value=1e9, dtype=np.float64)
    for i, trk in enumerate(tracks):
        px, py, pz = trk.position
        for j, (dx, dy, dz) in enumerate(detections):
            C[i, j] = math.sqrt((px - dx) ** 2 + (py - dy) ** 2 + (pz - dz) ** 2)
    return C


def _associate(
    tracks: List[KalmanTrack],
    detections: List[Tuple[float, float, float]],
    max_dist: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Returns:
        matched      : list of (track_idx, det_idx)
        unmatched_trk: track indices with no match
        unmatched_det: detection indices with no match
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    C = _build_cost_matrix(tracks, detections)

    if HAVE_SCIPY:
        row_ind, col_ind = linear_sum_assignment(C)
        pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        # Greedy fallback — good enough for < 50 objects
        pairs = []
        used_det: set = set()
        for i in range(len(tracks)):
            best_j, best_d = -1, 1e9
            for j in range(len(detections)):
                if j not in used_det and C[i, j] < best_d:
                    best_d, best_j = C[i, j], j
            if best_j >= 0:
                pairs.append((i, best_j))
                used_det.add(best_j)

    matched, unmatched_trk, unmatched_det = [], [], []
    matched_t, matched_d = set(), set()

    for i, j in pairs:
        if C[i, j] <= max_dist:
            matched.append((i, j))
            matched_t.add(i)
            matched_d.add(j)

    unmatched_trk = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_det = [j for j in range(len(detections)) if j not in matched_d]

    return matched, unmatched_trk, unmatched_det


# ─────────────────────────────────────────────────────────────────────────────
# ROS 2 Node
# ─────────────────────────────────────────────────────────────────────────────
class ObjectTrackerNode(Node):
    """
    Subscribes to a MarkerArray of bounding-box detections,
    runs nearest-neighbour Hungarian association + per-track Kalman filter,
    and publishes:
      - /tracks/markers  — bounding boxes coloured by track state
      - /tracks/text     — track ID + speed text labels
      - /tracks/velocity — velocity arrow markers
    """

    def __init__(self) -> None:
        super().__init__("object_tracker")

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter("input_topic",   "/detections/pcl/fused")
        self.declare_parameter("output_topic",  "/tracks/markers")
        self.declare_parameter("text_topic",    "/tracks/text")
        self.declare_parameter("vel_topic",     "/tracks/velocity")
        self.declare_parameter("target_frame",  "base_link")

        # Lifecycle thresholds
        self.declare_parameter("min_hits",      3)    # frames to confirm a track
        self.declare_parameter("max_misses",    5)    # frames missed before deletion
        self.declare_parameter("max_match_dist", 1.5) # metres — max association distance

        # Kalman tuning
        self.declare_parameter("dt",            0.1)  # seconds — matches 10 Hz fusion
        self.declare_parameter("process_noise", 0.5)  # Q diagonal scale
        self.declare_parameter("measure_noise", 0.3)  # R diagonal scale

        # Visualisation
        self.declare_parameter("marker_lifetime_sec", 0.5)
        self.declare_parameter("publish_velocity_arrows", True)

        # ── Read parameters ───────────────────────────────────────────
        self.input_topic   = str(self.get_parameter("input_topic").value)
        self.output_topic  = str(self.get_parameter("output_topic").value)
        self.text_topic    = str(self.get_parameter("text_topic").value)
        self.vel_topic     = str(self.get_parameter("vel_topic").value)
        self.target_frame  = str(self.get_parameter("target_frame").value)

        self.min_hits      = int(self.get_parameter("min_hits").value)
        self.max_misses    = int(self.get_parameter("max_misses").value)
        self.max_match_dist = float(self.get_parameter("max_match_dist").value)

        self.dt            = float(self.get_parameter("dt").value)
        self.process_noise = float(self.get_parameter("process_noise").value)
        self.measure_noise = float(self.get_parameter("measure_noise").value)

        self.marker_lifetime = float(self.get_parameter("marker_lifetime_sec").value)
        self.pub_vel_arrows  = bool(self.get_parameter("publish_velocity_arrows").value)

        # ── Track storage ─────────────────────────────────────────────
        self.tracks: Dict[int, KalmanTrack] = {}   # track_id -> KalmanTrack

        # ── QoS — matches existing nodes ─────────────────────────────
        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # ── Subscribers / Publishers ──────────────────────────────────
        self.sub = self.create_subscription(
            MarkerArray, self.input_topic, self._on_detections, det_qos
        )
        self.pub_boxes = self.create_publisher(MarkerArray, self.output_topic, 10)
        self.pub_text  = self.create_publisher(MarkerArray, self.text_topic,   10)
        self.pub_vel   = self.create_publisher(MarkerArray, self.vel_topic,    10)

        self.get_logger().info(
            f"object_tracker listening on {self.input_topic} | "
            f"min_hits={self.min_hits} max_misses={self.max_misses} "
            f"max_match_dist={self.max_match_dist}m dt={self.dt}s | "
            f"scipy={'yes' if HAVE_SCIPY else 'no (greedy fallback)'}"
        )

    # ── Main callback ─────────────────────────────────────────────────
    def _on_detections(self, msg: MarkerArray) -> None:
        now = self.get_clock().now()

        # ── 1. Extract detection centroids + sizes from MarkerArray ───
        detections: List[Tuple[float, float, float]] = []
        det_sizes:  List[Tuple[float, float, float]] = []

        for m in msg.markers:
            if m.action != Marker.ADD:
                continue
            detections.append((m.pose.position.x,
                                m.pose.position.y,
                                m.pose.position.z))
            det_sizes.append((m.scale.x, m.scale.y, m.scale.z))

        active_tracks = list(self.tracks.values())

        # ── 2. Kalman predict step for all existing tracks ─────────────
        for trk in active_tracks:
            trk.predict()

        # ── 3. Associate detections to predicted track positions ───────
        matched, unmatched_trk, unmatched_det = _associate(
            active_tracks, detections, self.max_match_dist
        )

        # ── 4. Update matched tracks ───────────────────────────────────
        for trk_idx, det_idx in matched:
            trk = active_tracks[trk_idx]
            cx, cy, cz = detections[det_idx]
            sx, sy, sz = det_sizes[det_idx]
            trk.update(cx, cy, cz, sx, sy, sz)

            # Promote tentative → confirmed
            if trk.state == TENTATIVE and trk.hits >= self.min_hits:
                trk.state = CONFIRMED
            # Re-confirm if it was lost but matched again
            elif trk.state == LOST:
                trk.state = CONFIRMED

        # ── 5. Handle unmatched tracks ─────────────────────────────────
        for trk_idx in unmatched_trk:
            trk = active_tracks[trk_idx]
            if trk.misses >= self.max_misses:
                # Mark for deletion — removed below
                trk.state = LOST + 99   # sentinel
            elif trk.state == CONFIRMED:
                trk.state = LOST        # faded but still shown

        # Remove dead tracks
        dead = [tid for tid, trk in self.tracks.items()
                if trk.state == LOST + 99]
        for tid in dead:
            del self.tracks[tid]

        # ── 6. Create new tracks for unmatched detections ──────────────
        for det_idx in unmatched_det:
            cx, cy, cz = detections[det_idx]
            sx, sy, sz = det_sizes[det_idx]
            new_trk = KalmanTrack(
                cx, cy, cz, sx, sy, sz,
                self.dt, self.process_noise, self.measure_noise,
            )
            self.tracks[new_trk.id] = new_trk

        # ── 7. Publish ──────────────────────────────────────────────────
        self._publish(now)

    # ── Publishing ────────────────────────────────────────────────────
    def _publish(self, now: rclpy.time.Time) -> None:
        now_msg  = now.to_msg()
        lifetime = rclpy.duration.Duration(seconds=self.marker_lifetime).to_msg()

        box_array = MarkerArray()
        txt_array = MarkerArray()
        vel_array = MarkerArray()

        for trk in self.tracks.values():
            # Only publish CONFIRMED and LOST tracks (not tentative, not dead)
            if trk.state == TENTATIVE:
                continue
            if trk.state >= LOST + 99:
                continue

            px, py, pz = trk.position
            sx, sy, sz = float(trk.size[0]), float(trk.size[1]), float(trk.size[2])

            is_lost = (trk.state == LOST)

            # ── Bounding box marker ──────────────────────────────────
            box = Marker()
            box.header.frame_id = self.target_frame
            box.header.stamp    = now_msg
            box.ns              = "tracks"
            box.id              = trk.id
            box.type            = Marker.CUBE
            box.action          = Marker.ADD
            box.lifetime        = lifetime

            box.pose.position.x    = px
            box.pose.position.y    = py
            box.pose.position.z    = pz
            box.pose.orientation.w = 1.0

            box.scale.x = max(0.05, sx)
            box.scale.y = max(0.05, sy)
            box.scale.z = max(0.05, sz)

            # Colour: confirmed=cyan, lost=grey
            if is_lost:
                box.color.r, box.color.g, box.color.b, box.color.a = 0.5, 0.5, 0.5, 0.3
            else:
                box.color.r, box.color.g, box.color.b, box.color.a = 0.0, 0.85, 1.0, 0.65

            box_array.markers.append(box)

            # ── Text label: ID + speed ───────────────────────────────
            txt = Marker()
            txt.header.frame_id = self.target_frame
            txt.header.stamp    = now_msg
            txt.ns              = "track_labels"
            txt.id              = trk.id
            txt.type            = Marker.TEXT_VIEW_FACING
            txt.action          = Marker.ADD
            txt.lifetime        = lifetime

            txt.pose.position.x    = px
            txt.pose.position.y    = py
            txt.pose.position.z    = pz + sz / 2.0 + 0.15   # above the box
            txt.pose.orientation.w = 1.0
            txt.scale.z            = 0.25   # text height metres

            speed_kmh = trk.speed * 3.6
            txt.text = f"ID:{trk.id}  {speed_kmh:.1f}km/h"

            txt.color.r = txt.color.g = txt.color.b = 1.0
            txt.color.a = 0.0 if is_lost else 1.0

            txt_array.markers.append(txt)

            # ── Velocity arrow ───────────────────────────────────────
            if self.pub_vel_arrows:
                vx, vy, vz = trk.velocity
                arrow = Marker()
                arrow.header.frame_id = self.target_frame
                arrow.header.stamp    = now_msg
                arrow.ns              = "track_velocity"
                arrow.id              = trk.id
                arrow.type            = Marker.ARROW
                arrow.action          = Marker.ADD
                arrow.lifetime        = lifetime

                # Arrow: from centre, pointing in velocity direction
                from geometry_msgs.msg import Point
                start = Point()
                start.x, start.y, start.z = px, py, pz
                end   = Point()
                end.x = px + vx * 1.0   # scale: 1 second of travel
                end.y = py + vy * 1.0
                end.z = pz + vz * 1.0
                arrow.points = [start, end]

                arrow.scale.x = 0.05   # shaft diameter
                arrow.scale.y = 0.10   # head diameter
                arrow.scale.z = 0.10   # head length

                arrow.color.r = 1.0
                arrow.color.g = 0.6
                arrow.color.b = 0.0
                arrow.color.a = 0.0 if is_lost else 0.9

                vel_array.markers.append(arrow)

        # ── Delete markers for tracks that were removed this cycle ────
        # RViz needs an explicit DELETE for old IDs not in the current set
        # We use a simple approach: send one DELETEALL per namespace
        # only when the track count drops significantly
        # (avoids marker ghosting in RViz)
        if not self.tracks:
            for ns, pub, marker_type in [
                ("tracks",         self.pub_boxes, Marker.CUBE),
                ("track_labels",   self.pub_text,  Marker.TEXT_VIEW_FACING),
                ("track_velocity", self.pub_vel,   Marker.ARROW),
            ]:
                clear = Marker()
                clear.ns     = ns
                clear.action = Marker.DELETEALL
                arr = MarkerArray()
                arr.markers.append(clear)
                pub.publish(arr)
            return

        self.pub_boxes.publish(box_array)
        self.pub_text.publish(txt_array)
        if self.pub_vel_arrows:
            self.pub_vel.publish(vel_array)

        confirmed = sum(1 for t in self.tracks.values() if t.state == CONFIRMED)
        self.get_logger().debug(
            f"tracks total={len(self.tracks)} confirmed={confirmed} "
            f"scipy={HAVE_SCIPY}"
        )


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObjectTrackerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()