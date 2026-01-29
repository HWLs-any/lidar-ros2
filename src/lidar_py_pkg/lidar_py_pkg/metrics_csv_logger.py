#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _time_to_float_sec(t: Time) -> float:
    # seconds as float
    return float(t.nanoseconds) / 1e9


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _msg_to_value(msg: Any) -> Any:
    """
    Best-effort conversion:
    - If msg has a 'data' field (std_msgs style), return msg.data
    - Else dump to dict-like representation (JSON serializable where possible)
    """
    if hasattr(msg, "data"):
        return getattr(msg, "data")

    # Fallback: try to convert recursively
    def convert(x: Any) -> Any:
        if is_dataclass(x):
            return {k: convert(v) for k, v in asdict(x).items()}
        if hasattr(x, "__slots__"):
            out: Dict[str, Any] = {}
            for s in getattr(x, "__slots__", []):
                try:
                    out[s] = convert(getattr(x, s))
                except Exception:
                    out[s] = str(getattr(x, s, ""))
            return out
        if isinstance(x, (list, tuple)):
            return [convert(i) for i in x]
        if isinstance(x, (int, float, str, bool)) or x is None:
            return x
        return str(x)

    return convert(msg)


class MetricsCsvLogger(Node):
    """
    Logs numeric/simple ROS messages from /metrics/* topics to a CSV file.

    Params:
      - output_dir (string): directory to store csv. default: ~/lidar_ws/results
      - topics (string[]): list of topics to log. If empty -> auto-log all /metrics/* topics found.
      - discovery_timeout_sec (double): how long to wait for topics to appear. default: 2.0
      - include_types (bool): whether to log message type column. default: true
    """

    def __init__(self) -> None:
        super().__init__("metrics_csv_logger")

        self.declare_parameter("output_dir", str(Path.home() / "lidar_ws" / "results"))
        self.declare_parameter("topics", [])
        self.declare_parameter("discovery_timeout_sec", 2.0)
        self.declare_parameter("include_types", True)

        out_dir = Path(self.get_parameter("output_dir").value)
        _safe_mkdir(out_dir)

        self.csv_path = out_dir / f"metrics_{_now_str()}.csv"
        self.include_types = bool(self.get_parameter("include_types").value)

        self._fh = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)

        header = ["stamp_sec", "topic", "value"]
        if self.include_types:
            header.insert(2, "type")
        self._writer.writerow(header)
        self._fh.flush()

        # Resolve topics to subscribe
        topics_param = list(self.get_parameter("topics").value)
        discovery_timeout = float(self.get_parameter("discovery_timeout_sec").value)

        topic_type_pairs = self._discover_topics(topics_param, discovery_timeout)
        if not topic_type_pairs:
            self.get_logger().error("No metrics topics found to subscribe to. Exiting.")
            raise RuntimeError("No metrics topics to log")

        self._subs = []
        for topic, type_str in topic_type_pairs:
            try:
                msg_cls = get_message(type_str)
            except Exception as e:
                self.get_logger().warning(f"Cannot import message type '{type_str}' for topic '{topic}': {e}")
                continue

            sub = self.create_subscription(
                msg_cls,
                topic,
                lambda m, t=topic, ty=type_str: self._on_msg(t, ty, m),
                10,
            )
            self._subs.append(sub)
            self.get_logger().info(f"Logging: {topic} [{type_str}]")

        self.get_logger().info(f"Writing metrics CSV to: {self.csv_path}")

    def _discover_topics(self, topics_param: List[str], timeout_sec: float) -> List[Tuple[str, str]]:
        """
        Returns list of (topic, type_str).
        type_str format matches rosidl_runtime_py.get_message, e.g. 'std_msgs/msg/Float64'
        """
        end_time = self.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)

        wanted = [t.strip() for t in topics_param if str(t).strip()]
        while rclpy.ok() and self.get_clock().now() < end_time:
            graph = self.get_topic_names_and_types()
            mapping: Dict[str, List[str]] = {name: types for (name, types) in graph}

            if wanted:
                pairs = []
                ok = True
                for t in wanted:
                    if t not in mapping or not mapping[t]:
                        ok = False
                        break
                    pairs.append((t, mapping[t][0]))
                if ok:
                    return pairs
            else:
                pairs = []
                for name, types in mapping.items():
                    if name.startswith("/metrics/") and types:
                        pairs.append((name, types[0]))
                if pairs:
                    return sorted(pairs)

            # small wait
            rclpy.spin_once(self, timeout_sec=0.1)

        # last attempt
        graph = self.get_topic_names_and_types()
        mapping = {name: types for (name, types) in graph}

        if wanted:
            pairs = []
            for t in wanted:
                if t in mapping and mapping[t]:
                    pairs.append((t, mapping[t][0]))
            return pairs
        else:
            pairs = [(n, ts[0]) for (n, ts) in mapping.items() if n.startswith("/metrics/") and ts]
            return sorted(pairs)

    def _on_msg(self, topic: str, type_str: str, msg: Any) -> None:
        stamp = self.get_clock().now()
        stamp_sec = _time_to_float_sec(stamp)

        val = _msg_to_value(msg)
        # keep CSV stable: serialize complex values into JSON
        if isinstance(val, (dict, list)):
            val_out = json.dumps(val, ensure_ascii=False)
        else:
            val_out = val

        row = [stamp_sec, topic, val_out]
        if self.include_types:
            row.insert(2, type_str)

        self._writer.writerow(row)
        # flush frequently to not lose data on crash
        self._fh.flush()

    def destroy_node(self) -> bool:
        try:
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MetricsCsvLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()   # ensures file gets closed
        rclpy.shutdown()


if __name__ == "__main__":
    main()
