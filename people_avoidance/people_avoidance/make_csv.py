#!/usr/bin/env python3
"""
rosbag2_to_csv.py
-----------------
Reads a rosbag2 SQLite3 (.db3) file and exports every topic with messages
to a separate CSV in a csvs/ subdirectory.

All CDR-serialised messages are decoded into human-readable columns.
The /scan topic (sensor_msgs/msg/LaserScan) is handled specially:
  - Each scan is flattened to one row per beam (angle_deg, range_m,
    intensity) so the data is immediately plottable.
  - A second wide-format file scan_wide.csv keeps one row per timestamp
    with range_0 … range_N and intensity_0 … intensity_N columns.

Dependencies (pip install):
    rosbags          # pure-Python CDR deserialiser, no ROS install needed
    numpy
    pandas
"""

import math
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    sys.exit("Missing dependency. Please run:\n" "  pip install rosbags numpy pandas")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BAG_PATH = "rosbag2_2026_03_24-16_07_41_0.db3"  # path to the .db3 file
OUT_DIR = Path("csvs")


# ---------------------------------------------------------------------------
# Helper: flatten any rosbags message object to a flat dict
# ---------------------------------------------------------------------------
def _flatten(obj, prefix=""):
    """Recursively flatten a deserialized rosbags message to a flat dict."""
    result = {}

    if hasattr(obj, "__slots__"):  # it's a message-like object
        for slot in obj.__slots__:
            val = getattr(obj, slot)
            key = f"{prefix}{slot}" if prefix else slot
            result.update(_flatten(val, prefix=key + "."))
    elif isinstance(obj, (list, tuple, np.ndarray)):
        # For arrays we serialise as a semicolon-separated string so the
        # entire row still fits in a single CSV cell.
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        result[prefix.rstrip(".")] = ";".join(str(v) for v in obj)
    else:
        result[prefix.rstrip(".")] = obj

    return result


# ---------------------------------------------------------------------------
# Special handler: /scan  (sensor_msgs/msg/LaserScan)
# ---------------------------------------------------------------------------
def handle_scan(rows):
    """
    Returns two DataFrames:
      long_df  – one row per beam  (timestamp_ns, seq?, angle_deg, range_m, intensity)
      wide_df  – one row per scan  (timestamp_ns + range_0…range_N + intensity_0…intensity_N)
    """
    long_records = []
    wide_records = []

    for ts_ns, msg in rows:
        ranges = list(msg.ranges)
        intensities = (
            list(msg.intensities)
            if len(msg.intensities)
            else [float("nan")] * len(ranges)
        )
        angle_min = msg.angle_min  # radians
        angle_inc = msg.angle_increment  # radians
        n_beams = len(ranges)

        # --- long format ---
        for i, (r, intensity) in enumerate(zip(ranges, intensities)):
            angle_rad = angle_min + i * angle_inc
            angle_deg = math.degrees(angle_rad)
            long_records.append(
                {
                    "timestamp_ns": ts_ns,
                    "timestamp_s": ts_ns * 1e-9,
                    "beam_index": i,
                    "angle_deg": round(angle_deg, 4),
                    "range_m": round(r, 6),
                    "intensity": round(intensity, 4),
                    "frame_id": msg.header.frame_id,
                }
            )

        # --- wide format ---
        wide_row = {
            "timestamp_ns": ts_ns,
            "timestamp_s": ts_ns * 1e-9,
            "frame_id": msg.header.frame_id,
            "angle_min_deg": round(math.degrees(msg.angle_min), 4),
            "angle_max_deg": round(math.degrees(msg.angle_max), 4),
            "angle_increment_deg": round(math.degrees(msg.angle_increment), 6),
            "time_increment_s": msg.time_increment,
            "scan_time_s": msg.scan_time,
            "range_min_m": msg.range_min,
            "range_max_m": msg.range_max,
            "n_beams": n_beams,
        }
        for i, r in enumerate(ranges):
            wide_row[f"range_{i}"] = round(r, 6)
        for i, iv in enumerate(intensities):
            wide_row[f"intensity_{i}"] = round(iv, 4)
        wide_records.append(wide_row)

    long_df = pd.DataFrame(long_records)
    wide_df = pd.DataFrame(wide_records)
    return long_df, wide_df


# ---------------------------------------------------------------------------
# Special handler: /tf and /tf_static  (tf2_msgs/msg/TFMessage)
# ---------------------------------------------------------------------------
def handle_tf(rows):
    records = []
    for ts_ns, msg in rows:
        for t in msg.transforms:
            records.append(
                {
                    "timestamp_ns": ts_ns,
                    "timestamp_s": ts_ns * 1e-9,
                    "header_stamp_sec": t.header.stamp.sec,
                    "header_stamp_nanosec": t.header.stamp.nanosec,
                    "header_frame_id": t.header.frame_id,
                    "child_frame_id": t.child_frame_id,
                    "tx": t.transform.translation.x,
                    "ty": t.transform.translation.y,
                    "tz": t.transform.translation.z,
                    "qx": t.transform.rotation.x,
                    "qy": t.transform.rotation.y,
                    "qz": t.transform.rotation.z,
                    "qw": t.transform.rotation.w,
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Special handler: /ir_intensity  (irobot_create_msgs/msg/IrIntensityVector)
# ---------------------------------------------------------------------------
def handle_ir_intensity(rows):
    records = []
    for ts_ns, msg in rows:
        row = {
            "timestamp_ns": ts_ns,
            "timestamp_s": ts_ns * 1e-9,
            "header_frame_id": msg.header.frame_id,
            "header_stamp_sec": msg.header.stamp.sec,
            "header_stamp_nanosec": msg.header.stamp.nanosec,
        }
        for i, reading in enumerate(msg.readings):
            row[f"sensor_{i}_header_frame_id"] = reading.header.frame_id
            row[f"sensor_{i}_value"] = reading.value
        records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Generic handler: uses _flatten for everything else
# ---------------------------------------------------------------------------
def handle_generic(rows):
    records = []
    for ts_ns, msg in rows:
        flat = _flatten(msg)
        flat["timestamp_ns"] = ts_ns
        flat["timestamp_s"] = ts_ns * 1e-9
        # move timestamps to front
        ordered = {k: flat[k] for k in ["timestamp_ns", "timestamp_s"]}
        ordered.update({k: v for k, v in flat.items() if k not in ordered})
        records.append(ordered)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    bag_path = Path(BAG_PATH)
    if not bag_path.exists():
        sys.exit(f"ERROR: bag file not found: {bag_path.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # rosbags Reader needs the *directory* that contains the db3 + metadata.yaml,
    # or the db3 directly if metadata is alongside it.
    bag_dir = bag_path.parent

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    print(f"Opening bag: {bag_path}")
    with Reader(bag_dir) as reader:
        # Build {topic_name: (msg_type_str, connection)} mapping
        connections_by_topic = {}
        for conn in reader.connections:
            connections_by_topic[conn.topic] = conn

        topics_with_data = {topic: conn for topic, conn in connections_by_topic.items()}

        print(f"Found {len(topics_with_data)} topics. Reading messages …\n")

        # Collect all messages per topic first
        topic_messages = {t: [] for t in topics_with_data}

        for conn, timestamp, rawdata in reader.messages():
            msg = deserialize(conn, rawdata, typestore)
            if msg is not None:
                topic_messages[conn.topic].append((timestamp, msg))

    # Now write CSVs
    for topic, rows in topic_messages.items():
        if not rows:
            print(f"  [skip]  {topic}  (0 messages)")
            continue

        safe_name = topic.lstrip("/").replace("/", "__")
        msg_type = topics_with_data[topic].msgtype

        print(f"  {topic}  ({len(rows)} msgs)  [{msg_type}]")

        # --- dispatch to specialised handlers ---
        if topic in ("/tf", "/tf_static"):
            df = handle_tf(rows)
            _save(df, OUT_DIR / f"{safe_name}.csv")

        elif topic == "/scan":
            long_df, wide_df = handle_scan(rows)
            _save(long_df, OUT_DIR / "scan_long.csv")
            _save(wide_df, OUT_DIR / "scan_wide.csv")
            print(f"           → scan_long.csv ({len(long_df)} beam rows)")
            print(f"           → scan_wide.csv ({len(wide_df)} scan rows)")

        elif topic == "/ir_intensity":
            df = handle_ir_intensity(rows)
            _save(df, OUT_DIR / f"{safe_name}.csv")

        else:
            df = handle_generic(rows)
            _save(df, OUT_DIR / f"{safe_name}.csv")

    print(f"\nDone. CSVs written to: {OUT_DIR.resolve()}")


def deserialize(conn, rawdata, typestore):
    """Deserialise a raw CDR bytes blob using rosbags typestore."""
    try:
        msgtype = typestore.types[conn.msgtype]
        return typestore.deserialize_cdr(rawdata, conn.msgtype)
    except Exception as exc:
        print(f"    WARNING: could not deserialise {conn.topic}: {exc}")
        return None


def _save(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
