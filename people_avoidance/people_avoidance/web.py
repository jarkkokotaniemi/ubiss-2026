"""
visualizer_web.py — Browser-based replay visualizer for the people-avoidance pipeline.

Replaces the matplotlib 4-panel window with a Flask web server that streams
frame data as JSON via Server-Sent Events (SSE).  The browser renders everything
with Canvas2D at ~30 fps with no Python rendering overhead.

Improvements over visualizer_offline.py
-----------------------------------------
- Flask SSE stream → no Tk/Qt/Agg dependency, runs headlessly on a server
- Instance tracking: each track ID gets a persistent colour so you can follow
  individuals across frames.  The leg/person ID is drawn on the canvas.
- Improved Kalman visualisation: draws the 2-σ covariance ellipse, velocity
  vector, and a short trail of the last N predicted positions per track.
- Track history panel: timeline strip showing when each track was alive.
- All pipeline tuning constants are exposed as sliders in the browser.

Usage
-----
    python3 visualizer_web.py --bag /path/to/rosbag2_dir_or_db3
    python3 visualizer_web.py          # searches current directory

Then open  http://localhost:5000  in any modern browser.

Dependencies
------------
    pip install flask rosbags numpy
    (No ROS installation required.)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import types
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Flask ─────────────────────────────────────────────────────────────────────
try:
    from flask import Flask, Response, render_template_string, request, jsonify
except ImportError:
    sys.exit("Missing Flask.  Install with:  pip install flask\n")

# ── rosbags ───────────────────────────────────────────────────────────────────
try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    sys.exit("Missing rosbags.  Install with:  pip install rosbags\n")


# ── Stub out ROS message types ────────────────────────────────────────────────
def _make_stub_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _stub_class(mod, clsname):
    cls = type(clsname, (), {"__init__": lambda self, *a, **kw: None})
    setattr(mod, clsname, cls)
    return cls


for _pkg in (
    "geometry_msgs",
    "geometry_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "nav_msgs",
    "nav_msgs.msg",
    "rclpy",
    "rclpy.node",
):
    _make_stub_module(_pkg)


class _Vec3:
    def __init__(self):
        self.x = self.y = self.z = 0.0


class _Twist:
    def __init__(self):
        self.linear = _Vec3()
        self.angular = _Vec3()


sys.modules["geometry_msgs.msg"].Twist = _Twist
_stub_class(sys.modules["sensor_msgs.msg"], "LaserScan")
_stub_class(sys.modules["nav_msgs.msg"], "Odometry")
_stub_class(sys.modules["rclpy.node"], "Node")

# ── Load pipeline modules from the same directory ────────────────────────────
_HERE = Path(__file__).resolve().parent


def _load_local(
    filename: str, module_name: str, package_alias: str = "people_avoidance"
):
    if package_alias not in sys.modules:
        pkg = types.ModuleType(package_alias)
        pkg.__path__ = [str(_HERE)]
        pkg.__package__ = package_alias
        sys.modules[package_alias] = pkg
    full_name = f"{package_alias}.{module_name}"
    spec = importlib.util.spec_from_file_location(
        full_name, _HERE / filename, submodule_search_locations=[]
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package_alias
    sys.modules[full_name] = mod
    sys.modules[module_name] = mod
    setattr(sys.modules[package_alias], module_name, mod)
    spec.loader.exec_module(mod)
    return mod


_leg_mod = _load_local("leg_detection.py", "leg_detection")
_track_mod = _load_local("tracking.py", "tracking")
_ctrl_mod = _load_local("controller.py", "controller")

detect_legs = _leg_mod.detect_legs
scan_fft = _leg_mod.scan_fft
scan_to_cartesian = _leg_mod.scan_to_cartesian
segment_scan = _leg_mod.segment_scan
LegMeasurement = _leg_mod.LegMeasurement
KalmanTracker = _track_mod.KalmanTracker
Track = _track_mod.Track
compute_velocity = _ctrl_mod.compute_velocity
obstacle_radius = _ctrl_mod.obstacle_radius

# ─────────────────────────────────────────────────────────────────────────────
# Default pipeline constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    distance_threshold=0.10,
    leg_radius=0.10,
    max_leg_width=0.25,
    dt=0.10,
    max_misses=5,
    max_linear_speed=0.20,
    max_angular_speed=1.00,
    obstacle_radius_scale=2.0,
    plot_range=4.0,
    curv_threshold=0.40,
    trail_length=20,  # frames of position trail per track
    play_fps=20,
)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight message objects
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FakeLaserScan:
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray


@dataclass
class FrameData:
    timestamp_s: float
    scan_index: int
    points: np.ndarray
    segments: List[np.ndarray]
    measurements: List[LegMeasurement]
    tracks: List[Track]
    linear_x: float
    angular_z: float
    robot_x: float
    robot_y: float
    robot_theta: float
    fft_freqs: np.ndarray
    fft_mags: np.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# Instance tracker  — assigns a persistent colour to each track ID
# ─────────────────────────────────────────────────────────────────────────────
# 12 visually distinct, colourblind-friendly hues (hex, no '#')
TRACK_PALETTE = [
    "4E79A7",
    "F28E2B",
    "59A14F",
    "E15759",
    "76B7B2",
    "EDC948",
    "B07AA1",
    "FF9DA7",
    "9C755F",
    "BAB0AC",
    "D4A6C8",
    "86BCB6",
]


class TrackRegistry:
    """Maintains stable colour assignment per track_id across frames."""

    def __init__(self):
        self._colour: Dict[int, str] = {}
        self._counter = 0

    def colour_for(self, track_id: int) -> str:
        if track_id not in self._colour:
            self._colour[track_id] = TRACK_PALETTE[self._counter % len(TRACK_PALETTE)]
            self._counter += 1
        return self._colour[track_id]

    def as_dict(self) -> Dict[int, str]:
        return dict(self._colour)


# Global registry (persists for the lifetime of the server process)
_registry = TrackRegistry()

# Per-track position trail  {track_id: [(x,y), ...]}
_trails: Dict[int, List[Tuple[float, float]]] = {}


def _update_trails(tracks: List[Track], max_len: int):
    seen = set()
    for tr in tracks:
        tid = tr.track_id
        seen.add(tid)
        trail = _trails.setdefault(tid, [])
        trail.append((float(tr.m[0]), float(tr.m[1])))
        if len(trail) > max_len:
            del trail[: len(trail) - max_len]
    # prune dead tracks after a while (keep last 200 frames of ghost trails)
    # – actually just let them live; they'll stop being appended to.


# ─────────────────────────────────────────────────────────────────────────────
# Bag loader
# ─────────────────────────────────────────────────────────────────────────────
def _yaw_from_quat(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy**2 + qz**2)
    return math.atan2(siny, cosy)


def load_bag(bag_path: Path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    bag_dir = bag_path if bag_path.is_dir() else bag_path.parent
    scans, scan_times, odom_map = [], [], {}
    print(f"Loading bag from {bag_dir} …", flush=True)
    with Reader(bag_dir) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == "/scan":
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                scans.append(
                    FakeLaserScan(
                        angle_min=float(msg.angle_min),
                        angle_max=float(msg.angle_max),
                        angle_increment=float(msg.angle_increment),
                        range_min=float(msg.range_min),
                        range_max=float(msg.range_max),
                        ranges=np.array(msg.ranges, dtype=float),
                    )
                )
                scan_times.append(timestamp * 1e-9)
            elif conn.topic == "/odom":
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                x = float(msg.pose.pose.position.x)
                y = float(msg.pose.pose.position.y)
                q = msg.pose.pose.orientation
                yaw = _yaw_from_quat(float(q.x), float(q.y), float(q.z), float(q.w))
                odom_map[timestamp] = (x, y, yaw)
    print(f"  Loaded {len(scans)} scans, {len(odom_map)} odom messages.")
    return scans, scan_times, odom_map


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(scans, scan_times, odom_map, cfg: dict) -> List[FrameData]:
    tracker = KalmanTracker(dt=cfg["dt"], max_misses=int(cfg["max_misses"]))
    frames: List[FrameData] = []
    odom_keys = sorted(odom_map.keys())

    def _nearest_odom(ts_s: float):
        if not odom_keys:
            return 0.0, 0.0, 0.0
        ts_ns = int(ts_s * 1e9)
        lo, hi = 0, len(odom_keys) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if odom_keys[mid] < ts_ns:
                lo = mid + 1
            else:
                hi = mid
        best = lo
        if lo > 0 and abs(odom_keys[lo - 1] - ts_ns) < abs(odom_keys[lo] - ts_ns):
            best = lo - 1
        return odom_map[odom_keys[best]]

    prev_ts = scan_times[0] - cfg["dt"] if scan_times else 0.0

    for i, (scan, ts) in enumerate(zip(scans, scan_times)):
        dt_i = ts - prev_ts
        if dt_i <= 0 or dt_i > 1.0:
            dt_i = cfg["dt"]
        tracker.dt = dt_i
        tracker.F[0, 2] = dt_i
        tracker.F[1, 3] = dt_i
        prev_ts = ts

        rx, ry, rth = _nearest_odom(ts)
        points = scan_to_cartesian(scan)
        segments = (
            segment_scan(
                points,
                distance_threshold=cfg["distance_threshold"],
                scan=scan,
                curv_threshold=cfg["curv_threshold"],
            )
            if points.shape[0] > 0
            else []
        )

        fft_freqs, fft_mags = scan_fft(scan)
        measurements = detect_legs(
            scan,
            distance_threshold=cfg["distance_threshold"],
            leg_radius=cfg["leg_radius"],
            max_leg_width=cfg["max_leg_width"],
            curv_threshold=cfg["curv_threshold"],
        )
        tracker.update(measurements)
        tracks = tracker.get_tracks()
        cmd = compute_velocity(
            tracks,
            robot_x=0.0,
            robot_y=0.0,
            robot_theta=0.0,
            max_linear_speed=cfg["max_linear_speed"],
            max_angular_speed=cfg["max_angular_speed"],
            obstacle_radius_scale=cfg["obstacle_radius_scale"],
        )
        frames.append(
            FrameData(
                timestamp_s=ts,
                scan_index=i,
                points=points,
                segments=list(segments),
                measurements=measurements,
                tracks=tracks,
                linear_x=cmd.linear.x,
                angular_z=cmd.angular.z,
                robot_x=rx,
                robot_y=ry,
                robot_theta=rth,
                fft_freqs=fft_freqs,
                fft_mags=fft_mags,
            )
        )
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(scans)} frames …", flush=True)

    print(f"  Pipeline complete — {len(frames)} frames ready.")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Frame → JSON serialiser
# ─────────────────────────────────────────────────────────────────────────────
def _covariance_ellipse(P2x2, n_std=2.0):
    """Return {cx,cy,rx,ry,angle} of the covariance ellipse."""
    try:
        vals, vecs = np.linalg.eigh(P2x2)
        vals = np.maximum(vals, 0)
        angle = math.atan2(float(vecs[1, -1]), float(vecs[0, -1]))
        return {
            "rx": float(n_std * math.sqrt(vals[0])),
            "ry": float(n_std * math.sqrt(vals[1])),
            "angle": float(angle),
        }
    except Exception:
        return {"rx": 0.05, "ry": 0.05, "angle": 0.0}


def frame_to_json(
    frame: FrameData, frame_idx: int, n_frames: int, t0: float, trail_len: int
) -> dict:
    # Update trails for this frame
    _update_trails(frame.tracks, trail_len)

    points = frame.points.tolist() if frame.points.shape[0] > 0 else []
    segments = [s.tolist() for s in frame.segments]

    measurements = [
        {"x": m.x, "y": m.y, "sigma": math.sqrt(max(m.Rxx, 0))}
        for m in frame.measurements
    ]

    tracks = []
    for tr in frame.tracks:
        tid = tr.track_id
        colour = _registry.colour_for(tid)
        ell = _covariance_ellipse(tr.P[:2, :2])
        trail = _trails.get(tid, [])
        r_safe = float(obstacle_radius(tr, DEFAULTS["obstacle_radius_scale"]))
        tracks.append(
            {
                "id": tid,
                "colour": colour,
                "x": float(tr.m[0]),
                "y": float(tr.m[1]),
                "vx": float(tr.m[2]),
                "vy": float(tr.m[3]),
                "misses": int(tr.misses),
                "ellipse": ell,
                "r_safe": r_safe,
                "trail": trail[-trail_len:],
            }
        )

    # FFT — downsample to 256 points for compact JSON
    fq = frame.fft_freqs
    mg = frame.fft_mags
    step = max(1, len(fq) // 256)
    fft_data = {"freqs": fq[::step].tolist(), "mags": mg[::step].tolist()}

    return {
        "frame_idx": frame_idx,
        "n_frames": n_frames,
        "t": round(frame.timestamp_s - t0, 3),
        "points": points,
        "segments": segments,
        "measurements": measurements,
        "tracks": tracks,
        "linear_x": round(frame.linear_x, 4),
        "angular_z": round(frame.angular_z, 4),
        "robot_theta": round(frame.robot_theta, 4),
        "fft": fft_data,
        "track_colours": _registry.as_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Shared state (set in main())
_frames: List[FrameData] = []
_t0: float = 0.0
_cfg = dict(DEFAULTS)

# Playback state
_play_state = {
    "idx": 0,
    "playing": False,
    "fps": DEFAULTS["play_fps"],
}
_play_lock = threading.Lock()
_sse_clients: List = []  # list of queue.Queue

import queue


def _broadcast(data: dict):
    msg = f"data: {json.dumps(data)}\n\n"
    dead = []
    for q in list(_sse_clients):
        try:
            q.put_nowait(msg)
        except queue.Full:
            dead.append(q)
    for q in dead:
        try:
            _sse_clients.remove(q)
        except ValueError:
            pass


def _playback_thread():
    while True:
        time.sleep(0.005)
        with _play_lock:
            if not _play_state["playing"] or not _frames:
                continue
            idx = _play_state["idx"]
            fps = max(1, _play_state["fps"])

        time.sleep(1.0 / fps)

        with _play_lock:
            if not _play_state["playing"]:
                continue
            idx = _play_state["idx"]
            if idx >= len(_frames) - 1:
                _play_state["playing"] = False
                continue
            _play_state["idx"] = idx + 1
            new_idx = _play_state["idx"]

        if _frames:
            data = frame_to_json(
                _frames[new_idx], new_idx, len(_frames), _t0, int(_cfg["trail_length"])
            )
            data["playing"] = True
            _broadcast(data)


threading.Thread(target=_playback_thread, daemon=True).start()


@app.route("/")
def index():
    return render_template_string(HTML_PAGE, defaults=json.dumps(_cfg))


@app.route("/api/frame/<int:idx>")
def get_frame(idx: int):
    if not _frames:
        return jsonify({"error": "no frames loaded"}), 404
    idx = max(0, min(idx, len(_frames) - 1))
    with _play_lock:
        _play_state["idx"] = idx
    data = frame_to_json(
        _frames[idx], idx, len(_frames), _t0, int(_cfg["trail_length"])
    )
    return jsonify(data)


@app.route("/api/play", methods=["POST"])
def api_play():
    body = request.get_json(silent=True) or {}
    with _play_lock:
        action = body.get("action", "toggle")
        if action == "play":
            _play_state["playing"] = True
        elif action == "pause":
            _play_state["playing"] = False
        elif action == "toggle":
            _play_state["playing"] = not _play_state["playing"]
        if "idx" in body:
            _play_state["idx"] = int(body["idx"])
        if "fps" in body:
            _play_state["fps"] = float(body["fps"])
        return jsonify({"playing": _play_state["playing"], "idx": _play_state["idx"]})


@app.route("/api/config", methods=["POST"])
def api_config():
    """Update pipeline config and re-run pipeline."""
    global _frames, _t0
    body = request.get_json(silent=True) or {}
    for k, v in body.items():
        if k in _cfg:
            _cfg[k] = float(v)
    # Reset trails and registry for a clean re-run
    _trails.clear()
    _registry._colour.clear()
    _registry._counter = 0
    with _play_lock:
        _play_state["playing"] = False
        _play_state["idx"] = 0

    # Re-run pipeline in background
    def _rerun():
        global _frames, _t0
        _frames = run_pipeline(_scans, _scan_times, _odom_map, _cfg)
        _t0 = _frames[0].timestamp_s if _frames else 0.0
        data = frame_to_json(
            _frames[0], 0, len(_frames), _t0, int(_cfg["trail_length"])
        )
        data["pipeline_done"] = True
        _broadcast(data)

    threading.Thread(target=_rerun, daemon=True).start()
    return jsonify({"status": "rerunning"})


@app.route("/stream")
def stream():
    """SSE endpoint — pushes frame JSON to subscribed browser tabs."""

    def event_gen():
        q: queue.Queue = queue.Queue(maxsize=10)
        _sse_clients.append(q)
        try:
            yield ": connected\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield ": ping\n\n"
        except GeneratorExit:
            pass
        finally:
            try:
                _sse_clients.remove(q)
            except ValueError:
                pass

    return Response(
        event_gen(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML / JS / CSS  (single-file, no external CDN required)
# ─────────────────────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>People-Avoidance Pipeline Visualizer</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0f1117; --panel: #1a1d27; --border: #2e3348;
    --text: #d0d4e8; --sub: #7a7f9a; --accent: #5b8dee;
    --green: #4ade80; --orange: #fb923c; --red: #f87171;
  }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 13px; }
  #app { display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  /* ── top bar ── */
  #topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 6px 14px; background: var(--panel);
    border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  #topbar h1 { font-size: 13px; font-weight: 600; color: var(--accent); white-space: nowrap; }
  #status { color: var(--sub); font-size: 11px; flex: 1; }
  .btn {
    padding: 4px 12px; border-radius: 5px; border: 1px solid var(--border);
    background: #252840; color: var(--text); cursor: pointer; font-size: 12px;
    transition: background .15s;
  }
  .btn:hover { background: #343860; }
  .btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }

  /* ── main layout ── */
  #main { display: flex; flex: 1; overflow: hidden; }

  /* ── canvases ── */
  #canvas-area {
    flex: 1; display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    gap: 6px; padding: 6px; min-width: 0;
  }
  .panel {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    display: flex; flex-direction: column; overflow: hidden; position: relative;
  }
  .panel-title {
    padding: 4px 10px; font-size: 11px; font-weight: 600; color: var(--sub);
    background: #1e2133; border-bottom: 1px solid var(--border); flex-shrink: 0;
  }
  canvas { flex: 1; display: block; width: 100%; height: 100%; }

  /* ── right sidebar ── */
  #sidebar {
    width: 280px; background: var(--panel);
    border-left: 1px solid var(--border);
    display: flex; flex-direction: column; overflow: hidden; flex-shrink: 0;
  }
  #sidebar-tabs { display: flex; border-bottom: 1px solid var(--border); }
  .stab {
    flex: 1; padding: 7px 4px; text-align: center; cursor: pointer;
    font-size: 11px; color: var(--sub); border-bottom: 2px solid transparent;
  }
  .stab.active { color: var(--accent); border-bottom-color: var(--accent); }
  .stab-content { display: none; flex: 1; overflow-y: auto; padding: 10px; }
  .stab-content.active { display: block; }

  /* Track list */
  .track-entry {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 0; border-bottom: 1px solid var(--border);
  }
  .track-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
  .track-info { flex: 1; font-size: 11px; line-height: 1.5; }
  .track-id { font-weight: 700; }
  .track-sub { color: var(--sub); }

  /* Config sliders */
  .cfg-row { margin-bottom: 10px; }
  .cfg-label { font-size: 11px; color: var(--sub); display: flex; justify-content: space-between; margin-bottom: 3px; }
  input[type=range] { width: 100%; accent-color: var(--accent); }

  /* Timeline canvas */
  #timeline-wrap { padding: 6px; }
  #timeline-canvas { width: 100%; border-radius: 4px; }

  /* ── bottom scrubber ── */
  #scrubber {
    padding: 6px 14px; background: var(--panel);
    border-top: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px; flex-shrink: 0;
  }
  #scrubber input[type=range] { flex: 1; accent-color: var(--accent); }
  #frame-label { font-size: 11px; color: var(--sub); white-space: nowrap; min-width: 130px; }
  #fps-label { font-size: 11px; color: var(--sub); white-space: nowrap; }
  #fps-input { width: 50px; text-align: center; background: #252840; border: 1px solid var(--border); color: var(--text); border-radius: 4px; padding: 2px 4px; }
</style>
</head>
<body>
<div id="app">

  <!-- Top bar -->
  <div id="topbar">
    <h1>🤖 People-Avoidance Pipeline</h1>
    <div id="status">Loading…</div>
    <button class="btn" id="btn-prev" title="Previous frame (←)">◀ Prev</button>
    <button class="btn" id="btn-play" title="Play/Pause (Space)">▶ Play</button>
    <button class="btn" id="btn-next" title="Next frame (→)">Next ▶</button>
  </div>

  <!-- Main content -->
  <div id="main">

    <!-- 4-panel canvas grid -->
    <div id="canvas-area">
      <div class="panel">
        <div class="panel-title" id="title1">① Raw Scan</div>
        <canvas id="c1"></canvas>
      </div>
      <div class="panel">
        <div class="panel-title" id="title2">② Segments</div>
        <canvas id="c2"></canvas>
      </div>
      <div class="panel">
        <div class="panel-title" id="title3">③ Leg Detection</div>
        <canvas id="c3"></canvas>
      </div>
      <div class="panel">
        <div class="panel-title" id="title4">④ Tracks + Kalman</div>
        <canvas id="c4"></canvas>
      </div>
    </div>

    <!-- Right sidebar -->
    <div id="sidebar">
      <div id="sidebar-tabs">
        <div class="stab active" data-tab="tracks">Tracks</div>
        <div class="stab" data-tab="config">Config</div>
        <div class="stab" data-tab="fft">FFT</div>
        <div class="stab" data-tab="timeline">Timeline</div>
      </div>

      <!-- Tracks tab -->
      <div class="stab-content active" id="tab-tracks">
        <div id="track-list"><span style="color:var(--sub)">No tracks</span></div>
      </div>

      <!-- Config tab -->
      <div class="stab-content" id="tab-config">
        <div id="cfg-sliders"></div>
        <button class="btn" id="btn-rerun" style="width:100%;margin-top:10px">↺ Re-run Pipeline</button>
        <div id="rerun-status" style="font-size:11px;color:var(--sub);margin-top:6px;text-align:center"></div>
      </div>

      <!-- FFT tab -->
      <div class="stab-content" id="tab-fft">
        <canvas id="fft-canvas" height="200" style="width:100%;margin-top:4px"></canvas>
        <div style="font-size:10px;color:var(--sub);margin-top:4px;text-align:center">cycles / beam</div>
      </div>

      <!-- Timeline tab -->
      <div class="stab-content" id="tab-timeline">
        <div id="timeline-wrap">
          <canvas id="timeline-canvas" height="160"></canvas>
          <div style="font-size:10px;color:var(--sub);margin-top:4px">Track lifetime across frames</div>
        </div>
      </div>
    </div>

  </div><!-- /main -->

  <!-- Bottom scrubber -->
  <div id="scrubber">
    <span id="frame-label">Frame — / —</span>
    <input type="range" id="scrub" min="0" max="0" value="0" step="1">
    <label id="fps-label">FPS: <input type="number" id="fps-input" value="20" min="1" max="60"></label>
  </div>

</div><!-- /app -->

<script>
// ─────────────────────────────────────────────────────────────────────────────
// Constants / palette
// ─────────────────────────────────────────────────────────────────────────────
const SEG_PALETTE = [
  '#5DCAA5','#378ADD','#D85A30','#7F77DD','#E24B4A',
  '#639922','#EF9F27','#D4537E','#1D9E75','#185FA5',
  '#993C1D','#534AB7',
];
const C_SCAN   = '#888780';
const C_ROBOT  = '#4ade80';
const C_LEG    = '#EF9F27';
const C_PERSON = '#D85A30';
const C_CMD    = '#f87171';

// World → canvas transform
const MARGIN = 30;  // px

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let currentFrame = null;
let nFrames = 0;
let playing = false;
let plotRange = {{ defaults | tojson }}.plot_range || 4.0;
const cfg = {{ defaults | tojson }};

// Timeline data: {trackId: [frameIdx, ...]}
const trackHistory = {};

// ─────────────────────────────────────────────────────────────────────────────
// Canvas helpers
// ─────────────────────────────────────────────────────────────────────────────
function getCanvases() {
  return ['c1','c2','c3','c4'].map(id => document.getElementById(id));
}

function resizeCanvas(c) {
  const r = c.parentElement.getBoundingClientRect();
  if (c.width !== Math.floor(r.width) || c.height !== Math.floor(r.height)) {
    c.width  = Math.floor(r.width);
    c.height = Math.floor(r.height);
  }
}

function makeTransform(c, plotRange) {
  // Returns functions to convert world (m) ↔ canvas (px)
  const w = c.width - 2*MARGIN;
  const h = c.height - 2*MARGIN;
  const scale = Math.min(w, h) / (2 * plotRange);
  const cx = c.width  / 2;
  const cy = c.height / 2;
  return {
    wx: x => cx + x * scale,
    wy: y => cy - y * scale,   // y-up in world, y-down in canvas
    ms: s => s * scale,         // scalar
    scale,
  };
}

function clearCanvas(c) {
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  // faint grid
  ctx.strokeStyle = '#1e2133';
  ctx.lineWidth = 1;
  const t = makeTransform(c, plotRange);
  // axes
  ctx.beginPath();
  ctx.moveTo(0, t.wy(0)); ctx.lineTo(c.width, t.wy(0));
  ctx.moveTo(t.wx(0), 0); ctx.lineTo(t.wx(0), c.height);
  ctx.strokeStyle = '#2a2e44';
  ctx.stroke();
  // concentric rings at 1m intervals
  ctx.strokeStyle = '#1a1e2e';
  for (let r = 1; r <= Math.ceil(plotRange); r++) {
    ctx.beginPath();
    ctx.arc(t.wx(0), t.wy(0), t.ms(r), 0, 2*Math.PI);
    ctx.stroke();
  }
}

// Draw robot at origin
function drawRobot(ctx, t, theta) {
  const rx = t.wx(0), ry = t.wy(0);
  const sz = Math.max(8, t.ms(0.15));
  ctx.strokeStyle = C_ROBOT;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(rx, ry, sz, 0, 2*Math.PI);
  ctx.stroke();
  // heading arrow
  const ex = rx + sz * Math.cos(-theta);
  const ey = ry + sz * Math.sin(-theta);
  ctx.beginPath();
  ctx.moveTo(rx, ry);
  ctx.lineTo(ex, ey);
  ctx.strokeStyle = C_ROBOT;
  ctx.stroke();
}

function drawCovarEllipse(ctx, t, x, y, ell, colour, alpha=0.5) {
  const cx = t.wx(x), cy = t.wy(y);
  const rx = Math.max(2, t.ms(ell.rx));
  const ry = Math.max(2, t.ms(ell.ry));
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(-ell.angle);  // note: canvas y-flip
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry, 0, 0, 2*Math.PI);
  ctx.strokeStyle = colour;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.globalAlpha = alpha * 0.15;
  ctx.fillStyle = colour;
  ctx.fill();
  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// Panel renderers
// ─────────────────────────────────────────────────────────────────────────────
function renderScan(c, frame) {
  clearCanvas(c);
  const ctx = c.getContext('2d');
  const t = makeTransform(c, plotRange);
  ctx.fillStyle = C_SCAN;
  for (const [x, y] of frame.points) {
    ctx.beginPath();
    ctx.arc(t.wx(x), t.wy(y), 1.5, 0, 2*Math.PI);
    ctx.fill();
  }
  drawRobot(ctx, t, frame.robot_theta);
}

function renderSegments(c, frame) {
  clearCanvas(c);
  const ctx = c.getContext('2d');
  const t = makeTransform(c, plotRange);
  frame.segments.forEach((seg, i) => {
    ctx.fillStyle = SEG_PALETTE[i % SEG_PALETTE.length];
    for (const [x, y] of seg) {
      ctx.beginPath();
      ctx.arc(t.wx(x), t.wy(y), 2, 0, 2*Math.PI);
      ctx.fill();
    }
  });
  drawRobot(ctx, t, frame.robot_theta);
}

function renderDetections(c, frame) {
  clearCanvas(c);
  const ctx = c.getContext('2d');
  const t = makeTransform(c, plotRange);
  // faint scan
  ctx.fillStyle = C_SCAN;
  ctx.globalAlpha = 0.25;
  for (const [x, y] of frame.points) {
    ctx.beginPath();
    ctx.arc(t.wx(x), t.wy(y), 1.5, 0, 2*Math.PI);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  // measurements
  for (const m of frame.measurements) {
    const px = t.wx(m.x), py = t.wy(m.y);
    // uncertainty ring
    ctx.strokeStyle = C_LEG;
    ctx.lineWidth = 1;
    ctx.setLineDash([4,3]);
    ctx.beginPath();
    ctx.arc(px, py, Math.max(4, t.ms(m.sigma)), 0, 2*Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
    // star marker
    ctx.fillStyle = C_PERSON;
    drawStar(ctx, px, py, 6, 5);
  }
  drawRobot(ctx, t, frame.robot_theta);
}

function drawStar(ctx, x, y, r, pts=5) {
  ctx.save();
  ctx.translate(x, y);
  ctx.beginPath();
  for (let i = 0; i < 2*pts; i++) {
    const ang = (i * Math.PI) / pts - Math.PI/2;
    const rad = i % 2 === 0 ? r : r * 0.45;
    i === 0 ? ctx.moveTo(rad*Math.cos(ang), rad*Math.sin(ang))
            : ctx.lineTo(rad*Math.cos(ang), rad*Math.sin(ang));
  }
  ctx.closePath();
  ctx.fillStyle = C_PERSON;
  ctx.fill();
  ctx.restore();
}

function renderTracks(c, frame) {
  clearCanvas(c);
  const ctx = c.getContext('2d');
  const t = makeTransform(c, plotRange);
  // faint scan
  ctx.fillStyle = C_SCAN;
  ctx.globalAlpha = 0.15;
  for (const [x, y] of frame.points) {
    ctx.beginPath();
    ctx.arc(t.wx(x), t.wy(y), 1.5, 0, 2*Math.PI);
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  for (const tr of frame.tracks) {
    const colour = '#' + tr.colour;
    const px = t.wx(tr.x), py = t.wy(tr.y);

    // ── Position trail ──────────────────────────────────────────────────────
    if (tr.trail.length > 1) {
      ctx.beginPath();
      const [tx0, ty0] = tr.trail[0];
      ctx.moveTo(t.wx(tx0), t.wy(ty0));
      for (let i = 1; i < tr.trail.length; i++) {
        ctx.lineTo(t.wx(tr.trail[i][0]), t.wy(tr.trail[i][1]));
      }
      ctx.strokeStyle = colour;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
    // trail dots fading
    tr.trail.forEach(([tx, ty], i) => {
      const alpha = 0.1 + 0.5 * (i / tr.trail.length);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(t.wx(tx), t.wy(ty), 2, 0, 2*Math.PI);
      ctx.fill();
    });
    ctx.globalAlpha = 1;

    // ── 2-σ covariance ellipse ───────────────────────────────────────────────
    drawCovarEllipse(ctx, t, tr.x, tr.y, tr.ellipse, colour, 0.7);

    // ── Safety radius (dashed) ───────────────────────────────────────────────
    ctx.setLineDash([5,4]);
    ctx.strokeStyle = colour;
    ctx.globalAlpha = 0.4;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(px, py, Math.max(4, t.ms(tr.r_safe)), 0, 2*Math.PI);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // ── Velocity arrow ───────────────────────────────────────────────────────
    const vx = tr.vx, vy = tr.vy;
    if (Math.abs(vx) + Math.abs(vy) > 0.02) {
      const scale = 0.5;
      const ex = t.wx(tr.x + vx * scale);
      const ey = t.wy(tr.y + vy * scale);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = colour;
      ctx.lineWidth = 2;
      ctx.stroke();
      // arrowhead
      const ang = Math.atan2(ey - py, ex - px);
      ctx.save();
      ctx.translate(ex, ey);
      ctx.rotate(ang);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(-8, -4);
      ctx.lineTo(-8, 4);
      ctx.closePath();
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.restore();
    }

    // ── Centre dot ──────────────────────────────────────────────────────────
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, 2*Math.PI);
    ctx.fill();

    // ── ID label ────────────────────────────────────────────────────────────
    ctx.fillStyle = '#fff';
    ctx.font = `bold 11px monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`${tr.id}`, px, py - t.ms(0.18));
    ctx.fillStyle = colour + 'cc';
    ctx.font = '9px monospace';
    ctx.fillText(`miss:${tr.misses}`, px, py + t.ms(0.22));
  }

  // ── cmd_vel arrow ──────────────────────────────────────────────────────────
  const v = frame.linear_x, w = frame.angular_z;
  if (Math.abs(v) > 0.005) {
    const ex = t.wx(v * 0.8 * Math.cos(0));
    const ey = t.wy(v * 0.8 * Math.sin(0));
    ctx.beginPath();
    ctx.moveTo(t.wx(0), t.wy(0));
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = C_CMD;
    ctx.lineWidth = 3;
    ctx.stroke();
  }
  drawRobot(ctx, t, frame.robot_theta);
}

// ─────────────────────────────────────────────────────────────────────────────
// FFT renderer
// ─────────────────────────────────────────────────────────────────────────────
function renderFFT(frame) {
  const c = document.getElementById('fft-canvas');
  const parent = c.parentElement;
  c.width = parent.clientWidth - 20;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const { freqs, mags } = frame.fft;
  if (!freqs.length) return;
  const maxM = Math.max(...mags) || 1;
  const w = c.width, h = c.height;
  const pad = { l: 28, r: 8, t: 8, b: 20 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;

  ctx.fillStyle = '#0f1117';
  ctx.fillRect(0, 0, w, h);

  // axes
  ctx.strokeStyle = '#2e3348';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t+ih);
  ctx.lineTo(pad.l+iw, pad.t+ih);
  ctx.stroke();

  // bars
  const n = freqs.length;
  const bw = Math.max(1, iw / n);
  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t+ih);
  grad.addColorStop(0, '#5b8dee');
  grad.addColorStop(1, '#1e3a6a');
  ctx.fillStyle = grad;
  for (let i = 0; i < n; i++) {
    const bh = (mags[i] / maxM) * ih;
    ctx.fillRect(pad.l + i * bw, pad.t + ih - bh, bw * 0.8, bh);
  }

  // labels
  ctx.fillStyle = '#7a7f9a';
  ctx.font = '9px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('|FFT|', 2, pad.t + 10);
  ctx.textAlign = 'center';
  const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
  const maxF = freqs[freqs.length-1] || 0.5;
  for (const tf of ticks) {
    if (tf > maxF) break;
    const fx = pad.l + (tf / maxF) * iw;
    ctx.fillText(tf.toFixed(1), fx, h - 2);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Timeline renderer
// ─────────────────────────────────────────────────────────────────────────────
function updateTimeline(frame) {
  // Record current tracks
  for (const tr of frame.tracks) {
    if (!trackHistory[tr.id]) trackHistory[tr.id] = { colour: tr.colour, frames: [] };
    trackHistory[tr.id].frames.push(frame.frame_idx);
  }
  const c = document.getElementById('timeline-canvas');
  c.width = c.parentElement.clientWidth - 12;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, c.width, c.height);
  const trackIds = Object.keys(trackHistory);
  if (!trackIds.length) return;
  const rowH = Math.min(20, (c.height - 20) / trackIds.length);
  const w = c.width - 60;
  const total = nFrames || 1;
  ctx.font = '9px monospace';
  trackIds.forEach((tid, row) => {
    const { colour, frames } = trackHistory[tid];
    const y = 10 + row * rowH;
    // label
    ctx.fillStyle = '#' + colour;
    ctx.textAlign = 'right';
    ctx.fillText(`id ${tid}`, 50, y + rowH*0.6);
    // presence bars
    ctx.fillStyle = '#' + colour;
    for (const fi of frames) {
      const x = 55 + (fi / total) * w;
      ctx.fillRect(x, y + 2, Math.max(1, w / total), rowH - 4);
    }
    // current frame marker
    const cur = 55 + (frame.frame_idx / total) * w;
    ctx.strokeStyle = '#ffffff44';
    ctx.beginPath();
    ctx.moveTo(cur, 8); ctx.lineTo(cur, c.height - 4);
    ctx.stroke();
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Track list sidebar
// ─────────────────────────────────────────────────────────────────────────────
function updateTrackList(frame) {
  const list = document.getElementById('track-list');
  if (!frame.tracks.length) {
    list.innerHTML = '<span style="color:var(--sub)">No active tracks</span>';
    return;
  }
  list.innerHTML = frame.tracks.map(tr => {
    const speed = Math.sqrt(tr.vx**2 + tr.vy**2).toFixed(2);
    const dist  = Math.sqrt(tr.x**2  + tr.y**2).toFixed(2);
    return `
      <div class="track-entry">
        <div class="track-swatch" style="background:#${tr.colour}"></div>
        <div class="track-info">
          <span class="track-id">Track #${tr.id}</span>
          <div class="track-sub">pos (${tr.x.toFixed(2)}, ${tr.y.toFixed(2)}) m · d=${dist}m</div>
          <div class="track-sub">vel (${tr.vx.toFixed(2)}, ${tr.vy.toFixed(2)}) → ${speed} m/s</div>
          <div class="track-sub">misses: ${tr.misses} · r_safe: ${tr.r_safe.toFixed(2)}m</div>
        </div>
      </div>`;
  }).join('');
}

// ─────────────────────────────────────────────────────────────────────────────
// Main render dispatcher
// ─────────────────────────────────────────────────────────────────────────────
function renderFrame(frame) {
  currentFrame = frame;
  nFrames = frame.n_frames;
  const [c1, c2, c3, c4] = getCanvases();
  [c1,c2,c3,c4].forEach(resizeCanvas);

  renderScan(c1, frame);
  renderSegments(c2, frame);
  renderDetections(c3, frame);
  renderTracks(c4, frame);
  renderFFT(frame);
  updateTimeline(frame);
  updateTrackList(frame);

  // Update titles
  document.getElementById('title1').textContent = `① Raw Scan  (${frame.points.length} pts)`;
  document.getElementById('title2').textContent = `② Segments  (${frame.segments.length} clusters)`;
  document.getElementById('title3').textContent = `③ Detections  (${frame.measurements.length} people)`;
  document.getElementById('title4').textContent = `④ Tracks + Kalman  (${frame.tracks.length} active)  v=${frame.linear_x.toFixed(2)} ω=${frame.angular_z.toFixed(2)}`;

  // Status bar
  document.getElementById('status').textContent =
    `frame ${frame.frame_idx+1}/${frame.n_frames}  ·  t=${frame.t.toFixed(2)}s  ·  ${frame.tracks.length} tracks  ·  v=${frame.linear_x.toFixed(2)}m/s  ω=${frame.angular_z.toFixed(2)}rad/s`;

  // Scrubber
  const scrub = document.getElementById('scrub');
  scrub.max = frame.n_frames - 1;
  scrub.value = frame.frame_idx;
  document.getElementById('frame-label').textContent =
    `Frame ${frame.frame_idx+1} / ${frame.n_frames}  |  t=${frame.t.toFixed(2)}s`;
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE stream
// ─────────────────────────────────────────────────────────────────────────────
const evtSrc = new EventSource('/stream');
evtSrc.onmessage = e => {
  const frame = JSON.parse(e.data);
  if (frame.pipeline_done) {
    document.getElementById('rerun-status').textContent = '✓ Pipeline re-run complete';
    setTimeout(() => document.getElementById('rerun-status').textContent = '', 3000);
  }
  renderFrame(frame);
  if (frame.playing !== undefined) {
    playing = frame.playing;
    document.getElementById('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
  }
};
evtSrc.onerror = () => document.getElementById('status').textContent = 'SSE disconnected — reload page';

// ─────────────────────────────────────────────────────────────────────────────
// Controls
// ─────────────────────────────────────────────────────────────────────────────
async function fetchFrame(idx) {
  const r = await fetch(`/api/frame/${idx}`);
  const data = await r.json();
  renderFrame(data);
}

document.getElementById('btn-prev').addEventListener('click', () => {
  const idx = currentFrame ? Math.max(0, currentFrame.frame_idx - 1) : 0;
  fetch('/api/play', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action:'pause', idx})}).then(() => fetchFrame(idx));
});
document.getElementById('btn-next').addEventListener('click', () => {
  const idx = currentFrame ? Math.min(nFrames-1, currentFrame.frame_idx + 1) : 0;
  fetch('/api/play', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action:'pause', idx})}).then(() => fetchFrame(idx));
});
document.getElementById('btn-play').addEventListener('click', () => {
  const fps = parseFloat(document.getElementById('fps-input').value) || 20;
  fetch('/api/play', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action:'toggle', fps})})
    .then(r => r.json()).then(d => {
      playing = d.playing;
      document.getElementById('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
    });
});

document.getElementById('scrub').addEventListener('input', function() {
  fetch('/api/play', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action:'pause', idx: parseInt(this.value)})})
    .then(() => fetchFrame(parseInt(this.value)));
});

document.getElementById('fps-input').addEventListener('change', function() {
  const fps = parseFloat(this.value) || 20;
  if (playing) fetch('/api/play', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({action:'play', fps})});
});

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft')  document.getElementById('btn-prev').click();
  if (e.key === 'ArrowRight') document.getElementById('btn-next').click();
  if (e.key === ' ')          { e.preventDefault(); document.getElementById('btn-play').click(); }
});

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar tabs
// ─────────────────────────────────────────────────────────────────────────────
document.querySelectorAll('.stab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.stab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.stab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    // re-render FFT if switching to that tab
    if (tab.dataset.tab === 'fft' && currentFrame) renderFFT(currentFrame);
    if (tab.dataset.tab === 'timeline' && currentFrame) updateTimeline(currentFrame);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Config sliders
// ─────────────────────────────────────────────────────────────────────────────
const SLIDER_DEFS = [
  { key: 'curv_threshold',       label: 'Curvature threshold',  min:0.05, max:1.0,  step:0.01 },
  { key: 'distance_threshold',   label: 'Distance threshold',   min:0.01, max:0.5,  step:0.01 },
  { key: 'max_leg_width',        label: 'Max leg width',        min:0.1,  max:1.0,  step:0.01 },
  { key: 'leg_radius',           label: 'Leg radius',           min:0.02, max:0.3,  step:0.01 },
  { key: 'obstacle_radius_scale',label: 'Obstacle radius scale',min:0.5,  max:5.0,  step:0.1  },
  { key: 'plot_range',           label: 'Plot range (m)',       min:1.0,  max:10.0, step:0.5  },
  { key: 'trail_length',         label: 'Trail length (frames)',min:5,    max:100,  step:1    },
  { key: 'max_misses',           label: 'Max misses',           min:1,    max:20,   step:1    },
];

const cfgContainer = document.getElementById('cfg-sliders');
const localCfg = Object.assign({}, {{ defaults | tojson }});

SLIDER_DEFS.forEach(({ key, label, min, max, step }) => {
  const row = document.createElement('div');
  row.className = 'cfg-row';
  const val = localCfg[key] ?? 1;
  row.innerHTML = `
    <div class="cfg-label">
      <span>${label}</span>
      <span id="lv-${key}">${val}</span>
    </div>
    <input type="range" min="${min}" max="${max}" step="${step}" value="${val}" data-key="${key}">
  `;
  cfgContainer.appendChild(row);
});

cfgContainer.querySelectorAll('input[type=range]').forEach(inp => {
  inp.addEventListener('input', function() {
    const k = this.dataset.key;
    localCfg[k] = parseFloat(this.value);
    document.getElementById(`lv-${k}`).textContent = this.value;
    if (k === 'plot_range') { plotRange = parseFloat(this.value); if (currentFrame) renderFrame(currentFrame); }
    if (k === 'trail_length' && currentFrame) renderFrame(currentFrame);
  });
});

document.getElementById('btn-rerun').addEventListener('click', () => {
  document.getElementById('rerun-status').textContent = '⏳ Re-running pipeline…';
  fetch('/api/config', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(localCfg)});
});

// ─────────────────────────────────────────────────────────────────────────────
// Initial load
// ─────────────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  fetchFrame(0);
});

window.addEventListener('resize', () => {
  if (currentFrame) renderFrame(currentFrame);
});
</script>
</body>
</html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Global bag data (set in main, read by re-run endpoint)
# ─────────────────────────────────────────────────────────────────────────────
_scans: List[FakeLaserScan] = []
_scan_times: List[float] = []
_odom_map: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _find_bag() -> Path:
    cwd = Path(".")
    for d in sorted(cwd.iterdir()):
        if d.is_dir() and (d / "metadata.yaml").exists():
            return d
    db3_files = list(cwd.glob("*.db3"))
    if db3_files:
        return db3_files[0]
    sys.exit(
        "No rosbag2 found in the current directory.\n"
        "Pass the path explicitly:  python3 visualizer_web.py --bag /path/to/bag\n"
    )


def main():
    global _frames, _t0, _scans, _scan_times, _odom_map

    parser = argparse.ArgumentParser(
        description="Browser-based replay visualizer for the people-avoidance pipeline."
    )
    parser.add_argument("--bag", "-b", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    bag_path = args.bag if args.bag is not None else _find_bag()
    if not bag_path.exists():
        sys.exit(f"Bag path not found: {bag_path}")

    _scans, _scan_times, _odom_map = load_bag(bag_path)
    if not _scans:
        sys.exit("No /scan messages found in the bag.")

    _frames = run_pipeline(_scans, _scan_times, _odom_map, _cfg)
    _t0 = _frames[0].timestamp_s if _frames else 0.0

    print(f"\n  ✓  Open  http://localhost:{args.port}  in your browser\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
