"""
visualizer_offline.py — Offline replay visualizer for the people-avoidance pipeline.

Reads /scan and /odom messages directly from a rosbag2 .db3 file using the
`rosbags` library (no ROS install required).  Runs the full detection →
tracking → control pipeline and displays the same 4-panel matplotlib window
as visualizer.py, with added playback controls.

Layout
------
    ┌─────────────────────┬──────────────────────┐
    │  1. Raw scan        │  2. Segments         │
    ├─────────────────────┼──────────────────────┤
    │  3. Leg detection   │  4. Tracks + control │
    └─────────────────────┴──────────────────────┘
    [ |< Prev ]  [ Next > ]  [ ▶ Play / ⏸ Pause ]  [ slider ]

Usage
-----
    python3 visualizer_offline.py --bag /path/to/rosbag2_dir_or_db3
    python3 visualizer_offline.py                  # looks for bag in cwd

Dependencies (pip install rosbags matplotlib numpy)
No ROS installation needed.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import math
import matplotlib

matplotlib.use("TkAgg")  # swap to "Qt5Agg" or "Agg" if needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ── rosbags import ────────────────────────────────────────────────────────────
try:
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore
except ImportError:
    sys.exit(
        "Missing dependency.\n" "Install with:  pip install rosbags numpy matplotlib\n"
    )

# ── Pipeline imports — load directly from sibling .py files ──────────────────
# Inserts the script's own directory at the front of sys.path so that
# leg_detection.py / tracking.py / controller.py are found regardless of
# where Python is invoked from.  Also stubs out the ROS message types that
# those modules import at the top level so they don't require a ROS install.

import types


# Stub out ROS message types used in the pipeline modules
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


# Twist needs .linear.x and .angular.z sub-objects
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

# Also patch the relative-import style used inside the modules:
# tracking.py does  `from .leg_detection import …`
# controller.py does `from .tracking import …`
# We load them as top-level modules and wire the dotted aliases ourselves.
import importlib.util, os

_HERE = Path(__file__).resolve().parent


def _load_local(
    filename: str, module_name: str, package_alias: str = "people_avoidance"
):
    """
    Load a local .py file, satisfying relative imports (from .sibling import X)
    by registering already-loaded siblings under a fake package alias.
    """
    if package_alias not in sys.modules:
        pkg = types.ModuleType(package_alias)
        pkg.__path__ = [str(_HERE)]
        pkg.__package__ = package_alias
        sys.modules[package_alias] = pkg

    full_name = f"{package_alias}.{module_name}"
    spec = importlib.util.spec_from_file_location(
        full_name,
        _HERE / filename,
        submodule_search_locations=[],
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

# Expose what the rest of this script needs
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
# Pipeline tuning constants  (edit here or expose as CLI args)
# ─────────────────────────────────────────────────────────────────────────────
DISTANCE_THRESHOLD = 0.2  # segmentation gap (m)
LEG_RADIUS = 0.10  # expected single-leg radius (m)
MAX_LEG_WIDTH = 1  # max leg-pair separation (m)
DT = 0.5  # KF time step (s) — approximate; see note below
MAX_MISSES = 12  # increased: legs move fast, allow more missed frames
MAX_LINEAR_SPEED = 2
MAX_ANGULAR_SPEED = 2.00
OBSTACLE_RADIUS_SCALE = 1.0
PLOT_RANGE = 4.0  # half-width of each panel (m)
CURV_THRESHOLD = 2  # |r''| threshold for segment break (m)
TRAIL_LENGTH = 40  # how many past positions to show per track

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (colourblind-friendly, matches visualizer.py)
# ─────────────────────────────────────────────────────────────────────────────
C_SCAN = "#888780"
C_SEG = [
    "#5DCAA5",
    "#378ADD",
    "#D85A30",
    "#7F77DD",
    "#E24B4A",
    "#639922",
    "#EF9F27",
    "#D4537E",
    "#1D9E75",
    "#185FA5",
    "#993C1D",
    "#534AB7",
]
C_LEG = "#EF9F27"
C_PERSON = "#D85A30"
C_TRACK = "#534AB7"
C_ROBOT = "#0F6E56"
C_CMD = "#D85A30"

# Per-track colors — enough for many simultaneous tracks
TRACK_COLORS = [
    "#5DCAA5",
    "#378ADD",
    "#EF9F27",
    "#E24B4A",
    "#D4537E",
    "#7F77DD",
    "#639922",
    "#D85A30",
    "#1D9E75",
    "#185FA5",
    "#993C1D",
    "#534AB7",
    "#F06292",
    "#4DD0E1",
    "#AED581",
    "#FFB300",
    "#AB47BC",
    "#26C6DA",
    "#EC407A",
    "#66BB6A",
]


def _track_color(track_id: int) -> str:
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight stand-in message objects
# (rosbags gives us its own typed objects; we wrap them so the pipeline
#  functions — which expect sensor_msgs/LaserScan — still work.)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakeLaserScan:
    """Minimal duck-type replacement for sensor_msgs/msg/LaserScan."""

    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    ranges: np.ndarray


@dataclass
class FrameData:
    """One processed scan frame ready for rendering."""

    timestamp_s: float
    scan_index: int
    # raw pipeline inputs
    points: np.ndarray
    segments: List[np.ndarray]
    measurements: List[LegMeasurement]
    tracks: List[Track]
    # controller output
    linear_x: float
    angular_z: float
    # robot pose
    robot_x: float
    robot_y: float
    robot_theta: float
    fft_freqs: np.ndarray
    fft_mags: np.ndarray
    # track_id → list of (x, y) positions up to this frame (for trail drawing)
    track_trails: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Bag loader
# ─────────────────────────────────────────────────────────────────────────────


def _yaw_from_quat(qx, qy, qz, qw) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def load_bag(bag_path: Path) -> Tuple[List[FakeLaserScan], List[float], dict]:
    """
    Read /scan and /odom from a rosbag2 db3.

    Returns
    -------
    scans      : list of FakeLaserScan  (568 messages in the provided bag)
    scan_times : matching timestamps in seconds
    odom_map   : {timestamp_ns: (x, y, theta)}  for nearest-odom lookup
    """
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    # rosbags Reader needs the directory containing the db3 + metadata.yaml
    bag_dir = bag_path if bag_path.is_dir() else bag_path.parent

    scans: List[FakeLaserScan] = []
    scan_times: List[float] = []
    odom_map: dict = {}  # ts_ns → (x, y, theta)

    print(f"Loading bag from {bag_dir} …", flush=True)
    with Reader(bag_dir) as reader:
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic == "/scan":
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                fake = FakeLaserScan(
                    angle_min=float(msg.angle_min),
                    angle_max=float(msg.angle_max),
                    angle_increment=float(msg.angle_increment),
                    range_min=float(msg.range_min),
                    range_max=float(msg.range_max),
                    ranges=np.array(msg.ranges, dtype=float),
                )
                scans.append(fake)
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


def nearest_odom(odom_map: dict, target_ns: int) -> Tuple[float, float, float]:
    """Return the odom pose (x, y, theta) closest in time to target_ns."""
    if not odom_map:
        return 0.0, 0.0, 0.0
    best = min(odom_map.keys(), key=lambda t: abs(t - target_ns))
    return odom_map[best]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner — processes all frames in order so tracking state is correct
# ─────────────────────────────────────────────────────────────────────────────


def run_pipeline(
    scans: List[FakeLaserScan],
    scan_times: List[float],
    odom_map: dict,
) -> List[FrameData]:
    """
    Run the full detection → tracking → control pipeline over every scan.

    The Kalman tracker is reset once and then updated sequentially so that
    track continuity across frames is maintained exactly as it would be on
    the live robot.  The per-frame dt is derived from consecutive scan
    timestamps; DT is used as a fallback for the first frame.
    """
    tracker = KalmanTracker(dt=DT, max_misses=MAX_MISSES)
    frames: List[FrameData] = []
    # Persistent trail history: track_id → deque of (x, y)
    from collections import deque

    trail_history: dict = {}  # track_id → deque[(x, y), ...]

    # Build sorted list of odom ts keys for fast nearest lookup
    odom_keys = sorted(odom_map.keys())

    def _nearest_odom(ts_s: float):
        ts_ns = int(ts_s * 1e9)
        # binary search for speed
        lo, hi = 0, len(odom_keys) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if odom_keys[mid] < ts_ns:
                lo = mid + 1
            else:
                hi = mid
        # check lo and lo-1
        best = lo
        if lo > 0 and abs(odom_keys[lo - 1] - ts_ns) < abs(odom_keys[lo] - ts_ns):
            best = lo - 1
        return odom_map[odom_keys[best]]

    prev_ts = scan_times[0] - DT if scan_times else 0.0

    for i, (scan, ts) in enumerate(zip(scans, scan_times)):
        # Derive dt and update tracker's motion model accordingly
        dt_i = ts - prev_ts
        if dt_i <= 0 or dt_i > 1.0:
            dt_i = DT
        tracker.dt = dt_i
        tracker.F[0, 2] = dt_i
        tracker.F[1, 3] = dt_i
        prev_ts = ts

        # Robot pose
        rx, ry, rth = _nearest_odom(ts)

        # Stage 1 — leg detection
        points = scan_to_cartesian(scan)
        segments = (
            segment_scan(
                points,
                distance_threshold=DISTANCE_THRESHOLD,
                scan=scan,
                curv_threshold=CURV_THRESHOLD,
            )
            if points.shape[0] > 0
            else []
        )

        fft_freqs, fft_mags = scan_fft(scan)

        measurements = detect_legs(
            scan,
            distance_threshold=DISTANCE_THRESHOLD,
            leg_radius=LEG_RADIUS,
            max_leg_width=MAX_LEG_WIDTH,
            curv_threshold=CURV_THRESHOLD,
        )

        # Stage 2 — tracking
        tracker.update(measurements)
        tracks = tracker.get_tracks()

        # Update trail history for active tracks
        for tr in tracks:
            tid = tr.track_id
            if tid not in trail_history:
                trail_history[tid] = deque(maxlen=TRAIL_LENGTH)
            trail_history[tid].append((float(tr.m[0]), float(tr.m[1])))

        # Snapshot of trails for this frame (copy so later frames don't mutate)
        trails_snapshot = {tid: list(pts) for tid, pts in trail_history.items()}

        # Stage 3 — control
        # Note: in the bag the robot isn't running avoidance, but we still
        # compute the command so we can visualize what the algorithm *would*
        # have done.
        # Measurements are in the laser frame; tracks are maintained in that
        # same frame because no odom transform is applied here.
        # If your tracker already converts to odom frame, pass (rx, ry, rth).
        # Otherwise pass (0, 0, 0) so the controller sees the laser-frame tracks.
        cmd = compute_velocity(
            tracks,
            robot_x=0.0,
            robot_y=0.0,
            robot_theta=0.0,
            max_linear_speed=MAX_LINEAR_SPEED,
            max_angular_speed=MAX_ANGULAR_SPEED,
            obstacle_radius_scale=OBSTACLE_RADIUS_SCALE,
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
                track_trails=trails_snapshot,
            )
        )

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(scans)} frames …", flush=True)

    print(f"  Pipeline complete — {len(frames)} frames ready.")
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers  (identical to visualizer.py)
# ─────────────────────────────────────────────────────────────────────────────


def _draw_robot(ax, rx, ry, theta, size=0.15):
    circle = plt.Circle(
        (rx, ry), size, color=C_ROBOT, fill=False, linewidth=1.5, zorder=5
    )
    ax.add_patch(circle)
    ex = rx + size * math.cos(theta)
    ey = ry + size * math.sin(theta)
    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(rx, ry),
        arrowprops=dict(arrowstyle="->", color=C_ROBOT, lw=1.5),
    )


def _draw_covariance_ellipse(ax, x, y, P2x2, n_std=2.0, **kwargs):
    try:
        vals, vecs = np.linalg.eigh(P2x2)
        vals = np.maximum(vals, 0)
        angle = math.degrees(math.atan2(vecs[1, -1], vecs[0, -1]))
        w, h = 2 * n_std * np.sqrt(vals)
        ellipse = mpatches.Ellipse((x, y), width=w, height=h, angle=angle, **kwargs)
        ax.add_patch(ellipse)
    except Exception:
        pass


def _axis_setup(ax, title, rx, ry, r=PLOT_RANGE):
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_aspect("equal")
    ax.set_xlim(rx - r, rx + r)
    ax.set_ylim(ry - r, ry + r)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.axhline(ry, color="#cccccc", lw=0.4, zorder=0)
    ax.axvline(rx, color="#cccccc", lw=0.4, zorder=0)


# ─────────────────────────────────────────────────────────────────────────────
# Fast renderer — clears only dynamic artists, keeps static axes alive
# ─────────────────────────────────────────────────────────────────────────────


class FastRenderer:
    """
    Manages all matplotlib artists for the 4-panel view.

    On first call _setup() draws the static elements (axes, grid lines, labels,
    legend) and captures the background with fig.canvas.copy_from_bbox().
    On every subsequent _show() call we:
      1. Restore the background (no redraw of axes/ticks/legend).
      2. Draw only the dynamic artists (scan points, segments, detections,
         tracks, robot marker, cmd arrow) via ax.draw_artist().
      3. blit() the updated region to the screen.

    This keeps rendering at ~30 fps even on a laptop.
    """

    def __init__(self, fig, axes):
        self.fig = fig
        self.axes = axes
        self.ax1, self.ax2, self.ax3, self.ax4 = axes.flat
        self._bg = None  # cached background (set after first draw)
        self._ready = False

    # ── one-time static setup ─────────────────────────────────────────────────
    def _setup_axes(self):
        for ax in self.axes.flat:
            ax.set_aspect("equal")
            ax.set_xlim(-PLOT_RANGE, PLOT_RANGE)
            ax.set_ylim(-PLOT_RANGE, PLOT_RANGE)
            ax.set_xlabel("x (m)", fontsize=7)
            ax.set_ylabel("y (m)", fontsize=7)
            ax.tick_params(labelsize=7)
            # static grid lines
            ax.axhline(0, color="#cccccc", lw=0.4, zorder=0)
            ax.axvline(0, color="#cccccc", lw=0.4, zorder=0)
            ax.set_animated(False)

        # placeholder titles (updated per-frame via set_text, not set_title)
        self._t1 = self.ax1.set_title("", fontsize=9, pad=4)
        self._t2 = self.ax2.set_title("", fontsize=9, pad=4)
        self._t3 = self.ax3.set_title("", fontsize=9, pad=4)
        self._t4 = self.ax4.set_title("", fontsize=9, pad=4)

        # ── legend (drawn once into the static background) ────────────────────
        handles = [
            mlines.Line2D(
                [], [], color=C_SCAN, marker="o", ls="", ms=4, label="raw scan"
            ),
            mpatches.Patch(color=C_SEG[0], label="segment"),
            mlines.Line2D(
                [], [], color=C_PERSON, marker="*", ls="", ms=8, label="leg detection"
            ),
            mlines.Line2D([], [], color=TRACK_COLORS[0], lw=2, label="track trail"),
            mlines.Line2D([], [], color=C_CMD, lw=2, label="cmd_vel"),
            mlines.Line2D([], [], color=C_ROBOT, lw=1.5, label="robot"),
        ]
        self.fig.legend(
            handles=handles,
            loc="lower center",
            ncol=6,
            fontsize=7,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

    def _capture_bg(self):
        self.fig.canvas.draw()  # full draw to get clean axes into the buffer
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._ready = True

    # ── per-frame update ──────────────────────────────────────────────────────
    def show(self, frame: FrameData, suptitle: str = "") -> None:
        if not self._ready:
            self._setup_axes()
            self._capture_bg()

        self.fig.canvas.restore_region(self._bg)

        pts = frame.points

        # helper: draw a collection of artists then throw them away
        def _draw(ax, artists):
            for a in artists:
                ax.add_artist(a)
                ax.draw_artist(a)
                a.remove()

        # ── Panel 1: raw scan ──────────────────────────────────────────────────
        if pts.shape[0] > 0:
            sc = self.ax1.scatter(
                pts[:, 0],
                pts[:, 1],
                s=2,
                color=C_SCAN,
                alpha=0.6,
                zorder=2,
                animated=True,
            )
            self.ax1.draw_artist(sc)
            sc.remove()
        _draw_robot_artists(self.ax1, 0, 0, 0)
        t = self._t1
        t.set_text(f"1 · raw scan   ({pts.shape[0]} pts)")
        self.ax1.draw_artist(t)

        # ── Panel 2: segments ──────────────────────────────────────────────────
        for idx, seg in enumerate(frame.segments):
            sc = self.ax2.scatter(
                seg[:, 0],
                seg[:, 1],
                s=3,
                color=C_SEG[idx % len(C_SEG)],
                alpha=0.8,
                zorder=2,
                animated=True,
            )
            self.ax2.draw_artist(sc)
            sc.remove()
        _draw_robot_artists(self.ax2, 0, 0, 0)
        t = self._t2
        t.set_text(f"2 · segments   ({len(frame.segments)} clusters)")
        self.ax2.draw_artist(t)

        # ── Panel 3: leg detection ─────────────────────────────────────────────
        if pts.shape[0] > 0:
            sc = self.ax3.scatter(
                pts[:, 0],
                pts[:, 1],
                s=2,
                color=C_SCAN,
                alpha=0.3,
                zorder=1,
                animated=True,
            )
            self.ax3.draw_artist(sc)
            sc.remove()
        for m in frame.measurements:
            sc = self.ax3.scatter(
                m.x, m.y, s=60, color=C_PERSON, zorder=4, marker="*", animated=True
            )
            self.ax3.draw_artist(sc)
            sc.remove()
            sigma = math.sqrt(max(m.Rxx, 0))
            circ = mpatches.Circle(
                (m.x, m.y),
                sigma,
                color=C_LEG,
                fill=False,
                linestyle="--",
                lw=0.8,
                zorder=3,
                animated=True,
            )
            self.ax3.add_patch(circ)
            self.ax3.draw_artist(circ)
            circ.remove()
        _draw_robot_artists(self.ax3, 0, 0, 0)
        t = self._t3
        t.set_text(f"3 · detections   ({len(frame.measurements)} people)")
        self.ax3.draw_artist(t)

        # ── Panel 4: tracks + control ──────────────────────────────────────────
        if pts.shape[0] > 0:
            sc = self.ax4.scatter(
                pts[:, 0],
                pts[:, 1],
                s=2,
                color=C_SCAN,
                alpha=0.2,
                zorder=1,
                animated=True,
            )
            self.ax4.draw_artist(sc)
            sc.remove()

        for tr in frame.tracks:
            tx, ty = tr.m[0], tr.m[1]
            vx, vy = tr.m[2], tr.m[3]
            col = _track_color(tr.track_id)

            # ── Trail path ────────────────────────────────────────────────────
            trail = frame.track_trails.get(tr.track_id, [])
            if len(trail) >= 2:
                xs = [p[0] for p in trail]
                ys = [p[1] for p in trail]
                n = len(xs)
                # Draw fading segments: newer = brighter/thicker
                for k in range(n - 1):
                    frac = (k + 1) / n  # 0→oldest, 1→newest
                    alpha = 0.15 + 0.75 * frac
                    lw = 0.6 + 1.8 * frac
                    line = mlines.Line2D(
                        [xs[k], xs[k + 1]],
                        [ys[k], ys[k + 1]],
                        color=col,
                        alpha=alpha,
                        linewidth=lw,
                        solid_capstyle="round",
                        zorder=2,
                        animated=True,
                    )
                    self.ax4.add_line(line)
                    self.ax4.draw_artist(line)
                    line.remove()

            # ── Velocity arrow ────────────────────────────────────────────────
            speed = math.hypot(vx, vy)
            if speed > 0.02:
                scale = min(speed * 0.4, 0.8)
                nx, ny = vx / speed * scale, vy / speed * scale
                arr = self.ax4.annotate(
                    "",
                    xy=(tx + nx, ty + ny),
                    xytext=(tx, ty),
                    arrowprops=dict(
                        arrowstyle="-|>", color=col, lw=1.5, mutation_scale=10
                    ),
                    animated=True,
                )
                self.ax4.draw_artist(arr)
                arr.remove()

            # ── Current position dot ──────────────────────────────────────────
            sc = self.ax4.scatter(
                tx,
                ty,
                s=50,
                color=col,
                zorder=5,
                animated=True,
                edgecolors="white",
                linewidths=0.5,
            )
            self.ax4.draw_artist(sc)
            sc.remove()

            # ── Label ─────────────────────────────────────────────────────────
            lbl = self.ax4.text(
                tx,
                ty + 0.18,
                f"{tr.track_id}",
                fontsize=7,
                ha="center",
                color=col,
                zorder=6,
                fontweight="bold",
                animated=True,
            )
            self.ax4.draw_artist(lbl)
            lbl.remove()

        # cmd_vel arrow
        v, w = frame.linear_x, frame.angular_z
        if abs(v) > 0.005:
            ex = 0.6 * v * math.cos(0)
            ey = 0.6 * v * math.sin(0)
            arr = self.ax4.annotate(
                "",
                xy=(ex, ey),
                xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="-|>", color=C_CMD, lw=2.0, mutation_scale=12
                ),
                animated=True,
            )
            self.ax4.draw_artist(arr)
            arr.remove()
        _draw_robot_artists(self.ax4, 0, 0, 0)
        t = self._t4
        t.set_text(f"4 · tracks={len(frame.tracks)}   v={v:.2f} m/s   ω={w:.2f} rad/s")
        self.ax4.draw_artist(t)

        # suptitle
        self.fig.suptitle(suptitle, fontsize=10)

        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()


def _draw_robot_artists(ax, rx, ry, theta, size=0.15):
    """Draw robot circle + heading arrow as animated artists (add, draw, remove)."""
    circ = mpatches.Circle(
        (rx, ry),
        size,
        color=C_ROBOT,
        fill=False,
        linewidth=1.5,
        zorder=5,
        animated=True,
    )
    ax.add_patch(circ)
    ax.draw_artist(circ)
    circ.remove()
    ex = rx + size * math.cos(theta)
    ey = ry + size * math.sin(theta)
    arr = ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(rx, ry),
        arrowprops=dict(arrowstyle="->", color=C_ROBOT, lw=1.5),
        animated=True,
    )
    ax.draw_artist(arr)
    arr.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Interactive player
# ─────────────────────────────────────────────────────────────────────────────


class Player:
    """
    Matplotlib-based playback controller with blitting for speed.

    Controls
    --------
    Buttons : ◀◀ Prev   Next ▶▶   ▶ Play / ⏸ Pause
    Slider  : jump to any frame
    Keyboard: ← / → step,  Space play/pause,  q quit
    """

    PLAY_INTERVAL_MS = 50  # 20 fps target; lower = faster

    def __init__(self, frames: List[FrameData]) -> None:
        self.frames = frames
        self.n = len(frames)
        self.idx = 0
        self.playing = False
        self._timer = None

        self.fig, axes_2x2 = plt.subplots(2, 2, figsize=(12, 9))
        self.axes = axes_2x2
        self.fig.subplots_adjust(bottom=0.13, top=0.95, hspace=0.35, wspace=0.3)

        self.renderer = FastRenderer(self.fig, self.axes)

        from matplotlib.widgets import Button, Slider

        ax_prev = self.fig.add_axes([0.05, 0.03, 0.10, 0.04])
        ax_play = self.fig.add_axes([0.17, 0.03, 0.10, 0.04])
        ax_next = self.fig.add_axes([0.29, 0.03, 0.10, 0.04])
        ax_slider = self.fig.add_axes([0.42, 0.03, 0.54, 0.03])

        self.btn_prev = Button(ax_prev, "◀◀ Prev")
        self.btn_play = Button(ax_play, "▶ Play")
        self.btn_next = Button(ax_next, "Next ▶▶")
        self.slider = Slider(
            ax_slider, "Frame", 0, self.n - 1, valinit=0, valstep=1, valfmt="%d"
        )

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_play.on_clicked(self._on_play_pause)
        self.btn_next.on_clicked(self._on_next)
        self.slider.on_changed(self._on_slider)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Connect resize so we can recapture the background if window is resized
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)

        self._show(0)

    # ── rendering ─────────────────────────────────────────────────────────────

    def _show(self, idx: int) -> None:
        self.idx = max(0, min(idx, self.n - 1))
        frame = self.frames[self.idx]
        t0 = self.frames[0].timestamp_s
        title = (
            f"People-avoidance pipeline — offline replay  │  "
            f"frame {self.idx + 1}/{self.n}  │  "
            f"t = {frame.timestamp_s - t0:+.2f} s"
        )
        self.renderer.show(frame, suptitle=title)

        self.slider.eventson = False
        self.slider.set_val(self.idx)
        self.slider.eventson = True

    def _on_resize(self, _event) -> None:
        # Force background recapture after resize
        self.renderer._ready = False

    # ── navigation ────────────────────────────────────────────────────────────

    def _on_prev(self, _):
        self._stop_timer()
        self._show(self.idx - 1)

    def _on_next(self, _):
        self._stop_timer()
        self._show(self.idx + 1)

    def _on_slider(self, val):
        self._stop_timer()
        self._show(int(val))

    def _on_play_pause(self, _):
        self._stop_timer() if self.playing else self._start_timer()

    def _on_key(self, event):
        if event.key in ("left", "a"):
            self._stop_timer()
            self._show(self.idx - 1)
        elif event.key in ("right", "d"):
            self._stop_timer()
            self._show(self.idx + 1)
        elif event.key == " ":
            self._on_play_pause(None)
        elif event.key in ("q", "escape"):
            plt.close("all")

    # ── timer ─────────────────────────────────────────────────────────────────

    def _start_timer(self):
        self.playing = True
        self.btn_play.label.set_text("⏸ Pause")
        self._timer = self.fig.canvas.new_timer(interval=self.PLAY_INTERVAL_MS)
        self._timer.add_callback(self._tick)
        self._timer.start()

    def _stop_timer(self):
        self.playing = False
        self.btn_play.label.set_text("▶ Play")
        if self._timer:
            self._timer.stop()
            self._timer = None

    def _tick(self):
        if self.idx >= self.n - 1:
            self._stop_timer()
            return
        self._show(self.idx + 1)

    def run(self):
        plt.show()


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
        "Pass the path explicitly:  python3 visualizer_offline.py --bag /path/to/bag\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline replay visualizer for the people-avoidance pipeline."
    )
    parser.add_argument(
        "--bag",
        "-b",
        type=Path,
        default=None,
        help="Path to the rosbag2 directory or .db3 file",
    )
    args = parser.parse_args()

    bag_path = args.bag if args.bag is not None else _find_bag()
    if not bag_path.exists():
        sys.exit(f"Bag path not found: {bag_path}")

    scans, scan_times, odom_map = load_bag(bag_path)
    if not scans:
        sys.exit("No /scan messages found in the bag.")

    frames = run_pipeline(scans, scan_times, odom_map)

    player = Player(frames)
    player.run()


if __name__ == "__main__":
    main()
