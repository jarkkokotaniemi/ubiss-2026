"""
visualizer.py — Real-time visualizer for the people-avoidance pipeline.

Subscribes to:
    /scan       sensor_msgs/LaserScan
    /odom       nav_msgs/Odometry

Runs the full pipeline internally (same functions as the node) and renders
four panels in a live matplotlib window:

    ┌─────────────────────┬──────────────────────┐
    │  1. Raw scan        │  2. Segments         │
    ├─────────────────────┼──────────────────────┤
    │  3. Leg detection   │  4. Tracks + control │
    └─────────────────────┴──────────────────────┘

Usage (from the package root, with ROS 2 sourced):
    python3 visualizer.py

Requires: matplotlib, numpy, rclpy (standard ROS 2 install).
The people_avoidance package must be on PYTHONPATH, e.g.:
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
"""

from __future__ import annotations

import math
import threading
from typing import List, Optional

import matplotlib

matplotlib.use("TkAgg")  # change to "Qt5Agg" if you prefer Qt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from people_avoidance.leg_detection import (
    LegMeasurement,
    detect_legs,
    scan_to_cartesian,
    segment_scan,
)
from people_avoidance.tracking import KalmanTracker, Track
from people_avoidance.controller import compute_velocity, obstacle_radius

# ── Tunable constants (match your launch parameters) ─────────────────────────
DISTANCE_THRESHOLD = 0.20  # segmentation gap (m)
LEG_RADIUS = 0.10  # expected single-leg radius (m)
MAX_LEG_WIDTH = 0.25  # max leg-pair separation (m)
DT = 0.10  # KF time step (s)
MAX_MISSES = 5
MAX_LINEAR_SPEED = 0.20
MAX_ANGULAR_SPEED = 1.00
OBSTACLE_RADIUS_SCALE = 2.0
PLOT_RANGE = 4.0  # metres shown in each panel

# ── Colour palette (colourblind-friendly) ─────────────────────────────────────
C_SCAN = "#888780"  # raw scan points — gray
C_SEG = [  # up to 12 segment colours, cycling
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
C_LEG = "#EF9F27"  # leg candidates — amber
C_PERSON = "#D85A30"  # person detections — coral
C_TRACK = "#534AB7"  # track ellipses — purple
C_ROBOT = "#0F6E56"  # robot heading arrow — teal
C_CMD = "#D85A30"  # commanded velocity arrow — coral


# ─────────────────────────────────────────────────────────────────────────────
# ROS subscriber node
# ─────────────────────────────────────────────────────────────────────────────


class PipelineVisualizer(Node):
    """
    ROS 2 node that subscribes to /scan and /odom, runs the full pipeline,
    and stores the per-scan pipeline state for the matplotlib renderer.
    """

    def __init__(self) -> None:
        super().__init__("pipeline_visualizer")

        self.tracker = KalmanTracker(dt=DT, max_misses=MAX_MISSES)

        # Latest pipeline state (written by ROS thread, read by plot thread)
        self._lock = threading.Lock()
        self._state: Optional[dict] = None

        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_theta = 0.0

        self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_theta = math.atan2(siny, cosy)

    def _scan_cb(self, scan: LaserScan) -> None:
        # ── Stage 1 — leg detection (with intermediate data captured) ─────────
        points = scan_to_cartesian(scan)
        segments = (
            segment_scan(points, distance_threshold=DISTANCE_THRESHOLD)
            if points.shape[0] > 0
            else []
        )
        segments = list(segments)  # materialise generator

        measurements = detect_legs(
            scan,
            distance_threshold=DISTANCE_THRESHOLD,
            leg_radius=LEG_RADIUS,
            max_leg_width=MAX_LEG_WIDTH,
        )

        # ── Stage 2 — tracking ────────────────────────────────────────────────
        self.tracker.update(measurements)
        tracks = self.tracker.get_tracks()

        # ── Stage 3 — control ─────────────────────────────────────────────────
        cmd = compute_velocity(
            tracks,
            robot_x=self._robot_x,
            robot_y=self._robot_y,
            robot_theta=self._robot_theta,
            max_linear_speed=MAX_LINEAR_SPEED,
            max_angular_speed=MAX_ANGULAR_SPEED,
            obstacle_radius_scale=OBSTACLE_RADIUS_SCALE,
        )

        state = dict(
            points=points,
            segments=segments,
            measurements=measurements,
            tracks=tracks,
            robot_x=self._robot_x,
            robot_y=self._robot_y,
            robot_theta=self._robot_theta,
            linear_x=cmd.linear.x,
            angular_z=cmd.angular.z,
        )
        with self._lock:
            self._state = state

    def get_state(self) -> Optional[dict]:
        with self._lock:
            return self._state


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib renderer
# ─────────────────────────────────────────────────────────────────────────────


def _draw_robot(ax, rx, ry, theta, size=0.15):
    """Draw the robot as a circle with a heading tick."""
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
    """Draw a 2-sigma error ellipse from a 2×2 covariance matrix."""
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


def render(fig, axes, state: dict) -> None:
    """Clear and redraw all four panels from pipeline state."""
    for ax in axes.flat:
        ax.cla()

    ax1, ax2, ax3, ax4 = axes.flat
    rx = state["robot_x"]
    ry = state["robot_y"]
    rth = state["robot_theta"]
    pts = state["points"]

    # ── Panel 1: raw scan ─────────────────────────────────────────────────────
    _axis_setup(ax1, "1 · raw lidar scan", rx, ry)
    if pts.shape[0] > 0:
        ax1.scatter(
            pts[:, 0] + rx, pts[:, 1] + ry, s=2, color=C_SCAN, alpha=0.6, zorder=2
        )
    _draw_robot(ax1, rx, ry, rth)
    ax1.set_title(f"1 · raw scan   ({pts.shape[0]} pts)", fontsize=9, pad=4)

    # ── Panel 2: segments ─────────────────────────────────────────────────────
    _axis_setup(ax2, "2 · segmentation", rx, ry)
    for idx, seg in enumerate(state["segments"]):
        col = C_SEG[idx % len(C_SEG)]
        ax2.scatter(seg[:, 0] + rx, seg[:, 1] + ry, s=3, color=col, alpha=0.8, zorder=2)
    _draw_robot(ax2, rx, ry, rth)
    ax2.set_title(
        f"2 · segments   ({len(state['segments'])} clusters)", fontsize=9, pad=4
    )

    # ── Panel 3: leg detection ────────────────────────────────────────────────
    _axis_setup(ax3, "3 · leg detection", rx, ry)
    if pts.shape[0] > 0:
        ax3.scatter(
            pts[:, 0] + rx, pts[:, 1] + ry, s=2, color=C_SCAN, alpha=0.3, zorder=1
        )
    for m in state["measurements"]:
        mx, my = m.x + rx, m.y + ry
        ax3.scatter(mx, my, s=60, color=C_PERSON, zorder=4, marker="*")
        sigma = math.sqrt(m.Rxx)
        circ = plt.Circle(
            (mx, my), sigma, color=C_LEG, fill=False, linestyle="--", lw=0.8, zorder=3
        )
        ax3.add_patch(circ)
    _draw_robot(ax3, rx, ry, rth)
    ax3.set_title(
        f"3 · detections   ({len(state['measurements'])} people)", fontsize=9, pad=4
    )

    # ── Panel 4: tracks + control ─────────────────────────────────────────────
    _axis_setup(ax4, "4 · tracks + control", rx, ry)
    if pts.shape[0] > 0:
        ax4.scatter(
            pts[:, 0] + rx, pts[:, 1] + ry, s=2, color=C_SCAN, alpha=0.2, zorder=1
        )

    for t in state["tracks"]:
        tx, ty = t.m[0] + rx, t.m[1] + ry
        # velocity arrow
        vx, vy = t.m[2], t.m[3]
        if abs(vx) + abs(vy) > 0.02:
            ax4.annotate(
                "",
                xy=(tx + vx * 0.5, ty + vy * 0.5),
                xytext=(tx, ty),
                arrowprops=dict(arrowstyle="->", color=C_TRACK, lw=1.0),
            )
        # uncertainty ellipse
        _draw_covariance_ellipse(
            ax4,
            tx,
            ty,
            t.P[:2, :2],
            color=C_TRACK,
            fill=False,
            linestyle="-",
            linewidth=1.0,
            alpha=0.5,
            zorder=3,
        )
        # safety radius
        r_safe = obstacle_radius(t, OBSTACLE_RADIUS_SCALE)
        safe_c = plt.Circle(
            (tx, ty), r_safe, color=C_TRACK, fill=True, alpha=0.08, zorder=2
        )
        ax4.add_patch(safe_c)
        safe_border = plt.Circle(
            (tx, ty),
            r_safe,
            color=C_TRACK,
            fill=False,
            linestyle=":",
            linewidth=0.9,
            zorder=3,
        )
        ax4.add_patch(safe_border)
        # label
        ax4.text(
            tx,
            ty + 0.12,
            f"id={t.track_id}  miss={t.misses}",
            fontsize=7,
            ha="center",
            color=C_TRACK,
            zorder=5,
        )
        ax4.scatter(tx, ty, s=40, color=C_TRACK, zorder=4)

    # commanded velocity arrow (in robot body frame → world frame)
    v = state["linear_x"]
    w = state["angular_z"]
    fwd = 0.6 * v
    ex = rx + fwd * math.cos(rth)
    ey = ry + fwd * math.sin(rth)
    if abs(fwd) > 0.005:
        ax4.annotate(
            "",
            xy=(ex, ey),
            xytext=(rx, ry),
            arrowprops=dict(arrowstyle="-|>", color=C_CMD, lw=2.0, mutation_scale=12),
        )

    _draw_robot(ax4, rx, ry, rth)
    ax4.set_title(
        f"4 · tracks={len(state['tracks'])}   " f"v={v:.2f} m/s   ω={w:.2f} rad/s",
        fontsize=9,
        pad=4,
    )

    # ── Legend (shared, below panels) ────────────────────────────────────────
    handles = [
        mlines.Line2D([], [], color=C_SCAN, marker="o", ls="", ms=4, label="raw scan"),
        mpatches.Patch(color=C_SEG[0], label="segment (sample)"),
        mlines.Line2D(
            [], [], color=C_PERSON, marker="*", ls="", ms=8, label="leg detection"
        ),
        mpatches.Patch(color=C_TRACK, alpha=0.3, label="track ellipse / safety radius"),
        mlines.Line2D([], [], color=C_CMD, lw=2, label="cmd_vel arrow"),
        mlines.Line2D([], [], color=C_ROBOT, lw=1.5, label="robot heading"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    rclpy.init()
    node = PipelineVisualizer()

    # Spin ROS in a background thread so matplotlib owns the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("People-avoidance pipeline — real-time visualizer", fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.ion()
    plt.show(block=False)

    print("Visualizer running — waiting for /scan messages …")
    try:
        while rclpy.ok():
            state = node.get_state()
            if state is not None:
                render(fig, axes, state)
            plt.pause(0.05)  # ~20 Hz redraw
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
        plt.close("all")


if __name__ == "__main__":
    main()
