"""
visualizer2.py — Lidar visualizer with click-to-goal A* navigation.

New features:
  - Left-click on the plot to set a navigation goal (green circle).
  - An occupancy grid is built from live LaserScan data every second.
  - A* pathfinding runs on that grid and the resulting waypoint list is:
      (a) drawn as a cyan path on the plot, and
      (b) broadcast over a ZMQ PUB socket so controller.py can consume it.

Dependencies (beyond existing ones):
    pip install pyzmq

ZMQ socket: tcp://*:5556  (PUB)
Message format: JSON  {"waypoints": [[x0,y0], [x1,y1], ...]}
                      or {"waypoints": []}  when no path exists.
"""

from __future__ import annotations

import heapq
import json
import math
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from people_avoidance_msgs.msg import LegMeasurementArray, TrackArray

from tf2_ros import Buffer, TransformListener, TransformException

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg

try:
    import zmq
    _ZMQ_OK = True
except ImportError:
    _ZMQ_OK = False
    print("[visualizer2] WARNING: pyzmq not installed. Path will not be sent to controller.")

# ---------------------------------------------------------------------------
# Occupancy-grid parameters (tune to your environment)
# ---------------------------------------------------------------------------
GRID_RESOLUTION   = 0.05   # metres per cell
GRID_HALF_EXTENT  = 10.0   # grid covers ±10 m around the origin in odom frame
GRID_SIZE         = int(2 * GRID_HALF_EXTENT / GRID_RESOLUTION)  # cells per side

OBSTACLE_INFLATE  = 0.18   # metres — inflate scan obstacles by robot half-width
INFLATE_CELLS     = max(1, int(OBSTACLE_INFLATE / GRID_RESOLUTION))

PERSON_INFLATE    = 0.50   # metres — much larger bubble around tracked people
PERSON_INFLATE_CELLS = max(1, int(PERSON_INFLATE / GRID_RESOLUTION))

PATH_REPLAN_HZ    = 4.0    # how often (seconds) to replan — fast enough to track moving people

# ---------------------------------------------------------------------------
# A* on a 2-D occupancy grid
# ---------------------------------------------------------------------------

def _astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """
    A* on a boolean occupancy grid (True = obstacle).

    Returns a list of (row, col) cells from start to goal (inclusive),
    or None if no path exists.
    """
    rows, cols = grid.shape
    sr, sc = start
    gr, gc = goal

    if grid[sr, sc] or grid[gr, gc]:
        return None

    def h(r, c):
        return math.hypot(r - gr, c - gc)

    # (f, g, r, c, parent)
    open_heap: list = []
    heapq.heappush(open_heap, (h(sr, sc), 0.0, sr, sc, None))

    came_from: dict[tuple, tuple | None] = {}
    g_score: dict[tuple, float] = {(sr, sc): 0.0}
    visited: set = set()

    NEIGHBORS_8 = [
        (-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0,  1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1,  1, 1.414),
    ]

    while open_heap:
        f, g, r, c, parent = heapq.heappop(open_heap)

        if (r, c) in visited:
            continue
        visited.add((r, c))
        came_from[(r, c)] = parent

        if (r, c) == (gr, gc):
            # Reconstruct path
            path = []
            node = (gr, gc)
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        for dr, dc, cost in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not grid[nr, nc]:
                ng = g + cost
                if ng < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = ng
                    heapq.heappush(open_heap, (ng + h(nr, nc), ng, nr, nc, (r, c)))

    return None  # no path


def _world_to_cell(wx: float, wy: float) -> tuple[int, int]:
    col = int((wx + GRID_HALF_EXTENT) / GRID_RESOLUTION)
    row = int((wy + GRID_HALF_EXTENT) / GRID_RESOLUTION)
    return (
        max(0, min(GRID_SIZE - 1, row)),
        max(0, min(GRID_SIZE - 1, col)),
    )


def _cell_to_world(row: int, col: int) -> tuple[float, float]:
    wx = col * GRID_RESOLUTION - GRID_HALF_EXTENT + GRID_RESOLUTION / 2
    wy = row * GRID_RESOLUTION - GRID_HALF_EXTENT + GRID_RESOLUTION / 2
    return wx, wy


def _simplify_path(waypoints: list[tuple[float, float]], tolerance: float = 0.15) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification to reduce waypoint count."""
    if len(waypoints) <= 2:
        return waypoints

    def perp_dist(pt, line_start, line_end):
        x0, y0 = pt
        x1, y1 = line_start
        x2, y2 = line_end
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(x0 - x1, y0 - y1)
        return abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.hypot(dx, dy)

    def rdp(pts, eps):
        if len(pts) < 3:
            return pts
        max_d, idx = 0.0, 0
        for i in range(1, len(pts) - 1):
            d = perp_dist(pts[i], pts[0], pts[-1])
            if d > max_d:
                max_d, idx = d, i
        if max_d > eps:
            left  = rdp(pts[:idx + 1], eps)
            right = rdp(pts[idx:], eps)
            return left[:-1] + right
        return [pts[0], pts[-1]]

    return rdp(waypoints, tolerance)


# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class LidarVisualizer(Node):

    def __init__(self) -> None:
        super().__init__("lidar_visualizer")

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.scan_sub  = self.create_subscription(LaserScan,             "/scan",   self.scan_callback,  10)
        self.leg_sub   = self.create_subscription(LegMeasurementArray,  "/legs",   self.leg_callback,   10)
        self.track_sub = self.create_subscription(TrackArray,            "/tracks", self.track_callback, 10)

        # Latest lidar points in odom frame
        self.scan_x: np.ndarray = np.array([])
        self.scan_y: np.ndarray = np.array([])

        # Latest leg detections in odom frame
        self.leg_x: np.ndarray = np.array([])
        self.leg_y: np.ndarray = np.array([])

        # Latest Kalman tracks: each entry is a dict with x,y,vx,vy,id,Pxx,Pxy,Pyy
        self.tracks: list[dict] = []
        self._track_lock = threading.Lock()

        # Robot pose in odom frame
        self.robot_x:   float = 0.0
        self.robot_y:   float = 0.0
        self.robot_yaw: float = 0.0

        # Navigation goal (odom frame) — set by mouse click
        self.goal_x: float | None = None
        self.goal_y: float | None = None

        # Current planned path (list of (x, y) in odom frame)
        self.path: list[tuple[float, float]] = []
        self._path_lock = threading.Lock()

        # Occupancy grid (built from scan points, shared with planner thread)
        self._grid: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        self._grid_lock = threading.Lock()

        # ZMQ publisher
        self._zmq_socket = None
        if _ZMQ_OK:
            ctx = zmq.Context()
            self._zmq_socket = ctx.socket(zmq.PUB)
            self._zmq_socket.bind("tcp://*:5556")
            self.get_logger().info("ZMQ PUB socket bound on tcp://*:5556")

        # Background planner thread
        self._planner_thread = threading.Thread(target=self._planner_loop, daemon=True)
        self._planner_thread.start()

    # ── Scan callback ─────────────────────────────────────────────────────────

    def scan_callback(self, scan: LaserScan) -> None:
        try:
            # Robot pose via TF
            try:
                robot_tf = self.tf_buffer.lookup_transform(
                    "odom", "base_link", scan.header.stamp, timeout=Duration(seconds=0.05)
                )
                self.robot_x   = robot_tf.transform.translation.x
                self.robot_y   = robot_tf.transform.translation.y
                self.robot_yaw = self._yaw_from_quaternion(
                    robot_tf.transform.rotation.x, robot_tf.transform.rotation.y,
                    robot_tf.transform.rotation.z, robot_tf.transform.rotation.w,
                )
            except TransformException:
                pass

            # Laser → odom transform
            trans = self.tf_buffer.lookup_transform(
                "odom", scan.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.1)
            )

            ranges = np.array(scan.ranges, dtype=np.float64)
            angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
            mask   = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
            r, a   = ranges[mask], angles[mask]

            x_local = r * np.cos(a)
            y_local = r * np.sin(a)

            tx  = trans.transform.translation.x
            ty  = trans.transform.translation.y
            yaw = self._yaw_from_quaternion(
                trans.transform.rotation.x, trans.transform.rotation.y,
                trans.transform.rotation.z, trans.transform.rotation.w,
            )
            self.scan_x = x_local * np.cos(yaw) - y_local * np.sin(yaw) + tx
            self.scan_y = x_local * np.sin(yaw) + y_local * np.cos(yaw) + ty

            # Rebuild occupancy grid from the latest scan
            self._rebuild_grid()

        except TransformException:
            return

    def _rebuild_grid(self) -> None:
        """Mark every scan point (and inflated neighbours) as occupied.
        Also stamp a large inflation bubble around each Kalman track so
        A* plans a path that avoids detected people."""
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)

        # Raw scan obstacles (wall/furniture inflation)
        for wx, wy in zip(self.scan_x, self.scan_y):
            row, col = _world_to_cell(wx, wy)
            r0 = max(0, row - INFLATE_CELLS)
            r1 = min(GRID_SIZE, row + INFLATE_CELLS + 1)
            c0 = max(0, col - INFLATE_CELLS)
            c1 = min(GRID_SIZE, col + INFLATE_CELLS + 1)
            grid[r0:r1, c0:c1] = True

        # Tracked-people inflation — larger bubble so A* routes around them
        with self._track_lock:
            current_tracks = list(self.tracks)
        for t in current_tracks:
            row, col = _world_to_cell(t["x"], t["y"])
            r0 = max(0, row - PERSON_INFLATE_CELLS)
            r1 = min(GRID_SIZE, row + PERSON_INFLATE_CELLS + 1)
            c0 = max(0, col - PERSON_INFLATE_CELLS)
            c1 = min(GRID_SIZE, col + PERSON_INFLATE_CELLS + 1)
            # Circular mask so corners aren't blocked unnecessarily
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if math.hypot(r - row, c - col) <= PERSON_INFLATE_CELLS:
                        grid[r, c] = True

        with self._grid_lock:
            self._grid = grid

    # ── Leg callback ──────────────────────────────────────────────────────────

    def leg_callback(self, msg: LegMeasurementArray) -> None:
        if msg.legs:
            self.leg_x = np.array([l.x for l in msg.legs])
            self.leg_y = np.array([l.y for l in msg.legs])
        else:
            self.leg_x = np.array([])
            self.leg_y = np.array([])

    # ── Track callback ────────────────────────────────────────────────────────

    def track_callback(self, msg: TrackArray) -> None:
        """Cache the latest Kalman track list for visualization."""
        tracks = []
        for t in msg.tracks:
            tracks.append({
                "x": t.x, "y": t.y,
                "vx": t.vx, "vy": t.vy,
                "id": t.id,
                # TrackMsg doesn't carry P; we'll draw a fixed-size ellipse
                # sized to a typical 2-sigma position uncertainty (~0.25 m).
                "sigma": 0.25,
            })
        with self._track_lock:
            self.tracks = tracks

    # ── Planner thread ────────────────────────────────────────────────────────

    def _planner_loop(self) -> None:
        """Runs A* every PATH_REPLAN_HZ seconds and publishes via ZMQ."""
        while True:
            time.sleep(1.0 / PATH_REPLAN_HZ)

            gx, gy = self.goal_x, self.goal_y
            if gx is None:
                continue

            with self._grid_lock:
                grid = self._grid.copy()

            start_cell = _world_to_cell(self.robot_x, self.robot_y)
            goal_cell  = _world_to_cell(gx, gy)

            cell_path = _astar(grid, start_cell, goal_cell)

            if cell_path is None:
                waypoints: list[tuple[float, float]] = []
            else:
                raw = [_cell_to_world(r, c) for r, c in cell_path]
                waypoints = _simplify_path(raw, tolerance=0.20)

            with self._path_lock:
                self.path = waypoints

            # Publish to controller
            msg = json.dumps({"waypoints": [[x, y] for x, y in waypoints]})
            if self._zmq_socket:
                try:
                    self._zmq_socket.send_string(msg, zmq.NOBLOCK)
                except zmq.Again:
                    pass

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _yaw_from_quaternion(x, y, z, w) -> float:
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------
# Qt main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self, ros_node: LidarVisualizer) -> None:
        super().__init__()
        self.ros_node = ros_node

        self.setWindowTitle("Robot Navigator — click to set goal")
        self.resize(900, 860)
        self.setStyleSheet("background:#0a0a0f;")

        # ── Layout ───────────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(4)

        # Status bar
        self._status = QLabel("Left-click on the map to set a navigation goal.")
        self._status.setStyleSheet(
            "color:#a0c8ff; font-family:monospace; font-size:12px; padding:4px 8px;"
        )
        vbox.addWidget(self._status)

        # Plot
        self.plot_widget = pg.PlotWidget(background="#0a0a0f")
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setXRange(-8, 8)
        self.plot_widget.setYRange(-8, 8)
        self.plot_widget.getAxis("bottom").setStyle(tickFont=pg.Qt.QtGui.QFont("monospace", 8))
        self.plot_widget.getAxis("left").setStyle(tickFont=pg.Qt.QtGui.QFont("monospace", 8))
        vbox.addWidget(self.plot_widget)

        # ── Plot items ────────────────────────────────────────────────────────
        # Scan points
        self._scan_scatter = pg.ScatterPlotItem(
            size=2, brush=pg.mkBrush(60, 200, 80), pen=pg.mkPen(None)
        )
        # Leg detections
        self._leg_scatter = pg.ScatterPlotItem(
            size=14, symbol="o",
            brush=pg.mkBrush(255, 60, 60, 180),
            pen=pg.mkPen(255, 120, 120, 200, width=1.5),
        )
        # A* path
        self._path_line = pg.PlotCurveItem(
            pen=pg.mkPen(0, 220, 255, 220, width=2, style=Qt.DashLine)
        )
        # Waypoint dots on path
        self._waypoint_scatter = pg.ScatterPlotItem(
            size=7, symbol="o",
            brush=pg.mkBrush(0, 220, 255, 160),
            pen=pg.mkPen(None),
        )
        # Goal marker
        self._goal_scatter = pg.ScatterPlotItem(
            size=18, symbol="x",
            brush=pg.mkBrush(255, 220, 0, 200),
            pen=pg.mkPen(255, 220, 0, 255, width=2),
        )
        # Robot arrow
        self._robot_arrow = pg.ArrowItem(
            angle=0, tipAngle=28, baseAngle=18, headLen=22, tailLen=22,
            brush=pg.mkBrush(40, 140, 255),
            pen=pg.mkPen(180, 220, 255, width=1),
        )

        # Kalman track centre dots
        self._track_scatter = pg.ScatterPlotItem(
            size=12, symbol="o",
            brush=pg.mkBrush(255, 153, 0, 200),
            pen=pg.mkPen(255, 200, 80, 255, width=1.5),
        )

        # Velocity arrow lines (one PlotCurveItem per track, managed dynamically)
        self._vel_items: list[pg.PlotCurveItem] = []

        # Covariance ellipses (one PlotCurveItem per track, managed dynamically)
        self._ellipse_items: list[pg.PlotCurveItem] = []

        # Track ID text labels (one TextItem per track, managed dynamically)
        self._id_labels: list[pg.TextItem] = []

        for item in (
            self._scan_scatter, self._leg_scatter,
            self._track_scatter,
            self._path_line, self._waypoint_scatter,
            self._goal_scatter, self._robot_arrow,
        ):
            self.plot_widget.addItem(item)

        # ── Legend ────────────────────────────────────────────────────────────
        legend_html = (
            '<span style="color:#3cc850">■</span> Scan &nbsp;'
            '<span style="color:#ff3c3c">●</span> Legs &nbsp;'
            '<span style="color:#ff9900">●</span> Track pos &nbsp;'
            '<span style="color:#ff9900">→</span> Velocity &nbsp;'
            '<span style="color:#ff6600">○</span> 2σ ellipse &nbsp;'
            '<span style="color:#00dcff">– –</span> Path &nbsp;'
            '<span style="color:#ffdc00">✕</span> Goal'
        )
        legend = QLabel(legend_html)
        legend.setStyleSheet("color:#888; font-family:monospace; font-size:11px; padding:2px 8px;")
        vbox.addWidget(legend)

        # ── Mouse click → goal ────────────────────────────────────────────────
        self.plot_widget.scene().sigMouseClicked.connect(self._on_click)

        # ── Refresh timer ─────────────────────────────────────────────────────
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_ui)
        self._timer.start(33)  # ~30 fps

    # ── Click handler ─────────────────────────────────────────────────────────

    def _on_click(self, event) -> None:
        pos = event.scenePos()
        if self.plot_widget.sceneBoundingRect().contains(pos):
            pt = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.ros_node.goal_x = pt.x()
            self.ros_node.goal_y = pt.y()
            self._status.setText(
                f"Goal set → ({pt.x():.2f}, {pt.y():.2f}) m  |  Planning…"
            )

    # ── UI refresh ────────────────────────────────────────────────────────────

    def _update_ui(self) -> None:
        node = self.ros_node

        # Scan
        if node.scan_x.size > 0:
            self._scan_scatter.setData(node.scan_x, node.scan_y)

        # Legs
        if node.leg_x.size > 0:
            self._leg_scatter.setData(node.leg_x, node.leg_y)
        else:
            self._leg_scatter.setData([], [])

        # Goal marker
        if node.goal_x is not None:
            self._goal_scatter.setData([node.goal_x], [node.goal_y])
        else:
            self._goal_scatter.setData([], [])

        # Path
        with node._path_lock:
            path = list(node.path)

        if len(path) >= 2:
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            self._path_line.setData(xs, ys)
            self._waypoint_scatter.setData(xs, ys)

            dist = math.hypot(xs[-1] - node.robot_x, ys[-1] - node.robot_y)
            self._status.setText(
                f"Goal ({node.goal_x:.2f}, {node.goal_y:.2f}) m  │  "
                f"{len(path)} waypoints  │  dist {dist:.1f} m"
            )
        elif node.goal_x is not None:
            self._path_line.setData([], [])
            self._waypoint_scatter.setData([], [])
            self._status.setText("No path found — goal may be inside an obstacle.")
        else:
            self._path_line.setData([], [])
            self._waypoint_scatter.setData([], [])

        # Robot arrow
        # pg.ArrowItem: angle is CCW from the +X axis in *screen* space where
        # Y points down, so we negate the ROS yaw (CCW from +X, Y up).
        deg = math.degrees(node.robot_yaw)
        self._robot_arrow.setPos(node.robot_x, node.robot_y)
        self._robot_arrow.setStyle(angle=-deg)   # fix: was +180, now correct

        # ── Kalman tracks ──────────────────────────────────────────────────────
        with node._track_lock:
            tracks = list(node.tracks)

        # Grow or shrink dynamic item pools
        while len(self._vel_items) < len(tracks):
            item = pg.PlotCurveItem(pen=pg.mkPen(255, 180, 0, 200, width=2))
            self.plot_widget.addItem(item)
            self._vel_items.append(item)

        while len(self._ellipse_items) < len(tracks):
            item = pg.PlotCurveItem(pen=pg.mkPen(255, 100, 0, 160, width=1.5))
            self.plot_widget.addItem(item)
            self._ellipse_items.append(item)

        while len(self._id_labels) < len(tracks):
            lbl = pg.TextItem("", color=(255, 220, 100), anchor=(0.5, 1.2))
            lbl.setFont(pg.Qt.QtGui.QFont("monospace", 9))
            self.plot_widget.addItem(lbl)
            self._id_labels.append(lbl)

        # Update track centres
        if tracks:
            self._track_scatter.setData(
                [t["x"] for t in tracks],
                [t["y"] for t in tracks],
            )
        else:
            self._track_scatter.setData([], [])

        # Update per-track velocity arrows, covariance ellipses, ID labels
        # Hide surplus items from previous frame
        for i in range(len(tracks), len(self._vel_items)):
            self._vel_items[i].setData([], [])
        for i in range(len(tracks), len(self._ellipse_items)):
            self._ellipse_items[i].setData([], [])
        for i in range(len(tracks), len(self._id_labels)):
            self._id_labels[i].setText("")

        VEL_SCALE = 1.0  # seconds of look-ahead for velocity arrow
        ELLIPSE_N = 64   # points on the 2-sigma ellipse
        t_vals = np.linspace(0, 2 * math.pi, ELLIPSE_N)
        cos_t, sin_t = np.cos(t_vals), np.sin(t_vals)

        for i, t in enumerate(tracks):
            tx, ty = t["x"], t["y"]
            sigma = t["sigma"]   # positional 1-sigma (m)

            # Velocity arrow: line from track centre to predicted position
            ex = tx + t["vx"] * VEL_SCALE
            ey = ty + t["vy"] * VEL_SCALE
            self._vel_items[i].setData([tx, ex], [ty, ey])

            # 2-sigma covariance circle (isotropic when we only have sigma)
            r = 2.0 * sigma
            self._ellipse_items[i].setData(
                tx + r * cos_t,
                ty + r * sin_t,
            )

            # ID label above the track
            self._id_labels[i].setPos(tx, ty + r + 0.05)
            self._id_labels[i].setText(f"T{t['id']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarVisualizer()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    window = MainWindow(node)
    window.show()

    exit_code = app.exec_()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
