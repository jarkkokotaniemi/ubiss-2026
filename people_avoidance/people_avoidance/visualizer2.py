"""
visualizer_direct.py — Lidar visualizer with direct click-to-goal publishing.

Features:
  - Left-click on the plot to publish a PoseStamped goal to ROS2.
  - Full visualization of Kalman tracks (velocity, covariance, IDs).
  - No path planning or ZMQ dependencies.
"""

from __future__ import annotations

import math
import sys
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from people_avoidance_msgs.msg import LegMeasurementArray, TrackArray

from tf2_ros import Buffer, TransformListener, TransformException

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg

# ---------------------------------------------------------------------------
# ROS node
# ---------------------------------------------------------------------------

class LidarVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("lidar_goal_publisher")

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.scan_sub  = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.leg_sub   = self.create_subscription(LegMeasurementArray, "/legs", self.leg_callback, 10)
        self.track_sub = self.create_subscription(TrackArray, "/tracks", self.track_callback, 10)

        # Publisher for the goal (Direct to ROS2)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # Data storage
        self.scan_x: np.ndarray = np.array([])
        self.scan_y: np.ndarray = np.array([])
        self.leg_x: np.ndarray = np.array([])
        self.leg_y: np.ndarray = np.array([])
        
        self.tracks: list[dict] = []
        self._track_lock = threading.Lock()

        # Robot pose
        self.robot_x:   float = 0.0
        self.robot_y:   float = 0.0
        self.robot_yaw: float = 0.0

        # Goal coordinates for visualization
        self.goal_x: float | None = None
        self.goal_y: float | None = None

    def publish_goal(self, x: float, y: float):
        """Publishes the clicked position as a ROS2 PoseStamped message."""
        self.goal_x = x
        self.goal_y = y
        
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.w = 1.0  # Neutral orientation
        
        self.goal_pub.publish(msg)
        self.get_logger().info(f"Goal Published: x={x:.2f}, y={y:.2f}")

    def scan_callback(self, scan: LaserScan) -> None:
        try:
            # Update Robot pose via TF
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

            # Transform Laser points to odom frame
            trans = self.tf_buffer.lookup_transform(
                "odom", scan.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.1)
            )
            ranges = np.array(scan.ranges, dtype=np.float64)
            angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
            mask = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
            r, a = ranges[mask], angles[mask]

            x_local, y_local = r * np.cos(a), r * np.sin(a)
            tx, ty = trans.transform.translation.x, trans.transform.translation.y
            yaw = self._yaw_from_quaternion(
                trans.transform.rotation.x, trans.transform.rotation.y,
                trans.transform.rotation.z, trans.transform.rotation.w,
            )
            self.scan_x = x_local * np.cos(yaw) - y_local * np.sin(yaw) + tx
            self.scan_y = x_local * np.sin(yaw) + y_local * np.cos(yaw) + ty

        except TransformException:
            pass

    def leg_callback(self, msg: LegMeasurementArray) -> None:
        self.leg_x = np.array([l.x for l in msg.legs]) if msg.legs else np.array([])
        self.leg_y = np.array([l.y for l in msg.legs]) if msg.legs else np.array([])

    def track_callback(self, msg: TrackArray) -> None:
        tracks = []
        for t in msg.tracks:
            tracks.append({
                "x": t.x, "y": t.y, "vx": t.vx, "vy": t.vy, "id": t.id, "sigma": 0.25
            })
        with self._track_lock:
            self.tracks = tracks

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
        self.setWindowTitle("Robot Goal Publisher")
        self.resize(900, 860)
        self.setStyleSheet("background:#0a0a0f;")

        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        self._status = QLabel("Left-click on the map to publish a goal to /goal_pose.")
        self._status.setStyleSheet("color:#a0c8ff; font-family:monospace; font-size:12px; padding:4px;")
        vbox.addWidget(self._status)

        self.plot_widget = pg.PlotWidget(background="#0a0a0f")
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setXRange(-8, 8)
        self.plot_widget.setYRange(-8, 8)
        vbox.addWidget(self.plot_widget)

        # Permanent Plot Items
        self._scan_scatter = pg.ScatterPlotItem(size=2, brush=pg.mkBrush(60, 200, 80))
        self._leg_scatter = pg.ScatterPlotItem(size=14, symbol="o", brush=pg.mkBrush(255, 60, 60, 180))
        self._track_scatter = pg.ScatterPlotItem(size=12, symbol="o", brush=pg.mkBrush(255, 153, 0, 200))
        self._goal_scatter = pg.ScatterPlotItem(size=18, symbol="x", brush=pg.mkBrush(255, 220, 0, 255))
        self._robot_arrow = pg.ArrowItem(brush=pg.mkBrush(40, 140, 255))

        for item in (self._scan_scatter, self._leg_scatter, self._track_scatter, 
                     self._goal_scatter, self._robot_arrow):
            self.plot_widget.addItem(item)

        # Dynamic Track Item Pools
        self._vel_items: list[pg.PlotCurveItem] = []
        self._ellipse_items: list[pg.PlotCurveItem] = []
        self._id_labels: list[pg.TextItem] = []

        # Legend
        legend_html = (
            '<span style="color:#3cc850">■</span> Scan &nbsp;'
            '<span style="color:#ff3c3c">●</span> Legs &nbsp;'
            '<span style="color:#ff9900">●</span> Track pos &nbsp;'
            '<span style="color:#ff9900">→</span> Velocity &nbsp;'
            '<span style="color:#ff6600">○</span> 2σ ellipse &nbsp;'
            '<span style="color:#ffdc00">✕</span> Goal'
        )
        legend = QLabel(legend_html)
        legend.setStyleSheet("color:#888; font-family:monospace; font-size:11px; padding:4px;")
        vbox.addWidget(legend)

        self.plot_widget.scene().sigMouseClicked.connect(self._on_click)

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_ui)
        self._timer.start(33)

    def _on_click(self, event) -> None:
        pos = event.scenePos()
        if self.plot_widget.sceneBoundingRect().contains(pos):
            pt = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            self.ros_node.publish_goal(pt.x(), pt.y())
            self._status.setText(f"Goal Published → ({pt.x():.2f}, {pt.y():.2f})")

    def _update_ui(self) -> None:
        node = self.ros_node

        # Update Scans and Legs
        self._scan_scatter.setData(node.scan_x, node.scan_y)
        self._leg_scatter.setData(node.leg_x, node.leg_y)
        
        # Update Goal
        if node.goal_x is not None:
            self._goal_scatter.setData([node.goal_x], [node.goal_y])

        # Update Robot
        self._robot_arrow.setPos(node.robot_x, node.robot_y)
        self._robot_arrow.setStyle(angle=-math.degrees(node.robot_yaw))

        # Update Tracks
        with node._track_lock:
            tracks = list(node.tracks)

        # Sync item pools
        while len(self._vel_items) < len(tracks):
            v_item = pg.PlotCurveItem(pen=pg.mkPen(255, 180, 0, 200, width=2))
            e_item = pg.PlotCurveItem(pen=pg.mkPen(255, 100, 0, 160, width=1.5))
            l_item = pg.TextItem("", color=(255, 220, 100), anchor=(0.5, 1.2))
            l_item.setFont(pg.Qt.QtGui.QFont("monospace", 9))
            self.plot_widget.addItem(v_item)
            self.plot_widget.addItem(e_item)
            self.plot_widget.addItem(l_item)
            self._vel_items.append(v_item)
            self._ellipse_items.append(e_item)
            self._id_labels.append(l_item)

        # Reset unused items
        for i in range(len(tracks), len(self._vel_items)):
            self._vel_items[i].setData([], [])
            self._ellipse_items[i].setData([], [])
            self._id_labels[i].setText("")

        # Update track centers
        self._track_scatter.setData([t["x"] for t in tracks], [t["y"] for t in tracks])

        # Draw Velocity and Ellipses
        t_vals = np.linspace(0, 2 * math.pi, 32)
        cos_t, sin_t = np.cos(t_vals), np.sin(t_vals)

        for i, t in enumerate(tracks):
            tx, ty = t["x"], t["y"]
            r = 2.0 * t["sigma"] # 2-sigma radius

            # Velocity arrow
            self._vel_items[i].setData([tx, tx + t["vx"]], [ty, ty + t["vy"]])
            # Covariance circle
            self._ellipse_items[i].setData(tx + r * cos_t, ty + r * sin_t)
            # ID Label
            self._id_labels[i].setPos(tx, ty + r + 0.05)
            self._id_labels[i].setText(f"ID:{t['id']}")

def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarVisualizer()
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QApplication(sys.argv)
    window = MainWindow(node)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()