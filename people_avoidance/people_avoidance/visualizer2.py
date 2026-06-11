import sys
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from people_avoidance_msgs.msg import LegMeasurementMsg, LegMeasurementArray

# TF2 Imports
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.duration import Duration

# UI Imports
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from PyQt5.QtCore import QTimer

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')
        
        # Setup TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # LIDAR Subscription
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        
        # Leg Subscription
        self.leg_sub = self.create_subscription(
            LegMeasurementArray,
            '/legs',
            self.leg_callback,
            10)
        
        # Storage for LIDAR data
        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)
        
        # Storage for Leg data
        self.lx = np.array([], dtype=np.float64)
        self.ly = np.array([], dtype=np.float64)

    def scan_callback(self, scan: LaserScan):
        try:
            # FIX 1: Use the EXACT timestamp of the scan (scan.header.stamp)
            # FIX 2: Add a small timeout (0.1s). This tells the node to wait 
            # slightly for the transform to arrive before giving up.
            trans = self.tf_buffer.lookup_transform(
                'odom', 
                scan.header.frame_id, 
                scan.header.stamp,
                timeout=Duration(seconds=0.1) 
            )
        except TransformException as ex:
            # This is now rare, but happens during startup
            return

        # Convert Scan to Cartesian
        ranges = np.array(scan.ranges, dtype=np.float64)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        
        mask = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        r = ranges[mask]
        a = angles[mask]
        
        x_local = r * np.cos(a)
        y_local = r * np.sin(a)
        
        # Apply Transform
        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        q = trans.transform.rotation
        yaw = self._yaw_from_quaternion(q.x, q.y, q.z, q.w)
        
        self.x = x_local * np.cos(yaw) - y_local * np.sin(yaw) + tx
        self.y = x_local * np.sin(yaw) + y_local * np.cos(yaw) + ty

    def leg_callback(self, msg):
        if len(msg.legs) > 0:
            self.lx = np.array([leg.x for leg in msg.legs], dtype=np.float64)
            self.ly = np.array([leg.y for leg in msg.legs], dtype=np.float64)
        else:
            self.lx, self.ly = np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    def _yaw_from_quaternion(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

class MainWindow(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("Synchronized Lidar Visualizer")
        self.resize(800, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.plot_widget = pg.PlotWidget(background='k')
        self.layout.addWidget(self.plot_widget)
        
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setXRange(-8, 8)
        self.plot_widget.setYRange(-8, 8)
        
        self.scan_scatter = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(0, 255, 0))
        self.leg_scatter = pg.ScatterPlotItem(size=15, brush=pg.mkBrush(255, 0, 0))
        self.plot_widget.addItem(self.scan_scatter)
        self.plot_widget.addItem(self.leg_scatter)

        # UI Update Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(33) 

    def update_ui(self):
        if self.ros_node.x.size > 0:
            self.scan_scatter.setData(self.ros_node.x, self.ros_node.y)
        if self.ros_node.lx.size > 0:
            self.leg_scatter.setData(self.ros_node.lx, self.ros_node.ly)

def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizer()
    
    # Run ROS in a background thread so the UI stays responsive
    # and the TF buffer stays updated constantly.
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    app = QApplication(sys.argv)
    window = MainWindow(node)
    window.show()
    
    exit_code = app.exec_()
    rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()