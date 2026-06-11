import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from people_avoidance_msgs.msg import LegMeasurementMsg

# TF2 Imports
from tf2_ros import Buffer, TransformListener, TransformException

# UI Imports
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import pyqtgraph as pg
from PyQt5.QtCore import QTimer

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')
        
        # 1. Setup TF2 to listen for the robot's position
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
            LegMeasurementMsg,
            '/legs',
            self.leg_callback,
            10)
        
        # Storage for LIDAR data (in odom frame)
        self.x = np.array([], dtype=np.float64)
        self.y = np.array([], dtype=np.float64)
        
        # Storage for Leg data (already in odom frame)
        self.lx = np.array([], dtype=np.float64)
        self.ly = np.array([], dtype=np.float64)

    def scan_callback(self, scan: LaserScan):
        try:
            # 2. Look up the transform from laser to odom at the time of the scan
            # We use 'odom' as the target frame
            trans = self.tf_buffer.lookup_transform(
                'odom', 
                scan.header.frame_id, 
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            )
        except TransformException as ex:
            self.get_logger().warn(f"Could not transform scan: {ex}")
            return

        # 3. Convert Scan to local Cartesian coordinates (NumPy vectorized)
        ranges = np.array(scan.ranges, dtype=np.float64)
        angles = np.arange(len(ranges), dtype=np.float64) * scan.angle_increment + scan.angle_min
        
        mask = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
        r = ranges[mask]
        a = angles[mask]
        
        # Local x, y (in laser frame)
        x_local = r * np.cos(a)
        y_local = r * np.sin(a)
        
        # 4. Apply the Transform to Odom frame
        # Extract translation
        tx = trans.transform.translation.x
        ty = trans.transform.translation.y
        
        # Extract rotation (yaw) from quaternion
        q = trans.transform.rotation
        yaw = self._yaw_from_quaternion(q.x, q.y, q.z, q.w)
        
        # Rotation Matrix application: 
        # x' = x*cos(theta) - y*sin(theta) + tx
        # y' = x*sin(theta) + y*cos(theta) + ty
        self.x = x_local * np.cos(yaw) - y_local * np.sin(yaw) + tx
        self.y = x_local * np.sin(yaw) + y_local * np.cos(yaw) + ty

    def leg_callback(self, msg):
        # Store legs (they are already published in odom frame by the detector node)
        # Append to arrays to show multiple legs if necessary
        self.lx = np.array([msg.x])
        self.ly = np.array([msg.y])

    def _yaw_from_quaternion(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

class MainWindow(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("Odom-Frame Lidar Visualizer")
        self.resize(900, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.plot_widget = pg.PlotWidget(background='k')
        self.layout.addWidget(self.plot_widget)
        
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Fixed range because odom frame moves. 
        # You might want to increase this if the robot drives far.
        self.plot_widget.setXRange(-10, 10)
        self.plot_widget.setYRange(-10, 10)
        
        self.scan_scatter = pg.ScatterPlotItem(
            size=3, pen=None, brush=pg.mkBrush(0, 255, 0, 255)
        )
        self.leg_scatter = pg.ScatterPlotItem(
            size=15, pen=pg.mkPen('w'), brush=pg.mkBrush(255, 0, 0, 255), symbol='o'
        )
        
        self.plot_widget.addItem(self.scan_scatter)
        self.plot_widget.addItem(self.leg_scatter)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_cycle)
        self.timer.start(30) # ~33 FPS

    def update_cycle(self):
        # 1. Check if ROS is still running before doing anything
        if not rclpy.ok():
            return

        try:
            # Process ROS events
            rclpy.spin_once(self.ros_node, timeout_sec=0)
            
            # 2. Check if the node hasn't been destroyed yet
            if self.ros_node.x.size > 0:
                self.scan_scatter.setData(self.ros_node.x, self.ros_node.y)
                
            if self.ros_node.lx.size > 0:
                self.leg_scatter.setData(self.ros_node.lx, self.ros_node.ly)
        except Exception as e:
            # Catch errors that happen during the window-closing transition
            print(f"Closing error: {e}")

    def closeEvent(self, event):
        self.timer.stop()  # Stop the timer immediately
        super().closeEvent(event)

def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizer()
    app = QApplication(sys.argv)
    window = MainWindow(node)
    window.show()
    
    # Run the app
    exit_code = app.exec_()
    
    # Clean shutdown
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
        
    sys.exit(exit_code)

if __name__ == '__main__':
    main()