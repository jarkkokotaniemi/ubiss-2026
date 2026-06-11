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
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscriptions
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.leg_sub = self.create_subscription(LegMeasurementArray, '/legs', self.leg_callback, 10)
        
        # Data Storage
        self.x, self.y = np.array([]), np.array([]) # Lidar
        self.lx, self.ly = np.array([]), np.array([]) # Legs
        
        # Robot Pose Storage (in odom frame)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

    def scan_callback(self, scan: LaserScan):
        try:
            # 1. Get Robot Transform (odom -> base_link) for the robot icon
            # Using base_link usually represents the center of the robot
            try:
                robot_trans = self.tf_buffer.lookup_transform(
                    'odom', 'base_link', scan.header.stamp, timeout=Duration(seconds=0.05))
                self.robot_x = robot_trans.transform.translation.x
                self.robot_y = robot_trans.transform.translation.y
                self.robot_yaw = self._yaw_from_quaternion(
                    robot_trans.transform.rotation.x,
                    robot_trans.transform.rotation.y,
                    robot_trans.transform.rotation.z,
                    robot_trans.transform.rotation.w
                )
            except TransformException:
                pass # Skip robot update if base_link isn't available

            # 2. Get Laser Transform (odom -> laser_frame) for the points
            trans = self.tf_buffer.lookup_transform(
                'odom', scan.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.1))
            
            # Processing Laser points
            ranges = np.array(scan.ranges, dtype=np.float64)
            angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
            mask = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
            r, a = ranges[mask], angles[mask]
            
            x_local = r * np.cos(a)
            y_local = r * np.sin(a)
            
            tx, ty = trans.transform.translation.x, trans.transform.translation.y
            yaw = self._yaw_from_quaternion(
                trans.transform.rotation.x, trans.transform.rotation.y,
                trans.transform.rotation.z, trans.transform.rotation.w)
            
            self.x = x_local * np.cos(yaw) - y_local * np.sin(yaw) + tx
            self.y = x_local * np.sin(yaw) + y_local * np.cos(yaw) + ty

        except TransformException:
            return

    def leg_callback(self, msg):
        if len(msg.legs) > 0:
            self.lx = np.array([leg.x for leg in msg.legs])
            self.ly = np.array([leg.y for leg in msg.legs])
        else:
            self.lx, self.ly = np.array([]), np.array([])

    def _yaw_from_quaternion(self, x, y, z, w):
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

class MainWindow(QMainWindow):
    def __init__(self, ros_node):
        super().__init__()
        self.ros_node = ros_node
        self.setWindowTitle("Robot & Lidar Visualizer")
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
        
        # 1. The Laser Points
        self.scan_scatter = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(0, 255, 0))
        # 2. The Leg Detections
        self.leg_scatter = pg.ScatterPlotItem(size=15, brush=pg.mkBrush(255, 0, 0))
        
        # 3. The Robot Arrow
        # angle=0 points to the right in pyqtgraph. 
        # We update its rotation and position in update_ui.
        self.robot_arrow = pg.ArrowItem(
            angle=0, 
            tipAngle=30, 
            baseAngle=20, 
            headLen=20, 
            tailLen=20, 
            brush=pg.mkBrush(0, 150, 255)
        )
        
        self.plot_widget.addItem(self.scan_scatter)
        self.plot_widget.addItem(self.leg_scatter)
        self.plot_widget.addItem(self.robot_arrow)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(33) 

    def update_ui(self):
        # Update Lidar
        if self.ros_node.x.size > 0:
            self.scan_scatter.setData(self.ros_node.x, self.ros_node.y)
        
        # Update Legs
        if self.ros_node.lx.size > 0:
            self.leg_scatter.setData(self.ros_node.lx, self.ros_node.ly)

        # Update Robot Position and Heading
        # Convert radians to degrees and offset by 180 because pg.ArrowItem 
        # calculates 'angle' differently than standard unit circles sometimes.
        deg_yaw = np.degrees(self.ros_node.robot_yaw)
        self.robot_arrow.setPos(self.ros_node.robot_x, self.ros_node.robot_y)
        self.robot_arrow.setStyle(angle=deg_yaw + 180) # +180 often aligns ArrowItem correctly

def main(args=None):
    rclpy.init(args=args)
    node = LidarVisualizer()
    
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