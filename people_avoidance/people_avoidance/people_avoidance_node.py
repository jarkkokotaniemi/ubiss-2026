"""
people_avoidance_node.py — ROS 2 node that wires the pipeline together.

Data flow (triggered by each incoming /scan message):

    /scan  ──► detect_legs()      ──► List[LegMeasurement] (laser frame)
                                           │
                                    transform to odom frame
                                           │
                                    tracker.update()
                                           │
                                     List[Track] (odom frame)
                                           │
                                  compute_velocity()  ◄──  robot pose (/odom)
                                           │
                                        /cmd_vel
"""

from __future__ import annotations

import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from people_avoidance_msgs.msg import LegMeasurementMsg, LegMeasurementArray, TrackMsg, TrackArray

# TF2 Imports
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs  # Allows transforming PointStamped directly

from .leg_detection import detect_legs, LegMeasurement
from .tracking import KalmanTracker
from .controller import compute_velocity

from rclpy.qos import qos_profile_sensor_data


class PeopleAvoidanceNode(Node):

    def __init__(self) -> None:
        super().__init__("people_avoidance_node")

        # ── Declare all tunable parameters ───────────────────────────────────
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("laser_frame", "rplidar_link")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("dt", 0.1)
        self.declare_parameter("max_misses", 5)

        # Leg detection parameters
        self.declare_parameter("distance_threshold", 0.1)
        self.declare_parameter("leg_radius", 0.10)
        self.declare_parameter("max_leg_width", 0.25)
        self.declare_parameter("curv_threshold", 0.15)          # NEW
        self.declare_parameter("min_apex_angle_deg", 90.0)      # NEW
        self.declare_parameter("apex_window", 20)               # NEW
        self.declare_parameter("max_flatness", 0.02)            # NEW

        # Controller parameters
        self.declare_parameter("max_linear_speed", 0.2)
        self.declare_parameter("max_angular_speed", 1.0)
        self.declare_parameter("obstacle_radius_scale", 2.0)
        self.declare_parameter("goal_pose_topic", "/goal_pose")
        self.declare_parameter("lookahead_distance", 0.3)
        self.declare_parameter("cbf_gamma", 2.0)
        self.declare_parameter("omega_weight", 0.1)
        self.declare_parameter("heading_gain", 2.5)
        self.declare_parameter("goal_tolerance", 0.15)

        p = self._params()

        # ── TF2 Setup ────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Kalman tracker (shared state across scans) ────────────────────────
        self.tracker = KalmanTracker(
            dt=p["dt"],
            max_misses=p["max_misses"],
        )

        # ── Latest robot pose — updated from /odom, consumed on each /scan ───
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_theta: float = 0.0

        # ── Latest destination — updated from /goal_pose, consumed on each /scan ─
        self._goal_x: float | None = None
        self._goal_y: float | None = None

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(LaserScan, p["scan_topic"], self._scan_cb, 10)
        self.create_subscription(Odometry, p["odom_topic"], self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, p["goal_pose_topic"], self._goal_cb, 10)

        # ── Publisher ─────────────────────────────────────────────────────────
        self._cmd_pub = self.create_publisher(Twist, p["cmd_vel_topic"], 10)
        self._leg_pub = self.create_publisher(LegMeasurementArray, "/legs", 10)
        self._track_pub = self.create_publisher(TrackArray, "/tracks", 10)

        self.get_logger().info(
            f"PeopleAvoidanceNode ready — "
            f"listening on '{p['scan_topic']}', publishing to '{p['cmd_vel_topic']}'"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        """Cache the latest robot pose from odometry."""
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        self._robot_theta = _yaw_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

    def _goal_cb(self, msg: PoseStamped) -> None:
        """Cache the latest destination point (odom frame)."""
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y

    def _transform_measurement_to_odom(self, m: LegMeasurement) -> LegMeasurement:
        """
        Transform a single LegMeasurement from the laser frame to the odom frame.
        The measurement covariance (R) is also rotated because the orientation
        of the laser frame relative to odom changes with robot_theta.
        """
        # Position transformation
        x_odom = self._robot_x + m.x * math.cos(self._robot_theta) - m.y * math.sin(self._robot_theta)
        y_odom = self._robot_y + m.x * math.sin(self._robot_theta) + m.y * math.cos(self._robot_theta)

        # Covariance rotation: R_odom = J * R_laser * J^T
        # Jacobian of the position transformation w.r.t. laser-frame coordinates
        J = np.array([
            [math.cos(self._robot_theta), -math.sin(self._robot_theta)],
            [math.sin(self._robot_theta),  math.cos(self._robot_theta)]
        ])
        R_laser = np.array([[m.Rxx, m.Rxy], [m.Rxy, m.Ryy]])
        R_odom = J @ R_laser @ J.T

        return LegMeasurement(
            x=x_odom,
            y=y_odom,
            Rxx=float(R_odom[0, 0]),
            Rxy=float(R_odom[0, 1]),
            Ryy=float(R_odom[1, 1]),
        )
    
    def _transform_with_tf(self, m: LegMeasurement, tf) -> LegMeasurement:
        """Uses a TF transform to move the measurement and rotate covariance."""
        # Create a PointStamped for the position
        p_laser = PointStamped()
        p_laser.point.x = m.x
        p_laser.point.y = m.y
        
        # Transform position
        p_odom = tf2_geometry_msgs.do_transform_point(p_laser, tf)

        # Rotate Covariance
        # Extract rotation matrix from TF quaternion
        q = tf.transform.rotation
        yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
        
        J = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw),  math.cos(yaw)]
        ])
        R_laser = np.array([[m.Rxx, m.Rxy], [m.Rxy, m.Ryy]])
        R_odom = J @ R_laser @ J.T

        return LegMeasurement(
            x=p_odom.point.x,
            y=p_odom.point.y,
            Rxx=float(R_odom[0, 0]),
            Rxy=float(R_odom[0, 1]),
            Ryy=float(R_odom[1, 1]),
        )

    def _scan_cb(self, scan: LaserScan) -> None:
        """Main pipeline callback — fires on every incoming LaserScan."""
        p = self._params()

        # ── Stage 1: leg detection (in laser frame) ───────────────────────────
        measurements_laser = detect_legs(
            scan,
            distance_threshold=p["distance_threshold"],
            leg_radius=p["leg_radius"],
            max_leg_width=p["max_leg_width"],
            curv_threshold=p["curv_threshold"],
            min_apex_angle_deg=p["min_apex_angle_deg"],
            apex_window=p["apex_window"],
            max_flatness=p["max_flatness"],
        )

        # ── Transform measurements from laser frame to odom frame ─────────────
        measurements_odom = []
        try:
            # Look up transform from laser to odom at the time of the scan
            transform = self.tf_buffer.lookup_transform(
                p["odom_frame"],
                scan.header.frame_id,
                scan.header.stamp,
                rclpy.duration.Duration(seconds=0.1) # Wait up to 100ms for TF
            )
            msg = LegMeasurementArray()
            for m_laser in measurements_laser:
                m_odom = self._transform_with_tf(m_laser, transform)
                measurements_odom.append(m_odom)
                
                # Publish for visualization/debug
                leg = LegMeasurementMsg()
                leg.x, leg.y = m_odom.x, m_odom.y
                leg.rxx, leg.rxy, leg.ryy = m_odom.Rxx, m_odom.Rxy, m_odom.Ryy
                msg.legs.append(leg)
            self._leg_pub.publish(msg)

        except TransformException as ex:
            self.get_logger().warning(f"Could not transform laser to odom: {ex}")
            return # Skip this frame if transform fails

        # ── Stage 2: Kalman tracking (all tracks are in odom frame) ───────────
        self.tracker.update(measurements_odom)
        tracks = self.tracker.get_tracks()
        
        msg = TrackArray()
        for track in tracks:
            trk = TrackMsg()
            trk.x = track.m[0]
            trk.y = track.m[1]
            trk.vx = track.m[2]
            trk.vy = track.m[3]

            pos_cov = track.P[:2, :2]
            eigenvals = np.linalg.eigvalsh(pos_cov)  # sorted ascending
            lambda_max = eigenvals[-1]
            trk.radius = 0.5 + 2.0 * math.sqrt(max(lambda_max, 0.0)) # first 2 numbers are PERSON_CLEARANCE and obstacle_radius_scale in controller.py

            trk.id = track.track_id
            msg.tracks.append(trk)
        self._track_pub.publish(msg)

        # ── Stage 3: avoidance control (controller expects odom‑frame tracks) ─
        cmd = compute_velocity(
            tracks,
            robot_x=self._robot_x,
            robot_y=self._robot_y,
            robot_theta=self._robot_theta,
            goal_x=self._goal_x,
            goal_y=self._goal_y,
            max_linear_speed=p["max_linear_speed"],
            max_angular_speed=p["max_angular_speed"],
            obstacle_radius_scale=p["obstacle_radius_scale"],
            lookahead_distance=p["lookahead_distance"],
            cbf_gamma=p["cbf_gamma"],
            omega_weight=p["omega_weight"],
            heading_gain=p["heading_gain"],
            goal_tolerance=p["goal_tolerance"],
        )

        self._cmd_pub.publish(cmd)

        self.get_logger().debug(
            f"{len(measurements_laser)} detections  {len(tracks)} tracks  "
            f"→  v={cmd.linear.x:.2f} m/s  ω={cmd.angular.z:.2f} rad/s"
        )

    # ── Helper ────────────────────────────────────────────────────────────────

    def _params(self) -> dict:
        return {
            "scan_topic": self.get_parameter("scan_topic").value,
            "cmd_vel_topic": self.get_parameter("cmd_vel_topic").value,
            "odom_topic": self.get_parameter("odom_topic").value,
            "laser_frame": self.get_parameter("laser_frame").value,
            "odom_frame": self.get_parameter("odom_frame").value,
            "dt": self.get_parameter("dt").value,
            "max_misses": self.get_parameter("max_misses").value,
            "distance_threshold": self.get_parameter("distance_threshold").value,
            "leg_radius": self.get_parameter("leg_radius").value,
            "max_leg_width": self.get_parameter("max_leg_width").value,
            "curv_threshold": self.get_parameter("curv_threshold").value,
            "min_apex_angle_deg": self.get_parameter("min_apex_angle_deg").value,
            "apex_window": self.get_parameter("apex_window").value,
            "max_flatness": self.get_parameter("max_flatness").value,
            "max_linear_speed": self.get_parameter("max_linear_speed").value,
            "max_angular_speed": self.get_parameter("max_angular_speed").value,
            "obstacle_radius_scale": self.get_parameter("obstacle_radius_scale").value,
            "goal_pose_topic": self.get_parameter("goal_pose_topic").value,
            "lookahead_distance": self.get_parameter("lookahead_distance").value,
            "cbf_gamma": self.get_parameter("cbf_gamma").value,
            "omega_weight": self.get_parameter("omega_weight").value,
            "heading_gain": self.get_parameter("heading_gain").value,
            "goal_tolerance": self.get_parameter("goal_tolerance").value,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation about Z) from a unit quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PeopleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()