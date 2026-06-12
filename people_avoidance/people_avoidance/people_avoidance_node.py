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
                                  WaypointManager.compute_velocity()
                                    │  A* replan on new goal
                                    │  CBF people-avoidance
                                    │  CBF boundary-exit
                                    │  velocity lerp
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
from people_avoidance_msgs.msg import (
    LegMeasurementMsg,
    LegMeasurementArray,
    TrackMsg,
    TrackArray,
)

# TF2 Imports
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs

from .leg_detection import detect_legs, LegMeasurement
from .tracking import KalmanTracker
from .controller import compute_velocity, WaypointManager

from rclpy.qos import qos_profile_sensor_data


class PeopleAvoidanceNode(Node):

    def __init__(self) -> None:
        super().__init__("people_avoidance_node")
        self._odom_received = False
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
        self.declare_parameter("curv_threshold", 0.15)
        self.declare_parameter("min_apex_angle_deg", 90.0)
        self.declare_parameter("apex_window", 20)
        self.declare_parameter("max_flatness", 0.02)

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
        self.declare_parameter("waypoint_tolerance", 0.25)

        p = self._params()

        # ── TF2 Setup ────────────────────────────────────────────────────────
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Kalman tracker ────────────────────────────────────────────────────
        self.tracker = KalmanTracker(
            dt=p["dt"],
            max_misses=p["max_misses"],
        )

        # ── Waypoint manager (owns A* state, smoothing state) ─────────────────
        self._waypoint_manager = WaypointManager()

        # ── Latest robot pose ─────────────────────────────────────────────────
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_theta: float = 0.0

        # ── Pending goal (set by goal callback, consumed in scan callback) ─────
        # We store it separately so we can trigger a replan exactly once.
        self._pending_goal: tuple[float, float] | None = None

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(LaserScan, p["scan_topic"], self._scan_cb, 10)
        self.create_subscription(
            Odometry, p["odom_topic"], self._odom_cb, qos_profile_sensor_data
        )
        self.create_subscription(PoseStamped, p["goal_pose_topic"], self._goal_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
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
        self._odom_received = True
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        self._robot_theta = _yaw_from_quaternion(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

    def _goal_cb(self, msg: PoseStamped) -> None:
        """
        Cache the new goal. The actual A* replan happens in _scan_cb
        once we have fresh track data, so the plan accounts for current
        obstacle positions.
        """
        self._pending_goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(
            f"New goal received: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}) "
            f"— will replan on next scan."
        )

    def _transform_with_tf(self, m: LegMeasurement, tf) -> LegMeasurement:
        """Uses a TF transform to move the measurement and rotate covariance."""
        p_laser = PointStamped()
        p_laser.point.x = m.x
        p_laser.point.y = m.y

        p_odom = tf2_geometry_msgs.do_transform_point(p_laser, tf)

        q = tf.transform.rotation
        yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)

        J = np.array(
            [
                [math.cos(yaw), -math.sin(yaw)],
                [math.sin(yaw), math.cos(yaw)],
            ]
        )
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
        if not self._odom_received:
            self.get_logger().warning(
                "Waiting for odometry...", throttle_duration_sec=2.0
            )
            return
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
        # Use rclpy.time.Time() (latest available TF) instead of scan.header.stamp
        # to avoid "extrapolation into the future" errors when the TF buffer
        # hasn't caught up to the scan timestamp yet.
        measurements_odom = []
        tf_ok = False
        try:
            transform = self.tf_buffer.lookup_transform(
                p["odom_frame"],
                scan.header.frame_id,
                rclpy.time.Time(),  # ← always use latest available TF
            )
            msg = LegMeasurementArray()
            for m_laser in measurements_laser:
                m_odom = self._transform_with_tf(m_laser, transform)
                measurements_odom.append(m_odom)

                leg = LegMeasurementMsg()
                leg.x, leg.y = m_odom.x, m_odom.y
                leg.rxx, leg.rxy, leg.ryy = m_odom.Rxx, m_odom.Rxy, m_odom.Ryy
                msg.legs.append(leg)
            self._leg_pub.publish(msg)
            tf_ok = True

        except TransformException as ex:
            self.get_logger().warning(
                f"Could not transform laser to odom: {ex} — skipping detections, "
                f"still running controller with last known tracks."
            )

        # ── Stage 2: Kalman tracking ──────────────────────────────────────────
        # Only feed new measurements into the tracker when TF succeeded.
        # If TF failed we still call update() with an empty list so the tracker
        # ages its miss-counts; stale tracks remain usable for avoidance.
        self.tracker.update(measurements_odom if tf_ok else [])
        tracks = self.tracker.get_tracks()

        # Publish tracks (with sigma-derived radius for visualizer)
        track_msg = TrackArray()
        for track in tracks:
            trk = TrackMsg()
            trk.x = track.m[0]
            trk.y = track.m[1]
            trk.vx = track.m[2]
            trk.vy = track.m[3]

            pos_cov = track.P[:2, :2]
            eigenvals = np.linalg.eigvalsh(pos_cov)
            lambda_max = eigenvals[-1]
            trk.radius = 0.0 + 1.0 * math.sqrt(max(lambda_max, 0.0))
            trk.id = track.track_id
            track_msg.tracks.append(trk)
        self._track_pub.publish(track_msg)

        # ── Stage 2.5: Replan A* if a new goal arrived ────────────────────────
        if self._pending_goal is not None:
            gx, gy = self._pending_goal
            self._pending_goal = None  # clear only after we've captured gx/gy
            self._waypoint_manager.set_goal(
                gx,
                gy,
                self._robot_x,
                self._robot_y,
                tracks,
                p["obstacle_radius_scale"],
            )
            wm = self._waypoint_manager
            self.get_logger().info(
                f"A* planned {len(wm.waypoints)} waypoints to ({gx:.2f}, {gy:.2f})"
            )

        # ── Stage 3: avoidance control ────────────────────────────────────────
        cmd = compute_velocity(
            tracks,
            robot_x=self._robot_x,
            robot_y=self._robot_y,
            robot_theta=self._robot_theta,
            waypoint_manager=self._waypoint_manager,
            max_linear_speed=p["max_linear_speed"],
            max_angular_speed=p["max_angular_speed"],
            obstacle_radius_scale=p["obstacle_radius_scale"],
            lookahead_distance=p["lookahead_distance"],
            cbf_gamma=p["cbf_gamma"],
            omega_weight=p["omega_weight"],
            heading_gain=p["heading_gain"],
            goal_tolerance=p["goal_tolerance"],
            waypoint_tolerance=p["waypoint_tolerance"],
        )

        self._cmd_pub.publish(cmd)

        wm = self._waypoint_manager
        self.get_logger().info(
            f"[{wm.state}] wp {wm.wp_index}/{len(wm.waypoints)}  "
            f"{len(measurements_laser)} det  {len(tracks)} tracks  "
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
            "waypoint_tolerance": self.get_parameter("waypoint_tolerance").value,
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _yaw_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> float:
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
