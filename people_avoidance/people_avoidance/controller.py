"""
controller.py — Stage 3 of the people-avoidance pipeline.

Input : List[Track], robot pose (x, y, theta)
Output: geometry_msgs/Twist  published on /cmd_vel

Navigation modes
----------------
1. IDLE      — no goal; stop.
2. FOLLOWING — pure-pursuit toward the next waypoint while reactively
               avoiding obstacles (tracked legs + raw scan points).
3. AVOIDING  — obstacle too close; stop and spin away; resume FOLLOWING
               once clear.

Waypoints are received from the visualizer over a ZMQ SUB socket
(tcp://localhost:5556).  The controller subscribes non-blockingly so
it never stalls the ROS spin loop.

Students still implement:
  - obstacle_radius()  : unchanged from the original spec.
  - compute_velocity() : now includes waypoint following + avoidance.
"""

from __future__ import annotations

import json
import math
import threading
from typing import List

import numpy as np
from geometry_msgs.msg import Twist

from .tracking import Track

try:
    import zmq
    _ZMQ_OK = True
except ImportError:
    _ZMQ_OK = False

# ---------------------------------------------------------------------------
# ZMQ waypoint receiver (runs in its own thread)
# ---------------------------------------------------------------------------

class WaypointReceiver:
    """
    Subscribes to the ZMQ PUB socket from the visualizer and maintains
    the latest waypoint list.

    Thread-safe: all public attributes guarded by self._lock.
    """

    def __init__(self, address: str = "tcp://localhost:5556") -> None:
        self._lock = threading.Lock()
        self._waypoints: list[tuple[float, float]] = []

        if not _ZMQ_OK:
            print("[controller] WARNING: pyzmq not installed — waypoint following disabled.")
            return

        self._ctx    = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.connect(address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # Non-blocking polling: recv only returns when data is available
        self._socket.setsockopt(zmq.RCVTIMEO, 100)  # ms

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        while True:
            try:
                raw = self._socket.recv_string()
                data = json.loads(raw)
                wps  = [tuple(p) for p in data.get("waypoints", [])]
                with self._lock:
                    self._waypoints = wps  # type: ignore[assignment]
            except zmq.Again:
                pass  # timeout — no new message, keep going
            except Exception as exc:
                print(f"[WaypointReceiver] error: {exc}")

    @property
    def waypoints(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._waypoints)

    @waypoints.setter
    def waypoints(self, value: list[tuple[float, float]]) -> None:
        with self._lock:
            self._waypoints = value


# Module-level singleton — created once when the module is first imported.
_receiver: WaypointReceiver | None = None


def get_receiver() -> WaypointReceiver:
    global _receiver
    if _receiver is None:
        _receiver = WaypointReceiver()
    return _receiver


# ---------------------------------------------------------------------------
# Stage 3a — uncertainty-aware obstacle radius
# ---------------------------------------------------------------------------


def obstacle_radius(track: Track, sigma_scale: float) -> float:
    """
    Derive a conservative obstacle radius from the track's positional uncertainty.

    Returns:  sigma_scale × √(λ_max),  where λ_max is the largest eigenvalue
              of the 2×2 positional covariance sub-block  P[:2, :2].
    """
    pos_cov    = track.P[:2, :2]
    eigenvals  = np.linalg.eigvalsh(pos_cov)   # sorted ascending
    lambda_max = eigenvals[-1]
    return float(sigma_scale * math.sqrt(max(lambda_max, 0.0)))


# ---------------------------------------------------------------------------
# Stage 3b — avoidance + waypoint-following velocity controller
# ---------------------------------------------------------------------------

# Pure-pursuit tuning
LOOKAHEAD_DISTANCE = 0.5   # metres — how far ahead on the path to aim for
GOAL_TOLERANCE     = 0.25  # metres — distance to declare a waypoint reached

# Obstacle avoidance tuning
STOP_DISTANCE      = 0.60  # metres — stop + spin if any obstacle is closer than this


def _pure_pursuit_angular(
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    target_x: float,
    target_y: float,
) -> float:
    """
    Compute the angular velocity needed to steer toward (target_x, target_y)
    using the pure-pursuit heading error as a proportional controller.

    Returns ω in rad/s (positive = left / counter-clockwise).
    """
    angle_to_target = math.atan2(target_y - robot_y, target_x - robot_x)
    heading_error   = angle_to_target - robot_theta
    # Wrap to [-π, π]
    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
    # Proportional gain — tune to taste
    K_ang = 1.8
    return K_ang * heading_error


def _select_lookahead(
    waypoints: list[tuple[float, float]],
    robot_x: float,
    robot_y: float,
) -> tuple[float, float] | None:
    """
    Walk the waypoint list and find the first point that is at least
    LOOKAHEAD_DISTANCE away from the robot.  If all waypoints are closer
    than the lookahead (i.e. we are almost at the goal), return the last one.

    Returns None if waypoints is empty.
    """
    if not waypoints:
        return None

    # Drop waypoints we have already passed
    remaining = [
        wp for wp in waypoints
        if math.hypot(wp[0] - robot_x, wp[1] - robot_y) > GOAL_TOLERANCE
    ]
    if not remaining:
        return None  # reached the goal

    for wp in remaining:
        if math.hypot(wp[0] - robot_x, wp[1] - robot_y) >= LOOKAHEAD_DISTANCE:
            return wp

    return remaining[-1]  # all closer than lookahead → aim for the last one


def compute_velocity(
    tracks: List[Track],
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    max_linear_speed: float = 0.2,
    max_angular_speed: float = 1.0,
    obstacle_radius_scale: float = 2.0,
) -> Twist:
    """
    Compute a Twist that follows the A* path from the visualizer while
    reactively avoiding detected people.

    Logic
    -----
    1. Read the latest waypoints from WaypointReceiver (ZMQ).
    2. For every tracked person compute obstacle_radius(t) and dist to robot.
       If any person is within their radius → AVOID (stop + turn away).
    3. Otherwise → FOLLOW: pure-pursuit toward the lookahead point.
    4. Clip and return the Twist.
    """
    receiver  = get_receiver()
    waypoints = receiver.waypoints

    cmd = Twist()

    # ── 1. Check for nearby obstacles (people) ────────────────────────────────
    nearest_dist  = float("inf")
    nearest_angle = 0.0
    obstacle_near = False

    for t in tracks:
        px, py = float(t.m[0]), float(t.m[1])
        r      = obstacle_radius(t, obstacle_radius_scale)
        dist   = math.hypot(px - robot_x, py - robot_y)

        # Use the larger of the KF-derived radius and a hard minimum stop distance
        effective_radius = max(r, STOP_DISTANCE)

        if dist < effective_radius:
            obstacle_near = True
            if dist < nearest_dist:
                nearest_dist  = dist
                odom_angle    = math.atan2(py - robot_y, px - robot_x)
                nearest_angle = (odom_angle - robot_theta + math.pi) % (2 * math.pi) - math.pi

    # ── 2. AVOID mode ─────────────────────────────────────────────────────────
    if obstacle_near:
        v = 0.0
        # Turn away from the obstacle
        omega = -max_angular_speed if nearest_angle > 0 else max_angular_speed
        cmd.linear.x  = float(np.clip(v,     0.0,              max_linear_speed))
        cmd.angular.z = float(np.clip(omega, -max_angular_speed, max_angular_speed))
        return cmd

    # ── 3. FOLLOW mode ────────────────────────────────────────────────────────
    target = _select_lookahead(waypoints, robot_x, robot_y)

    if target is None:
        # No waypoints or goal reached — stop
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        return cmd

    # Compute heading error to the lookahead point
    omega = _pure_pursuit_angular(robot_x, robot_y, robot_theta, target[0], target[1])

    # Scale forward speed down when we need a big heading correction
    heading_error = abs(math.atan2(target[1] - robot_y, target[0] - robot_x) - robot_theta)
    heading_error = abs((heading_error + math.pi) % (2 * math.pi) - math.pi)

    # Slow down for sharp turns; stop and spin in place for very sharp angles
    if heading_error > math.radians(60):
        v = 0.0          # spin in place first
    elif heading_error > math.radians(25):
        v = max_linear_speed * 0.4
    else:
        v = max_linear_speed

    cmd.linear.x  = float(np.clip(v,     0.0,              max_linear_speed))
    cmd.angular.z = float(np.clip(omega, -max_angular_speed, max_angular_speed))
    return cmd
