"""
controller.py — Stage 3 of the people-avoidance pipeline.

Input : List[Track], robot pose (x, y, theta)
Output: geometry_msgs/Twist  published on /cmd_vel

Architecture
------------
1. Goal selection (from the pathfinding teammate): the visualizer's A*
   planner publishes path waypoints over a ZMQ PUB socket
   (tcp://localhost:5556). WaypointReceiver subscribes non-blockingly,
   and _select_lookahead() picks a local goal `p_goal` on that path.

2. Nominal controller (ported from exp_cbf_solution.ipynb): pure-pursuit
   toward p_goal — u_nominal_dubins().

3. Safety filter (ported from exp_cbf_solution.ipynb): a control-barrier-
   function QP — cbf_qp — projects the nominal (v, omega) onto the
   nearest command that keeps every tracked person outside their safety
   radius, with one CBF constraint per track. obstacle_radius() supplies
   each person's radius from the Kalman covariance.

If no path is available, or the final waypoint has been reached (within
GOAL_TOLERANCE), the robot stops (zero Twist) rather than running the
nominal controller into its near-goal singularity.
"""
from __future__ import annotations

import json
import math
import threading
from typing import List

import numpy as np
from cvxopt import matrix, solvers
from geometry_msgs.msg import Twist

from .tracking import Track

try:
    import zmq
    _ZMQ_OK = True
except ImportError:
    _ZMQ_OK = False

solvers.options["show_progress"] = False


# ---------------------------------------------------------------------------
# ZMQ waypoint receiver (runs in its own thread) — goal from the visualizer UI
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

    The radius grows when the Kalman filter is uncertain (large P) and
    shrinks as the estimate converges — giving an implicit safety margin
    that inflates when we are unsure where the person is.

    Returns:  sigma_scale × √(λ_max),  where λ_max is the largest eigenvalue
              of the 2×2 positional covariance sub-block  P[:2, :2].
    """
    pos_cov    = track.P[:2, :2]
    eigenvals  = np.linalg.eigvalsh(pos_cov)   # sorted ascending
    lambda_max = eigenvals[-1]
    return float(sigma_scale * math.sqrt(max(lambda_max, 0.0)))


# ---------------------------------------------------------------------------
# Goal selection: lookahead point on the A* path
# ---------------------------------------------------------------------------

PATH_LOOKAHEAD_DIST = 0.5   # metres — how far ahead on the path to aim for
GOAL_TOLERANCE      = 0.25  # metres — distance to declare a waypoint reached


def _select_lookahead(
    waypoints: list[tuple[float, float]],
    robot_x: float,
    robot_y: float,
) -> tuple[float, float] | None:
    """
    Walk the waypoint list and find the first point that is at least
    PATH_LOOKAHEAD_DIST away from the robot.  If all waypoints are closer
    than the lookahead (i.e. we are almost at the goal), return the last one.

    Returns None if waypoints is empty or the goal has been reached.
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
        if math.hypot(wp[0] - robot_x, wp[1] - robot_y) >= PATH_LOOKAHEAD_DIST:
            return wp

    return remaining[-1]  # all closer than lookahead → aim for the last one


# ---------------------------------------------------------------------------
# Nominal controller + CBF safety filter (ported from exp_cbf_solution.ipynb)
# ---------------------------------------------------------------------------

CBF_PROBE_DIST = 0.3   # L — lookahead-probe distance for the CBF (metres)
CBF_GAMMA      = 2.0   # gamma — class-K gain (small = conservative)
CBF_OMEGA_W    = 0.1   # w_omega — steer-before-brake weight in the QP cost
NOMINAL_K_OM   = 2.5   # heading-error gain for u_nominal_dubins


def angle_wrap(a: float) -> float:
    """Wrap an angle (radians) into (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def lookahead_pt(x: np.ndarray, L: float = CBF_PROBE_DIST) -> np.ndarray:
    """Lookahead point P = p + L * [cos(theta), sin(theta)]."""
    return x[:2] + L * np.array([math.cos(x[2]), math.sin(x[2])])


def h_la(x: np.ndarray, p_obs: np.ndarray, r: float, L: float = CBF_PROBE_DIST) -> float:
    """Lookahead barrier: h_L(x) = ||P - p_obs||^2 - (r + L)^2."""
    P = lookahead_pt(x, L)
    d = P - p_obs
    return float(d @ d - (r + L) ** 2)


def u_nominal_dubins(
    x: np.ndarray,
    p_goal: np.ndarray,
    k_om: float = NOMINAL_K_OM,
    v_des: float = 0.2,
) -> np.ndarray:
    """Pure-pursuit nominal controller: turn toward the goal, drive at v_des."""
    th_d = math.atan2(p_goal[1] - x[1], p_goal[0] - x[0])
    om = k_om * angle_wrap(th_d - x[2])
    d = float(np.linalg.norm(p_goal - x[:2]))
    v = min(v_des, max(d, 0.0))
    return np.array([v, om])


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
    Pure-pursuit toward the next path waypoint, filtered through a
    per-person CBF-QP safety layer.
    """
    cmd = Twist()

    waypoints = get_receiver().waypoints
    target = _select_lookahead(waypoints, robot_x, robot_y)
    if target is None:
        return cmd  # no path, or goal reached -> stop

    x = np.array([robot_x, robot_y, robot_theta])
    p_goal = np.array(target, dtype=float)
    u_nom = u_nominal_dubins(x, p_goal, k_om=NOMINAL_K_OM, v_des=max_linear_speed)

    # ----- one CBF row per tracked person -----
    L = CBF_PROBE_DIST
    A_cbf, b_cbf = [], []
    for t in tracks:
        p_obs = np.array([float(t.m[0]), float(t.m[1])])
        r = obstacle_radius(t, obstacle_radius_scale)

        e = x[:2] - p_obs
        s =  e[0] * math.cos(x[2]) + e[1] * math.sin(x[2])  # forward projection
        q = -e[0] * math.sin(x[2]) + e[1] * math.cos(x[2])  # lateral projection
        hL = h_la(x, p_obs, r, L=L)

        A_cbf.append([-2.0 * (s + L), -2.0 * L * q])
        b_cbf.append(CBF_GAMMA * hL)

    # ----- control-box rows: v in [0, max_linear_speed], |omega| <= max_angular_speed -----
    A_box = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    b_box = np.array([max_linear_speed, 0.0, max_angular_speed, max_angular_speed])

    if A_cbf:
        A = np.vstack([np.array(A_cbf), A_box])
        b = np.concatenate([np.array(b_cbf), b_box])
    else:
        A, b = A_box, b_box

    # ----- QP: min (v - v_nom)^2 + w_w (omega - omega_nom)^2  s.t.  A u <= b -----
    P    = matrix(np.diag([2.0, 2.0 * CBF_OMEGA_W]))
    q_qp = matrix(np.array([-2.0 * u_nom[0], -2.0 * CBF_OMEGA_W * u_nom[1]]))
    G    = matrix(A)
    h_qp = matrix(b)

    try:
        sol = solvers.qp(P, q_qp, G, h_qp)
        u_safe = np.array(sol["x"]).flatten()
    except Exception:
        u_safe = np.zeros(2)

    cmd.linear.x  = float(np.clip(u_safe[0], 0.0, max_linear_speed))
    cmd.angular.z = float(np.clip(u_safe[1], -max_angular_speed, max_angular_speed))
    return cmd
