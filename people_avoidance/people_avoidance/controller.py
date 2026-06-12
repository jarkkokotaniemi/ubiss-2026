"""
controller.py — Stage 3 of the people-avoidance pipeline.

Input : List[Track], robot pose (x, y, theta), destination (goal_x, goal_y)
Output: geometry_msgs/Twist  published on /cmd_vel

Drives toward (goal_x, goal_y) with a pure-pursuit nominal controller,
then projects that command onto the nearest control that keeps every
tracked person at least PERSON_CLEARANCE away, via a per-track
control-barrier-function (CBF) QP (cvxopt).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from cvxopt import matrix, solvers
from geometry_msgs.msg import Twist

from .tracking import Track

solvers.options["show_progress"] = False

# Minimum standoff distance the robot must keep from any tracked person (m).
# This is the floor of obstacle_radius(): even a perfectly-converged track
# (lambda_max -> 0) still inflates to this radius.
PERSON_CLEARANCE = 0.0


# ---------------------------------------------------------------------------
# Stage 3a — uncertainty-aware obstacle radius
# ---------------------------------------------------------------------------

def obstacle_radius(track: Track, sigma_scale: float) -> float:
    """
    Derive a conservative obstacle radius from the track's positional uncertainty.

    Returns  PERSON_CLEARANCE + sigma_scale * sqrt(lambda_max),  where
    lambda_max is the largest eigenvalue of the positional covariance
    sub-block P[:2, :2]. The constant term enforces the minimum standoff
    clearance even for a fully-converged track; the sigma term inflates
    the bubble further while the Kalman filter is still uncertain about
    where the person is.
    """
    pos_cov = track.P[:2, :2]
    eigenvals = np.linalg.eigvalsh(pos_cov)  # sorted ascending
    lambda_max = eigenvals[-1]
    return PERSON_CLEARANCE + sigma_scale * math.sqrt(max(lambda_max, 0.0))


# ---------------------------------------------------------------------------
# Stage 3b — avoidance policy
# ---------------------------------------------------------------------------

def compute_velocity(
    tracks: List[Track],
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    goal_x: Optional[float] = None,
    goal_y: Optional[float] = None,
    max_linear_speed: float = 0.2,
    max_angular_speed: float = 1.0,
    obstacle_radius_scale: float = 2.0,
    lookahead_distance: float = 0.3,
    cbf_gamma: float = 2.0,
    omega_weight: float = 0.1,
    heading_gain: float = 2.5,
    goal_tolerance: float = 0.15,
    v_prev: float = 0.0,
    omega_prev: float = 0.0,
    v_smoothness_weight: float = 0.1,
    omega_smoothness_weight: float = 0.3,
) -> Twist:
    """
    Compute a velocity command that drives toward (goal_x, goal_y) while
    avoiding all tracked people, via a pure-pursuit nominal controller
    filtered through a per-track control-barrier-function (CBF) QP.

    Inputs
    ------
    tracks               : Active person tracks from KalmanTracker.get_tracks().
    robot_x, robot_y     : Robot position in the odometry frame (m).
    robot_theta          : Robot heading in the odometry frame (rad).
    goal_x, goal_y       : Destination in the odometry frame (m), or None if
                           no goal has been received yet (robot stays idle).
    max_linear_speed     : Forward speed cap (m/s); also the pure-pursuit
                           cruise speed.
    max_angular_speed    : Rotation rate cap (rad/s).
    obstacle_radius_scale: Passed to obstacle_radius(); scales how much
                           positional uncertainty inflates each person's
                           safety bubble.
    lookahead_distance   : L — distance ahead of the robot at which the CBF
                           is evaluated (m). Fixes the relative-degree issue
                           so both v and ω appear in the CBF constraint.
    cbf_gamma            : γ — class-K gain; how aggressively the filter may
                           let the safety margin shrink while moving toward
                           the goal. Smaller -> earlier/wider avoidance.
    omega_weight         : Weight on ω in the QP cost (<1 => "steer before
                           brake": the filter prefers turning over stopping).
    heading_gain         : Proportional gain on heading error for the
                           pure-pursuit nominal controller.
    goal_tolerance       : Stop once within this distance of the goal (m).
    v_prev, omega_prev   : The (v, ω) command issued on the previous cycle.
                           Used to penalise large jumps so the path looks
                           smooth ("momentum").
    v_smoothness_weight  : Weight on (v - v_prev)^2 in the QP cost.
    omega_smoothness_weight: Weight on (ω - ω_prev)^2 in the QP cost.

    Output
    ------
    geometry_msgs/Twist published on /cmd_vel:
        twist.linear.x   — forward velocity (m/s); always >= 0.
        twist.angular.z  — rotation rate   (rad/s); positive = left turn.
    """
    cmd = Twist()

    if goal_x is None or goal_y is None:
        return cmd

    dx = goal_x - robot_x
    dy = goal_y - robot_y
    dist_to_goal = math.hypot(dx, dy)
    if dist_to_goal < goal_tolerance:
        return cmd

    # ---- nominal pure-pursuit control: turn toward the goal, drive at v_des ----
    heading_to_goal = math.atan2(dy, dx)
    heading_error = (heading_to_goal - robot_theta + math.pi) % (2 * math.pi) - math.pi
    v_nom = min(max_linear_speed, dist_to_goal)
    omega_nom = heading_gain * heading_error

    # ---- CBF-QP safety filter: one constraint row per tracked person ----
    L = lookahead_distance
    cos_th, sin_th = math.cos(robot_theta), math.sin(robot_theta)
    lookahead_x = robot_x + L * cos_th
    lookahead_y = robot_y + L * sin_th

    A_rows = []
    b_rows = []
    for t in tracks:
        person_x, person_y = t.m[0], t.m[1]
        r = obstacle_radius(t, obstacle_radius_scale)

        # Lookahead barrier h_L = ||P - p_person||^2 - (r + L)^2
        ex, ey = lookahead_x - person_x, lookahead_y - person_y
        h_la = ex * ex + ey * ey - (r + L) ** 2

        # Forward/lateral projection of (robot - person) onto the heading frame
        rel_x, rel_y = robot_x - person_x, robot_y - person_y
        s = rel_x * cos_th + rel_y * sin_th
        q = -rel_x * sin_th + rel_y * cos_th

        # CBF condition  2(s+L)v + 2Lq*ω >= -γ h_L  rewritten as A·u <= b
        A_rows.append([-2.0 * (s + L), -2.0 * L * q])
        b_rows.append(cbf_gamma * h_la)

    # Box constraints: 0 <= v <= max_linear_speed, |ω| <= max_angular_speed
    A_rows += [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    b_rows += [max_linear_speed, 0.0, max_angular_speed, max_angular_speed]

    # min (v - v_nom)^2 + omega_weight*(ω - ω_nom)^2
    #   + v_smoothness_weight*(v - v_prev)^2 + omega_smoothness_weight*(ω - ω_prev)^2
    # s.t. A·[v,ω]^T <= b
    P = matrix(np.diag([
        2.0 * (1.0 + v_smoothness_weight),
        2.0 * (omega_weight + omega_smoothness_weight),
    ]))
    q_cost = matrix(np.array([
        -2.0 * (v_nom + v_smoothness_weight * v_prev),
        -2.0 * (omega_weight * omega_nom + omega_smoothness_weight * omega_prev),
    ]))
    G = matrix(np.array(A_rows, dtype=float))
    h_qp = matrix(np.array(b_rows, dtype=float))

    try:
        sol = solvers.qp(P, q_cost, G, h_qp)
        if sol["status"] == "optimal":
            v, omega = np.array(sol["x"]).flatten()
        else:
            v, omega = 0.0, 0.0
    except Exception:
        v, omega = 0.0, 0.0

    cmd.linear.x = float(np.clip(v, 0.0, max_linear_speed))
    cmd.angular.z = float(np.clip(omega, -max_angular_speed, max_angular_speed))
    return cmd
