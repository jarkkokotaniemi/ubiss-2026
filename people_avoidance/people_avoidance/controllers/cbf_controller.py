"""
controllers/cbf_controller.py — reactive pure-pursuit + CBF-QP controller.

Drives toward (goal_x, goal_y) with a pure-pursuit nominal controller, then
projects that command onto the nearest control that keeps every tracked
person at least PERSON_CLEARANCE away, via a per-track control-barrier-
function (CBF) QP (cvxopt).
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from cvxopt import matrix, solvers
from geometry_msgs.msg import Twist

from ..tracking import Track
from .base import BaseController, obstacle_radius

solvers.options["show_progress"] = False


class CBFController(BaseController):
    """
    Pure-pursuit nominal controller filtered through a per-track CBF QP.

    Args
    ----
    max_linear_speed     : Forward speed cap (m/s); also the pure-pursuit
                           cruise speed.
    max_angular_speed    : Rotation rate cap (rad/s).
    obstacle_radius_scale: Passed to obstacle_radius(); scales how much
                           positional uncertainty inflates each person's
                           safety bubble.
    lookahead_distance   : L — distance ahead of the robot at which the CBF
                           is evaluated (m). Fixes the relative-degree issue
                           so both v and omega appear in the CBF constraint.
    cbf_gamma            : gamma — class-K gain; how aggressively the filter
                           may let the safety margin shrink while moving
                           toward the goal. Smaller -> earlier/wider avoidance.
    omega_weight         : Weight on omega in the QP cost (<1 => "steer before
                           brake": the filter prefers turning over stopping).
    heading_gain         : Proportional gain on heading error for the
                           pure-pursuit nominal controller.
    goal_tolerance       : Stop once within this distance of the goal (m).
    v_smoothness_weight  : Weight on (v - v_prev)^2 in the QP cost.
    omega_smoothness_weight: Weight on (omega - omega_prev)^2 in the QP cost.
    """

    def __init__(
        self,
        max_linear_speed: float = 0.2,
        max_angular_speed: float = 1.0,
        obstacle_radius_scale: float = 2.0,
        lookahead_distance: float = 0.3,
        cbf_gamma: float = 2.0,
        omega_weight: float = 0.1,
        heading_gain: float = 2.5,
        goal_tolerance: float = 0.15,
        v_smoothness_weight: float = 0.1,
        omega_smoothness_weight: float = 0.3,
    ) -> None:
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.obstacle_radius_scale = obstacle_radius_scale
        self.lookahead_distance = lookahead_distance
        self.cbf_gamma = cbf_gamma
        self.omega_weight = omega_weight
        self.heading_gain = heading_gain
        self.goal_tolerance = goal_tolerance
        self.v_smoothness_weight = v_smoothness_weight
        self.omega_smoothness_weight = omega_smoothness_weight

    def compute(
        self,
        tracks: List[Track],
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        goal_x: Optional[float],
        goal_y: Optional[float],
        v_prev: float,
        omega_prev: float,
    ) -> Twist:
        cmd = Twist()

        if goal_x is None or goal_y is None:
            return cmd

        dx = goal_x - robot_x
        dy = goal_y - robot_y
        dist_to_goal = math.hypot(dx, dy)
        if dist_to_goal < self.goal_tolerance:
            return cmd

        # ---- nominal pure-pursuit control: turn toward the goal, drive at v_des ----
        heading_to_goal = math.atan2(dy, dx)
        heading_error = (heading_to_goal - robot_theta + math.pi) % (2 * math.pi) - math.pi
        v_nom = min(self.max_linear_speed, dist_to_goal)
        omega_nom = self.heading_gain * heading_error

        # ---- CBF-QP safety filter: one constraint row per tracked person ----
        L = self.lookahead_distance
        cos_th, sin_th = math.cos(robot_theta), math.sin(robot_theta)
        lookahead_x = robot_x + L * cos_th
        lookahead_y = robot_y + L * sin_th

        A_rows = []
        b_rows = []
        for t in tracks:
            person_x, person_y = t.m[0], t.m[1]
            r = obstacle_radius(t, self.obstacle_radius_scale)

            # Lookahead barrier h_L = ||P - p_person||^2 - (r + L)^2
            ex, ey = lookahead_x - person_x, lookahead_y - person_y
            h_la = ex * ex + ey * ey - (r + L) ** 2

            # Forward/lateral projection of (robot - person) onto the heading frame
            rel_x, rel_y = robot_x - person_x, robot_y - person_y
            s = rel_x * cos_th + rel_y * sin_th
            q = -rel_x * sin_th + rel_y * cos_th

            # CBF condition  2(s+L)v + 2Lq*omega >= -gamma h_L  rewritten as A.u <= b
            A_rows.append([-2.0 * (s + L), -2.0 * L * q])
            b_rows.append(self.cbf_gamma * h_la)

        # Box constraints: 0 <= v <= max_linear_speed, |omega| <= max_angular_speed
        A_rows += [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        b_rows += [self.max_linear_speed, 0.0, self.max_angular_speed, self.max_angular_speed]

        # min (v - v_nom)^2 + omega_weight*(omega - omega_nom)^2
        #   + v_smoothness_weight*(v - v_prev)^2 + omega_smoothness_weight*(omega - omega_prev)^2
        # s.t. A.[v,omega]^T <= b
        P = matrix(np.diag([
            2.0 * (1.0 + self.v_smoothness_weight),
            2.0 * (self.omega_weight + self.omega_smoothness_weight),
        ]))
        q_cost = matrix(np.array([
            -2.0 * (v_nom + self.v_smoothness_weight * v_prev),
            -2.0 * (self.omega_weight * omega_nom + self.omega_smoothness_weight * omega_prev),
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

        cmd.linear.x = float(np.clip(v, 0.0, self.max_linear_speed))
        cmd.angular.z = float(np.clip(omega, -self.max_angular_speed, self.max_angular_speed))
        return cmd
