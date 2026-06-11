"""
controller.py — Stage 3 of the people-avoidance pipeline.

Input : List[Track], robot pose (x, y, theta)
Output: geometry_msgs/Twist  published on /cmd_vel

Students implement:
  - obstacle_radius()    : derive a safety radius from track covariance
  - compute_velocity()   : avoidance policy → linear + angular velocity command
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from geometry_msgs.msg import Twist

from .tracking import Track

# ---------------------------------------------------------------------------
# TODO Stage 3a — uncertainty-aware obstacle radius
# ---------------------------------------------------------------------------


def obstacle_radius(track: Track, sigma_scale: float) -> float:
    """
    Derive a conservative obstacle radius from the track's positional uncertainty.

    The radius should grow when the Kalman filter is uncertain (large P) and
    shrink as the estimate converges — giving an implicit safety margin that
    inflates when we are unsure where the person is.

    Args:
        track:        An active Track with state covariance P (4×4).
                      The positional sub-block is  P[:2, :2].
        sigma_scale:  Scaling factor k.  The returned radius equals
                      k × √(λ_max),  where λ_max is the largest eigenvalue
                      of P[:2, :2].

    Returns:
        Obstacle radius in metres (always ≥ 0).

    TODO(student): implement this function.
        Steps:
            pos_cov   = track.P[:2, :2]
            eigenvals = np.linalg.eigvalsh(pos_cov)   # sorted ascending
            lambda_max = eigenvals[-1]
            return sigma_scale * math.sqrt(max(lambda_max, 0.0))

    Alternative strategies to explore:
        - Frobenius norm:    sigma_scale * np.linalg.norm(pos_cov, 'fro')
        - Trace:             sigma_scale * math.sqrt(np.trace(pos_cov))
        - Constant + sigma:  base_radius + sigma_scale * math.sqrt(lambda_max)
    """
    # TODO(student): replace this placeholder with the real computation.
    # Returning sigma_scale directly (units: metres) keeps the node runnable
    # before the function is implemented.
    pos_cov = track.P[:2, :2]
    eigenvals = np.linalg.eigvalsh(pos_cov)
    lambda_max = eigenvals[-1]
    return float(sigma_scale * math.sqrt(max(lambda_max, 0.0)))


# ---------------------------------------------------------------------------
# TODO Stage 3b — avoidance policy
# ---------------------------------------------------------------------------


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
    Compute a velocity command that avoids all tracked people.

    Inputs
    ------
    tracks               : Active person tracks from KalmanTracker.get_tracks().
    robot_x              : Robot position x in the odometry frame (m).
    robot_y              : Robot position y in the odometry frame (m).
    robot_theta          : Robot heading in the odometry frame (rad).
    max_linear_speed     : Forward speed cap (m/s).
    max_angular_speed    : Rotation rate cap (rad/s).
    obstacle_radius_scale: Passed to obstacle_radius(); scales how much
                           positional uncertainty inflates the safety bubble.

    Per-track inputs (derived inside this function)
    ------------------------------------------------
    For each track  t  in tracks:
        person_x, person_y  =  t.m[0], t.m[1]       # position in odom frame
        r_i                 =  obstacle_radius(t, obstacle_radius_scale)

    Output
    ------
    geometry_msgs/Twist published on /cmd_vel each control cycle:
        twist.linear.x   — forward velocity (m/s);  positive = forward.
        twist.angular.z  — rotation rate   (rad/s); positive = left turn.

    TODO(student): implement an avoidance policy.
        Suggested approaches (choose one):

        A. Simple reactive rule
           If any obstacle is within its radius r_i of the robot:
               stop (v = 0) and turn away from the nearest obstacle.

        B. Potential fields
           Repulsive force from each person:
               F_rep_i = k_rep / dist_i² × (robot_pos - person_pos) / dist_i
               (only active when dist_i < influence_radius)
           Sum forces → convert to (v, ω) via differential-drive kinematics.

        C. VFH / DWA
           Build a polar obstacle histogram and select the best heading.

        Regardless of approach, clip final commands:
            v   = np.clip(v,   0.0, max_linear_speed)
            ω   = np.clip(ω, -max_angular_speed, max_angular_speed)
    """

    # TODO(student): compute obstacle positions and radii from tracks, e.g.:
    #   for t in tracks:
    #       px, py = t.m[0], t.m[1]
    #       r = obstacle_radius(t, obstacle_radius_scale)
    #       dist = math.hypot(px - robot_x, py - robot_y)
    #       ...

    # TODO(student): implement avoidance logic and set cmd.linear.x / cmd.angular.z

    # Safe default: zero Twist (stop) until logic is implemented.
    cmd = Twist()
    nearest_obstacle_dist = float("inf")
    nearest_obstacle_angle = 0.0
    obstacle_triggered = False

    for t in tracks:
        px, py = t.m[0], t.m[1]
        r = obstacle_radius(t, obstacle_radius_scale)
        dist = math.hypot(px - robot_x, py - robot_y)

        if dist < r:
            obstacle_triggered = True
            if dist < nearest_obstacle_dist:
                nearest_obstacle_dist = dist
                odom_angle = math.atan2(py - robot_y, px - robot_x)
                nearest_obstacle_angle = odom_angle - robot_theta
                # Normalize angle heading boundaries to [-pi, pi]
                nearest_obstacle_angle = (nearest_obstacle_angle + math.pi) % (
                    2 * math.pi
                ) - math.pi

    if obstacle_triggered:
        v = 0.0
        # Obstacle to the left -> turn right, obstacle to the right -> turn left
        omega = -max_angular_speed if nearest_obstacle_angle > 0 else max_angular_speed
    else:
        v = max_linear_speed
        omega = 0.0

    cmd.linear.x = float(np.clip(v, 0.0, max_linear_speed))
    cmd.angular.z = float(np.clip(omega, -max_angular_speed, max_angular_speed))
    return cmd
