"""
controllers/base.py — shared interface and helpers for avoidance controllers.

Both CBFController (cbf_controller.py) and MPCController (mpc_controller.py)
implement BaseController.compute(), so people_avoidance_node.py can swap
between them via the make_controller() factory without caring which one it
got.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import Twist

from ..tracking import Track

# Minimum standoff distance the robot must keep from any tracked person (m).
# This is the floor of obstacle_radius(): even a perfectly-converged track
# (lambda_max -> 0) still inflates to this radius.
PERSON_CLEARANCE = 0.0


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


def angle_wrap(angle: float) -> float:
    """Wrap an angle (rad) to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


class BaseController(ABC):
    """Common interface for goal-seeking, people-avoiding velocity controllers."""

    @abstractmethod
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
        """
        Compute a velocity command that drives toward (goal_x, goal_y) while
        avoiding all tracked people.

        Args:
            tracks      : Active person tracks from KalmanTracker.get_tracks().
            robot_x, robot_y, robot_theta: Robot pose in the odometry frame.
            goal_x, goal_y: Destination in the odometry frame, or None if no
                            goal has been received yet (robot stays idle).
            v_prev, omega_prev: The (v, omega) command issued on the previous
                            cycle, for controllers that penalise large jumps.

        Returns:
            geometry_msgs/Twist with linear.x >= 0 and angular.z the
            commanded rotation rate (rad/s, positive = left turn).
        """
        raise NotImplementedError
