"""
controllers/mpc_controller.py — MPC-CBF controller (CasADi / IPOPT).

Adapted from exp_mpc_solution.ipynb (Sec. 5/7, "MPC-CBF"): a receding-horizon
controller over the discrete Dubins model

    x_{k+1} = x_k + dt * [v cos(theta_k), v sin(theta_k), omega]

that drives toward (goal_x, goal_y) while enforcing a discrete control-
barrier-function constraint

    h(x_{k+1}) >= (1 - gamma) * h(x_k),   k = 0 .. N-1

per tracked person, generalized from the notebook's single static obstacle
to up to `mpc_max_tracked_people` people whose positions are predicted
forward with their tracked (constant) velocity, with uncertainty-scaled
radii from obstacle_radius().
"""
from __future__ import annotations

import math
from typing import List, Optional

import casadi as ca
import numpy as np
from geometry_msgs.msg import Twist

from ..tracking import Track
from .base import BaseController, angle_wrap, obstacle_radius


class MPCController(BaseController):
    """
    MPC-CBF controller built once at construction time.

    Args
    ----
    dt                    : Prediction/control time step (s) — should match
                            the pipeline's scan/control period.
    mpc_horizon           : N — number of predicted steps.
    max_linear_speed      : Forward speed cap (m/s); v in [0, max_linear_speed].
    max_angular_speed     : Rotation rate cap (rad/s); |omega| <= max_angular_speed.
    obstacle_radius_scale : Passed to obstacle_radius(); scales how much
                            positional uncertainty inflates each person's
                            safety bubble.
    mpc_cbf_gamma         : gamma in (0, 1] — discrete CBF decay rate. gamma
                            -> 1 is close to naive h(x_{k+1}) >= 0; gamma -> 0
                            forces h to stay (almost) non-decreasing along the
                            whole horizon (wider, earlier detours).
    mpc_max_tracked_people: K — fixed number of obstacle slots built into the
                            NLP. Each scan, the K closest tracks fill these
                            slots; unused slots become a far-away dummy
                            obstacle (radius 0) so their CBF row stays slack.
    goal_tolerance        : Stop once within this distance of the goal (m).
    mpc_q_pos             : Stage/terminal cost weight on (x, y) tracking error.
    mpc_q_theta           : Stage/terminal cost weight on heading tracking error.
    mpc_r_v               : Stage cost weight on v^2 (control effort).
    mpc_r_omega           : Stage cost weight on omega^2 (control effort).
    """

    # Obstacle slots further than this from the robot are treated as "dummy"
    # (radius 0) -- keeps padded slots from ever binding the CBF constraint.
    _DUMMY_DISTANCE = 1.0e3

    def __init__(
        self,
        dt: float = 0.1,
        mpc_horizon: int = 15,
        max_linear_speed: float = 0.2,
        max_angular_speed: float = 1.0,
        obstacle_radius_scale: float = 2.0,
        mpc_cbf_gamma: float = 0.3,
        mpc_max_tracked_people: int = 5,
        goal_tolerance: float = 0.15,
        mpc_q_pos: float = 10.0,
        mpc_q_theta: float = 0.1,
        mpc_r_v: float = 0.1,
        mpc_r_omega: float = 0.1,
    ) -> None:
        self.dt = dt
        self.N = mpc_horizon
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.obstacle_radius_scale = obstacle_radius_scale
        self.gamma = mpc_cbf_gamma
        self.K = mpc_max_tracked_people
        self.goal_tolerance = goal_tolerance

        Q = np.diag([mpc_q_pos, mpc_q_pos, mpc_q_theta])
        R = np.diag([mpc_r_v, mpc_r_omega])
        Qf = Q

        self._build_opti(Q, R, Qf)
        self._warmup_solve()

    # ------------------------------------------------------------------
    # NLP construction (once, at startup)
    # ------------------------------------------------------------------

    def _build_opti(self, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray) -> None:
        N, K, dt = self.N, self.K, self.dt

        opti = ca.Opti()
        X = opti.variable(3, N + 1)
        U = opti.variable(2, N)
        x0_p = opti.parameter(3)
        xref_p = opti.parameter(3)
        obs_p0 = opti.parameter(2, K)  # current (x, y) of each obstacle slot
        obs_v = opti.parameter(2, K)  # constant velocity (vx, vy) of each slot
        obs_r = opti.parameter(K)  # inflated radius of each slot

        # ---- stage cost + forward-Euler Dubins dynamics ----
        cost = 0
        for k in range(N):
            dx = X[:, k] - xref_p
            cost += dx.T @ Q @ dx + U[:, k].T @ R @ U[:, k]
            x_next = X[:, k] + dt * ca.vertcat(
                U[0, k] * ca.cos(X[2, k]),
                U[0, k] * ca.sin(X[2, k]),
                U[1, k],
            )
            opti.subject_to(X[:, k + 1] == x_next)
        dxN = X[:, N] - xref_p
        cost += dxN.T @ Qf @ dxN
        opti.minimize(cost)

        # ---- initial state + box constraints ----
        opti.subject_to(X[:, 0] == x0_p)
        opti.subject_to(opti.bounded(0.0, U[0, :], self.max_linear_speed))
        opti.subject_to(opti.bounded(-self.max_angular_speed, U[1, :], self.max_angular_speed))

        # ---- discrete CBF safety: one row per (obstacle slot, predicted step) ----
        for i in range(K):
            for k in range(N):
                p_obs_k = obs_p0[:, i] + k * dt * obs_v[:, i]
                p_obs_k1 = obs_p0[:, i] + (k + 1) * dt * obs_v[:, i]
                e_k = X[0:2, k] - p_obs_k
                e_k1 = X[0:2, k + 1] - p_obs_k1
                h_k = e_k.T @ e_k - obs_r[i] ** 2
                h_k1 = e_k1.T @ e_k1 - obs_r[i] ** 2
                opti.subject_to(h_k1 >= (1 - self.gamma) * h_k)

        opti.solver("ipopt", {
            "print_time": 0,
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "ipopt.tol": 1e-4,
        })

        self.opti = opti
        self.X = X
        self.U = U
        self.x0_p = x0_p
        self.xref_p = xref_p
        self.obs_p0 = obs_p0
        self.obs_v = obs_v
        self.obs_r = obs_r

    def _warmup_solve(self) -> None:
        """
        Run one throwaway solve at construction time.

        CasADi/IPOPT pay a one-off JIT/compilation cost (hundreds of ms) on
        the first solve of a given Opti graph. Paying it here, during node
        startup, keeps it off the first real scan callback.
        """
        K = self.K
        self.opti.set_value(self.x0_p, [0.0, 0.0, 0.0])
        self.opti.set_value(self.xref_p, [0.0, 0.0, 0.0])
        self.opti.set_value(self.obs_p0, np.full((2, K), self._DUMMY_DISTANCE))
        self.opti.set_value(self.obs_v, np.zeros((2, K)))
        self.opti.set_value(self.obs_r, np.zeros(K))
        try:
            sol = self.opti.solve()
            self.opti.set_initial(self.X, sol.value(self.X))
            self.opti.set_initial(self.U, sol.value(self.U))
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Per-cycle solve
    # ------------------------------------------------------------------

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

        # Reference heading, unwrapped to stay within +/-pi of robot_theta so
        # the heading cost doesn't see a spurious +/-2*pi jump near the wrap.
        heading_to_goal = math.atan2(dy, dx)
        theta_ref = robot_theta + angle_wrap(heading_to_goal - robot_theta)

        self.opti.set_value(self.x0_p, [robot_x, robot_y, robot_theta])
        self.opti.set_value(self.xref_p, [goal_x, goal_y, theta_ref])

        # ---- fill the K closest tracks into obstacle slots, pad the rest ----
        K = self.K
        p0 = np.full((2, K), self._DUMMY_DISTANCE)
        v = np.zeros((2, K))
        r = np.zeros(K)

        closest = sorted(
            tracks,
            key=lambda t: (t.m[0] - robot_x) ** 2 + (t.m[1] - robot_y) ** 2,
        )[:K]
        for i, t in enumerate(closest):
            p0[:, i] = (t.m[0], t.m[1])
            v[:, i] = (t.m[2], t.m[3])
            r[i] = obstacle_radius(t, self.obstacle_radius_scale)

        self.opti.set_value(self.obs_p0, p0)
        self.opti.set_value(self.obs_v, v)
        self.opti.set_value(self.obs_r, r)

        try:
            sol = self.opti.solve()
            u0 = np.array(sol.value(self.U[:, 0])).flatten()
            v_cmd, omega_cmd = float(u0[0]), float(u0[1])
            self.opti.set_initial(self.X, sol.value(self.X))
            self.opti.set_initial(self.U, sol.value(self.U))
        except RuntimeError:
            v_cmd, omega_cmd = 0.0, 0.0
            # Re-seed the warm start with a "stay put" guess so the next
            # solve doesn't keep replaying the same infeasible trajectory.
            x0 = np.array([robot_x, robot_y, robot_theta])
            self.opti.set_initial(self.X, np.tile(x0.reshape(3, 1), (1, self.N + 1)))
            self.opti.set_initial(self.U, np.zeros((2, self.N)))

        cmd.linear.x = float(np.clip(v_cmd, 0.0, self.max_linear_speed))
        cmd.angular.z = float(np.clip(omega_cmd, -self.max_angular_speed, self.max_angular_speed))
        return cmd
