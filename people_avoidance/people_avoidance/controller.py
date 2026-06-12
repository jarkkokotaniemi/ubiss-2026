"""
controller.py — Stage 3 of the people-avoidance pipeline.

Input : List[Track], robot pose (x, y, theta), destination (goal_x, goal_y)
Output: geometry_msgs/Twist  published on /cmd_vel

Navigation flow:
  1. On new goal → run A* from robot to goal, extract waypoints (path corners).
  2. Drive toward next waypoint using pure-pursuit + people-avoidance CBF.
  3. When waypoint reached → advance to next; when all done → arrived.
  4. Boundary check: if robot enters a circular keep-out zone, switch to
     boundary-exit mode — a second CBF drives the robot radially outward
     until it clears the exit threshold (> entry threshold, hysteresis).
     After clearing, replan A* and resume.
  5. All velocity/omega commands are low-pass lerped for smooth motion.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np
from cvxopt import matrix, solvers
from geometry_msgs.msg import Twist

from .tracking import Track

solvers.options["show_progress"] = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERSON_CLEARANCE = 0.0  # base clearance added on top of sigma term

# Boundary (keep-out circle) thresholds — hysteresis prevents chattering.
# Robot enters "exit mode" when dist < BOUNDARY_ENTER_RADIUS,
# and resumes normal nav only after dist > BOUNDARY_EXIT_RADIUS.
BOUNDARY_ENTER_RADIUS: float = 1.0  # m  — trip into exit-mode
BOUNDARY_EXIT_RADIUS: float = 1.4  # m  — clear to resume (must be > enter)

# A* grid
GRID_RESOLUTION: float = 0.15  # m per cell
GRID_HALF_EXTENT: float = 12.0  # grid covers ±12 m around origin
ASTAR_INFLATE: float = 0.35  # extra inflation around obstacles in A* grid (m)

# Velocity lerp smoothing factor per control tick (0 = no smoothing, 1 = frozen).
LERP_ALPHA: float = 0.25  # new_cmd = alpha*prev + (1-alpha)*desired

# Waypoint pruning: keep only corners where heading changes by at least this.
WAYPOINT_MIN_TURN_RAD: float = math.radians(12.0)

# ---------------------------------------------------------------------------
# Stage 3a — uncertainty-aware obstacle radius
# ---------------------------------------------------------------------------


def obstacle_radius(track: Track, sigma_scale: float) -> float:
    """PERSON_CLEARANCE + sigma_scale * sqrt(lambda_max of pos covariance)."""
    pos_cov = track.P[:2, :2]
    eigenvals = np.linalg.eigvalsh(pos_cov)
    lambda_max = eigenvals[-1]
    return PERSON_CLEARANCE + sigma_scale * math.sqrt(max(lambda_max, 0.0))


# ---------------------------------------------------------------------------
# A* path planner
# ---------------------------------------------------------------------------


def _world_to_grid(
    wx: float, wy: float, origin: Tuple[float, float], res: float
) -> Tuple[int, int]:
    return (int((wx - origin[0]) / res), int((wy - origin[1]) / res))


def _grid_to_world(
    gx: int, gy: int, origin: Tuple[float, float], res: float
) -> Tuple[float, float]:
    return (gx * res + origin[0] + res * 0.5, gy * res + origin[1] + res * 0.5)


def plan_astar(
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    tracks: List[Track],
    sigma_scale: float,
    boundary_center: Optional[Tuple[float, float]] = None,
    boundary_radius: float = BOUNDARY_EXIT_RADIUS,
) -> List[Tuple[float, float]]:
    """
    Return a list of (x, y) world-frame waypoints from start to goal,
    avoiding inflated track obstacles and optionally a circular boundary zone.
    Returns [goal] if no obstacles or trivial path.
    """
    res = GRID_RESOLUTION
    half = GRID_HALF_EXTENT

    # Grid origin (bottom-left corner) chosen to keep both start and goal inside.
    min_x = min(start_x, goal_x) - half
    min_y = min(start_y, goal_y) - half
    max_x = max(start_x, goal_x) + half
    max_y = max(start_y, goal_y) + half
    origin = (min_x, min_y)
    cols = int((max_x - min_x) / res) + 1
    rows = int((max_y - min_y) / res) + 1

    # Build obstacle grid
    blocked = np.zeros((cols, rows), dtype=bool)

    inflate_cells = int(math.ceil(ASTAR_INFLATE / res))

    for t in tracks:
        r = obstacle_radius(t, sigma_scale) + ASTAR_INFLATE
        cx, cy = _world_to_grid(t.m[0], t.m[1], origin, res)
        ir = int(math.ceil(r / res))
        for dx in range(-ir, ir + 1):
            for dy in range(-ir, ir + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    wx, wy = _grid_to_world(nx, ny, origin, res)
                    if math.hypot(wx - t.m[0], wy - t.m[1]) <= r:
                        blocked[nx, ny] = True

    # Optionally block boundary circle
    if boundary_center is not None:
        bx, by = boundary_center
        ir = int(math.ceil(boundary_radius / res))
        bcx, bcy = _world_to_grid(bx, by, origin, res)
        for dx in range(-ir, ir + 1):
            for dy in range(-ir, ir + 1):
                nx, ny = bcx + dx, bcy + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    wx, wy = _grid_to_world(nx, ny, origin, res)
                    if math.hypot(wx - bx, wy - by) <= boundary_radius:
                        blocked[nx, ny] = True

    s_grid = _world_to_grid(start_x, start_y, origin, res)
    g_grid = _world_to_grid(goal_x, goal_y, origin, res)

    # Clamp to grid bounds
    def clamp(g):
        return (max(0, min(cols - 1, g[0])), max(0, min(rows - 1, g[1])))

    s_grid = clamp(s_grid)
    g_grid = clamp(g_grid)

    # A* search (8-connected)
    DIRS = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, 1.414),
        (1, -1, 1.414),
        (-1, 1, 1.414),
        (-1, -1, 1.414),
    ]

    def h(n):
        return math.hypot(n[0] - g_grid[0], n[1] - g_grid[1])

    open_heap = [(h(s_grid), 0.0, s_grid, None)]
    came_from = {}
    g_cost = {s_grid: 0.0}

    found = False
    while open_heap:
        _, cost, node, parent = heapq.heappop(open_heap)
        if node in came_from:
            continue
        came_from[node] = parent
        if node == g_grid:
            found = True
            break
        for ddx, ddy, step in DIRS:
            nb = (node[0] + ddx, node[1] + ddy)
            if not (0 <= nb[0] < cols and 0 <= nb[1] < rows):
                continue
            if blocked[nb[0], nb[1]]:
                continue
            nc = cost + step
            if nc < g_cost.get(nb, 1e18):
                g_cost[nb] = nc
                heapq.heappush(open_heap, (nc + h(nb), nc, nb, node))

    if not found:
        # Fallback: straight line to goal (CBF will handle avoidance)
        return [(goal_x, goal_y)]

    # Reconstruct grid path
    path_cells = []
    cur = g_grid
    while cur is not None:
        path_cells.append(cur)
        cur = came_from[cur]
    path_cells.reverse()

    # Convert to world coords
    path_world = [_grid_to_world(c[0], c[1], origin, res) for c in path_cells]

    # Prune to corners only (where heading changes significantly)
    waypoints = _extract_waypoints(path_world)

    # Always end exactly at the requested goal
    if (
        not waypoints
        or math.hypot(waypoints[-1][0] - goal_x, waypoints[-1][1] - goal_y) > res
    ):
        waypoints.append((goal_x, goal_y))

    return waypoints if waypoints else [(goal_x, goal_y)]


def _extract_waypoints(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Keep only points where the heading changes by more than the threshold."""
    if len(path) <= 2:
        return list(path)

    waypoints = [path[0]]
    prev_heading = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])

    for i in range(1, len(path) - 1):
        heading = math.atan2(
            path[i + 1][1] - path[i][1],
            path[i + 1][0] - path[i][0],
        )
        delta = abs((heading - prev_heading + math.pi) % (2 * math.pi) - math.pi)
        if delta > WAYPOINT_MIN_TURN_RAD:
            waypoints.append(path[i])
            prev_heading = heading

    waypoints.append(path[-1])
    return waypoints


# ---------------------------------------------------------------------------
# Stage 3b — Waypoint manager (stateful, owned by the node)
# ---------------------------------------------------------------------------


class WaypointManager:
    """
    Holds the current A* waypoint list and tracks which waypoint the robot
    is currently heading toward.

    State machine:
      IDLE        — no goal set
      NAVIGATING  — following waypoints
      BOUNDARY_EXIT — inside keep-out zone, driving outward
      ARRIVED     — within goal_tolerance of final goal
    """

    IDLE = "IDLE"
    NAVIGATING = "NAVIGATING"
    BOUNDARY_EXIT = "BOUNDARY_EXIT"
    ARRIVED = "ARRIVED"

    def __init__(self) -> None:
        self.state: str = self.IDLE
        self.waypoints: List[Tuple[float, float]] = []
        self.wp_index: int = 0
        self.final_goal: Optional[Tuple[float, float]] = None
        self.boundary_center: Optional[Tuple[float, float]] = None
        # Smoothing state
        self._prev_v: float = 0.0
        self._prev_omega: float = 0.0
        # Cooldown counter: how many ticks to suppress boundary re-entry after clearing
        self._boundary_cooldown: int = 0
        # Suppress waypoint-tolerance check on the very first tick after a replan,
        # so the robot doesn't declare ARRIVED before it has moved anywhere.
        self._just_replanned: bool = False

    def set_goal(
        self,
        goal_x: float,
        goal_y: float,
        robot_x: float,
        robot_y: float,
        tracks: List[Track],
        sigma_scale: float,
    ) -> None:
        """Called when a new goal arrives. Fully resets state and replans A*."""
        self.final_goal = (goal_x, goal_y)
        # Clear any stale boundary / arrival state so the robot never stays
        # stuck in BOUNDARY_EXIT or ARRIVED after a new goal is set.
        self.boundary_center = None
        self.state = self.IDLE  # _replan will set NAVIGATING
        self._prev_v = 0.0
        self._prev_omega = 0.0
        self._boundary_cooldown = 0
        self._just_replanned = False
        self._replan(robot_x, robot_y, tracks, sigma_scale)

    def _replan(
        self,
        robot_x: float,
        robot_y: float,
        tracks: List[Track],
        sigma_scale: float,
        avoid_boundary: bool = False,
    ) -> None:
        if self.final_goal is None:
            return
        gx, gy = self.final_goal
        bc = self.boundary_center if avoid_boundary else None
        self.waypoints = plan_astar(
            robot_x,
            robot_y,
            gx,
            gy,
            tracks,
            sigma_scale,
            boundary_center=bc,
            boundary_radius=BOUNDARY_EXIT_RADIUS,
        )
        self.wp_index = 0
        self.state = self.NAVIGATING
        self._just_replanned = True

    @property
    def current_waypoint(self) -> Optional[Tuple[float, float]]:
        if self.wp_index < len(self.waypoints):
            return self.waypoints[self.wp_index]
        return None

    def advance_waypoint(self) -> None:
        self.wp_index += 1
        if self.wp_index >= len(self.waypoints):
            self.state = self.ARRIVED
            self._prev_v = 0.0
            self._prev_omega = 0.0

    def enter_boundary_exit(self, center_x: float, center_y: float) -> None:
        self.boundary_center = (center_x, center_y)
        self.state = self.BOUNDARY_EXIT

    def clear_boundary(
        self,
        robot_x: float,
        robot_y: float,
        tracks: List[Track],
        sigma_scale: float,
    ) -> None:
        """Called once robot has cleared the exit radius. Replan and resume."""
        self.boundary_center = None  # clear before replan so it can't be re-triggered immediately
        self._boundary_cooldown = 20  # suppress re-entry for ~2 s at 10 Hz
        self._replan(robot_x, robot_y, tracks, sigma_scale, avoid_boundary=True)


# ---------------------------------------------------------------------------
# Stage 3c — CBF QP helpers
# ---------------------------------------------------------------------------


def _solve_cbf_qp(
    v_nom: float,
    omega_nom: float,
    A_rows: List[List[float]],
    b_rows: List[float],
    max_linear_speed: float,
    max_angular_speed: float,
    omega_weight: float,
) -> Tuple[float, float]:
    """Solve min (v-v_nom)^2 + w*(ω-ω_nom)^2  s.t. A·[v,ω]^T <= b."""
    # Box constraints
    A_rows = list(A_rows) + [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    b_rows = list(b_rows) + [
        max_linear_speed,
        0.0,
        max_angular_speed,
        max_angular_speed,
    ]

    P = matrix(np.diag([2.0, 2.0 * omega_weight]))
    q_cost = matrix(np.array([-2.0 * v_nom, -2.0 * omega_weight * omega_nom]))
    G = matrix(np.array(A_rows, dtype=float))
    h_qp = matrix(np.array(b_rows, dtype=float))

    try:
        sol = solvers.qp(P, q_cost, G, h_qp)
        if sol["status"] == "optimal":
            v, omega = np.array(sol["x"]).flatten()
            return float(v), float(omega)
    except Exception:
        pass
    # QP infeasible or failed — fall back to clipped nominal so the robot
    # keeps moving rather than freezing. CBF constraints are best-effort.
    v_safe = float(np.clip(v_nom, 0.0, max_linear_speed))
    omega_safe = float(np.clip(omega_nom, -max_angular_speed, max_angular_speed))
    return v_safe, omega_safe


def _people_cbf_constraints(
    tracks: List[Track],
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    lookahead_distance: float,
    cbf_gamma: float,
    sigma_scale: float,
) -> Tuple[List[List[float]], List[float]]:
    """
    Per-track CBF constraint rows.
    Barrier: h_L = ||lookahead - p_person||^2 - (r+L)^2
    Constraint: -2(s+L)v - 2Lq*ω <= γ·h_L
    """
    L = lookahead_distance
    cos_th, sin_th = math.cos(robot_theta), math.sin(robot_theta)
    lx = robot_x + L * cos_th
    ly = robot_y + L * sin_th

    A_rows, b_rows = [], []
    for t in tracks:
        px, py = t.m[0], t.m[1]
        r = obstacle_radius(t, sigma_scale)

        ex, ey = lx - px, ly - py
        h_la = ex * ex + ey * ey - (r + L) ** 2

        rel_x, rel_y = robot_x - px, robot_y - py
        s = rel_x * cos_th + rel_y * sin_th
        q = -rel_x * sin_th + rel_y * cos_th

        A_rows.append([-2.0 * (s + L), -2.0 * L * q])
        b_rows.append(cbf_gamma * h_la)

    return A_rows, b_rows


# ---------------------------------------------------------------------------
# Stage 3d — main compute_velocity (now waypoint-aware + boundary-aware)
# ---------------------------------------------------------------------------


def compute_velocity(
    tracks: List[Track],
    robot_x: float,
    robot_y: float,
    robot_theta: float,
    waypoint_manager: "WaypointManager",
    max_linear_speed: float = 0.2,
    max_angular_speed: float = 1.0,
    obstacle_radius_scale: float = 2.0,
    lookahead_distance: float = 0.3,
    cbf_gamma: float = 2.0,
    omega_weight: float = 0.1,
    heading_gain: float = 2.5,
    goal_tolerance: float = 0.15,
    waypoint_tolerance: float = 0.25,
) -> Twist:
    """
    Main velocity controller. Uses WaypointManager for stateful A* navigation.

    States
    ------
    IDLE / ARRIVED  → zero velocity.
    BOUNDARY_EXIT   → radial-outward CBF; once clear, manager replans.
    NAVIGATING      → pure-pursuit toward current waypoint, filtered by
                      people-avoidance CBF; advances waypoints automatically.
    """
    cmd = Twist()
    wm = waypoint_manager
    alpha = LERP_ALPHA

    # ── IDLE / ARRIVED ────────────────────────────────────────────────────────
    if wm.state in (WaypointManager.IDLE, WaypointManager.ARRIVED):
        v_out = _lerp(wm._prev_v, 0.0, alpha)
        omega_out = _lerp(wm._prev_omega, 0.0, alpha)
        wm._prev_v, wm._prev_omega = v_out, omega_out
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        return cmd  # return-path: IDLE/ARRIVED — zero velocity

    # ── Boundary entrance detection ───────────────────────────────────────────
    # Check tracked people as potential boundary centers (closest one inside radius)
    if wm.state == WaypointManager.NAVIGATING:
        if wm._boundary_cooldown > 0:
            wm._boundary_cooldown -= 1
        else:
            for t in tracks:
                px, py = t.m[0], t.m[1]
                dist = math.hypot(robot_x - px, robot_y - py)
                if dist < BOUNDARY_ENTER_RADIUS:
                    wm.enter_boundary_exit(px, py)
                    break

    # ── BOUNDARY_EXIT: drive radially outward from boundary center ────────────
    if wm.state == WaypointManager.BOUNDARY_EXIT and wm.boundary_center is not None:
        bcx, bcy = wm.boundary_center
        dist_to_center = math.hypot(robot_x - bcx, robot_y - bcy)

        if dist_to_center >= BOUNDARY_EXIT_RADIUS:
            # Cleared — replan and resume
            wm.clear_boundary(robot_x, robot_y, tracks, obstacle_radius_scale)
        else:
            # Nominal: drive straight forward; steer heading away from center
            away_heading = math.atan2(robot_y - bcy, robot_x - bcx)
            heading_error = (away_heading - robot_theta + math.pi) % (
                2 * math.pi
            ) - math.pi

            # Barrier: h = dist^2 - BOUNDARY_ENTER_RADIUS^2
            # We want h to increase → push outward.
            # Simple CBF: constraint that v_radial >= γ * (-h)
            # Implemented as: steer toward away_heading at max speed.
            v_nom = max_linear_speed
            omega_nom = heading_gain * heading_error

            # People avoidance constraints still apply
            A_rows, b_rows = _people_cbf_constraints(
                tracks,
                robot_x,
                robot_y,
                robot_theta,
                lookahead_distance,
                cbf_gamma,
                obstacle_radius_scale,
            )

            # Boundary exit CBF: ensure forward progress outward.
            # h_b = (rx - bcx)*cos_th + (ry - bcy)*sin_th  (signed dist ahead)
            # dh/dt = v + ... we want v >= cbf_gamma*(EXIT_R - dist_to_center)
            # Constraint: -v <= -cbf_gamma*(EXIT_R - dist_to_center)
            cos_th, sin_th = math.cos(robot_theta), math.sin(robot_theta)
            outward_dot = (robot_x - bcx) * cos_th + (robot_y - bcy) * sin_th
            h_boundary = dist_to_center - BOUNDARY_ENTER_RADIUS
            A_rows.append([-1.0, 0.0])
            b_rows.append(-cbf_gamma * max(0.0, -h_boundary))

            v_raw, omega_raw = _solve_cbf_qp(
                v_nom,
                omega_nom,
                A_rows,
                b_rows,
                max_linear_speed,
                max_angular_speed,
                omega_weight,
            )

            v_out = _lerp(wm._prev_v, v_raw, alpha)
            omega_out = _lerp(wm._prev_omega, omega_raw, alpha)
            wm._prev_v, wm._prev_omega = v_out, omega_out
            cmd.linear.x = float(np.clip(v_out, 0.0, max_linear_speed))
            cmd.angular.z = float(
                np.clip(omega_out, -max_angular_speed, max_angular_speed)
            )
            return cmd

    # ── NAVIGATING: drive toward current waypoint ─────────────────────────────
    wp = wm.current_waypoint
    if wp is None:
        wm.state = WaypointManager.ARRIVED
        return cmd

    wp_x, wp_y = wp
    dist_to_wp = math.hypot(wp_x - robot_x, wp_y - robot_y)

    # Check if we've reached this waypoint — but skip on the very first tick
    # after a replan to avoid declaring ARRIVED before the robot has moved.
    tol = goal_tolerance if wm.wp_index == len(wm.waypoints) - 1 else waypoint_tolerance
    if wm._just_replanned:
        wm._just_replanned = False  # consume flag; skip advance this tick
    elif dist_to_wp < tol:
        wm.advance_waypoint()
        wp = wm.current_waypoint
        if wp is None:
            return cmd
        wp_x, wp_y = wp
        dist_to_wp = math.hypot(wp_x - robot_x, wp_y - robot_y)

    # Pure-pursuit nominal command toward waypoint
    heading_to_wp = math.atan2(wp_y - robot_y, wp_x - robot_x)
    heading_error = (heading_to_wp - robot_theta + math.pi) % (2 * math.pi) - math.pi
    v_nom = min(dist_to_wp, max_linear_speed)
    omega_nom = heading_gain * heading_error

    # People-avoidance CBF constraints
    A_rows, b_rows = _people_cbf_constraints(
        tracks,
        robot_x,
        robot_y,
        robot_theta,
        lookahead_distance,
        cbf_gamma,
        obstacle_radius_scale,
    )

    v_raw, omega_raw = _solve_cbf_qp(
        v_nom,
        omega_nom,
        A_rows,
        b_rows,
        max_linear_speed,
        max_angular_speed,
        omega_weight,
    )

    # Lerp for smoothness
    v_out = _lerp(wm._prev_v, v_raw, alpha)
    omega_out = _lerp(wm._prev_omega, omega_raw, alpha)
    wm._prev_v, wm._prev_omega = v_out, omega_out

    cmd.linear.x = float(np.clip(v_out, 0.0, max_linear_speed))
    cmd.angular.z = float(np.clip(omega_out, -max_angular_speed, max_angular_speed))
    return cmd


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _lerp(a: float, b: float, alpha: float) -> float:
    """Exponential moving average: alpha=0 → instant, alpha=1 → frozen."""
    return alpha * a + (1.0 - alpha) * b
