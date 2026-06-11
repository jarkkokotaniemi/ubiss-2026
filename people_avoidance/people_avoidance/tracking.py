"""
tracking.py — Stage 2 of the people-avoidance pipeline.

Input : List[LegMeasurement]  (one per scan, from leg_detection.py)
Output: List[Track]           (maintained across scans)

Each track i models one person as a Gaussian:
    X^i_t ~ N(m^i_t, P^i_t)

State vector  m = [x, y, vx, vy]   (position + velocity in odom frame)
Covariance    P is 4 × 4.

Students implement:
  - KalmanTracker.__init__()   : define F and Q
  - KalmanTracker.predict()    : constant-velocity propagation
  - KalmanTracker.associate()  : measurement-to-track matching
  - KalmanTracker.update()     : KF update + track lifecycle management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .leg_detection import LegMeasurement

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Track:
    """
    Single-person track: state X^i_t ~ N(m^i_t, P^i_t).

    Attributes
    ----------
    m        : Mean state vector [x, y, vx, vy], shape (4,).
    P        : State covariance matrix, shape (4, 4).
    track_id : Unique integer identifier assigned at track creation.
    misses   : Number of consecutive scans without a matched measurement.
               The tracker deletes a track when misses > max_misses.

    Observation model
    -----------------
    Only position is observed.  The measurement matrix H projects the 4-D
    state to a 2-D observation:

        H = [[1, 0, 0, 0],
             [0, 1, 0, 0]]

    so that  z = H @ m + noise,  noise ~ N(0, R).
    """

    m: np.ndarray  # shape (4,): [x, y, vx, vy]
    P: np.ndarray  # shape (4, 4)
    track_id: int
    misses: int = 0


# ---------------------------------------------------------------------------
# Kalman tracker
# ---------------------------------------------------------------------------


class KalmanTracker:
    """
    Multi-target constant-velocity Kalman filter with data association.

    Lifecycle of each call to update()
    -----------------------------------
    1. predict()    — propagate all tracks forward by dt.
    2. associate()  — match measurements to tracks.
    3. KF update    — correct matched tracks with measurements.
    4. Spawn        — create new tracks for unmatched measurements.
    5. Prune        — delete tracks that have been missed too many times.

    Typical usage
    -------------
    tracker = KalmanTracker(dt=0.1)
    for measurements in scan_stream:          # measurements: List[LegMeasurement]
        tracker.update(measurements)
        active_tracks = tracker.get_tracks()  # List[Track]
    """

    # Observation matrix H: extracts (x, y) from the 4-D state.
    H: np.ndarray = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=float,
    )

    def __init__(self, dt: float = 0.1, max_misses: int = 5) -> None:
        """
        Args:
            dt:         Time step between scans (seconds).
            max_misses: Delete a track after this many consecutive missed frames.
        """
        self.dt = dt
        self.max_misses = max_misses
        self.tracks: List[Track] = []
        self._next_id: int = 0

        # TODO(student): define the motion-model matrices here.
        #
        # Constant-velocity state transition (F):
        #
        #     F = [[1, 0, dt,  0],
        #          [0, 1,  0, dt],
        #          [0, 0,  1,  0],
        #          [0, 0,  0,  1]]
        #
        # Process noise covariance (Q) — tune experimentally:
        #
        #     A diagonal initialisation is a good starting point:
        #     Q = diag([σ_pos², σ_pos², σ_vel², σ_vel²])
        #     where σ_pos ≈ 0.05 m,  σ_vel ≈ 0.1 m/s.
        #
        # Store them as:
        #     self.F = ...   (shape 4×4)
        #     self.Q = ...   (shape 4×4)

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        sigma_pos = 0.05
        sigma_vel = 0.10
        self.Q = np.diag([sigma_pos**2, sigma_pos**2, sigma_vel**2, sigma_vel**2])

    # ------------------------------------------------------------------
    # TODO Stage 2a — predict
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """
        Propagate every active track forward one time step.

        For each track i apply the constant-velocity model:

            m^i_t|t-1  =  F  @  m^i_t-1
            P^i_t|t-1  =  F  @  P^i_t-1  @  F.T  +  Q

        This is called automatically at the start of every update() cycle.
        """
        # TODO(student): loop over self.tracks and apply the predict equations.
        #   Use self.F and self.Q defined in __init__.
        for track in self.tracks:
            track.m = self.F @ track.m
            track.P = self.F @ track.P @ self.F.T + self.Q
        pass

    # ------------------------------------------------------------------
    # TODO Stage 2b — data association
    # ------------------------------------------------------------------

    def associate(
        self,
        measurements: List[LegMeasurement],
    ) -> List[Tuple[int, int]]:
        """
        Match measurements to existing tracks (global nearest-neighbour).

        Args:
            measurements: LegMeasurement list from the current scan.

        Returns:
            List of (track_index, measurement_index) pairs where
            track_index  indexes into self.tracks and
            meas_index   indexes into measurements.

            Unmatched measurements → passed to update() for track spawning.
            Unmatched tracks       → miss counter incremented in update().

        Implementation steps
        --------------------
        1. Build a cost matrix  C  of shape (n_tracks, n_measurements).
           A simple Euclidean distance between the track's predicted position
           (track.m[:2]) and each measurement's (x, y) is a correct first step.
           A more principled approach uses Mahalanobis distance:

               d²_ij = innovation.T @ inv(S_ij) @ innovation
               where innovation = [meas.x - m[0], meas.y - m[1]]
                     S_ij       = H @ P @ H.T + R_j

        2. Solve the linear assignment problem:

               from scipy.optimize import linear_sum_assignment
               row_ind, col_ind = linear_sum_assignment(C)

        3. Gate: reject any assignment (i, j) where C[i, j] > gate_threshold
           to prevent associating a track with a far-away measurement.
           A typical gate for Euclidean cost: 1.0–2.0 m.

        If there are no tracks or no measurements, return [].
        """
        # TODO(student): build cost matrix and run linear_sum_assignment
        if not self.tracks or not measurements:
            return []

        n_tracks = len(self.tracks)
        n_meas = len(measurements)
        C = np.zeros((n_tracks, n_meas))

        for i, track in enumerate(self.tracks):
            for j, meas in enumerate(measurements):
                C[i, j] = np.hypot(track.m[0] - meas.x, track.m[1] - meas.y)

        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(C)

        gate_threshold = 1.5
        assignments = []
        for r, c in zip(row_ind, col_ind):
            if C[r, c] <= gate_threshold:
                assignments.append((int(r), int(c)))
        return assignments

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_track(self, meas: LegMeasurement) -> None:
        """Initialise a new Track from an unmatched measurement."""
        m = np.array([meas.x, meas.y, 0.0, 0.0], dtype=float)

        # TODO(student): set a sensible initial covariance P0.
        #   A common diagonal initialisation:
        #   P0 = diag([Rxx, Ryy, σ_vel², σ_vel²])
        #   using the measurement's own Rxx, Ryy as position uncertainty seeds.
        m = np.array([meas.x, meas.y, 0.0, 0.0], dtype=float)
        sigma_vel = 1.0
        P = np.diag([meas.Rxx, meas.Ryy, sigma_vel**2, sigma_vel**2])

        self.tracks.append(Track(m=m, P=P, track_id=self._next_id))
        self._next_id += 1

    # ------------------------------------------------------------------
    # TODO Stage 2c — full update cycle
    # ------------------------------------------------------------------

    def update(self, measurements: List[LegMeasurement]) -> None:
        """
        Run one complete tracking cycle: predict → associate → KF update.

        Steps
        -----
        1. Call self.predict() to propagate all tracks.
        2. Call self.associate(measurements) to get matched pairs.
        3. For each matched (track_index, meas_index) pair, apply the KF
           update equations:

               z   =  np.array([meas.x, meas.y])
               R   =  np.array([[meas.Rxx, meas.Rxy],
                                 [meas.Rxy, meas.Ryy]])
               y   =  z  -  H @ m              # innovation
               S   =  H  @  P  @  H.T  +  R   # innovation covariance
               K   =  P  @  H.T  @  np.linalg.inv(S)   # Kalman gain
               m   =  m  +  K  @  y            # updated mean
               P   =  (I  -  K  @  H)  @  P   # updated covariance
                                               # (Joseph form is numerically safer)

        4. For matched tracks: reset track.misses = 0.
           For unmatched tracks: increment track.misses by 1.

        5. Spawn a new Track (via self._spawn_track) for every measurement
           that was NOT matched to any existing track.

        6. Remove tracks from self.tracks where track.misses > self.max_misses.

        The stub below performs steps 1–2 and identifies the matched/unmatched
        sets so students can fill in steps 3–6 without having to re-derive those
        sets themselves.
        """

        # TODO(student): step 3 — apply KF update for each (ti, mi) in assignments
        # TODO(student): step 4 — update miss counters
        # TODO(student): step 5 — spawn tracks for unmatched measurements
        #   for mi, meas in enumerate(measurements):
        #       if mi not in matched_meas_idxs:
        #           self._spawn_track(meas)
        # TODO(student): step 6 — prune stale tracks
        #   self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        self.predict()
        assignments = self.associate(measurements)

        matched_track_idxs = {ti for ti, _ in assignments}
        matched_meas_idxs = {mi for _, mi in assignments}

        I = np.eye(4)
        for ti, mi in assignments:
            track = self.tracks[ti]
            meas = measurements[mi]

            z = np.array([meas.x, meas.y])
            R = np.array([[meas.Rxx, meas.Rxy], [meas.Rxy, meas.Ryy]])

            y = z - self.H @ track.m
            S = self.H @ track.P @ self.H.T + R
            K = track.P @ self.H.T @ np.linalg.inv(S)

            # KF Mean correction
            track.m = track.m + K @ y

            # Joseph Form covariance update
            ImKH = I - K @ self.H
            track.P = ImKH @ track.P @ ImKH.T + K @ R @ K.T

            track.misses = 0

        # Account for unmatched stale tracks
        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_idxs:
                track.misses += 1

        # Spawn tracks for brand new entries
        for mi, meas in enumerate(measurements):
            if mi not in matched_meas_idxs:
                self._spawn_track(meas)

        # Clear expired tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    def get_tracks(self) -> List[Track]:
        """Return a snapshot of the current active track list."""
        return list(self.tracks)
