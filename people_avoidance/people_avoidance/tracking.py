"""
tracking.py — Stage 2 of the people-avoidance pipeline.

Input : List[LegMeasurement]  (one per scan, from leg_detection.py)
Output: List[Track]           (maintained across scans)

Each track i models one person as a Gaussian:
    X^i_t ~ N(m^i_t, P^i_t)

State vector  m = [x, y, vx, vy]   (position + velocity in odom frame)
Covariance    P is 4 × 4.

KalmanTracker maintains a constant-velocity Kalman filter per tracked person,
with global-nearest-neighbour data association (Mahalanobis distance +
Hungarian algorithm) so multiple people can be tracked simultaneously.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment

from .leg_detection import LegMeasurement


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Continuous white-noise-acceleration spectral density (m^2/s^3) used to build
# Q in KalmanTracker.__init__. Larger -> tracks adapt to velocity changes
# faster but are noisier; smaller -> smoother but laggier. Lowered from the
# notebook's q=1 (tuned for synthetic walkers whose velocity actually changes
# at that rate) to 0.5: real static clutter (furniture legs, wall corners)
# has ~zero acceleration, and a smaller q keeps a coasting track's covariance
# from ballooning during misses -- which otherwise lets it "jump" onto an
# unrelated detection several metres away on real scans. Going much lower
# (e.g. 0.1) makes the filter lag too much behind a person walking at a
# normal ~1.4 m/s, breaking the two-people regression scenario.
PROCESS_NOISE_Q = 0.5

# Initial velocity std-dev (m/s) for newly spawned tracks. Velocity is
# unobserved at spawn time, so this seeds P0's velocity block with a large
# uncertainty until a second matched measurement lets the filter estimate it.
# Lowered from 1.0 to 0.3 (a brisk walking pace): a large initial velocity
# uncertainty mixes into position uncertainty on the very next predict(),
# immediately widening a brand-new track's association gate -- exactly when
# most spurious clutter-blip tracks live their entire (short) life.
SPAWN_VELOCITY_SIGMA = 0.3

# Mahalanobis distance-squared gate used in associate(). Chi-square critical
# value for 2 DOF; 13.8 = 99.9% confidence. Pairs whose cost exceeds this are
# rejected even if they were the globally optimal assignment. Tighter gates
# (5.991 = 95%, 9.21 = 99%) reject a non-trivial fraction of *correct*
# associations purely by chance -- especially in the first few steps after
# spawn when the velocity estimate is still poor -- which causes a track to
# be dropped, pruned, and respawned under a new ID. The very permissive 99.9%
# gate trades a slightly higher chance of mis-associating two tracks that
# pass close together for much more stable track IDs over time.
ASSOCIATION_GATE_CHI2 = 13.8


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
    m:        np.ndarray   # shape (4,): [x, y, vx, vy]
    P:        np.ndarray   # shape (4, 4)
    track_id: int
    misses:   int = 0


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
        [[1, 0, 0, 0],
         [0, 1, 0, 0]],
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

        # Constant-velocity state transition.
        self.F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Continuous white-noise-acceleration process noise (see PROCESS_NOISE_Q).
        q = PROCESS_NOISE_Q
        self.Q = q * np.array([
            [dt**3 / 3, 0,         dt**2 / 2, 0        ],
            [0,         dt**3 / 3, 0,         dt**2 / 2],
            [dt**2 / 2, 0,         dt,        0        ],
            [0,         dt**2 / 2, 0,         dt       ],
        ])

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self) -> None:
        """
        Propagate every active track forward one time step.

        For each track i apply the constant-velocity model:

            m^i_t|t-1  =  F  @  m^i_t-1
            P^i_t|t-1  =  F  @  P^i_t-1  @  F.T  +  Q

        This is called automatically at the start of every update() cycle.
        """
        for track in self.tracks:
            track.m = self.F @ track.m
            track.P = self.F @ track.P @ self.F.T + self.Q

    # ------------------------------------------------------------------
    # Data association
    # ------------------------------------------------------------------

    def associate(
        self,
        measurements: List[LegMeasurement],
    ) -> List[Tuple[int, int]]:
        """
        Match measurements to existing tracks (global nearest-neighbour).

        Builds a Mahalanobis distance-squared cost matrix
        (d²_ij = innovation.T @ inv(S_ij) @ innovation, with
        S_ij = H @ P_i @ H.T + R_j), solves the linear assignment problem,
        and gates out pairs whose cost exceeds ASSOCIATION_GATE_CHI2.

        Args:
            measurements: LegMeasurement list from the current scan.

        Returns:
            List of (track_index, measurement_index) pairs where
            track_index  indexes into self.tracks and
            meas_index   indexes into measurements.

            Unmatched measurements → passed to update() for track spawning.
            Unmatched tracks       → miss counter incremented in update().
        """
        if not self.tracks or not measurements:
            return []

        n_tracks = len(self.tracks)
        n_meas = len(measurements)
        C = np.zeros((n_tracks, n_meas))

        for i, track in enumerate(self.tracks):
            P_pos = self.H @ track.P @ self.H.T
            for j, meas in enumerate(measurements):
                innovation = np.array([meas.x, meas.y]) - self.H @ track.m
                R = np.array([[meas.Rxx, meas.Rxy], [meas.Rxy, meas.Ryy]])
                S = P_pos + R
                C[i, j] = innovation @ linalg.solve(S, innovation, assume_a="pos")

        row_ind, col_ind = linear_sum_assignment(C)

        return [
            (int(r), int(c))
            for r, c in zip(row_ind, col_ind)
            if C[r, c] <= ASSOCIATION_GATE_CHI2
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_track(self, meas: LegMeasurement) -> None:
        """Initialise a new Track from an unmatched measurement."""
        m = np.array([meas.x, meas.y, 0.0, 0.0])
        P = np.diag([meas.Rxx, meas.Ryy, SPAWN_VELOCITY_SIGMA**2, SPAWN_VELOCITY_SIGMA**2])

        self.tracks.append(Track(m=m, P=P, track_id=self._next_id))
        self._next_id += 1

    # ------------------------------------------------------------------
    # Full update cycle
    # ------------------------------------------------------------------

    def update(self, measurements: List[LegMeasurement]) -> None:
        """
        Run one complete tracking cycle: predict → associate → KF update.

        1. Propagate all tracks via predict().
        2. Match measurements to tracks via associate().
        3. For each matched pair, apply the KF correction:
               innovation = z - H @ m
               S = H @ P @ H.T + R
               K = S^-1 @ (H @ P).T            # via linalg.solve
               m = m + K @ innovation
               P = P - K @ S @ K.T
        4. Reset misses for matched tracks; increment for unmatched tracks.
        5. Spawn a new track for every unmatched measurement.
        6. Drop tracks with misses > max_misses.
        """
        self.predict()
        assignments = self.associate(measurements)

        matched_track_idxs = {ti for ti, _ in assignments}
        matched_meas_idxs = {mi for _, mi in assignments}

        for ti, mi in assignments:
            track = self.tracks[ti]
            meas = measurements[mi]

            z = np.array([meas.x, meas.y])
            R = np.array([[meas.Rxx, meas.Rxy], [meas.Rxy, meas.Ryy]])

            S = self.H @ track.P @ self.H.T + R
            K = linalg.solve(S, self.H @ track.P.T, assume_a="pos").T

            innovation = z - self.H @ track.m
            track.m = track.m + K @ innovation
            track.P = track.P - K @ S @ K.T

            track.misses = 0

        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_idxs:
                track.misses += 1

        for mi, meas in enumerate(measurements):
            if mi not in matched_meas_idxs:
                self._spawn_track(meas)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    def get_tracks(self) -> List[Track]:
        """Return a snapshot of the current active track list."""
        return list(self.tracks)
