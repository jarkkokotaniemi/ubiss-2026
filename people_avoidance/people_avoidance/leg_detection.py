"""
leg_detection.py — Stage 1 of the people-avoidance pipeline.

Input : sensor_msgs/LaserScan
Output: List[LegMeasurement]

Segmentation  — 2nd derivative of the range signal in polar coordinates.
                A large |r''| means the surface curved sharply → new segment.

Leg detection — within each segment, find zero-crossings of r'' (the
                inflection points of the range curve).  A concave bump
                (r'' goes +→−) marks the closest point of a cylindrical
                leg.  Pairs of such bumps within max_leg_width are
                reported as a person.

FFT           — scan_fft() returns the magnitude spectrum of the full
                360° range signal so you can inspect it externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np
from sensor_msgs.msg import LaserScan


@dataclass
class LegMeasurement:
    """
    One detected person expressed as a 2-D position with observation covariance.

    Coordinate frame: the laser frame (x forward, y left).

        R = [[Rxx, Rxy],
             [Rxy, Ryy]]
    """

    x: float
    y: float
    Rxx: float
    Rxy: float
    Ryy: float


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def scan_to_cartesian(scan: LaserScan) -> np.ndarray:
    """Convert LaserScan polar readings to (x, y) Cartesian in the laser frame."""
    angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
    ranges = np.array(scan.ranges, dtype=float)
    valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
    r = ranges[valid]
    a = angles[valid]
    return np.column_stack((r * np.cos(a), r * np.sin(a)))  # (N, 2)


def scan_fft(scan: LaserScan) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the magnitude spectrum of the full range signal.

    Invalid ranges are replaced by linear interpolation so the FFT sees a
    continuous signal without NaN / inf spikes.

    Returns
    -------
    freqs : spatial frequencies in cycles-per-beam  (length N//2 + 1)
    mags  : magnitude of each frequency bin          (same length)

    Usage example (in your offline visualizer or a notebook):
        freqs, mags = scan_fft(scan)
        plt.plot(freqs, mags)
        plt.xlabel("cycles / beam"); plt.ylabel("|FFT|")
    """
    r = np.array(scan.ranges, dtype=float)

    # Replace invalid readings with interpolated values
    invalid = ~(np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max))
    if invalid.all():
        n = len(r)
        return np.fft.rfftfreq(n), np.zeros(n // 2 + 1)
    if invalid.any():
        idx = np.arange(len(r))
        r[invalid] = np.interp(idx[invalid], idx[~invalid], r[~invalid])

    window = np.hanning(len(r))
    spectrum = np.fft.rfft(r * window)
    mags = np.abs(spectrum) * 2.0 / window.sum()  # amplitude-corrected
    freqs = np.fft.rfftfreq(len(r))
    return freqs, mags


# ---------------------------------------------------------------------------
# Stage 1a — curvature-based segmentation
# ---------------------------------------------------------------------------


def _range_second_derivative(ranges: np.ndarray) -> np.ndarray:
    """
    Central-difference 2nd derivative of the range array.

    r''[i] = r[i+1] - 2*r[i] + r[i-1]

    Edge values are set to 0 (no curvature assumed at the boundary).
    """
    d2 = np.zeros_like(ranges)
    d2[1:-1] = ranges[2:] - 2.0 * ranges[1:-1] + ranges[:-2]
    return d2


# ---------------------------------------------------------------------------
# Stage 1b — zero-crossing leg detection
# ---------------------------------------------------------------------------


def _zero_crossings(d2: np.ndarray, kind: str = "pos_to_neg") -> np.ndarray:
    """
    Indices where the 2nd derivative crosses zero.

    kind="pos_to_neg"  →  d2 goes from positive to negative  (concave peak,
                          i.e. the closest point of a curved surface toward
                          the sensor — a leg bump).
    kind="any"         →  all zero-crossings.

    Returns array of crossing indices (the index just before the crossing).
    """
    if kind == "pos_to_neg":
        crossings = np.where((d2[:-1] > 0) & (d2[1:] <= 0))[0]
    else:
        crossings = np.where(np.sign(d2[:-1]) != np.sign(d2[1:]))[0]
    return crossings


# Module-level state for previous-frame predictions
_prev_detections: List[np.ndarray] = []  # list of (r, theta) polar coords


def _range_second_derivative(ranges: np.ndarray) -> np.ndarray:
    """
    Central-difference 2nd derivative of the range array.

    r''[i] = r[i+1] - 2*r[i] + r[i-1]

    Edge values are set to 0 (no curvature assumed at the boundary).
    """
    d2 = np.zeros_like(ranges)
    d2[1:-1] = ranges[2:] - 2.0 * ranges[1:-1] + ranges[:-2]
    return d2


def segment_scan(
    points: np.ndarray,
    distance_threshold: float = 0.1,
    scan: LaserScan = None,
    curv_threshold: float = 0.15,
) -> List[np.ndarray]:
    """
    Split the scan into contiguous segments using the 2nd derivative of range.
    Now operates on polar (r, theta) arrays extracted from the scan directly.

    A segment boundary is placed wherever |r''[i]| > curv_threshold, meaning
    the range signal curved sharply — a wall corner, the edge of a leg, etc.

    The Euclidean distance fallback (original method) is also applied so that
    large physical gaps always break segments even if curvature is low.

    Args:
        points:             (N, 2) polar (r, theta) scan points in scan order.
        distance_threshold: Maximum range gap before a new segment (m).
        scan:               Original LaserScan message. When supplied the
                            curvature method is active; otherwise falls back
                            to the pure range-gap method.
        curv_threshold:     |r''| threshold for a segment break (m).
                            Tune this: lower -> more splits, higher -> fewer.

    Returns:
        List of (K_i, 2) polar arrays, one per segment.
    """
    if points.size == 0:
        return []

    N = len(points)

    # points[:, 0] is r, points[:, 1] is theta (polar coords)
    ranges = points[:, 0]

    # ── Curvature-based breaks ────────────────────────────────────────────────
    if scan is not None:
        d2 = _range_second_derivative(ranges)
        curv_breaks = np.where(np.abs(d2) > curv_threshold)[0]
    else:
        curv_breaks = np.array([], dtype=int)

    # ── Range-gap breaks (replaces Euclidean gap in polar space) ─────────────
    range_diffs = np.abs(np.diff(ranges))
    gap_breaks = np.where(range_diffs > distance_threshold)[0] + 1

    # ── Merge and split ───────────────────────────────────────────────────────
    all_breaks = np.union1d(curv_breaks, gap_breaks)
    all_breaks = all_breaks[(all_breaks > 0) & (all_breaks < N)]

    segments = np.split(points, all_breaks)

    return [s for s in segments if len(s) > 3]


def _angular_distance_to_prev(
    r: float,
    theta: float,
    prev_detections: List[np.ndarray],
    max_range: float,
) -> float:
    """
    Return the minimum predicted angular distance from (r, theta) to any
    previous detection (propagated forward as stationary, i.e. same polar
    position). Returns infinity if no previous detections exist.

    Args:
        r:               Range of current point (m).
        theta:           Bearing of current point (rad).
        prev_detections: List of (r_prev, theta_prev) arrays from last frame.
        max_range:       Used to normalise; ignored here but kept for signature.

    Returns:
        Minimum angular separation (rad) to the nearest previous detection.
    """
    if not prev_detections:
        return float("inf")

    min_dist = float("inf")
    for pd in prev_detections:
        r_p, theta_p = float(pd[0]), float(pd[1])
        # Angular distance in polar (wrap-safe)
        d_theta = abs(theta - theta_p)
        if d_theta > math.pi:
            d_theta = 2 * math.pi - d_theta
        # Scale by range so the neighbourhood is metric (~arc length)
        arc = r * d_theta
        min_dist = min(min_dist, arc)

    return min_dist


def detect_legs(
    scan: LaserScan,
    distance_threshold: float = 0.1,
    leg_radius: float = 0.10,
    max_leg_width: float = 2,
    curv_threshold: float = 2,
    min_apex_angle_deg: float = 90.0,
    apex_window: int = 10,
    max_flatness: float = 0.02,
    max_range: float = 2.5,
    prediction_radius: float = 0.3,
    prediction_curv_scale: float = 0.7,
    prediction_flatness_scale: float = 0.7,
    prediction_angle_scale: float = 0.8,
) -> List[LegMeasurement]:
    """
    Detect people using 2nd-derivative segmentation + zero-crossing leg bumps.
    All intermediate processing is done in polar coordinates (r, theta).

    Pipeline
    --------
    1. Build polar array (r, theta) directly from scan ranges + angle info.
    2. segment_scan()        -> segments split on |r''| > curv_threshold.
    3. Per segment: compute r'' of the range values (already polar r).
       Find pos->neg zero-crossings — each is a leg-bump apex candidate.
    4. For each apex candidate, take a local window of +-apex_window points.
       Apply four filters:
         a. Max range: apex range must be <= max_range.
         b. "Away from robot": apex range < both window-endpoint ranges
            (bump protrudes toward sensor, not a concave wall corner).
         c. Apex angle >= min_apex_angle_deg: measured in Cartesian at the
            apex only for the angle test (arc geometry), using window
            endpoints converted on-the-fly.
         d. Curvature / straightness: max perpendicular deviation of points
            in the window computed in polar arc-length space.
       If the apex falls within prediction_radius (metric arc) of a previous
       detection, thresholds for (c) and (d) are relaxed by their respective
       scale factors.
    5. Pair leg candidates <= max_leg_width apart (arc distance); person =
       midpoint in polar, then converted to Cartesian for output.
    6. Assign range-dependent isotropic covariance.
    7. Store current detections as _prev_detections for next frame.

    New Parameters
    --------------
    max_range            : Legs beyond this range (m) are ignored.
    prediction_radius    : Arc-distance (m) within which a candidate is
                           considered "near a previous detection" and gets
                           relaxed thresholds.
    prediction_curv_scale: Multiply max_flatness by 1/this when near a
                           previous detection (lower floor -> easier pass).
                           Values < 1 make flatness test easier.
    prediction_flatness_scale: Multiply max_flatness by this when near a
                           previous detection (< 1 -> easier to pass).
    prediction_angle_scale:  Multiply min_apex_angle_deg by this when near
                           a previous detection (< 1 -> easier to pass).
    """
    global _prev_detections

    # ── Build polar array directly from scan ──────────────────────────────────
    raw_ranges = np.array(scan.ranges, dtype=float)
    N_raw = len(raw_ranges)
    angles = scan.angle_min + np.arange(N_raw) * scan.angle_increment

    valid_mask = (
        np.isfinite(raw_ranges)
        & (raw_ranges > scan.range_min)
        & (raw_ranges < scan.range_max)
    )
    r_valid = raw_ranges[valid_mask]
    theta_valid = angles[valid_mask]

    # polar_points[:, 0] = r,  polar_points[:, 1] = theta
    polar_points = np.column_stack((r_valid, theta_valid))

    if polar_points.shape[0] == 0:
        _prev_detections = []
        return []

    segments = segment_scan(
        polar_points,
        distance_threshold=distance_threshold,
        scan=scan,
        curv_threshold=curv_threshold,
    )

    min_apex_angle_rad = np.deg2rad(min_apex_angle_deg)
    leg_candidates_polar: List[np.ndarray] = []  # each entry: (r, theta)

    for seg in segments:
        if len(seg) < 4 or len(seg) > 150:
            # print("segment length rejected: ", len(seg))
            continue

        r_seg = seg[:, 0]  # ranges in polar segment
        theta_seg = seg[:, 1]  # bearings

        d2 = _range_second_derivative(r_seg)
        crossings = _zero_crossings(d2, kind="pos_to_neg")

        for ci in crossings:
            r_apex = r_seg[ci]
            theta_apex = theta_seg[ci]

            # ── (a) Max range filter ───────────────────────────────────────────
            if r_apex > max_range:
                continue

            # ── Check if near a previous detection (polar arc distance) ────────
            near_prev = (
                _angular_distance_to_prev(
                    r_apex, theta_apex, _prev_detections, max_range
                )
                <= prediction_radius
            )

            # Relaxed thresholds when near a previous detection
            eff_flatness = max_flatness * (
                prediction_flatness_scale if near_prev else 1.0
            )
            eff_angle_deg = min_apex_angle_deg * (
                prediction_angle_scale if near_prev else 1.0
            )

            # ── Local window around the crossing ──────────────────────────────
            i_lo = max(0, ci - apex_window)
            i_hi = min(len(seg) - 1, ci + apex_window)
            if i_hi <= i_lo:
                continue

            r_lo = r_seg[i_lo]
            r_hi = r_seg[i_hi]
            theta_lo = theta_seg[i_lo]
            theta_hi = theta_seg[i_hi]

            # ── (b) Away-from-robot in polar: apex r < both endpoint r ────────
            if not (r_apex < r_lo and r_apex < r_hi):
                continue

            # ── (c) Apex angle in local Cartesian (on-the-fly conversion) ─────
            # Convert only these three points to Cartesian for angle geometry
            apex_xy = np.array(
                [r_apex * math.cos(theta_apex), r_apex * math.sin(theta_apex)]
            )
            p_lo_xy = np.array([r_lo * math.cos(theta_lo), r_lo * math.sin(theta_lo)])
            p_hi_xy = np.array([r_hi * math.cos(theta_hi), r_hi * math.sin(theta_hi)])

            v_lo = p_lo_xy - apex_xy
            v_hi = p_hi_xy - apex_xy
            len_lo = np.linalg.norm(v_lo)
            len_hi = np.linalg.norm(v_hi)
            if len_lo < 1e-6 or len_hi < 1e-6:
                continue

            cos_a = float(np.clip(np.dot(v_lo, v_hi) / (len_lo * len_hi), -1.0, 1.0))
            apex_angle = math.acos(cos_a)
            if apex_angle < np.deg2rad(eff_angle_deg) or apex_angle > np.deg2rad(175):
                # print("Angle: ", np.rad2deg(apex_angle), " Rejected")
                continue

            # ── (d) Flatness rejection in polar arc-length space ───────────────
            # Arc length between consecutive polar points approximated as
            # r * delta_theta.  We project perpendicular to the chord in
            # (arc_cumulative, r) 2-D space — a coordinate that stays polar.
            window_r = r_seg[i_lo : i_hi + 1]
            window_theta = theta_seg[i_lo : i_hi + 1]

            # Cumulative arc length along the window
            d_theta = np.diff(window_theta)
            r_mid = 0.5 * (window_r[:-1] + window_r[1:])
            arc_steps = np.abs(r_mid * d_theta)
            arc_cum = np.concatenate(([0.0], np.cumsum(arc_steps)))

            # 2-D space: (arc, r).  Perpendicular deviation from chord.
            pts_2d = np.column_stack((arc_cum, window_r))
            chord_2d = pts_2d[-1] - pts_2d[0]
            chord_len_2d = np.linalg.norm(chord_2d)
            if chord_len_2d < 1e-6:
                continue
            chord_unit_2d = chord_2d / chord_len_2d
            vecs_2d = pts_2d - pts_2d[0]
            perp = np.abs(
                vecs_2d[:, 0] * chord_unit_2d[1] - vecs_2d[:, 1] * chord_unit_2d[0]
            )
            if perp.max() < eff_flatness:
                continue

            leg_candidates_polar.append(np.array([r_apex, theta_apex]))

    # ── Pair candidates into person detections (polar arc distance) ───────────
    measurements: List[LegMeasurement] = []
    new_prev_detections: List[np.ndarray] = []
    paired: set = set()

    for i in range(len(leg_candidates_polar)):
        if i in paired:
            continue
        r_i = leg_candidates_polar[i][0]
        theta_i = leg_candidates_polar[i][1]
        for j in range(i + 1, len(leg_candidates_polar)):
            if j in paired:
                continue
            r_j = leg_candidates_polar[j][0]
            theta_j = leg_candidates_polar[j][1]

            # Arc distance between the two candidates
            d_theta_ij = abs(theta_i - theta_j)
            if d_theta_ij > math.pi:
                d_theta_ij = 2 * math.pi - d_theta_ij
            arc_dist = 0.5 * (r_i + r_j) * d_theta_ij  # mean-r arc approx

            if arc_dist <= max_leg_width:
                # Midpoint in polar then convert to Cartesian for output
                r_mid = 0.5 * (r_i + r_j)
                theta_mid = 0.5 * (theta_i + theta_j)  # simple angular mean
                x_mid = r_mid * math.cos(theta_mid)
                y_mid = r_mid * math.sin(theta_mid)
                paired.add(i)
                paired.add(j)

                rng = r_mid
                r_val = float(max((rng**2) * 0.01, 0.001))

                measurements.append(
                    LegMeasurement(
                        x=float(x_mid),
                        y=float(y_mid),
                        Rxx=r_val,
                        Rxy=0.0,
                        Ryy=r_val,
                    )
                )
                # Store midpoint in polar for next frame prediction
                new_prev_detections.append(np.array([r_mid, theta_mid]))
                break

    # ── Persist detections for next frame ─────────────────────────────────────
    _prev_detections = new_prev_detections

    return measurements
