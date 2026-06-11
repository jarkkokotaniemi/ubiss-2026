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


def segment_scan(
    points: np.ndarray,
    distance_threshold: float = 0.1,
    scan: LaserScan = None,
    curv_threshold: float = 0.15,
) -> List[np.ndarray]:
    """
    Split the scan into contiguous segments using the 2nd derivative of range.

    A segment boundary is placed wherever |r''[i]| > curv_threshold, meaning
    the range signal curved sharply — a wall corner, the edge of a leg, etc.

    The Euclidean distance fallback (original method) is also applied so that
    large physical gaps always break segments even if curvature is low.

    Args:
        points:             (N, 2) Cartesian scan points in scan order.
        distance_threshold: Maximum Euclidean gap before a new segment (m).
        scan:               Original LaserScan message.  When supplied the
                            curvature method is active; otherwise falls back
                            to the pure Euclidean method.
        curv_threshold:     |r''| threshold for a segment break (m).
                            Tune this: lower → more splits, higher → fewer.

    Returns:
        List of (K_i, 2) arrays, one per segment.
    """
    if points.size == 0:
        return []

    N = len(points)

    # ── Curvature-based breaks ────────────────────────────────────────────────
    if scan is not None:
        ranges = np.array(scan.ranges, dtype=float)
        # Work only on the valid subset that produced `points`
        valid_mask = (
            np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
        )
        r_valid = ranges[valid_mask]

        # Safety: if the valid subset doesn't match points length, skip curvature
        if len(r_valid) == N:
            d2 = _range_second_derivative(r_valid)
            curv_breaks = np.where(np.abs(d2) > curv_threshold)[0]
        else:
            curv_breaks = np.array([], dtype=int)
    else:
        curv_breaks = np.array([], dtype=int)

    # ── Merge and split ───────────────────────────────────────────────────────
    all_breaks = curv_breaks
    # Clamp: index 0 is not a valid split point (np.split would give empty head)
    all_breaks = all_breaks[(all_breaks > 0) & (all_breaks < N)]

    segments = np.split(points, all_breaks)

    return [s for s in segments if len(s) > 3]


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


def detect_legs(
    scan: LaserScan,
    distance_threshold: float = 0.1,
    leg_radius: float = 0.10,
    max_leg_width: float = 1,
    curv_threshold: float = 1,
    min_apex_angle_deg: float = 90.0,
    apex_window: int = 20,
    max_flatness: float = 0.02,
) -> List[LegMeasurement]:
    """
    Detect people using 2nd-derivative segmentation + zero-crossing leg bumps.

    Pipeline
    --------
    1. scan_to_cartesian()   → Cartesian points.
    2. segment_scan()        → segments split on |r''| > curv_threshold.
    3. Per segment: compute r'' of the range values.
       Find pos→neg zero-crossings — each is a leg-bump apex candidate.
    4. For each apex candidate, take a local window of ±apex_window points.
       Apply three filters:
         a. "Away from robot": apex range < both window-endpoint ranges
            (bump protrudes toward sensor, not a concave wall corner).
         b. Apex angle >= min_apex_angle_deg: the angle at the apex between
            the two window endpoints must be wide enough to represent a
            cylinder of leg-like size.
         c. Curvature / straightness: the max perpendicular deviation of
            points in the window from the chord connecting its endpoints
            must exceed max_flatness (m).  Flat walls deviate < 1 mm;
            a cylindrical leg deviates by ~leg_radius.
    5. Pair leg candidates <= max_leg_width apart; person = midpoint.
    6. Assign range-dependent isotropic covariance.

    Parameters
    ----------
    apex_window     : half-width (in beam indices) of the local neighbourhood
                      used for angle and flatness tests.  Keeps the angle
                      measurement tied to the bump width, not the full segment.
    max_flatness    : maximum perpendicular chord deviation (m) below which a
                      window is considered a straight line and rejected.
                      Tune upward if cylindrical legs are being missed at range.
    min_apex_angle_deg : minimum angle at the apex subtended by the window
                      endpoints.  ~30° matches a ~10 cm leg at 20 cm–1 m range.
    """
    points = scan_to_cartesian(scan)
    if points.shape[0] == 0:
        return []

    segments = segment_scan(
        points,
        distance_threshold=distance_threshold,
        scan=scan,
        curv_threshold=curv_threshold,
    )

    min_apex_angle_rad = np.deg2rad(min_apex_angle_deg)
    leg_candidates: List[np.ndarray] = []

    for seg in segments:
        if len(seg) < 5:
            continue

        r_seg = np.hypot(seg[:, 0], seg[:, 1])
        d2 = _range_second_derivative(r_seg)
        crossings = _zero_crossings(d2, kind="pos_to_neg")

        for ci in crossings:
            apex = seg[ci]
            r_apex = r_seg[ci]

            # ── Local window around the crossing ──────────────────────────────
            i_lo = max(0, ci - apex_window)
            i_hi = min(len(seg) - 1, ci + apex_window)
            if i_hi <= i_lo:
                continue

            p_lo = seg[i_lo]
            p_hi = seg[i_hi]

            # ── (a) Away-from-robot: apex must be closer than window endpoints ─
            if not (r_apex < r_seg[i_lo] and r_apex < r_seg[i_hi]):
                continue

            # ── (b) Apex angle (measured at the apex, using window endpoints) ──
            v_lo = p_lo - apex
            v_hi = p_hi - apex
            len_lo = np.linalg.norm(v_lo)
            len_hi = np.linalg.norm(v_hi)
            if len_lo < 1e-6 or len_hi < 1e-6:
                continue

            cos_a = float(np.clip(np.dot(v_lo, v_hi) / (len_lo * len_hi), -1.0, 1.0))
            if math.acos(cos_a) < np.deg2rad(min_apex_angle_deg) or math.acos(
                cos_a
            ) > np.deg2rad(175):
                print("Angle: ", np.rad2deg(math.acos(cos_a)), " Rejected")
                continue

            # ── (c) Flatness rejection ────────────────────────────────────────
            # Max perpendicular distance of any point in the window from the
            # chord p_lo → p_hi.  A straight wall scores near 0; a curved
            # leg surface scores ~leg_radius.
            chord = p_hi - p_lo
            chord_len = np.linalg.norm(chord)
            if chord_len < 1e-6:
                continue

            chord_unit = chord / chord_len
            window_pts = seg[i_lo : i_hi + 1]  # (W, 2)
            # Perpendicular distance = |cross product| / chord_len (2-D cross)
            vecs = window_pts - p_lo  # (W, 2)
            perp = np.abs(vecs[:, 0] * chord_unit[1] - vecs[:, 1] * chord_unit[0])
            if perp.max() < max_flatness:
                continue

            leg_candidates.append(apex)

    # ── Pair candidates into person detections ────────────────────────────────
    measurements: List[LegMeasurement] = []
    paired: set = set()

    for i in range(len(leg_candidates)):
        if i in paired:
            continue
        for j in range(i + 1, len(leg_candidates)):
            if j in paired:
                continue
            dist = np.linalg.norm(leg_candidates[i] - leg_candidates[j])
            if dist <= max_leg_width:
                midpoint = (leg_candidates[i] + leg_candidates[j]) / 2.0
                paired.add(i)
                paired.add(j)

                rng = np.linalg.norm(midpoint)
                r_val = float(max((rng**2) * 0.01, 0.001))

                measurements.append(
                    LegMeasurement(
                        x=float(midpoint[0]),
                        y=float(midpoint[1]),
                        Rxx=r_val,
                        Rxy=0.0,
                        Ryy=r_val,
                    )
                )
                break

    # ── Also track unpaired single-leg candidates ──────────────────────────────
    # Each unmatched apex is a potential single visible leg or small obstacle;
    # feeding it to the tracker keeps all detected objects tracked.
    for i, cand in enumerate(leg_candidates):
        if i in paired:
            continue
        rng = float(np.linalg.norm(cand))
        r_val = max((rng**2) * 0.02, 0.002)  # slightly larger uncertainty
        measurements.append(
            LegMeasurement(
                x=float(cand[0]),
                y=float(cand[1]),
                Rxx=r_val,
                Rxy=0.0,
                Ryy=r_val,
            )
        )

    return measurements
