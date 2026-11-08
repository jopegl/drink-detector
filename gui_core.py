"""
Logic behind the interface (nothing visual): hand rotation, estimated path and detection.
Used by app.py.
"""
import io

import numpy as np
import pandas as pd

import kmeans_windows as kw


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        path_or_bytes = io.BytesIO(path_or_bytes)
    return pd.read_csv(path_or_bytes, parse_dates=["timestamp"])


def gestures(d):
    """The movement segments from the 'event' column, with start, end and label."""
    return (d[d.event > 0]
            .groupby("event")
            .agg(start=("timestamp", "first"), end=("timestamp", "last"), label=("label", "first"))
            .reset_index())


# ---------------------------------------------------------------------------
# Detection (k-means on windows)
# ---------------------------------------------------------------------------
def detect(d, window_s=4.0, k=3):
    W = kw.make_windows(d, window_s)
    groups, names = kw.run_kmeans(W, k=k, verbose=False)
    W["group"] = groups
    W["class"] = W.group.map(names)
    day = d.timestamp.iloc[0].strftime("%Y-%m-%d")
    E, found, rh, extras = kw.compare(kw.detected_events(W), day)
    return W, E, found, rh, extras


def flag_detected_gestures(G, E):
    """For each gesture, say whether any detected sip overlaps it."""
    G = G.copy()
    G["detected_as_sip"] = [bool(((E.start <= g.end) & (E.end >= g.start)).any()) if len(E) else False
                            for _, g in G.iterrows()]
    return G


# ---------------------------------------------------------------------------
# Rotation: roll, pitch, yaw (degrees) -> rotation matrix
# Z-Y-X convention (yaw, then pitch, then roll), the most common one for IMUs.
# ---------------------------------------------------------------------------
def rotation_matrices(roll, pitch, yaw):
    r, p, y = (np.deg2rad(np.asarray(v, dtype=float)) for v in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty((len(r), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


# A simple "hand": flat palm (box) + a finger pointing forward (+x)
_PALM = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.35, 0.35) for z in (-0.06, 0.06)])
_TRI_I = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
_TRI_J = [1, 3, 5, 7, 1, 5, 3, 7, 2, 6, 3, 7]
_TRI_K = [3, 2, 7, 6, 5, 4, 7, 6, 6, 4, 7, 5]
_FINGER = np.array([[0.5, 0.0, 0.0], [1.1, 0.0, 0.0]])
_AXES = np.eye(3) * 0.9  # sensor x, y, z

TRIANGLES = (_TRI_I, _TRI_J, _TRI_K)


def hand_geometry(R):
    """Apply one rotation (3x3) to the hand. Returns palm vertices, finger and axes."""
    return _PALM @ R.T, _FINGER @ R.T, _AXES @ R.T


# ---------------------------------------------------------------------------
# Estimated path from the acceleration (tran)
# ---------------------------------------------------------------------------
def estimated_path(segment):
    """
    Estimate the hand path over a short segment (one gesture).
    1. Rotate the sensor acceleration into the "world" frame using the orientation.
    2. Integrate once -> velocity. Force zero velocity at the start and end
       (the hand is still before and after the gesture) by removing the linear drift.
    3. Integrate again -> position.
    The result is APPROXIMATE and in arbitrary units: errors build up with each integration.
    """
    t = (segment.timestamp - segment.timestamp.iloc[0]).dt.total_seconds().values
    dt = np.diff(t, prepend=t[0])
    R = rotation_matrices(segment.gyro_roll, segment.gyro_pitch, segment.gyro_yaw)
    a_world = np.einsum("nij,nj->ni", R, segment[["tran_x", "tran_y", "tran_z"]].values)
    v = np.cumsum(a_world * dt[:, None], axis=0)
    if t[-1] > 0:
        v = v - np.outer(t / t[-1], v[-1])
    p = np.cumsum(v * dt[:, None], axis=0)
    return pd.DataFrame({"t_s": t, "x": p[:, 0], "y": p[:, 1], "z": p[:, 2]})


# ---------------------------------------------------------------------------
# Pre-computed results (written by build_results.py)
# ---------------------------------------------------------------------------
def load_detection(results_dir, window_s, k):
    """Detected events, window cross-table and group means for one (window, k) setting."""
    tag = f"w{int(window_s)}_k{int(k)}"
    E = pd.read_csv(f"{results_dir}/events_{tag}.csv", parse_dates=["start", "end"])
    crosstab = pd.read_csv(f"{results_dir}/crosstab_{tag}.csv")
    means = pd.read_csv(f"{results_dir}/group_means_{tag}.csv")
    found = E[E.andrew_note.str.startswith("water")]
    rh = E[E.andrew_note.str.startswith("red_herring")]
    extras = E[E.andrew_note == "—"]
    return E, crosstab, means, found, rh, extras


def load_analysis(results_dir):
    import json
    with open(f"{results_dir}/summary.json") as f:
        summary = json.load(f)
    return {
        "summary": summary,
        "robustness": pd.read_csv(f"{results_dir}/window_robustness.csv"),
        "gesture_summary": pd.read_csv(f"{results_dir}/gesture_kmeans_summary.csv"),
        "gestures": pd.read_csv(f"{results_dir}/gestures.csv"),
        "removal": pd.read_csv(f"{results_dir}/feature_removal.csv"),
    }
