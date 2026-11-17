"""
Event pipeline for recordings where the hand is rarely still (from experiment 06 on).

The steps are the same for every method, so they live here:
  1. remove_yaw_drift   - the BMI160 has no magnetometer, so yaw drifts at an almost constant
                          rate (~70 deg/min in the 11/11 recording). A straight line is fitted
                          and subtracted.
  2. detect_events      - a small state machine on the movement intensity (1 s mean of tran_mag):
                          REST -> ACTIVE when it goes above HIGH, back to REST when it falls
                          below LOW. Each ACTIVE stretch around a strong peak is one event.
  3. event_features     - a feature card per event. Angles are measured from the median of the
                          few seconds BEFORE the event (local baseline), not from a long rest.
  4. match_truth        - attach the video/notes ground truth to the detected events
                          (used ONLY to evaluate, never to train or group).
  5. evaluate_groups    - score a grouping against the truth.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HIGH = 55.0         # intensity (1 s mean of tran_mag) that starts an event
LOW = 15.0          # intensity below which the event ends
MIN_GAP_S = 20.0    # two peaks closer than this belong to the same event
MAX_SIDE_S = 6.0    # an event extends at most this far before / after its peak
PRE_S = 3.0         # seconds before the event used as the local baseline
GUARD_S = 0.5       # gap between the baseline and the event start
PAD_S = 1.0         # extra seconds around the event when computing features

FEATURES = ["yaw_min", "yaw_max", "roll_min", "roll_max", "pitch_min", "pitch_max",
            "rate_x_min", "rate_x_max", "rate_y_min", "rate_y_max", "rate_z_min", "rate_z_max",
            "peak", "duration_s"]
MOVEMENT_LABELS = ("R", "L")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_processed(path):
    return pd.read_csv(path, parse_dates=["timestamp"])


def remove_yaw_drift(d):
    """Adds yaw_dt (yaw without the linear drift). Returns (d, drift in degrees per minute)."""
    d = d.copy()
    slope, intercept = np.polyfit(d.t_s.values, d.gyro_yaw_unwrapped.values, 1)
    d["yaw_dt"] = d.gyro_yaw_unwrapped - (slope * d.t_s + intercept)
    return d, float(slope * 60)


def intensity(d, win_s=1.0):
    """1 s rolling mean of the movement intensity, aligned with d."""
    return d.set_index("timestamp").tran_mag.rolling(f"{int(win_s * 1000)}ms").mean().values


# ---------------------------------------------------------------------------
# Detection: REST <-> ACTIVE state machine
# ---------------------------------------------------------------------------
def detect_events(d, high=HIGH, low=LOW, min_gap_s=MIN_GAP_S, max_side_s=MAX_SIDE_S):
    s = intensity(d)
    t = d.t_s.values
    peaks = []                                  # indices of the strongest point of each event
    for i in np.where(s > high)[0]:
        if not peaks or t[i] - t[peaks[-1]] > min_gap_s:
            peaks.append(i)
        elif s[i] > s[peaks[-1]]:
            peaks[-1] = i
    rows = []
    for k, p in enumerate(peaks, start=1):
        a = p
        while a > 0 and s[a - 1] > low and t[p] - t[a - 1] <= max_side_s:
            a -= 1
        b = p
        while b < len(s) - 1 and s[b + 1] > low and t[b + 1] - t[p] <= max_side_s:
            b += 1
        rows.append({"event": k, "start": d.timestamp.iloc[a], "end": d.timestamp.iloc[b],
                     "peak_time": d.timestamp.iloc[p], "peak": float(s[p])})
    return pd.DataFrame(rows, columns=["event", "start", "end", "peak_time", "peak"])


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def event_features(d, E, pre_s=PRE_S, guard_s=GUARD_S, pad_s=PAD_S):
    rows = []
    for e in E.itertuples():
        pre = d[(d.timestamp >= e.start - pd.Timedelta(seconds=pre_s + guard_s))
                & (d.timestamp < e.start - pd.Timedelta(seconds=guard_s))]
        w = d[(d.timestamp >= e.start - pd.Timedelta(seconds=pad_s))
              & (d.timestamp <= e.end + pd.Timedelta(seconds=pad_s))]
        if len(pre) < 3:
            pre = w.iloc[:5]
        base = {c: pre[c].median() for c in ["yaw_dt", "gyro_roll", "gyro_pitch"]}
        y, r, p = w.yaw_dt - base["yaw_dt"], w.gyro_roll - base["gyro_roll"], w.gyro_pitch - base["gyro_pitch"]
        rows.append({"event": e.event,
                     "yaw_min": y.min(), "yaw_max": y.max(), "roll_min": r.min(), "roll_max": r.max(),
                     "pitch_min": p.min(), "pitch_max": p.max(),
                     "rate_x_min": w.tran_x.min(), "rate_x_max": w.tran_x.max(),
                     "rate_y_min": w.tran_y.min(), "rate_y_max": w.tran_y.max(),
                     "rate_z_min": w.tran_z.min(), "rate_z_max": w.tran_z.max(),
                     "duration_s": (e.end - e.start).total_seconds()})
    return E.merge(pd.DataFrame(rows), on="event")


def relative_angles_around(d, start, end, pad_s=PAD_S, pre_s=PRE_S, guard_s=GUARD_S):
    """Angles relative to the local baseline, from start - pad to end + pad (used by DTW)."""
    pre = d[(d.timestamp >= start - pd.Timedelta(seconds=pre_s + guard_s))
            & (d.timestamp < start - pd.Timedelta(seconds=guard_s))]
    w = d[(d.timestamp >= start - pd.Timedelta(seconds=pad_s)) & (d.timestamp <= end + pd.Timedelta(seconds=pad_s))]
    if len(pre) < 3:
        pre = w.iloc[:5]
    out = pd.DataFrame({"yaw": w.yaw_dt - pre.yaw_dt.median(),
                        "roll": w.gyro_roll - pre.gyro_roll.median(),
                        "pitch": w.gyro_pitch - pre.gyro_pitch.median(),
                        "rate_x": w.tran_x, "rate_y": w.tran_y, "rate_z": w.tran_z})
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ground truth (video + Andrew's notes) - evaluation only
# ---------------------------------------------------------------------------
def load_truth(path, day):
    T = pd.read_csv(path, dtype=str).fillna("")
    to_ts = lambda s: pd.Timestamp(f"{day} {s}") if s else pd.NaT  # noqa: E731
    T["start"] = T.log_start.map(to_ts)
    T["end"] = T.log_end.map(to_ts)
    return T


def match_truth(E, T, day, tol_s=2.0):
    """
    Each truth movement goes to ONE detected event: the strongest event overlapping
    [start - tol, end + tol]. If the video gave no time, Andrew's minute is used instead.
    """
    E = E.copy()
    E["truth_id"], E["truth_label"] = "—", "—"
    for r in T.itertuples():
        if pd.notna(r.start) or pd.notna(r.end):
            a = (r.start if pd.notna(r.start) else r.end - pd.Timedelta(seconds=4)) - pd.Timedelta(seconds=tol_s)
            b = (r.end if pd.notna(r.end) else r.start + pd.Timedelta(seconds=4)) + pd.Timedelta(seconds=tol_s)
        else:
            minute = r.andrew_note.split(" ")[0]
            a = pd.Timestamp(f"{day} {minute}:00")
            b = a + pd.Timedelta(seconds=59)
        cand = E[(E.start <= b) & (E.end >= a) & (E.truth_id == "—")]
        if len(cand):
            i = cand.peak.idxmax()
            E.loc[i, "truth_id"] = r.id
            E.loc[i, "truth_label"] = r.label
    return E


def detection_summary(E, T):
    found = int(E.truth_id.isin(T.id).sum())
    movements = T[T.label.isin(MOVEMENT_LABELS)]
    return {"truth_total": len(T), "truth_found": found,
            "rl_total": len(movements), "rl_found": int(E.truth_id.isin(movements.id).sum()),
            "events": len(E), "events_without_truth": int((E.truth_id == "—").sum())}


# ---------------------------------------------------------------------------
# Evaluation of a grouping (labels only used here)
# ---------------------------------------------------------------------------
def evaluate_groups(E, group_col="group"):
    """
    Each group is named after the majority R/L label among its members (evaluation only).
    Returns accuracy on the R/L movements, the adjusted Rand index and the mapping.
    """
    from sklearn.metrics import adjusted_rand_score

    rl = E[E.truth_label.isin(MOVEMENT_LABELS)]
    mapping = {}
    for g, x in E.groupby(group_col):
        votes = x.truth_label[x.truth_label.isin(MOVEMENT_LABELS)].value_counts()
        if str(g) == "-1":
            mapping[g] = "unusual"          # DBSCAN noise is never named R or L
        elif len(votes) == 0:
            mapping[g] = "no R/L"
        elif len(votes) > 1 and votes.iloc[0] == votes.iloc[1]:
            mapping[g] = "tie"
        else:
            mapping[g] = votes.index[0]
    pred = rl[group_col].map(mapping)
    correct = int((pred == rl.truth_label).sum())
    ari = float(adjusted_rand_score(rl.truth_label, rl[group_col])) if len(rl) > 1 else float("nan")
    return {"correct": correct, "n": len(rl), "accuracy": correct / len(rl) if len(rl) else float("nan"),
            "ari": ari, "mapping": {str(k): v for k, v in mapping.items()}}


def where_others_went(E, group_col="group", mapping=None):
    """Where the unlabeled movements and the red herrings ended up."""
    x = E[~E.truth_label.isin(MOVEMENT_LABELS) & (E.truth_label != "—")]
    out = {}
    for r in x.itertuples():
        g = getattr(r, group_col)
        out[r.truth_id] = {"truth": r.truth_label, "group": str(g),
                           "named_as": (mapping or {}).get(str(g), "")}
    return out


# ---------------------------------------------------------------------------
# Common preparation used by every build_results.py
# ---------------------------------------------------------------------------
def load_experiment_data(data_dir, day, high=HIGH):
    """Processed log (with yaw_dt), detected events with features and truth, and the truth table."""
    data_dir = Path(data_dir)
    d = load_processed(data_dir / "processed_log.csv")
    if "yaw_dt" not in d:
        d, _ = remove_yaw_drift(d)
    T = load_truth(data_dir / "annotations.csv", day)
    E = match_truth(event_features(d, detect_events(d, high=high)), T, day)
    return d, E, T


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.strftime("%H:%M:%S")
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=conv)


def save_events(E, path):
    out = E.copy()
    for c in ["start", "end", "peak_time"]:
        out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    out.round(3).to_csv(path, index=False)


def load_events(path):
    return pd.read_csv(path, parse_dates=["start", "end", "peak_time"], dtype={"truth_id": str, "group": str})
