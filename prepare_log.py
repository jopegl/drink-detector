"""
Step 1: clean the raw sensor log and split it into gestures.

Usage:
    python prepare_log.py accel_gryo_log_20261104_183756.txt

Requires: pandas, numpy
Output:   processed_log.csv (one row per sensor reading)

Columns of processed_log.csv
    timestamp           reading time (sub-second part estimated, see below)
    tran_x/y/z          translation (acceleration), first group in the raw log
    gyro_roll/pitch/yaw rotation (degrees), second group in the raw log
    gyro_yaw_unwrapped  yaw without the artificial jumps at +-180 degrees
    tran_mag            movement intensity = sqrt(tran_x^2 + tran_y^2 + tran_z^2)
    t_s                 seconds since the start of the log
    event               gesture number (1..N) or 0 when the hand is still
    label               idle / water / water (+red herring?) / red_herring / unknown
                        (from Andrew's notes; used ONLY to check results, never to train)
"""
import re
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MOVE_THRESHOLD = 5.0   # tran_mag above this = hand is moving
MAX_PAUSE_S = 3.0      # pauses shorter than this do not split a gesture
MIN_DURATION_S = 1.0   # gestures shorter than this are dropped (bumps)
MATCH_TOLERANCE_S = 150

# Andrew's notes (hour:minute). 17:49 was noted as "water and red herring directly after".
NOTES = {
    "16:53": "water", "16:59": "water", "17:19": "water", "17:25": "water",
    "17:49": "water (+red herring?)", "18:10": "water", "18:20": "water",
    "18:30": "water", "18:35": "red_herring", "18:37": "water",
}


def read_raw_log(path):
    pattern = re.compile(
        r"\{time:\s*(\d+),(\d+),(\d+)\s*\|\s*(\d+)/(\d+)/(\d+),\s*"
        r"accel:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*"
        r"gyro:\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\}"
    )
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append(m.groups())
            elif line.strip():
                bad += 1
    cols = ["h", "m", "s", "month", "day", "year",
            "tran_x", "tran_y", "tran_z", "gyro_roll", "gyro_pitch", "gyro_yaw"]
    d = pd.DataFrame(rows, columns=cols)
    for c in cols[6:]:
        d[c] = d[c].astype(float)
    print(f"Rows read: {len(d)} | malformed rows: {bad}")
    return d


def clean(d):
    sec = pd.to_datetime(d.year + "-" + d.month + "-" + d.day + " " + d.h + ":" + d.m + ":" + d.s)
    # The log has no milliseconds: spread the readings of each second evenly inside it.
    n = sec.groupby(sec).cumcount()
    total = sec.groupby(sec).transform("size")
    d["timestamp"] = sec + pd.to_timedelta(n / total, unit="s")
    d["gyro_yaw_unwrapped"] = np.rad2deg(np.unwrap(np.deg2rad(d.gyro_yaw)))
    d["tran_mag"] = np.sqrt(d.tran_x**2 + d.tran_y**2 + d.tran_z**2)
    d["t_s"] = (d.timestamp - d.timestamp.iloc[0]).dt.total_seconds()

    full = pd.date_range(sec.min(), sec.max(), freq="s")
    print(f"Sampling rate: {len(d) / d.t_s.iloc[-1]:.1f} readings/s | "
          f"missing seconds: {len(full.difference(sec.unique()))}")
    cols = ["timestamp", "tran_x", "tran_y", "tran_z", "gyro_roll", "gyro_pitch", "gyro_yaw",
            "gyro_yaw_unwrapped", "tran_mag", "t_s"]
    return d[cols]


def split_gestures(d):
    t = d.t_s.values[d.tran_mag.values > MOVE_THRESHOLD]
    segments, start, prev = [], t[0], t[0]
    for x in t[1:]:
        if x - prev > MAX_PAUSE_S:
            segments.append((start, prev))
            start = x
        prev = x
    segments.append((start, prev))
    segments = [(a, b) for a, b in segments if b - a >= MIN_DURATION_S]

    # Each gesture covers whole seconds: from the second it starts to the end of the
    # second it ends (the raw log only has whole-second timestamps).
    t0 = d.timestamp.iloc[0]
    d["event"] = 0
    for i, (a, b) in enumerate(segments, start=1):
        start = (t0 + pd.Timedelta(seconds=a)).floor("s")
        end = (t0 + pd.Timedelta(seconds=b)).floor("s") + pd.Timedelta(seconds=1)
        d.loc[(d.timestamp >= start) & (d.timestamp < end), "event"] = i
    print(f"Gestures found: {len(segments)}")
    return d


def add_labels(d):
    """Each of Andrew's notes goes to ONE gesture: the one starting closest to it."""
    g = d[d.event > 0].groupby("event").timestamp.first()
    day = d.timestamp.iloc[0].strftime("%Y-%m-%d")
    labels = {e: "unknown" for e in g.index}
    for minute, lab in NOTES.items():
        target = pd.Timestamp(f"{day} {minute}:30")  # middle of the noted minute
        dist = (g - target).abs().dt.total_seconds()
        if dist.min() <= MATCH_TOLERANCE_S:
            labels[dist.idxmin()] = lab
    d["label"] = d.event.map(labels).fillna("idle")
    return d


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python prepare_log.py <raw_log_file>")
    d = add_labels(split_gestures(clean(read_raw_log(sys.argv[1]))))
    summary = d[d.event > 0].groupby("event").agg(
        start=("timestamp", "first"), end=("timestamp", "last"), label=("label", "first"))
    summary["start"] = summary.start.dt.strftime("%H:%M:%S")
    summary["end"] = summary.end.dt.strftime("%H:%M:%S")
    print(summary.to_string())
    d.to_csv("processed_log.csv", index=False, float_format="%.3f", date_format="%Y-%m-%d %H:%M:%S.%f")
    print("Saved: processed_log.csv")
