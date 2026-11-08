"""
Read a raw wrist-sensor log, clean it and split it into gestures.

Raw log line (both label styles are accepted):
    {time: H,M,S | MM/DD/YYYY, accel: x, y, z, gyro: roll, pitch, yaw}
    {time: H,M,S | MM/DD/YYYY, trans: x, y, z, rot: roll, pitch, yaw}

Columns produced
    timestamp           reading time (sub-second part estimated, see clean())
    tran_x/y/z          translation, first group in the raw log
    gyro_roll/pitch/yaw rotation (degrees), second group in the raw log
    gyro_yaw_unwrapped  yaw without the artificial jumps at +-180 degrees
    tran_mag            movement intensity = sqrt(tran_x^2 + tran_y^2 + tran_z^2)
    t_s                 seconds since the start of the log
    event               gesture number (1..N) or 0 when the hand is still (split_gestures)
"""
import re

import numpy as np
import pandas as pd

MOVE_THRESHOLD = 5.0   # tran_mag above this = hand is moving
MAX_PAUSE_S = 3.0      # pauses shorter than this do not split a gesture
MIN_DURATION_S = 1.0   # gestures shorter than this are dropped (bumps)

_PATTERN = re.compile(
    r"\{time:\s*(\d+),(\d+),(\d+)\s*\|\s*(\d+)/(\d+)/(\d+),\s*"
    r"(?:accel|trans):\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+),\s*"
    r"(?:gyro|rot):\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\}"
)


def read_raw_log(path):
    rows, bad = [], 0
    with open(path) as f:
        for line in f:
            m = _PATTERN.search(line)
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


def split_gestures(d, move_threshold=MOVE_THRESHOLD, max_pause_s=MAX_PAUSE_S,
                   min_duration_s=MIN_DURATION_S):
    t = d.t_s.values[d.tran_mag.values > move_threshold]
    segments, start, prev = [], t[0], t[0]
    for x in t[1:]:
        if x - prev > max_pause_s:
            segments.append((start, prev))
            start = x
        prev = x
    segments.append((start, prev))
    segments = [(a, b) for a, b in segments if b - a >= min_duration_s]

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


def save_processed(d, path):
    d.to_csv(path, index=False, float_format="%.3f", date_format="%Y-%m-%d %H:%M:%S.%f")
    print(f"Saved: {path}")


def print_gestures(d):
    s = d[d.event > 0].groupby("event").agg(start=("timestamp", "first"), end=("timestamp", "last"),
                                            label=("label", "first"))
    s["start"] = s.start.dt.strftime("%H:%M:%S")
    s["end"] = s.end.dt.strftime("%H:%M:%S")
    print(s.to_string())
