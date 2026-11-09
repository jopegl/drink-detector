"""
Experiment 02, step 1: clean the raw log (right / left movements) and split it into gestures.

Usage (from anywhere):
    python experiments/experiment_02_right_left_movements/prepare.py
    python experiments/experiment_02_right_left_movements/prepare.py <raw_log_file>

Default input:  data/gyro_log_20261108_115151.txt
Output:         data/processed_movements.csv (same columns as experiment 01, plus 'label')

Andrew's notes here have seconds (HH:MM:SS), so each note is matched to the gesture
whose time span is closest to it (0 s if the note falls inside the gesture).
Notes are used ONLY to check results, never to train.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core import log_io  # noqa: E402

RAW_LOG = HERE / "data" / "gyro_log_20261108_115151.txt"
OUTPUT = HERE / "data" / "processed_movements.csv"
MATCH_TOLERANCE_S = 10

DATE = "2026-11-08"
NOTES = {
    "11:47:20": "right",
    "11:47:57": "right",
    "11:48:23": "up-left-down",
    "11:49:27": "right",
    "11:50:16": "left",
    "11:50:47": "left",
    "11:51:39": "right",
}


def add_labels(d):
    g = d[d.event > 0].groupby("event").timestamp.agg(["first", "last"])
    labels = {e: "unknown" for e in g.index}
    for hms, lab in NOTES.items():
        t = pd.Timestamp(f"{DATE} {hms}")
        before = (g["first"] - t).dt.total_seconds().clip(lower=0)
        after = (t - g["last"]).dt.total_seconds().clip(lower=0)
        dist = before + after
        if dist.min() <= MATCH_TOLERANCE_S:
            labels[dist.idxmin()] = lab
    d["label"] = d.event.map(labels).fillna("idle")
    return d


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else RAW_LOG
    d = add_labels(log_io.split_gestures(log_io.clean(log_io.read_raw_log(raw))))
    log_io.print_gestures(d)
    log_io.save_processed(d, OUTPUT)
