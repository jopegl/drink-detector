"""
Experiment 01, step 1: clean the raw log and split it into gestures.

Usage (from anywhere):
    python experiments/experiment_01_water_sips/prepare.py
    python experiments/experiment_01_water_sips/prepare.py <raw_log_file>

Default input:  data/accel_gryo_log_20261104_183756.txt
Output:         data/processed_log.csv (one row per reading; columns in core/log_io.py, plus 'label')

'label' = idle / water / water (+red herring?) / red_herring / unknown, from Andrew's notes.
It is used ONLY to check results, never to train.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core import log_io  # noqa: E402

RAW_LOG = HERE / "data" / "accel_gryo_log_20261104_183756.txt"
OUTPUT = HERE / "data" / "processed_log.csv"
MATCH_TOLERANCE_S = 150

# Andrew's notes (hour:minute). 17:49 was noted as "water and red herring directly after".
NOTES = {
    "16:53": "water", "16:59": "water", "17:19": "water", "17:25": "water",
    "17:49": "water (+red herring?)", "18:10": "water", "18:20": "water",
    "18:30": "water", "18:35": "red_herring", "18:37": "water",
}


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
    raw = sys.argv[1] if len(sys.argv) > 1 else RAW_LOG
    d = add_labels(log_io.split_gestures(log_io.clean(log_io.read_raw_log(raw))))
    log_io.print_gestures(d)
    log_io.save_processed(d, OUTPUT)
