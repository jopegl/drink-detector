"""
Experiment 06, step 1: clean the 11/11 recording (right / left movements) and remove the yaw drift.
Experiments 06 to 12 all use this same data folder.

Usage (from anywhere):
    python experiments/experiment_06_rl_1111_kmeans/prepare.py
    python experiments/experiment_06_rl_1111_kmeans/prepare.py <raw_log_file>

Input:   data/gyro_log_20261111_151520.txt  (raw log, git-ignored)
         data/annotations.csv               (ground truth: times from the video, R/L from Andrew's notes)
Output:  data/processed_log.csv              (one row per reading; adds yaw_dt = yaw without drift,
                                              and 'label' = truth label of the annotated movement, else 'none')
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core import events as ev  # noqa: E402
from core import log_io  # noqa: E402

RAW_LOG = HERE / "data" / "gyro_log_20261111_151520.txt"
OUTPUT = HERE / "data" / "processed_log.csv"
TRUTH = HERE / "data" / "annotations.csv"
DAY = "2026-11-11"


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else RAW_LOG
    d = log_io.clean(log_io.read_raw_log(raw))
    d, drift = ev.remove_yaw_drift(d)
    print(f"Yaw drift removed: {drift:.1f} degrees per minute")

    # label each reading with the truth movement it belongs to (for plots only)
    T = ev.load_truth(TRUTH, DAY)
    d["label"] = "none"
    for r in T.itertuples():
        if pd.notna(r.start) or pd.notna(r.end):
            a = r.start if pd.notna(r.start) else r.end - pd.Timedelta(seconds=4)
            b = r.end if pd.notna(r.end) else r.start + pd.Timedelta(seconds=4)
            d.loc[(d.timestamp >= a) & (d.timestamp <= b), "label"] = r.label

    # 'event' column (detected events) so the shared dashboard pieces work like in earlier experiments
    E = ev.detect_events(d)
    d["event"] = 0
    for e in E.itertuples():
        d.loc[(d.timestamp >= e.start) & (d.timestamp <= e.end), "event"] = e.event
    print(f"Events detected: {len(E)}")
    log_io.save_processed(d, OUTPUT)
