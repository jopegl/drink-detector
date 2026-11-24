"""
Experiment 13, step 1: clean the 11/20 recording (right / left movements + red herrings) and remove the yaw drift.

Usage (from anywhere):
    python experiments/experiment_13_rh_1120_red_herrings/prepare.py
    python experiments/experiment_13_rh_1120_red_herrings/prepare.py <raw_log_file>

Input:   data/gyro_log_20261120_153532.txt  (raw log, git-ignored)
         data/annotations.csv               (Andrew's notes: time to the second + R / L / RH)
Output:  data/processed_log.csv              (one row per reading; adds yaw_dt = yaw without drift,
                                              'label' = note label around each matched movement, else 'none',
                                              'event' = detected event number, 0 = none)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import red_herrings as rh  # noqa: E402
from core import events as ev  # noqa: E402
from core import log_io  # noqa: E402

RAW_LOG = HERE / "data" / "gyro_log_20261120_153532.txt"
OUTPUT = HERE / "data" / "processed_log.csv"


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else RAW_LOG
    d = log_io.clean(log_io.read_raw_log(raw))
    d, drift = ev.remove_yaw_drift(d)
    print(f"Yaw drift removed: {drift:.1f} degrees per minute")

    T = rh.load_notes(HERE / "data" / "annotations.csv")
    E = rh.match_notes(rh.detect(d, rh.DEFAULT["high"], rh.DEFAULT["min_gap_s"]), T)
    d["label"], d["event"] = "none", 0
    for e in E.itertuples():
        m = (d.timestamp >= e.start) & (d.timestamp <= e.end)
        d.loc[m, "event"] = e.event
        if e.truth_label != "—":
            d.loc[m, "label"] = e.truth_label
    print(f"Events detected: {len(E)} | notes matched: {(E.truth_id != '—').sum()}/{len(T)}")
    log_io.save_processed(d, OUTPUT)
