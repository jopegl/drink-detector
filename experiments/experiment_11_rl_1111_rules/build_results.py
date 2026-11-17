"""
Experiment 11, step 3: rules for every detection setting, saved in results/ for the dashboard.

Usage (from anywhere):
    python experiments/experiment_11_rl_1111_rules/build_results.py

Settings tested: state machine HIGH 40, 55, 70  x  LOW 10, 15, 20.
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import rules as ru  # noqa: E402
from core import events as ev  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "results"

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d, _, T = ev.load_experiment_data(ru.DATA_DIR, ru.DAY)
    E, res = ru.run(d, T)
    det = ev.detection_summary(E, T)
    ev.save_events(E, OUT / "events.csv")
    rows = []
    for high in ru.HIGHS:
        for low in ru.LOWS:
            Et, rt = ru.run(d, T, high=high, low=low)
            dt = ev.detection_summary(Et, T)
            rows.append({"high": high, "low": low, **dt, "n": rt["n"], "exp02_correct": rt["exp02_correct"],
                         "loo_correct": rt["loo_correct"], "loo_accuracy": round(rt["loo_correct"] / rt["n"], 3),
                         "text": f"{dt['truth_found']}/{dt['truth_total']} found, +{dt['events_without_truth']}"
                                 f"<br>rule {rt['loo_correct']}/{rt['n']}"})
            print(f"high {high}, low {low}: found {dt['truth_found']}/{dt['truth_total']}, "
                  f"extra {dt['events_without_truth']} | exp02 rule {rt['exp02_correct']}/{rt['n']} "
                  f"| one-threshold rule (LOO) {rt['loo_correct']}/{rt['n']}")
    pd.DataFrame(rows).to_csv(OUT / "robustness.csv", index=False)
    ev.save_json({"detection": det, "rules": res,
                  "others": {r.truth_id: {"truth": r.truth_label, "one_rule": r.one_rule}
                             for r in E.itertuples() if r.truth_label in ("?", "other?")}},
                 OUT / "summary.json")
    print(f"Saved everything in {OUT}/")
