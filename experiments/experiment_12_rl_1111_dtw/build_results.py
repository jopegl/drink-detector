"""
Experiment 12, step 3: DTW for every setting, saved in results/ for the dashboard.

Usage (from anywhere):
    python experiments/experiment_12_rl_1111_dtw/build_results.py

Settings tested: channels (angles / rotation rates) x normalisation (raw / z-normalised)
                 x detection threshold (40, 55, 70).
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import dtw as dt  # noqa: E402
from core import events as ev  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "results"

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d, E, T = ev.load_experiment_data(dt.DATA_DIR, dt.DAY)
    det = ev.detection_summary(E, T)
    Eo, res, M = dt.run(d, E, **dt.DEFAULT)
    ev.save_events(Eo, OUT / "events.csv")
    pd.DataFrame(M, index=Eo.event, columns=Eo.event).round(3).to_csv(OUT / "dtw_matrix.csv")
    print(f"Default {dt.DEFAULT}: 1-NN {res['correct_1nn']}/{res['n']}, "
          f"clustering {res['cluster']['correct']}/{res['cluster']['n']}")

    rows = []
    for high in [40, 55, 70]:
        _, Eh, _ = ev.load_experiment_data(dt.DATA_DIR, dt.DAY, high=high)
        for ch in dt.GRID["channels"]:
            for norm in dt.GRID["norm"]:
                _, r, _ = dt.run(d, Eh, ch, norm)
                rows.append({"high": high, "channels": ch, "norm": norm, "setting": f"{ch}, {norm}",
                             "n": r["n"], "correct_1nn": r["correct_1nn"], "acc_1nn": round(r["correct_1nn"] / r["n"], 3),
                             "cluster_correct": r["cluster"]["correct"], "cluster_ari": round(r["cluster"]["ari"], 3),
                             "text": f"1-NN {r['correct_1nn']}/{r['n']}<br>clusters {r['cluster']['correct']}/{r['n']}"})
                print(f"  high {high}, {ch}, {norm}: 1-NN {r['correct_1nn']}/{r['n']}, "
                      f"clustering {r['cluster']['correct']}/{r['n']}")
    pd.DataFrame(rows).to_csv(OUT / "robustness.csv", index=False)
    others = {r.truth_id: {"truth": r.truth_label, "nn_label": r.nn_label, "unusualness": r.unusualness}
              for r in Eo.itertuples() if r.truth_label in ("?", "other?")}
    ev.save_json({"default": dt.DEFAULT, "detection": det, "dtw": res, "others": others}, OUT / "summary.json")
    print(f"Saved everything in {OUT}/")
