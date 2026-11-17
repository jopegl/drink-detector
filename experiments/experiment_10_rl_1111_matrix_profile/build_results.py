"""
Experiment 10, step 3: Matrix Profile for every setting, saved in results/ for the dashboard.

Usage (from anywhere):
    python experiments/experiment_10_rl_1111_matrix_profile/build_results.py

Settings tested: signal (angles / rotation rates) x window length (2, 3, 4, 6 s). Takes ~1-2 minutes.
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import matrix_profile as mp  # noqa: E402
from core import events as ev  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "results"

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d, E, T = ev.load_experiment_data(mp.DATA_DIR, mp.DAY)
    det = ev.detection_summary(E, T)

    R, res, disc, mps = mp.evaluate(d, E, **mp.DEFAULT)
    ev.save_events(R.assign(nn_time=R.nn_time), OUT / "events.csv")
    disc.assign(time=disc.time.dt.strftime("%Y-%m-%d %H:%M:%S.%f")).to_csv(OUT / "discords.csv", index=False)
    mps.assign(time=mps.time.dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
               nn_time=mps.nn_time.dt.strftime("%Y-%m-%d %H:%M:%S.%f")).round(3).to_csv(OUT / "matrix_profile.csv",
                                                                                      index=False)
    print(f"Default {mp.DEFAULT}: nearest neighbour on a movement {res['nn_on_event']}/{res['n_rl']}, "
          f"same type {res['nn_same_label']}/{res['nn_on_rl']}")

    rows = []
    for signal in mp.GRID["signal"]:
        for m_s in mp.GRID["m_s"]:
            _, r, dsc, _ = mp.evaluate(d, E, signal, m_s)
            rows.append({"signal": signal, "m_s": m_s, **r,
                         "discords_on_events": int((dsc.event != 0).sum()),
                         "share_same": round(r["nn_same_label"] / r["nn_on_rl"], 3) if r["nn_on_rl"] else 0.0,
                         "text": f"{r['nn_same_label']}/{r['nn_on_rl']} same<br>{r['nn_on_event']}/{r['n_rl']} on a movement"})
            print(f"  {signal}, {m_s} s: same type {r['nn_same_label']}/{r['nn_on_rl']}, "
                  f"on a movement {r['nn_on_event']}/{r['n_rl']}")
    Rb = pd.DataFrame(rows)
    Rb.to_csv(OUT / "robustness.csv", index=False)

    others = {r.truth_id: {"truth": r.truth_label, "mp_percentile": round(r.mp_percentile, 1),
                           "nn": r.nn_label} for r in R.itertuples() if r.truth_label in ("?", "other?")}
    ev.save_json({"default": mp.DEFAULT, "detection": det, "nn": res, "others": others,
                  "discords_on_events": int((disc.event != 0).sum()), "n_discords": len(disc)},
                 OUT / "summary.json")
    print(f"Saved everything in {OUT}/")
