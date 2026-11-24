"""
Experiment 13, step 3: every approach for every setting, saved in results/ for the dashboard.

Usage (from anywhere):
    python experiments/experiment_13_rh_1120_red_herrings/build_results.py

Settings tested: detection threshold (30, 40, 55) x minimum gap between peaks (5, 6, 8 s)
                 x DTW channels (angles / rotation rates).
"""
import sys
import warnings
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import red_herrings as rh  # noqa: E402
from core import events as ev  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE / "results"
HIGHS, GAPS = [30.0, 40.0, 55.0], [5.0, 6.0, 8.0]

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d, E, T = rh.load_data()
    det = rh.detection_summary(E, T)
    Eo, M, res = rh.run(d, E, rh.DEFAULT["channels"])
    rh.print_results(Eo, res)
    ev.save_events(Eo, OUT / "events.csv")
    pd.DataFrame(M, index=Eo.event, columns=Eo.event).round(3).to_csv(OUT / "dtw_matrix.csv")

    rows = []
    for high in HIGHS:
        for gap in GAPS:
            _, Eh, _ = rh.load_data(high, gap)
            dh = rh.detection_summary(Eh, T)
            for ch in rh.CHANNELS:
                _, _, r = rh.run(d, Eh, ch)
                u = r["unusual"]
                rows.append({"high": high, "min_gap_s": gap, "channels": ch, "setting": f"gap {gap:g} s, {ch}",
                             "events": dh["events"], "notes_matched": dh["notes_matched"],
                             "rule_correct": r["rule"]["correct"], "logreg_correct": r["logreg"]["correct"],
                             "dtw_correct": r["dtw"]["correct"], "n": r["dtw"]["n"],
                             "unusual_tp": u["tp"], "unusual_fp": u["fp"], "unusual_auc": u["auc"],
                             "transfer_auc": r["transfer"]["auc"],
                             "text": f"{dh['notes_matched']}/24 notes<br>RH {u['rh_caught']}, "
                                     f"false {u['false_alarms']}<br>AUC {u['auc']:.2f}"})
                print(f"  high {high:g}, gap {gap:g}, {ch}: notes {dh['notes_matched']}/24 | unusualness RH "
                      f"{u['rh_caught']}, false alarms {u['false_alarms']}, AUC {u['auc']:.2f} | DTW 3-class "
                      f"{r['dtw']['correct']}/{r['dtw']['n']} | logreg {r['logreg']['correct']}/{r['logreg']['n']}")
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "robustness.csv", index=False)

    unnoted = {str(r.event): {"time": r.start.strftime("%H:%M:%S"), "unusualness": r.unusualness,
                              "dtw_pred": r.dtw_pred} for r in Eo.itertuples() if r.truth_label == "—"}
    ev.save_json({"default": rh.DEFAULT, "threshold": rh.UNUSUAL_THRESHOLD, "detection": det, "results": res,
                  "unnoted": unnoted,
                  "robustness": {"settings": len(R), "auc_min": float(R.unusual_auc.min()),
                                 "auc_median": float(R.unusual_auc.median()), "auc_max": float(R.unusual_auc.max())}},
                 OUT / "summary.json")
    print(f"Saved everything in {OUT}/")
