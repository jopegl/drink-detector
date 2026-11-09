import json
import sys
import warnings
from pathlib import Path

import pandas as pd

import movements_dbscan as ma   # sibling file: works when the script is run from any folder

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "processed_movements.csv"
OUT = HERE / "results"

WINDOWS = [2, 3, 4, 5, 6]
EPS = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
MIN_SAMPLES = [3, 4, 5]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d = ma.add_relative_angles(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else DATA,
                                           parse_dates=["timestamp"]))

    # angles relative to rest, for the dashboard timeline
    d[["timestamp", "yaw_rel", "roll_rel", "pitch_rel"]].round(2).to_csv(
        OUT / "relative_angles.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")

    rows = []
    for ms in MIN_SAMPLES:
        for eps in EPS:
            for ws in WINDOWS:
                W, M = ma.analyse(d, ws, eps, ms)
                tag = f"w{ws}_eps{eps}_ms{ms}"
                M.to_csv(OUT / f"movements_{tag}.csv", index=False)
                pd.crosstab(W.label, W["class"]).to_csv(OUT / f"crosstab_{tag}.csv")
                means = W.groupby("group")[["yaw_min", "yaw_max", "roll_max", "pitch_min", "tran_mean"]].mean()
                means.insert(0, "name", [W.loc[W.group == i, "class"].iloc[0] for i in means.index])
                means.insert(1, "windows", W.group.value_counts().sort_index().values)
                means.round(1).to_csv(OUT / f"group_means_{tag}.csv")

                r = {"window_s": ws, "eps": eps, "min_samples": ms,
                     "n_groups": int((means.index != -1).sum()),          # groups found (noise excluded)
                     "n_noise": int((W.group == -1).sum()),               # windows marked as noise
                     **ma.classify_noted(M)}
                r["all_correct"] = (r["right_as_yaw_down"] == r["n_right"]
                                    and r["left_as_small"] == r["n_left"]
                                    and r["uld_as_yaw_up"] == r["n_uld"])
                rows.append(r)
                print(f"{ws} s, eps={eps}, min_samples={ms}: {r['n_groups']} groups, {r['n_noise']} noise | "
                      f"right {r['right_as_yaw_down']}/{r['n_right']}, left {r['left_as_small']}/{r['n_left']}, "
                      f"up-left-down {r['uld_as_yaw_up']}/{r['n_uld']}")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "robustness.csv", index=False)

    summary = {"n_settings": len(R), "all_correct": int(R.all_correct.sum()),
               "default_window_s": ma.WINDOW_S, "default_eps": ma.EPS, "default_min_samples": ma.MIN_SAMPLES,
               "n_windows": len(ma.make_windows(d, ma.WINDOW_S))}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll 7 correct in {summary['all_correct']}/{summary['n_settings']} settings")
    print(f"Saved everything in {OUT}/")