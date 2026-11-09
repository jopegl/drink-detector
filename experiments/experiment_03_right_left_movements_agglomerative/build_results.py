import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import movements_agglomerative as ma

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "processed_movements.csv"
OUT = HERE / "results"
KS = [3, 4, 5, 6]
WINDOWS = [2, 3, 4, 5, 6]


def seed_stability(d, window_s, k, n_seeds=20):
    """In how many k-means runs (different random starts) are the noted movements classified the same way?"""
    W = ma.make_windows(d, window_s)
    X = StandardScaler().fit_transform(W[ma.FEATURES])
    outcomes = []
    for s in range(n_seeds):
        g = KMeans(k, n_init=10, random_state=s).fit_predict(X)
        names = ma.name_groups(W.assign(g=g).groupby("g")[ma.FEATURES].mean())
        M = ma.compare(ma.detected_movements(W.assign(**{"class": pd.Series(g).map(names).values})))
        outcomes.append(tuple(M[M.andrew_note != "—"].sort_values("start").type))
    counts = pd.Series(outcomes).value_counts()
    return int(counts.iloc[0]), n_seeds


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d = ma.add_relative_angles(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else DATA,
                                           parse_dates=["timestamp"]))

    # angles relative to rest, for the dashboard timeline
    d[["timestamp", "yaw_rel", "roll_rel", "pitch_rel"]].round(2).to_csv(
        OUT / "relative_angles.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")

    rows = []
    for k in KS:
        for ws in WINDOWS:
            W = ma.make_windows(d, ws)
            g, names = ma.run_agglomerative(W, k, verbose=False)
            W["group"] = g
            W["class"] = W.group.map(names)
            M = ma.compare(ma.detected_movements(W))
            tag = f"w{ws}_k{k}"
            M.to_csv(OUT / f"movements_{tag}.csv", index=False)
            pd.crosstab(W.label, W["class"]).to_csv(OUT / f"crosstab_{tag}.csv")
            means = W.groupby("group")[["yaw_min", "yaw_max", "roll_max", "pitch_min", "tran_mean"]].mean()
            means.insert(0, "name", [names[i] for i in means.index])
            means.insert(1, "windows", W.group.value_counts().sort_index().values)
            means.round(1).to_csv(OUT / f"group_means_{tag}.csv")
            r = {"k": k, "window_s": ws, **ma.classify_noted(M)}
            rows.append(r)
            print(f"{ws} s, k={k}: right {r['right_as_yaw_down']}/{r['n_right']}, "
                  f"left {r['left_as_small']}/{r['n_left']}, up-left-down {r['uld_as_yaw_up']}/{r['n_uld']}")
    pd.DataFrame(rows).to_csv(OUT / "robustness.csv", index=False)

    stable, n = seed_stability(d, ma.WINDOW_S, ma.K)
    summary = {"default_window_s": ma.WINDOW_S, "default_k": ma.K,
               "stable_runs": stable, "n_runs": n,
               "n_windows": len(ma.make_windows(d, ma.WINDOW_S))}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Default ({ma.WINDOW_S:.0f} s, k={ma.K}): same classification in {stable}/{n} runs")
    print(f"Saved everything in {OUT}/")
