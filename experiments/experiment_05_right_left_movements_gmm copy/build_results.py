"""
Experiment 05, step 3: run the Gaussian Mixture Model for every window length and number of
components, and save the results in results/ for the dashboard.

Usage (from anywhere):
    python experiments/experiment_05_right_left_movements_gmm/build_results.py
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import movements_gmm as mg

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "processed_movements.csv"
OUT = HERE / "results"
KS = [3, 4, 5, 6]
WINDOWS = [2, 3, 4, 5, 6]
COV = getattr(mg, "COVARIANCE_TYPE", "diag")   # same covariance type as movements_gmm.py


def gmm(k, seed):
    return GaussianMixture(n_components=k, covariance_type=COV, n_init=10, reg_covar=1e-3,
                           random_state=seed)


def seed_stability(d, window_s, k, n_seeds=20):
    """In how many GMM runs (different random starts) are the noted movements classified the same way?"""
    W = mg.make_windows(d, window_s)
    X = StandardScaler().fit_transform(W[mg.FEATURES])
    outcomes = []
    for s in range(n_seeds):
        g = gmm(k, s).fit_predict(X)
        names = mg.name_groups(W.assign(g=g).groupby("g")[mg.FEATURES].mean())
        M = mg.compare(mg.detected_movements(W.assign(**{"class": pd.Series(g).map(names).values})))
        outcomes.append(tuple(M[M.andrew_note != "—"].sort_values("start").type))
    counts = pd.Series(outcomes).value_counts()
    return int(counts.iloc[0]), n_seeds


def bic(W, k):
    """BIC of a GMM with k components on these windows: lower = better balance of fit vs. complexity."""
    X = StandardScaler().fit_transform(W[mg.FEATURES])
    return float(gmm(k, 0).fit(X).bic(X))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d = mg.add_relative_angles(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else DATA,
                                           parse_dates=["timestamp"]))

    # angles relative to rest, for the dashboard timeline
    d[["timestamp", "yaw_rel", "roll_rel", "pitch_rel"]].round(2).to_csv(
        OUT / "relative_angles.csv", index=False, date_format="%Y-%m-%d %H:%M:%S.%f")

    rows = []
    for k in KS:
        for ws in WINDOWS:
            W = mg.make_windows(d, ws)
            g, names = mg.run_gaussian(W, k, verbose=False)
            W["group"] = g
            W["class"] = W.group.map(names)
            M = mg.compare(mg.detected_movements(W))
            tag = f"w{ws}_k{k}"
            M.to_csv(OUT / f"movements_{tag}.csv", index=False)
            pd.crosstab(W.label, W["class"]).to_csv(OUT / f"crosstab_{tag}.csv")
            means = W.groupby("group")[["yaw_min", "yaw_max", "roll_max", "pitch_min", "tran_mean"]].mean()
            means.insert(0, "name", [names[i] for i in means.index])
            means.insert(1, "windows", W.group.value_counts().sort_index().values)
            means.round(1).to_csv(OUT / f"group_means_{tag}.csv")
            r = {"k": k, "window_s": ws, "bic": round(bic(W, k), 1), **mg.classify_noted(M)}
            rows.append(r)
            print(f"{ws} s, k={k}: BIC {r['bic']:.0f} | right {r['right_as_yaw_down']}/{r['n_right']}, "
                  f"left {r['left_as_small']}/{r['n_left']}, up-left-down {r['uld_as_yaw_up']}/{r['n_uld']}")

    R = pd.DataFrame(rows)
    R.to_csv(OUT / "robustness.csv", index=False)

    best_k = {int(ws): int(x.loc[x.bic.idxmin(), "k"]) for ws, x in R.groupby("window_s")}
    stable, n = seed_stability(d, mg.WINDOW_S, mg.K)
    summary = {"default_window_s": mg.WINDOW_S, "default_k": mg.K,
               "stable_runs": stable, "n_runs": n,
               "best_k_by_bic": best_k,
               "n_windows": len(mg.make_windows(d, mg.WINDOW_S))}
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Default ({mg.WINDOW_S:.0f} s, k={mg.K}): same classification in {stable}/{n} runs")
    print(f"Best k by BIC for each window length: {best_k}")
    print(f"Saved everything in {OUT}/")