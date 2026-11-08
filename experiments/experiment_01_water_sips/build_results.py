"""
Experiment 01, step 4: run every analysis once and save the results in results/,
so the dashboard (dashboard.py) opens with everything ready.

Usage (from anywhere):
    python experiments/experiment_01_water_sips/build_results.py   (reads data/processed_log.csv)

Run it again whenever processed_log.csv or any analysis script changes.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "processed_log.csv"
OUT = HERE / "results"
import warnings

import pandas as pd

import evaluate_kmeans as ev
import kmeans_gestures as kg
import kmeans_windows as kw

warnings.filterwarnings("ignore")


def window_analysis(d, window_s, k):
    W = kw.make_windows(d, window_s)
    groups, names = kw.run_kmeans(W, k=k, verbose=False)
    W["group"] = groups
    W["class"] = W.group.map(names)
    return W, kw.detected_events(W)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else DATA, parse_dates=["timestamp"])
    day = d.timestamp.iloc[0].strftime("%Y-%m-%d")

    # 1. Main detector (sliding windows) for every window length and k
    robustness = []
    for k in [2, 3, 4, 5, 6]:
        for ws in [2, 3, 4, 5, 6, 7, 8]:
            W, E = window_analysis(d, ws, k)
            E, found, rh, extras = kw.compare(E, day)
            tag = f"w{ws}_k{k}"
            E.to_csv(f"{OUT}/events_{tag}.csv", index=False)
            pd.crosstab(W.label, W["class"]).to_csv(f"{OUT}/crosstab_{tag}.csv")
            (W.groupby("class")[["pitch_min", "pitch_std", "roll_std", "tran_mean"]].mean().round(1)
             .to_csv(f"{OUT}/group_means_{tag}.csv"))
            robustness.append({"window_s": ws, "k": k, "sips_found": len(found),
                               "red_herring_as_sip": len(rh), "extra_events": len(extras),
                               "extra_times": ", ".join(t.strftime("%H:%M:%S") for t in extras.start)})
            print(f"windows {ws} s, k={k}: {len(found)}/9 sips, {len(rh)}/1 red herring, "
                  f"{len(extras)} extra")
    pd.DataFrame(robustness).to_csv(f"{OUT}/window_robustness.csv", index=False)

    # 2. Gesture-level k-means with the three feature sets
    F = kg.features(d)
    sets = {"all": [c for c in F.columns if c not in ("event", "start", "label")],
            "rotation_only": kg.ROTATION_FEATURES,
            "translation_only": kg.TRANSLATION_FEATURES}
    gk = []
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    for name, cols in sets.items():
        X = StandardScaler().fit_transform(F[cols])
        runs = pd.Series([tuple(pd.factorize(KMeans(2, n_init=10, random_state=s).fit_predict(X))[0])
                          for s in range(kg.N_SEEDS)]).value_counts()
        groups = pd.Series(runs.index[0])
        F[f"group_{name}"] = groups.values
        water_group = groups[F.label.str.startswith("water").values].mode()[0]
        gk.append({"feature_set": name, "n_features": len(cols),
                   "same_split_runs": f"{runs.iloc[0]}/{kg.N_SEEDS}",
                   "silhouette": round(silhouette_score(X, groups), 2),
                   "water_group": ", ".join(F.start[groups == water_group].str[:5]),
                   "other_group": ", ".join(F.start[groups != water_group].str[:5])})
    pd.DataFrame(gk).to_csv(f"{OUT}/gesture_kmeans_summary.csv", index=False)

    # 3. Robustness checks of the gesture-level k-means
    F, res, removal = ev.evaluate(F)
    F.round(3).to_csv(f"{OUT}/gestures.csv", index=False)
    removal.to_csv(f"{OUT}/feature_removal.csv", index=False)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved everything in {OUT}/")
