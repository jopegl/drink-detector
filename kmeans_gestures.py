"""
Step 2a: k-means on the gestures (one row per gesture).

Usage:
    python kmeans_gestures.py processed_log.csv

Requires: pandas, numpy, scikit-learn

k-means does NOT use the 'label' column. Labels appear only at the end,
to check whether the groups k-means found match Andrew's notes.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

K = 2            # number of groups
N_SEEDS = 20     # how many times to run k-means to check stability
RATE_HZ = 14.0   # sensor readings per second

ROTATION_FEATURES = ["pitch_min", "pitch_max", "roll_min", "roll_max", "pitch_std",
                     "roll_std", "largest_angle_range", "time_pitch_below_-60_s"]
TRANSLATION_FEATURES = ["tran_mean", "tran_max", "tran_p90", "duration_s"]


# ---------------------------------------------------------------------------
# 1. Read the processed log
# ---------------------------------------------------------------------------
def read(path):
    d = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Rows: {len(d)} | gestures ('event' column): {d.event[d.event > 0].nunique()}")
    return d


# ---------------------------------------------------------------------------
# 2. One feature "card" per gesture
# ---------------------------------------------------------------------------
def features(d):
    cards = []
    for ev, x in d[d.event > 0].groupby("event"):
        cards.append({
            "event": ev,
            "start": x.timestamp.iloc[0].strftime("%H:%M:%S"),
            "label": x.label.iloc[0],  # kept only for the final check
            # duration and movement (tran)
            "duration_s": x.t_s.iloc[-1] - x.t_s.iloc[0],
            "tran_mean": x.tran_mag.mean(),
            "tran_max": x.tran_mag.max(),
            "tran_p90": x.tran_mag.quantile(0.9),
            # rotation (gyro)
            "pitch_min": x.gyro_pitch.min(),
            "pitch_max": x.gyro_pitch.max(),
            "roll_min": x.gyro_roll.min(),
            "roll_max": x.gyro_roll.max(),
            "pitch_std": x.gyro_pitch.std(),
            "roll_std": x.gyro_roll.std(),
            "yaw_range": x.gyro_yaw_unwrapped.max() - x.gyro_yaw_unwrapped.min(),
            "time_pitch_below_-60_s": (x.gyro_pitch < -60).sum() / RATE_HZ,
            "largest_angle_range": max(x.gyro_pitch.max() - x.gyro_pitch.min(),
                                       x.gyro_roll.max() - x.gyro_roll.min()),
        })
    return pd.DataFrame(cards)


# ---------------------------------------------------------------------------
# 3. k-means (no labels)
# ---------------------------------------------------------------------------
def run_kmeans(F, columns, k=K):
    X = StandardScaler().fit_transform(F[columns])  # put every feature on the same scale

    def canon(r):  # renumber groups so different runs can be compared
        m = {}
        return tuple(m.setdefault(v, len(m)) for v in r)

    runs = pd.Series([
        canon(KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X))
        for s in range(N_SEEDS)
    ]).value_counts()
    groups = np.array(runs.index[0])
    print(f"  same split in {runs.iloc[0]}/{N_SEEDS} runs | "
          f"silhouette = {silhouette_score(X, groups):.2f}")
    return groups


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python kmeans_gestures.py processed_log.csv")

    F = features(read(sys.argv[1]))

    feature_sets = {
        "all": [c for c in F.columns if c not in ("event", "start", "label")],
        "rotation_only": ROTATION_FEATURES,
        "translation_only": TRANSLATION_FEATURES,
    }

    for name, cols in feature_sets.items():
        print(f"\n=== Features: {name} | k={K}")
        F[f"group_{name}"] = run_kmeans(F, cols)
        # 4. Only now do we look at the labels, to check
        for g, x in F.groupby(f"group_{name}"):
            items = ", ".join(f"{s[:5]} [{lab}]" for s, lab in zip(x.start, x.label))
            print(f"  group {g}: {items}")

    F.round(2).to_csv("gestures_kmeans.csv", index=False)
    print("\nSaved: gestures_kmeans.csv")
