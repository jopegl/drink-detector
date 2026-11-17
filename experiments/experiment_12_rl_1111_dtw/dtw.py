"""
Experiment 12, step 2: Dynamic Time Warping (DTW) between the events' curves.

Usage (from anywhere):
    python experiments/experiment_12_rl_1111_dtw/dtw.py
    (reads the data of experiment 06, writes outputs/)

What DTW does
    It compares two curves by stretching / squeezing time so that their shapes line up as well as
    possible, then measures what is left. Two movements with the same shape but done at different
    speeds get a small distance.

How it is used here
    For every detected event: the curves of yaw / roll / pitch relative to the 3 s before the event
    (drift removed), from 1 s before to 1 s after the event.
      - Template matching (1-nearest neighbour, leave-one-out): each R/L movement gets the label of
        its most similar OTHER labeled movement.
      - Unsupervised: agglomerative clustering (average linkage) on the DTW distances, k = 2.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core import events as ev  # noqa: E402
from core.experiments import load_info  # noqa: E402

warnings.filterwarnings("ignore")

INFO = load_info(HERE)
DATA_DIR = (HERE / INFO["data_dir"]).resolve()
OUT = HERE / "outputs"
DAY = INFO["date"]

STEP = 2                      # keep 1 reading out of 2 (~7 per second) to speed DTW up
DEFAULT = {"channels": "angles", "norm": "raw"}
GRID = {"channels": ["angles", "rates"], "norm": ["raw", "z"]}
CHANNELS = {"angles": ["yaw", "roll", "pitch"], "rates": ["rate_x", "rate_y", "rate_z"]}


def dtw(a, b):
    """DTW distance between two multichannel curves (rows = time), divided by the path length."""
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        Di, Dp, ci = D[i], D[i - 1], cost[i - 1]
        for j in range(1, m + 1):
            Di[j] = ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    return D[n, m] / (n + m)


def curves(d, E, channels, norm):
    out = []
    for e in E.itertuples():
        c = ev.relative_angles_around(d, e.start, e.end)[CHANNELS[channels]].values[::STEP]
        if norm == "z":
            c = (c - c.mean(axis=0)) / (c.std(axis=0) + 1e-9)
        out.append(c)
    return out


def distance_matrix(C):
    k = len(C)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            M[i, j] = M[j, i] = dtw(C[i], C[j])
    return M


def evaluate(E, M):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    E = E.copy()
    lab = E.truth_label.values
    rl = np.isin(lab, ev.MOVEMENT_LABELS)
    # template matching: nearest OTHER labeled movement
    nn_idx, nn_lab, nn_dist = [], [], []
    for i in range(len(E)):
        cand = [j for j in np.where(rl)[0] if j != i]
        j = cand[int(np.argmin(M[i, cand]))]
        nn_idx.append(int(E.event.iloc[j])); nn_lab.append(lab[j]); nn_dist.append(float(M[i, j]))
    E["nn_event"], E["nn_label"], E["nn_dist"] = nn_idx, nn_lab, np.round(nn_dist, 3)
    correct_1nn = int((E.nn_label[rl] == E.truth_label[rl]).sum())
    # how unusual is each event: its nearest-labeled distance vs. the typical one among R/L movements
    typical = float(np.median(np.array(nn_dist)[rl]))
    E["unusualness"] = np.round(np.array(nn_dist) / typical, 2)
    # unsupervised: average-linkage clustering on the DTW distances
    Z = linkage(squareform(M, checks=False), "average")
    E["group"] = (fcluster(Z, 2, criterion="maxclust") - 1).astype(str)
    res = ev.evaluate_groups(E)
    E["named_as"] = E.group.map(res["mapping"])
    return E, {"n": int(rl.sum()), "correct_1nn": correct_1nn, "cluster": res, "typical_nn_dist": typical}


def run(d, E, channels, norm):
    M = distance_matrix(curves(d, E, channels, norm))
    Eo, res = evaluate(E, M)
    return Eo, res, M


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d, E, T = ev.load_experiment_data(DATA_DIR, DAY)
    Eo, res, M = run(d, E, **DEFAULT)
    print(f"DTW {DEFAULT}: {len(E)} events")
    print(f"Template matching (1-nearest neighbour, leave-one-out): {res['correct_1nn']}/{res['n']} correct")
    c = res["cluster"]
    print(f"Clustering on DTW distances (k = 2): {c['correct']}/{c['n']} correct, ARI {c['ari']:.2f}, "
          f"groups {c['mapping']}")
    print(Eo[["event", "start", "truth_id", "truth_label", "nn_event", "nn_label", "nn_dist", "unusualness", "group"]]
          .assign(start=Eo.start.dt.strftime("%H:%M:%S")).to_string(index=False))
    pd.DataFrame(M, index=Eo.event, columns=Eo.event).round(3).to_csv(OUT / "dtw_matrix_default.csv")
    ev.save_events(Eo, OUT / "events_default.csv")
    print(f"\nSaved: {OUT / 'dtw_matrix_default.csv'} and {OUT / 'events_default.csv'}")
