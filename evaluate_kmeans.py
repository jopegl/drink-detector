"""
Step 2b: robustness checks for the gesture-level k-means (kmeans_gestures.py).

Usage:
    python evaluate_kmeans.py processed_log.csv
    (kmeans_gestures.py must be in the same folder)

Requires: pandas, numpy, scikit-learn, matplotlib
"""
import itertools
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kmeans_gestures import ROTATION_FEATURES as ROT, features, read

warnings.filterwarnings("ignore")


def evaluate(F):
    """Run every check. Returns (F with extra columns, dict of results, feature-removal table)."""
    F = F.copy()
    y = F.label.str.startswith("water").astype(int).values   # 1 = water (evaluation only)
    known = ~F.label.eq("unknown").values                     # gestures Andrew annotated

    def km(cols, k=2, idx=None, seed=0):
        idx = np.arange(len(F)) if idx is None else idx
        sc = StandardScaler().fit(F.loc[idx, cols])
        m = KMeans(k, n_init=10, random_state=seed).fit(sc.transform(F.loc[idx, cols]))
        return m, sc

    def score(groups):
        """Call 'water' the group holding most water gestures; return predictions and accuracy."""
        water = pd.Series(groups[y == 1]).mode()[0]
        pred = (groups == water).astype(int)
        return pred, (pred[known] == y[known]).mean()

    res = {"n_annotated": int(known.sum())}

    # 1) Accuracy on the annotated gestures
    m, sc = km(ROT)
    g = m.labels_
    pred, acc = score(g)
    F["kmeans_says"] = np.where(pred == 1, "water", "not water")
    res["accuracy"] = float(acc)

    # 2) Leave-one-out: fit without one gesture, then classify it by the nearest centre
    ok, out = 0, []
    for i in range(len(F)):
        idx = np.array([j for j in range(len(F)) if j != i])
        m2, sc2 = km(ROT, idx=idx)
        w = pd.Series(m2.labels_[y[idx] == 1]).mode()[0]
        p = int(m2.predict(sc2.transform(F.loc[[i], ROT]))[0] == w)
        out.append(p)
        if known[i]:
            ok += p == y[i]
    F["leave_one_out_says"] = np.where(np.array(out) == 1, "water", "not water")
    res["loo_correct"] = int(ok)

    # 3) Robustness: remove 1 and 2 features
    rows = []
    for r in [1, 2]:
        for drop in itertools.combinations(ROT, r):
            pr, a = score(km([c for c in ROT if c not in drop])[0].labels_)
            wrong = [F.start[i][:5] for i in range(len(F)) if known[i] and pr[i] != y[i]]
            rows.append({"n_removed": r, "removed": ", ".join(drop), "accuracy": a,
                         "wrong": ", ".join(wrong)})
    removal = pd.DataFrame(rows)

    # 4) Margin: distance to the 'not water' centre minus distance to the 'water' centre
    X = sc.transform(F[ROT])
    c = m.cluster_centers_
    water = pd.Series(g[y == 1]).mode()[0]
    F["margin"] = (np.linalg.norm(X - c[1 - water], axis=1)
                   - np.linalg.norm(X - c[water], axis=1)).round(2)

    # 5) k = 3
    g3 = km(ROT, k=3)[0].labels_
    res["k3_groups"] = [list(F.start[g3 == k].str[:5]) for k in range(3)]

    # 6) PCA map
    pca = PCA(2).fit(X)
    P = pca.transform(X)
    F["pca_1"], F["pca_2"] = P[:, 0], P[:, 1]
    res["pca_variance"] = [float(v) for v in pca.explained_variance_ratio_]
    F["in_water_group"] = g == water
    return F, res, removal


def plot_pca(F, res, filename="kmeans_pca.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"water": "#2b8cbe", "water (+red herring?)": "#2b8cbe",
              "red_herring": "#e34a33", "unknown": "#888888"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, r in F.iterrows():
        ax.scatter(r.pca_1, r.pca_2, s=110, c=colors[r.label],
                   marker="o" if r.in_water_group else "s", edgecolor="k")
        ax.annotate(f"{r.start[:5]} ({r.margin:+.1f})", (r.pca_1, r.pca_2), xytext=(6, 4),
                    textcoords="offset points", fontsize=8)
    ev = res["pca_variance"]
    ax.set_xlabel(f"component 1 ({ev[0] * 100:.0f}% of variance)")
    ax.set_ylabel(f"component 2 ({ev[1] * 100:.0f}%)")
    ax.set_title("k-means on the gestures (rotation features)\n"
                 "colour = Andrew's note · ● 'water' group · ■ 'not water' group · (margin)", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    print(f"Saved: {filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python evaluate_kmeans.py processed_log.csv")

    F, res, removal = evaluate(features(read(sys.argv[1])))
    i1729 = F.start.str[:5].eq("17:29")
    print(f"1) Accuracy on the {res['n_annotated']} annotated gestures: {res['accuracy'] * 100:.0f}% "
          f"| 17:29 -> {F.kmeans_says[i1729].iloc[0]}")
    print(f"2) Leave-one-out (unseen gesture): {res['loo_correct']}/{res['n_annotated']} correct "
          "| unknown gestures:",
          dict(zip(F.start[F.label == "unknown"].str[:5], F.leave_one_out_says[F.label == "unknown"])))
    for r, x in removal.groupby("n_removed"):
        for _, row in x[x.accuracy < 1].iterrows():
            print(f"   fails without ({row.removed}) -> wrong: {row.wrong}")
        print(f"3) Removing {r} feature(s) ({len(x)} combinations): 100% accuracy in "
              f"{(x.accuracy == 1).sum()}/{len(x)}, minimum {x.accuracy.min() * 100:.0f}%")
    print("4) Margin (positive = water side; close to 0 = uncertain):")
    print(F[["start", "label", "margin"]].to_string(index=False))
    print("5) k=3:", dict(enumerate(res["k3_groups"])))
    print("6) PCA explained variance:", np.round(res["pca_variance"], 2))
    plot_pca(F, res)
