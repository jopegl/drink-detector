"""
Experiment 02, step 2 (main analysis): k-means on sliding windows to tell movement types apart.

Usage (from anywhere):
    python experiments/experiment_02_right_left_movements/movements_kmeans.py
    (reads data/processed_movements.csv, writes outputs/)

Requires: pandas, numpy, scikit-learn, matplotlib

How it works:
  1. For each reading, the angles are measured RELATIVE to the hand's resting position
     just before the movement (last moment the hand stayed still for STILL_MIN_S). This gives the
     DIRECTION of a turn, and it does not use any label.
  2. The whole log is cut into WINDOW_S-second windows (50% overlap) and each window
     gets a feature card (how far yaw / roll / pitch went up or down, how agitated, how intense).
  3. k-means splits the windows into K groups WITHOUT seeing any label.
  4. The groups are named WITHOUT labels, from their centres:
       - lowest movement intensity                  -> "idle"
       - yaw goes down by more than 45 degrees      -> "yaw down"   (turn one way)
       - yaw goes up by more than 45 degrees        -> "yaw up"     (turn the other way)
       - anything else                              -> "small movement"
  5. Neighbouring non-idle windows become one detected movement, named after its
     strongest window type (yaw down / yaw up > small movement).
  6. Only at the end are the detected movements compared with Andrew's notes.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "processed_movements.csv"
OUT = HERE / "outputs"

WINDOW_S = 4.0
OVERLAP = 0.5
K = 4
N_SEEDS = 20
RATE_HZ = 13.7
STILL_THRESHOLD = 1.0   # tran_mag below this = hand not moving
STILL_MIN_S = 1.5       # the hand must stay still this long to count as "at rest"
MERGE_S = 3.0           # non-idle windows closer than this = same movement

FEATURES = ["yaw_min", "yaw_max", "roll_min", "roll_max", "pitch_min", "pitch_max",
            "yaw_std", "roll_std", "pitch_std", "tran_mean", "tran_max"]

DATE = "2026-11-08"
NOTES = {"11:47:20": "right", "11:47:57": "right", "11:48:23": "up-left-down", "11:49:27": "right",
         "11:50:16": "left", "11:50:47": "left", "11:51:39": "right"}


# ---------------------------------------------------------------------------
# 1. Angles relative to the resting position
# ---------------------------------------------------------------------------
def add_relative_angles(d):
    d = d.copy()
    n = int(STILL_MIN_S * RATE_HZ)
    # at rest = no movement during the last STILL_MIN_S seconds (short pauses mid-gesture don't count)
    still = (d.tran_mag < STILL_THRESHOLD).astype(int).rolling(n, min_periods=n).min().fillna(0).astype(bool)
    for name, col in [("roll", "gyro_roll"), ("pitch", "gyro_pitch"), ("yaw", "gyro_yaw_unwrapped")]:
        resting = d[col].where(still).ffill().bfill()   # last value seen while the hand was at rest
        d[f"{name}_rel"] = d[col] - resting
    return d


# ---------------------------------------------------------------------------
# 2. Windows
# ---------------------------------------------------------------------------
def make_windows(d, window_s=WINDOW_S):
    n = int(window_s * RATE_HZ)
    step = max(1, int(n * (1 - OVERLAP)))
    rows = []
    for i in range(0, len(d) - n, step):
        x = d.iloc[i:i + n]
        rows.append({
            "start": x.timestamp.iloc[0], "end": x.timestamp.iloc[-1],
            "label": x.label.mode()[0],   # for checking only
            "yaw_min": x.yaw_rel.min(), "yaw_max": x.yaw_rel.max(),
            "roll_min": x.roll_rel.min(), "roll_max": x.roll_rel.max(),
            "pitch_min": x.pitch_rel.min(), "pitch_max": x.pitch_rel.max(),
            "yaw_std": x.gyro_yaw_unwrapped.std(), "roll_std": x.gyro_roll.std(),
            "pitch_std": x.gyro_pitch.std(),
            "tran_mean": x.tran_mag.mean(), "tran_max": x.tran_mag.max(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. k-means + naming the groups from their centres (no labels)
# ---------------------------------------------------------------------------
def name_groups(means):
    """Name every group from its centre (several groups can share a name). No labels used."""
    idle = means.tran_mean.idxmin()
    names = {}
    for g, c in means.iterrows():
        if g == idle:
            names[g] = "idle"
        elif c.yaw_min < -45 and -c.yaw_min > c.yaw_max:
            names[g] = "yaw down"
        elif c.yaw_max > 45:
            names[g] = "yaw up"
        else:
            names[g] = "small movement"
    return names


def run_kmeans(W, k=K, verbose=True):
    X = StandardScaler().fit_transform(W[FEATURES])

    def canon(r):
        m = {}
        return tuple(m.setdefault(v, len(m)) for v in r)

    runs = pd.Series([canon(KMeans(k, n_init=10, random_state=s).fit_predict(X))
                      for s in range(N_SEEDS)]).value_counts()
    groups = np.array(runs.index[0])
    means = W.assign(g=groups).groupby("g")[FEATURES].mean()

    names = name_groups(means)

    if verbose:
        print(f"  same split in {runs.iloc[0]}/{N_SEEDS} runs | silhouette = {silhouette_score(X, groups):.2f}")
        print("  Mean of each group (angles relative to rest, degrees):")
        show = means[["yaw_min", "yaw_max", "roll_max", "pitch_min", "tran_mean"]].round(0)
        show.index = [f"{g} = {names[g]}" for g in show.index]
        print("    " + show.to_string().replace("\n", "\n    "))
    return groups, names


# ---------------------------------------------------------------------------
# 4. Merge windows into movements and name them
# ---------------------------------------------------------------------------
PRIORITY = {"yaw down": 3, "yaw up": 3, "small movement": 1}


def detected_movements(W):
    m = W[W["class"] != "idle"].sort_values("start")
    if m.empty:
        return pd.DataFrame(columns=["start", "end", "type", "n_windows", "window_types"])
    groups, current = [], [m.iloc[0]]
    for _, r in m.iloc[1:].iterrows():
        if (r.start - current[-1].end).total_seconds() <= MERGE_S:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)
    rows = []
    for gr in groups:
        counts = pd.Series([r["class"] for r in gr]).value_counts()
        strong = [c for c in counts.index if PRIORITY[c] == 3]
        kind = max(strong, key=lambda c: counts[c]) if strong else "small movement"
        rows.append({"start": gr[0].start, "end": gr[-1].end, "type": kind, "n_windows": len(gr),
                     "window_types": ", ".join(f"{c} x{n}" for c, n in counts.items())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Compare with Andrew's notes (labels only come in now)
# ---------------------------------------------------------------------------
def compare(M):
    M = M.copy()
    M["andrew_note"] = "—"
    for hms, lab in NOTES.items():
        if M.empty:
            continue
        t = pd.Timestamp(f"{DATE} {hms}")
        dist = ((M.start - t).dt.total_seconds().clip(lower=0)
                + (t - M.end).dt.total_seconds().clip(lower=0))
        i = dist.idxmin()
        if dist[i] <= 10:
            M.loc[i, "andrew_note"] = f"{lab} ({hms[3:]})"
    return M


def mapping_table(M):
    """How each kind of noted movement was classified."""
    x = M[M.andrew_note != "—"].copy()
    x["note"] = x.andrew_note.str.split(" ").str[0]
    return pd.crosstab(x.note, x.type)


def plot(d, W, M, filename="movements_kmeans.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"yaw down": "#2b8cbe", "yaw up": "#756bb1", "small movement": "#e6550d", "idle": "#dddddd"}
    fig, ax = plt.subplots(3, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    ax[0].plot(d.timestamp, d.yaw_rel, c="#9467bd", lw=1, label="yaw (relative to rest)")
    ax[0].plot(d.timestamp, d.roll_rel, c="#1f77b4", lw=.8, label="roll (relative to rest)")
    ax[0].plot(d.timestamp, d.pitch_rel, c="#d62728", lw=.8, label="pitch (relative to rest)")
    ax[0].set_ylabel("angle (°)")
    ax[0].legend(loc="lower right", fontsize=8)
    for _, m in M.iterrows():
        ax[0].axvspan(m.start, m.end, color=colors[m.type], alpha=.25)
        ax[0].text(m.start, ax[0].get_ylim()[1] * .9, m.type, fontsize=8, color=colors[m.type])
    for c, col in colors.items():
        x = W[W["class"] == c]
        ax[1].scatter(x.start, [0] * len(x), c=col, s=40, marker="|", label=c)
    ax[1].set_yticks([])
    ax[1].legend(loc="upper right", ncol=4, fontsize=8)
    ax[1].set_title("Class of each window (k-means)", loc="left", fontsize=9)
    note_colors = {"right": "#2b8cbe", "left": "#e6550d", "up-left-down": "#756bb1"}
    for hms, lab in NOTES.items():
        t = pd.Timestamp(f"{DATE} {hms}")
        ax[2].axvline(t, c=note_colors[lab], lw=2)
        ax[2].text(t, 0.5, f" {lab}", color=note_colors[lab], fontsize=8)
    ax[2].set_yticks([])
    ax[2].set_title("Andrew's notes", loc="left", fontsize=9)
    ax[0].set_title("Movements detected by k-means (without using the notes)")
    plt.tight_layout()
    plt.savefig(filename, dpi=110)
    print(f"  Plot saved: {filename}")


def analyse(d, window_s, k, verbose=False):
    W = make_windows(d, window_s)
    g, names = run_kmeans(W, k, verbose=verbose)
    W["class"] = pd.Series(g).map(names).values
    return W, compare(detected_movements(W))


def classify_noted(M):
    """How each noted movement was classified, as counts."""
    noted = M[M.andrew_note != "—"]
    res = {"right": [], "left": [], "up-left-down": []}
    for _, r in noted.iterrows():
        res[r.andrew_note.split(" ")[0]].append(r.type)
    total = pd.Series(list(NOTES.values())).value_counts()   # how many of each were noted
    return {"right_as_yaw_down": res["right"].count("yaw down"), "n_right": int(total["right"]),
            "left_as_small": res["left"].count("small movement"), "n_left": int(total["left"]),
            "uld_as_yaw_up": res["up-left-down"].count("yaw up"), "n_uld": int(total["up-left-down"]),
            "noted_found": len(noted), "extra": int((M.andrew_note == "—").sum())}


def robustness(d, ks=(3, 4, 5, 6), windows=(2, 3, 4, 5, 6)):
    rows = []
    for k in ks:
        for ws in windows:
            _, Mt = analyse(d, ws, k)
            rows.append({"k": k, "window_s": ws, **classify_noted(Mt)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = add_relative_angles(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else DATA,
                                        parse_dates=["timestamp"]))
    print(f"Rows: {len(d)}")

    print(f"\n=== {WINDOW_S:.0f} s windows, k={K}")
    W, M = analyse(d, WINDOW_S, K, verbose=True)
    print(f"  Windows: {len(W)}")
    print("\n  Windows by note (rows) x k-means class (columns):")
    print("    " + pd.crosstab(W.label, W["class"]).to_string().replace("\n", "\n    "))
    print("\n  Detected movements:")
    print("    " + M.assign(start=M.start.dt.strftime("%H:%M:%S"), end=M.end.dt.strftime("%H:%M:%S"))
          .to_string(index=False).replace("\n", "\n    "))
    print("\n  Noted movement (rows) x detected type (columns):")
    print("    " + mapping_table(M).to_string().replace("\n", "\n    "))
    OUT.mkdir(exist_ok=True)
    plot(d, W, M, OUT / "movements_kmeans.png")
    W.to_csv(OUT / "movement_windows.csv", index=False)
    M.to_csv(OUT / "detected_movements.csv", index=False)

    print("\n=== Robustness: window length x number of groups")
    print("  How the 7 noted movements were classified")
    R = robustness(d)
    show = pd.DataFrame({
        "k": R.k, "window_s": R.window_s,
        "right->yaw down": R.right_as_yaw_down.astype(str) + "/" + R.n_right.astype(str),
        "left->small": R.left_as_small.astype(str) + "/" + R.n_left.astype(str),
        "uld->yaw up": R.uld_as_yaw_up.astype(str) + "/" + R.n_uld.astype(str),
        "noted found": R.noted_found.astype(str) + "/7", "extra": R.extra})
    print("    " + show.to_string(index=False).replace("\n", "\n    "))
    show.to_csv(OUT / "movement_robustness.csv", index=False)
    print(f"\nSaved in {OUT}: movement_windows.csv, detected_movements.csv, movement_robustness.csv, "
          "movements_kmeans.png")
