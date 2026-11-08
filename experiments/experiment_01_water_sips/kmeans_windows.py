"""
Experiment 01, step 3 (main detector): k-means on sliding windows over the whole log (does NOT use the gesture split).

Usage (from anywhere):
    python experiments/experiment_01_water_sips/kmeans_windows.py   (reads data/processed_log.csv)

Requires: pandas, numpy, scikit-learn, matplotlib

How it works:
  1. Cut the whole log into WINDOW_S-second windows with 50% overlap.
  2. Compute a feature card for each window.
  3. k-means splits the windows into K groups WITHOUT seeing any label.
  4. The "sip" group is chosen WITHOUT labels: the group whose centre has the
     wrist tilted back the most (lowest pitch_min).
  5. Neighbouring "sip" windows are merged into one detected event.
  6. Only at the end are the events compared with Andrew's notes.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "processed_log.csv"
OUT = HERE / "outputs"

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
WINDOW_S = 4.0       # window length in seconds
OVERLAP = 0.5        # 50% overlap between windows
K = 3                # groups: idle / moving / sip
N_SEEDS = 20         # k-means runs to check stability
RATE_HZ = 14.0       # readings per second
MERGE_S = 10.0       # "sip" windows closer than this = same event

WATER_NOTES = ["16:53", "16:59", "17:19", "17:25", "17:49", "18:10", "18:20", "18:30", "18:37"]
RED_HERRING_NOTES = ["18:35"]
TOLERANCE_S = 150    # maximum distance between a detected event and a note

FEATURES = ["pitch_min", "pitch_max", "roll_min", "roll_max",
            "pitch_std", "roll_std", "tran_mean", "tran_max"]


# ---------------------------------------------------------------------------
# 1. Windowing
# ---------------------------------------------------------------------------
def make_windows(d, window_s=WINDOW_S):
    n = int(window_s * RATE_HZ)
    step = max(1, int(n * (1 - OVERLAP)))
    cards = []
    for i in range(0, len(d) - n, step):
        x = d.iloc[i:i + n]
        cards.append({
            "start": x.timestamp.iloc[0],
            "end": x.timestamp.iloc[-1],
            "label": x.label.mode()[0],  # for checking only; k-means does not use it
            "pitch_min": x.gyro_pitch.min(),
            "pitch_max": x.gyro_pitch.max(),
            "roll_min": x.gyro_roll.min(),
            "roll_max": x.gyro_roll.max(),
            "pitch_std": x.gyro_pitch.std(),
            "roll_std": x.gyro_roll.std(),
            "tran_mean": x.tran_mag.mean(),
            "tran_max": x.tran_mag.max(),
        })
    return pd.DataFrame(cards)


# ---------------------------------------------------------------------------
# 2. k-means (no labels) + choosing the "sip" group (no labels)
# ---------------------------------------------------------------------------
def run_kmeans(W, k=K, verbose=True):
    X = StandardScaler().fit_transform(W[FEATURES])

    def canon(r):
        m = {}
        return tuple(m.setdefault(v, len(m)) for v in r)

    runs = pd.Series([
        canon(KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(X))
        for s in range(N_SEEDS)
    ]).value_counts()
    groups = np.array(runs.index[0])

    # "sip" group = the one where, on average, the wrist is tilted back the most
    means = W.assign(g=groups).groupby("g")[FEATURES].mean()
    sip_group = means.pitch_min.idxmin()
    # "idle" group = the one with the least movement
    idle_group = means.tran_mean.idxmin()

    if verbose:
        print(f"  same split in {runs.iloc[0]}/{N_SEEDS} runs | "
              f"silhouette = {silhouette_score(X, groups, sample_size=2000, random_state=0):.2f}")
        print("  Mean of each group:")
        print(means.round(1).to_string().replace("\n", "\n    "))
    names = {g: ("sip" if g == sip_group else "idle" if g == idle_group else "moving")
             for g in range(k)}
    return groups, names


# ---------------------------------------------------------------------------
# 3. Merge "sip" windows into events
# ---------------------------------------------------------------------------
def detected_events(W):
    s = W[W["class"] == "sip"].sort_values("start")
    if s.empty:
        return pd.DataFrame(columns=["start", "end", "n_windows"])
    events, current = [], [s.iloc[0]]
    for _, r in s.iloc[1:].iterrows():
        if (r.start - current[-1].end).total_seconds() <= MERGE_S:
            current.append(r)
        else:
            events.append(current)
            current = [r]
    events.append(current)
    return pd.DataFrame([{"start": e[0].start, "end": e[-1].end, "n_windows": len(e)}
                         for e in events])


# ---------------------------------------------------------------------------
# 4. Compare with Andrew's notes (labels only come in now)
# ---------------------------------------------------------------------------
def compare(E, day):
    def dist(ev, note):
        target = pd.Timestamp(f"{day} {note}:30")
        if ev.start <= target <= ev.end:
            return 0.0
        return min(abs((ev.start - target).total_seconds()), abs((ev.end - target).total_seconds()))

    E = E.copy()
    E["andrew_note"] = "—"
    used = set()
    for notes, name in [(WATER_NOTES, "water"), (RED_HERRING_NOTES, "red_herring")]:
        for n in notes:
            if E.empty:
                continue
            ds = E.apply(lambda ev: dist(ev, n), axis=1)
            i = ds.idxmin()
            if ds[i] <= TOLERANCE_S and i not in used:
                E.loc[i, "andrew_note"] = f"{name} ({n})"
                used.add(i)
    found = [n for n in WATER_NOTES if any(E.andrew_note == f"water ({n})")]
    rh = [n for n in RED_HERRING_NOTES if any(E.andrew_note == f"red_herring ({n})")]
    extras = E[E.andrew_note == "—"]
    return E, found, rh, extras


def summary(W, E, found, rh, extras):
    print(f"\n  Annotated sips found: {len(found)}/{len(WATER_NOTES)}"
          + ("" if len(found) == len(WATER_NOTES)
             else f"  (missed: {sorted(set(WATER_NOTES) - set(found))})"))
    print(f"  Red herrings flagged as sips: {len(rh)}/{len(RED_HERRING_NOTES)}")
    print(f"  Detected events with no note: {len(extras)}"
          + (f"  -> {[t.strftime('%H:%M:%S') for t in extras.start]}" if len(extras) else ""))
    print("\n  Windows by label (rows) x k-means class (columns):")
    print(pd.crosstab(W.label, W["class"]).to_string().replace("\n", "\n    "))


def plot(d, W, E, filename="kmeans_windows.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    day = d.timestamp.dt.normalize().iloc[0]
    fig, ax = plt.subplots(3, 1, figsize=(16, 8), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1, 1]})
    ax[0].plot(d.timestamp, d.gyro_pitch, lw=.5, c="#d62728", label="gyro pitch")
    ax[0].plot(d.timestamp, d.gyro_roll, lw=.5, c="#1f77b4", label="gyro roll")
    ax[0].set_ylabel("angle (°)")
    ax[0].legend(loc="upper right")
    for _, e in E.iterrows():
        ax[0].axvspan(e.start, e.end, color="#2ca02c", alpha=.3)
    colors = {"idle": "#dddddd", "moving": "#ff9f40", "sip": "#2ca02c"}
    for c, col in colors.items():
        x = W[W["class"] == c]
        ax[1].scatter(x.start, [0] * len(x), c=col, s=12, marker="|", label=c)
    ax[1].set_yticks([])
    ax[1].set_title("Class of each window (k-means)", loc="left", fontsize=9)
    ax[1].legend(loc="upper right", ncol=3, fontsize=8)
    for n in WATER_NOTES:
        ax[2].axvline(pd.Timestamp(f"{day.date()} {n}:30"), c="#2b8cbe", lw=2)
    for n in RED_HERRING_NOTES:
        ax[2].axvline(pd.Timestamp(f"{day.date()} {n}:30"), c="#e34a33", lw=2)
    ax[2].set_yticks([])
    ax[2].set_title("Andrew's notes (blue = water, red = red herring)", loc="left", fontsize=9)
    ax[0].set_title("Green bands = sips detected by k-means (without using the notes)")
    plt.tight_layout()
    plt.savefig(filename, dpi=110)
    print(f"  Plot saved: {filename}")


if __name__ == "__main__":
    d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else DATA, parse_dates=["timestamp"])
    day = d.timestamp.iloc[0].strftime("%Y-%m-%d")
    print(f"Rows: {len(d)}")

    # --- Main analysis -------------------------------------------------------
    print(f"\n=== {WINDOW_S:.0f} s windows, k={K}")
    W = make_windows(d)
    print(f"  Windows: {len(W)}")
    groups, names = run_kmeans(W)
    W["group"] = groups
    W["class"] = W.group.map(names)
    E, found, rh, extras = compare(detected_events(W), day)
    summary(W, E, found, rh, extras)
    print("\n  Detected events:")
    print(E.assign(start=E.start.dt.strftime("%H:%M:%S"), end=E.end.dt.strftime("%H:%M:%S"))
          .to_string(index=False).replace("\n", "\n    "))
    OUT.mkdir(exist_ok=True)
    plot(d, W, E, OUT / "kmeans_windows.png")
    W.to_csv(OUT / "windows_kmeans.csv", index=False)
    E.to_csv(OUT / "detected_events.csv", index=False)

    # --- Robustness: other window lengths -----------------------------------
    print("\n=== Robustness: other window lengths")
    print("  window | sips found | red herring as sip | events with no note")
    for ws in [2, 3, 4, 5, 6, 8]:
        Wt = make_windows(d, ws)
        g, n = run_kmeans(Wt, verbose=False)
        Wt["class"] = pd.Series(g).map(n).values
        _, f, r, x = compare(detected_events(Wt), day)
        print(f"   {ws:>2} s  |    {len(f)}/{len(WATER_NOTES)}     |"
              f"        {len(r)}/{len(RED_HERRING_NOTES)}         |"
              f"  {len(x)} {[t.strftime('%H:%M') for t in x.start]}")

    print(f"\nSaved in {OUT}: windows_kmeans.csv, detected_events.csv, kmeans_windows.png")
