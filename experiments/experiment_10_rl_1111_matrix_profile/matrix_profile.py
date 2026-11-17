"""
Experiment 10, step 2: Matrix Profile on the continuous signal of the 11/11 recording.

Usage (from anywhere):
    python experiments/experiment_10_rl_1111_matrix_profile/matrix_profile.py
    (reads the data of experiment 06, writes outputs/)

What the Matrix Profile does
    For every window of M seconds in the recording, it finds the MOST SIMILAR other window anywhere in the
    recording (comparing the shape of the curves, z-normalised) and stores that distance.
      - low value  = this stretch repeats somewhere else  -> MOTIF (a repeated movement)
      - high value = this stretch looks like nothing else -> DISCORD (an odd movement, e.g. a red herring)
    It uses no labels and no number of groups.

How it is evaluated (labels only here)
    For every detected event, look at where its nearest neighbour is:
      - is it another movement at all?             (nn_on_event)
      - is it a movement of the SAME type (R/L)?   (nn_same_label)  -> like 1-nearest-neighbour template matching
    Discords: which events are the most unusual.

Implemented with numpy (MASS algorithm, FFT-based); checked against a brute-force computation.
No extra library is needed.
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

RESAMPLE_HZ = 4            # the signal is averaged to 4 readings per second (enough for 2-6 s movements)
DEFAULT = {"signal": "angles", "m_s": 4}
GRID = {"signal": ["angles", "rates"], "m_s": [2, 3, 4, 6]}
SIGNALS = {"angles": ["yaw_dt", "gyro_roll", "gyro_pitch"],     # drift-free yaw + roll + pitch
           "rates": ["tran_x", "tran_y", "tran_z"]}             # rotation rates (no drift by nature)
N_DISCORDS = 10


# ---------------------------------------------------------------------------
# Matrix Profile (multi-channel: distances of each channel are added)
# ---------------------------------------------------------------------------
def _mp_numpy(X, m, batch=512):
    n, c = X.shape
    k = n - m + 1
    N = 1 << int(np.ceil(np.log2(n + m)))
    total = np.zeros(k)
    prof = np.full(k, np.inf)
    idx = np.zeros(k, dtype=int)
    stats = []
    for ch in range(c):
        x = X[:, ch]
        cs, cs2 = np.cumsum(np.r_[0, x]), np.cumsum(np.r_[0, x * x])
        mu = (cs[m:] - cs[:-m]) / m
        sd = np.sqrt(np.maximum((cs2[m:] - cs2[:-m]) / m - mu ** 2, 1e-12))
        stats.append((np.fft.rfft(x, N), mu, sd, x))
    excl = m
    for b0 in range(0, k, batch):
        b1 = min(k, b0 + batch)
        D = np.zeros((b1 - b0, k))
        for Fx, mu, sd, x in stats:
            Q = np.stack([x[i:i + m][::-1] for i in range(b0, b1)])
            QT = np.fft.irfft(np.fft.rfft(Q, N, axis=1) * Fx[None, :], N, axis=1)[:, m - 1:m - 1 + k]
            num = QT - m * mu[b0:b1, None] * mu[None, :]
            corr = num / (m * sd[b0:b1, None] * sd[None, :])
            D += np.maximum(2 * m * (1 - corr), 0)
        for r, i in enumerate(range(b0, b1)):
            D[r, max(0, i - excl):min(k, i + excl + 1)] = np.inf
        j = D.argmin(axis=1)
        prof[b0:b1] = np.sqrt(D[np.arange(b1 - b0), j])
        idx[b0:b1] = j
    return prof, idx


def matrix_profile(X, m):
    """Distance from every window to its nearest non-overlapping neighbour, and that neighbour's index."""
    return _mp_numpy(X, m)


def resample(d, cols):
    s = d.set_index("timestamp")[cols].resample(f"{int(1000 / RESAMPLE_HZ)}ms").mean().interpolate()
    return s


# ---------------------------------------------------------------------------
# Evaluation against the detected events (labels only here)
# ---------------------------------------------------------------------------
def evaluate(d, E, signal, m_s):
    s = resample(d, SIGNALS[signal])
    m = int(m_s * RESAMPLE_HZ)
    prof, nn = matrix_profile(s.values, m)
    times = s.index[:len(prof)]
    centre = times + pd.Timedelta(seconds=m_s / 2)

    def event_at(t):
        hit = E[(E.start - pd.Timedelta(seconds=1) <= t) & (E.end + pd.Timedelta(seconds=1) >= t)]
        return hit.index[0] if len(hit) else None

    rows = []
    for i, e in E.iterrows():
        q = int(np.clip(np.searchsorted(times, e.peak_time - pd.Timedelta(seconds=m_s / 2)), 0, len(prof) - 1))
        j = nn[q]
        hit = event_at(centre[j])
        rows.append({"event": e.event, "mp": float(prof[q]),
                     "mp_percentile": float((prof < prof[q]).mean() * 100),
                     "nn_time": centre[j], "nn_event": E.loc[hit, "event"] if hit is not None else 0,
                     "nn_label": E.loc[hit, "truth_label"] if hit is not None else "no event"})
    R = E.merge(pd.DataFrame(rows), on="event")
    rl = R[R.truth_label.isin(ev.MOVEMENT_LABELS)]
    on_rl = rl[rl.nn_label.isin(ev.MOVEMENT_LABELS)]
    res = {"n_rl": len(rl), "nn_on_event": int((rl.nn_event != 0).sum()),
           "nn_on_rl": len(on_rl), "nn_same_label": int((on_rl.nn_label == on_rl.truth_label).sum())}

    # discords: highest profile values, at least m apart
    order = np.argsort(prof)[::-1]
    picked = []
    for i in order:
        if all(abs(i - p) > m for p in picked):
            picked.append(i)
        if len(picked) == N_DISCORDS:
            break
    disc = []
    for rank, i in enumerate(picked, start=1):
        hit = event_at(centre[i])
        disc.append({"rank": rank, "time": centre[i], "mp": float(prof[i]),
                     "event": E.loc[hit, "event"] if hit is not None else 0,
                     "truth": E.loc[hit, "truth_label"] if hit is not None else "no event"})
    mp_series = pd.DataFrame({"time": centre, "mp": prof, "nn_time": centre[nn]})
    return R, res, pd.DataFrame(disc), mp_series


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    d, E, T = ev.load_experiment_data(DATA_DIR, DAY)
    R, res, disc, mps = evaluate(d, E, **DEFAULT)
    print(f"Matrix Profile {DEFAULT}: {len(E)} events")
    print(f"R/L movements whose nearest neighbour is another movement: {res['nn_on_event']}/{res['n_rl']}")
    print(f"... of those landing on an R/L movement, same type: {res['nn_same_label']}/{res['nn_on_rl']}")
    print("\nTop discords (most unusual stretches):")
    print(disc.assign(time=disc.time.dt.strftime("%H:%M:%S")).to_string(index=False))
    print("\nEvents:")
    print(R[["event", "start", "truth_id", "truth_label", "mp", "mp_percentile", "nn_event", "nn_label"]]
          .assign(start=R.start.dt.strftime("%H:%M:%S")).round(1).to_string(index=False))

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(mps.time, mps.mp, lw=.6, c="#444")
    for e in R.itertuples():
        c = {"R": "#2b8cbe", "L": "#e6550d"}.get(e.truth_label, "#31a354")
        ax.axvspan(e.start, e.end, color=c, alpha=.3)
    for r in disc.itertuples():
        ax.annotate(f"D{r.rank}", (r.time, r.mp), fontsize=8, color="red")
    ax.set_ylabel("matrix profile (distance to nearest neighbour)")
    ax.set_title("Low = repeated stretch (motif), high = unusual (discord). Bands: events (blue R, orange L, green other)")
    plt.tight_layout()
    plt.savefig(OUT / "matrix_profile.png", dpi=110)
    ev.save_events(R, OUT / "events_default.csv")
    print(f"\nSaved: {OUT / 'matrix_profile.png'} and {OUT / 'events_default.csv'}")
