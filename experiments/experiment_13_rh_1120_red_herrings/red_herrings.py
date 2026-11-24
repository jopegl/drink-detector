"""
Experiment 13, step 2: can we tell the red herrings apart from the right / left movements?

Usage (from anywhere):
    python experiments/experiment_13_rh_1120_red_herrings/red_herrings.py
    (reads data/processed_log.csv made by prepare.py, writes outputs/)

The 11/20 recording has 24 notes from Andrew, to the second: 9 right, 7 left, 8 red herrings (RH).

Shared steps
    - Yaw drift removed, events detected with the same state machine as experiments 06-12, but two peaks only
      merge if they are less than 6 s apart (here movements can be 10 s apart; 20 s would merge them).
    - Each note goes to ONE detected event: the closest one whose peak is between 3 s before and 12 s after
      the note. Andrew's notes come ~4 s before the peak in almost every case, so "closest" means closest to
      note + 4 s. Events with no note get the label "—".

Approaches (all scored with leave-one-out: each event is predicted by a model that never saw it)
    1. One rule          : the single feature + threshold that best splits RH from R/L (OneR).
    2. Logistic regression on the feature card (R / L / RH).
    3. DTW template matching: each event gets the label of its most similar other labeled event (R / L / RH).
    4. DTW unusualness, WITHOUT any red herring example: distance to the most similar R/L movement divided
       by the typical one. Above 1.5 (threshold fixed in experiment 12, before seeing this data) = "not a
       known movement" = red herring.
    5. Transfer: the R/L movements of the 11/11 recording (experiment 06) used as the only templates.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from core import events as ev  # noqa: E402
from core.experiments import load_info  # noqa: E402

warnings.filterwarnings("ignore")

INFO = load_info(HERE)
DATA_DIR = (HERE / INFO["data_dir"]).resolve()
OUT = HERE / "outputs"
DAY = INFO["date"]

LABELS = ("R", "L", "RH")
DEFAULT = {"high": 40.0, "min_gap_s": 6.0, "channels": "angles"}
NOTE_OFFSET_S = 4.0            # notes come ~4 s before the peak
NOTE_WINDOW_S = (-3.0, 12.0)   # a note can match an event whose peak is in [note - 3 s, note + 12 s]
UNUSUAL_THRESHOLD = 1.5        # fixed in experiment 12
STEP = 2                       # DTW: keep 1 reading out of 2 (~7 per second)
CHANNELS = {"angles": ["yaw", "roll", "pitch"], "rates": ["rate_x", "rate_y", "rate_z"]}
TRANSFER_DATA = ROOT / "experiments" / "experiment_06_rl_1111_kmeans" / "data"
TRANSFER_DAY = "2026-11-11"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_notes(path, day=DAY):
    T = pd.read_csv(path, dtype=str)
    T["time"] = pd.to_datetime(day + " " + T.note_time)
    return T


def detect(d, high, min_gap_s):
    return ev.event_features(d, ev.detect_events(d, high=high, min_gap_s=min_gap_s))


def match_notes(E, T):
    """One note -> one event, closest to note + NOTE_OFFSET_S inside NOTE_WINDOW_S (global greedy)."""
    E = E.copy()
    E["truth_id"], E["truth_label"], E["note_lag_s"] = "—", "—", np.nan
    pairs = []
    for r in T.itertuples():
        lag = (E.peak_time - r.time).dt.total_seconds()
        for i, x in lag.items():
            if NOTE_WINDOW_S[0] <= x <= NOTE_WINDOW_S[1]:
                pairs.append((abs(x - NOTE_OFFSET_S), i, r.id, r.label, x))
    used_e, used_n = set(), set()
    for _, i, nid, lab, x in sorted(pairs):
        if i in used_e or nid in used_n:
            continue
        used_e.add(i); used_n.add(nid)
        E.loc[i, ["truth_id", "truth_label", "note_lag_s"]] = [nid, lab, round(x, 1)]
    return E


def load_data(high=DEFAULT["high"], min_gap_s=DEFAULT["min_gap_s"]):
    d = ev.load_processed(DATA_DIR / INFO["processed_log"])
    T = load_notes(DATA_DIR / "annotations.csv")
    return d, match_notes(detect(d, high, min_gap_s), T), T


def detection_summary(E, T):
    return {"notes": len(T), "notes_matched": int(E.truth_id.isin(T.id).sum()),
            "by_label": {l: f"{int((E.truth_label == l).sum())}/{int((T.label == l).sum())}" for l in LABELS},
            "events": len(E), "events_without_note": int((E.truth_id == "—").sum()),
            "note_lag_median_s": float(E.note_lag_s.median())}


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------
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


def curves(d, E, channels):
    return [ev.relative_angles_around(d, e.start, e.end)[CHANNELS[channels]].values[::STEP] for e in E.itertuples()]


def distance_matrix(A, B=None):
    if B is None:
        k = len(A)
        M = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                M[i, j] = M[j, i] = dtw(A[i], A[j])
        return M
    return np.array([[dtw(a, b) for b in B] for a in A])


# ---------------------------------------------------------------------------
# Approaches (leave-one-out on the labeled events)
# ---------------------------------------------------------------------------
def _best_rule(X, y):
    """Best single feature + threshold for y (bool). Returns (feature, threshold, direction)."""
    best = (-1, None, None, None)
    for f in X.columns:
        v = np.sort(X[f].unique())
        for t in (v[:-1] + v[1:]) / 2:
            for sign in (1, -1):
                acc = ((sign * (X[f] - t) > 0) == y).mean()
                if acc > best[0]:
                    best = (acc, f, t, sign)
    return best[1:]


def one_rule(E):
    """OneR for RH vs movement. Returns predictions ('RH' / 'move') for every event and the full-data rule."""
    lab = E[E.truth_label.isin(LABELS)]
    X, y = lab[ev.FEATURES], (lab.truth_label == "RH").values
    pred = pd.Series("", index=E.index)
    for k, i in enumerate(lab.index):
        f, t, s = _best_rule(X.drop(i), np.delete(y, k))
        pred[i] = "RH" if s * (E.loc[i, f] - t) > 0 else "move"
    f, t, s = _best_rule(X, y)
    rest = E.index.difference(lab.index)
    pred[rest] = np.where(s * (E.loc[rest, f] - t) > 0, "RH", "move")
    return pred, {"feature": f, "threshold": round(float(t), 1), "direction": ">" if s > 0 else "<"}


def logistic(E):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))  # noqa: E731
    lab = E[E.truth_label.isin(LABELS)]
    pred = pd.Series("", index=E.index)
    for i in lab.index:
        tr = lab.drop(i)
        pred[i] = model().fit(tr[ev.FEATURES], tr.truth_label).predict(E.loc[[i], ev.FEATURES])[0]
    rest = E.index.difference(lab.index)
    if len(rest):
        pred[rest] = model().fit(lab[ev.FEATURES], lab.truth_label).predict(E.loc[rest, ev.FEATURES])
    return pred


def dtw_methods(E, M):
    """3-class 1-NN and unusualness (distance to the nearest OTHER R/L movement / typical one)."""
    lab = E.truth_label.values
    is_lab, is_rl = np.isin(lab, LABELS), np.isin(lab, ev.MOVEMENT_LABELS)
    nn_lab, nn_ev, nn_rl = [], [], []
    for i in range(len(E)):
        c = [j for j in np.where(is_lab)[0] if j != i]
        j = c[int(np.argmin(M[i, c]))]
        nn_lab.append(lab[j]); nn_ev.append(int(E.event.iloc[j]))
        c = [j for j in np.where(is_rl)[0] if j != i]
        nn_rl.append(float(M[i, c].min()))
    nn_rl = np.array(nn_rl)
    typical = float(np.median(nn_rl[is_rl]))
    return pd.Series(nn_lab, index=E.index), pd.Series(nn_ev, index=E.index), \
        pd.Series(np.round(nn_rl / typical, 2), index=E.index), typical


def transfer(d, E, channels):
    """Templates = R/L movements of the 11/11 recording. 1-NN label and unusualness for every 11/20 event."""
    d0, E0, _ = ev.load_experiment_data(TRANSFER_DATA, TRANSFER_DAY)
    E0 = E0[E0.truth_label.isin(ev.MOVEMENT_LABELS)].reset_index(drop=True)
    C0 = curves(d0, E0, channels)
    typical = float(np.median([np.delete(row, k).min() for k, row in enumerate(distance_matrix(C0))]))
    X = distance_matrix(curves(d, E, channels), C0)
    nn = pd.Series(E0.truth_label.values[X.argmin(axis=1)], index=E.index)
    return nn, pd.Series(np.round(X.min(axis=1) / typical, 2), index=E.index)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def auc(score, positive):
    """Probability that a random RH scores higher than a random R/L movement (0.5 = chance, 1 = perfect)."""
    p, n = score[positive], score[~positive]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in p for b in n]))


def binary_score(E, pred_rh):
    """pred_rh: bool per event. Scored on the labeled events, RH = positive."""
    lab = E.truth_label.isin(LABELS)
    y, p = (E.truth_label == "RH")[lab], pred_rh[lab]
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    return {"correct": tp + tn, "n": int(lab.sum()), "rh_caught": f"{tp}/{tp + fn}",
            "false_alarms": f"{fp}/{fp + tn}", "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def three_class_score(E, pred):
    lab = E.truth_label.isin(LABELS)
    return {"correct": int((pred[lab] == E.truth_label[lab]).sum()), "n": int(lab.sum()),
            "confusion": pd.crosstab(E.truth_label[lab], pred[lab]).to_dict()}


def run(d, E, channels="angles", with_transfer=True):
    """All approaches on one set of events. Returns (events with predictions, M, results)."""
    E = E.copy()
    res = {}
    E["rule_pred"], res["rule_found"] = one_rule(E)
    res["rule"] = binary_score(E, E.rule_pred == "RH")

    E["logreg_pred"] = logistic(E)
    res["logreg"] = three_class_score(E, E.logreg_pred)
    res["logreg_binary"] = binary_score(E, E.logreg_pred == "RH")

    M = distance_matrix(curves(d, E, channels))
    E["dtw_pred"], E["dtw_nn_event"], E["unusualness"], typical = dtw_methods(E, M)
    res["dtw"] = three_class_score(E, E.dtw_pred)
    res["dtw_binary"] = binary_score(E, E.dtw_pred == "RH")

    lab = E.truth_label.isin(LABELS)
    res["unusual"] = binary_score(E, E.unusualness > UNUSUAL_THRESHOLD)
    res["unusual"]["auc"] = round(auc(E.unusualness[lab].values, (E.truth_label == "RH")[lab].values), 3)
    res["unusual"]["typical_dist"] = typical

    if with_transfer:
        E["transfer_pred"], E["transfer_unusualness"] = transfer(d, E, channels)
        rl = E.truth_label.isin(ev.MOVEMENT_LABELS)
        res["transfer"] = {"rl_correct": int((E.transfer_pred[rl] == E.truth_label[rl]).sum()), "rl_n": int(rl.sum()),
                           "auc": round(auc(E.transfer_unusualness[lab].values,
                                            (E.truth_label == "RH")[lab].values), 3)}
        res["transfer_unusual"] = binary_score(E, E.transfer_unusualness > UNUSUAL_THRESHOLD)
    return E, M, res


def print_results(E, res):
    print(f"1. One rule (RH vs movement): {res['rule']['correct']}/{res['rule']['n']} correct | RH caught "
          f"{res['rule']['rh_caught']}, false alarms {res['rule']['false_alarms']} | rule on all data: "
          f"{res['rule_found']['feature']} {res['rule_found']['direction']} {res['rule_found']['threshold']} -> RH")
    for k, name in [("logreg", "2. Logistic regression"), ("dtw", "3. DTW template matching")]:
        b = res[k + "_binary"]
        print(f"{name}: {res[k]['correct']}/{res[k]['n']} correct (R/L/RH) | RH caught {b['rh_caught']}, "
              f"false alarms {b['false_alarms']}")
    u = res["unusual"]
    print(f"4. DTW unusualness (no RH examples): RH caught {u['rh_caught']}, false alarms {u['false_alarms']} "
          f"(threshold {UNUSUAL_THRESHOLD}) | AUC {u['auc']:.2f}")
    if "transfer" in res:
        t, tu = res["transfer"], res["transfer_unusual"]
        print(f"5. Transfer from 11/11 templates: R/L {t['rl_correct']}/{t['rl_n']} | RH caught {tu['rh_caught']}, "
              f"false alarms {tu['false_alarms']} | AUC {t['auc']:.2f}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d, E, T = load_data()
    det = detection_summary(E, T)
    print(f"{det['events']} events | notes matched {det['notes_matched']}/{det['notes']} {det['by_label']} | "
          f"{det['events_without_note']} events with no note | note -> peak: median {det['note_lag_median_s']:.1f} s")
    Eo, M, res = run(d, E, DEFAULT["channels"])
    print_results(Eo, res)
    cols = ["event", "start", "truth_id", "truth_label", "note_lag_s", "peak", "rule_pred", "logreg_pred", "dtw_pred",
            "unusualness", "transfer_pred", "transfer_unusualness"]
    print(Eo[cols].assign(start=Eo.start.dt.strftime("%H:%M:%S")).to_string(index=False))
    ev.save_events(Eo, OUT / "events_default.csv")
    print(f"\nSaved: {OUT / 'events_default.csv'}")
