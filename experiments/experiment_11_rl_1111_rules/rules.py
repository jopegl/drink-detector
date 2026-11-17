"""
Experiment 11, step 2: simple rules, the baseline the other methods must beat.

Usage (from anywhere):
    python experiments/experiment_11_rl_1111_rules/rules.py
    (reads the data of experiment 06, writes outputs/)

Three parts
  1. Detection with a state machine (core/events.py): REST -> ACTIVE above HIGH, back to REST below LOW.
  2. Transfer test: the rule found in experiment 02 ("large yaw turn down = right, otherwise left")
     applied as is to this recording.
  3. One-threshold rule (OneR): pick the single feature and threshold that best separate R from L.
     Because this rule IS learned from the labels, it is tested with leave-one-out: for each movement,
     the rule is chosen WITHOUT it, then applied to it.
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

HIGHS = [40, 55, 70]
LOWS = [10, 15, 20]


# ---------------------------------------------------------------------------
# 2. Rule from experiment 02
# ---------------------------------------------------------------------------
def exp02_rule(row):
    if row.yaw_min < -45 and -row.yaw_min > row.yaw_max:
        return "R"          # "yaw down" was right in experiment 02
    return "L"              # "small movement" was left in experiment 02


# ---------------------------------------------------------------------------
# 3. One-threshold rule
# ---------------------------------------------------------------------------
def fit_one_rule(F, y):
    """Best (feature, threshold, side) on the training events. side = label given to values >= threshold."""
    best = None
    for f in ev.FEATURES:
        v = F[f].values
        cuts = np.unique(v)
        cuts = (cuts[:-1] + cuts[1:]) / 2 if len(cuts) > 1 else cuts
        for c in cuts:
            for side in ("R", "L"):
                other = "L" if side == "R" else "R"
                pred = np.where(v >= c, side, other)
                acc = (pred == y).mean()
                if best is None or acc > best[0]:
                    best = (acc, f, float(c), side)
    return {"train_accuracy": best[0], "feature": best[1], "threshold": best[2], "side": best[3]}


def apply_rule(rule, F):
    other = "L" if rule["side"] == "R" else "R"
    return np.where(F[rule["feature"]].values >= rule["threshold"], rule["side"], other)


def leave_one_out(E):
    rl = E[E.truth_label.isin(ev.MOVEMENT_LABELS)].reset_index(drop=True)
    preds, chosen = [], []
    for i in range(len(rl)):
        tr = rl.drop(index=i)
        rule = fit_one_rule(tr, tr.truth_label.values)
        preds.append(apply_rule(rule, rl.loc[[i]])[0])
        chosen.append(rule["feature"])
    rl["loo_pred"] = preds
    rl["loo_rule_feature"] = chosen
    return rl


def run(d, T, high=ev.HIGH, low=ev.LOW):
    E = ev.match_truth(ev.event_features(d, ev.detect_events(d, high=high, low=low)), T, DAY)
    E["exp02_rule"] = [exp02_rule(r) for r in E.itertuples()]
    rl = leave_one_out(E)
    rule = fit_one_rule(rl, rl.truth_label.values)       # final rule on all labeled movements (for display)
    E["one_rule"] = apply_rule(rule, E)
    E = E.merge(rl[["event", "loo_pred", "loo_rule_feature"]], on="event", how="left")
    m = E.truth_label.isin(ev.MOVEMENT_LABELS)
    res = {"n": int(m.sum()),
           "exp02_correct": int((E.exp02_rule[m] == E.truth_label[m]).sum()),
           "loo_correct": int((E.loo_pred[m] == E.truth_label[m]).sum()),
           "rule": rule,
           "loo_features": rl.loo_rule_feature.value_counts().to_dict()}
    return E, res


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    d, _, T = ev.load_experiment_data(DATA_DIR, DAY)
    E, res = run(d, T)
    det = ev.detection_summary(E, T)
    print(f"Detection (state machine, high {ev.HIGH}, low {ev.LOW}): {det['truth_found']}/{det['truth_total']} "
          f"annotated movements, {det['events_without_truth']} extra events")
    print(f"Rule from experiment 02 (yaw down = R): {res['exp02_correct']}/{res['n']} correct")
    r = res["rule"]
    print(f"One-threshold rule (all data): {r['side']} if {r['feature']} >= {r['threshold']:.1f} "
          f"(training accuracy {r['train_accuracy']:.0%})")
    print(f"One-threshold rule, leave-one-out: {res['loo_correct']}/{res['n']} correct "
          f"| features chosen: {res['loo_features']}")
    print(E[["event", "start", "truth_id", "truth_label", "exp02_rule", "one_rule", "loo_pred", r["feature"]]]
          .assign(start=E.start.dt.strftime("%H:%M:%S")).round(1).to_string(index=False))
    ev.save_events(E, OUT / "events_default.csv")
