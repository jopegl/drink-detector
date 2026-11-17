"""
Shared build steps for the clustering experiments (06 k-means, 07 agglomerative, 08 DBSCAN, 09 GMM).

Each experiment only provides a function   group_fn(X, **params) -> (groups, extra)
  - X       : standardised feature matrix, one row per detected event
  - groups  : integer group per event (-1 = noise, DBSCAN only)
  - extra   : {"per_event": {column: values}, "scalars": {name: value}}  (optional extras, e.g. GMM confidence)
"""
import itertools

import pandas as pd
from sklearn.preprocessing import StandardScaler

from core import events as ev

HIGHS = [40, 55, 70]   # detection thresholds tested in the robustness grid


def run_setting(d, T, day, group_fn, params, high=ev.HIGH):
    E = ev.match_truth(ev.event_features(d, ev.detect_events(d, high=high)), T, day)
    X = StandardScaler().fit_transform(E[ev.FEATURES])
    groups, extra = group_fn(X, **params)
    E["group"] = pd.Series(groups).astype(int).astype(str).values
    res = ev.evaluate_groups(E)
    E["named_as"] = E.group.map(res["mapping"])
    for col, values in (extra or {}).get("per_event", {}).items():
        E[col] = values
    return E, res, (extra or {}).get("scalars", {})


def build(data_dir, out_dir, day, group_fn, default, grid, describe_params, extra_summary=None):
    """
    default : dict of the default parameters
    grid    : dict {param: [values]} tested together with the detection threshold
    describe_params(params) -> short text used in the robustness table
    """
    out_dir.mkdir(exist_ok=True)
    d, _, T = ev.load_experiment_data(data_dir, day)

    E, res, scalars = run_setting(d, T, day, group_fn, default)
    det = ev.detection_summary(E, T)
    ev.save_events(E, out_dir / "events.csv")
    print(f"Default {describe_params(default)}: {res['correct']}/{res['n']} R/L correct, ARI {res['ari']:.2f} "
          f"| detected {det['truth_found']}/{det['truth_total']} annotated movements, "
          f"{det['events_without_truth']} extra events")

    rows = []
    keys = list(grid)
    for high in HIGHS:
        for combo in itertools.product(*[grid[k] for k in keys]):
            p = dict(zip(keys, combo))
            Et, rt, st = run_setting(d, T, day, group_fn, p, high=high)
            dt = ev.detection_summary(Et, T)
            rows.append({"high": high, **p, "setting": describe_params(p),
                         "correct": rt["correct"], "n": rt["n"], "accuracy": round(rt["accuracy"], 3),
                         "ari": round(rt["ari"], 3), "n_groups": len(set(Et.group) - {"-1"}),
                         "n_noise": int((Et.group == "-1").sum()),
                         "events": dt["events"], "rl_found": dt["rl_found"],
                         "text": f"{rt['correct']}/{rt['n']}<br>ARI {rt['ari']:.2f}", **st})
    R = pd.DataFrame(rows)
    R.to_csv(out_dir / "robustness.csv", index=False)
    best = R.sort_values(["accuracy", "ari"], ascending=False).iloc[0]
    print(f"Robustness: {len(R)} settings | accuracy min {R.accuracy.min():.2f}, median {R.accuracy.median():.2f}, "
          f"max {R.accuracy.max():.2f} (best: high {best.high}, {best.setting})")

    summary = {"default": default, "default_high": ev.HIGH, "detection": det, "evaluation": res,
               "others": ev.where_others_went(E, mapping=res["mapping"]), "scalars": scalars,
               "robustness": {"settings": len(R), "accuracy_min": float(R.accuracy.min()),
                              "accuracy_median": float(R.accuracy.median()),
                              "accuracy_max": float(R.accuracy.max()),
                              "share_at_least_90pct": float((R.accuracy >= 0.9).mean())}}
    if extra_summary:
        summary.update(extra_summary(d, T, E))
    ev.save_json(summary, out_dir / "summary.json")
    print(f"Saved: {out_dir}/events.csv, robustness.csv, summary.json")
    return E, R, summary


def analyse(data_dir, out_dir, day, group_fn, params, method_name):
    """Run one setting, print the results and save a PCA map of the events in outputs/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    out_dir.mkdir(exist_ok=True)
    d, _, T = ev.load_experiment_data(data_dir, day)
    E, res, scalars = run_setting(d, T, day, group_fn, params)
    print(f"{method_name} {params}: {len(E)} events")
    print(f"R/L classified correctly: {res['correct']}/{res['n']} | adjusted Rand index {res['ari']:.2f}")
    print("Groups named by majority (evaluation only):", res["mapping"])
    print("\nTruth (rows) x group (columns):")
    print(pd.crosstab(E.truth_label, E.group).to_string())
    print("\nUnlabeled movements and possible red herrings:", ev.where_others_went(E, mapping=res["mapping"]))
    if scalars:
        print("Extra:", scalars)

    X = StandardScaler().fit_transform(E[ev.FEATURES])
    P = PCA(2).fit(X)
    Z = P.transform(X)
    colors = {"R": "#2b8cbe", "L": "#e6550d", "?": "#9e9ac8", "other?": "#31a354", "—": "#bdbdbd"}
    markers = "osD^vP*Xhp<>"
    groups = sorted(E.group.unique(), key=lambda g: int(g))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for gi, g in enumerate(groups):
        m = (E.group == g).values
        ax.scatter(Z[m, 0], Z[m, 1], s=110, c=[colors.get(t, "#bdbdbd") for t in E.truth_label[m]],
                   marker=markers[gi % len(markers)], edgecolor="k", label=f"group {g} ({res['mapping'].get(g, '')})")
    for i, r in enumerate(E.itertuples()):
        ax.annotate(f"{r.start:%H:%M:%S}", Z[i], xytext=(5, 3), textcoords="offset points", fontsize=7)
    ev_ = P.explained_variance_ratio_
    ax.set_xlabel(f"component 1 ({ev_[0] * 100:.0f}%)")
    ax.set_ylabel(f"component 2 ({ev_[1] * 100:.0f}%)")
    ax.set_title(f"{method_name}: events (colour = truth: blue R, orange L; marker = group)", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"{method_name.lower().replace(' ', '_')}_events.png"
    plt.savefig(path, dpi=110)
    E_out = E.copy()
    ev.save_events(E_out, out_dir / "events_default.csv")
    print(f"\nSaved: {path} and {out_dir / 'events_default.csv'}")
    return E, res
