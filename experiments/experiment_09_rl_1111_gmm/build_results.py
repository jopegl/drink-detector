"""
Experiment 09, step 3: run gmm for every setting and save results/ for the dashboard.

Usage (from anywhere):
    python experiments/experiment_09_rl_1111_gmm/build_results.py

Settings tested: k = 2, 3, 4  x  detection threshold 40, 55, 70.
"""
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import movements_gmm as mm  # noqa: E402
from core import cluster_build as cb  # noqa: E402
from core import events as ev  # noqa: E402,F401

warnings.filterwarnings("ignore")


def stability(d, T, E):
    """In how many random starts are the R/L movements grouped the same way (up to renaming)?"""
    from sklearn.metrics import adjusted_rand_score
    rl = E.truth_label.isin(ev.MOVEMENT_LABELS).values
    ref, same = None, 0
    for s in range(mm.N_SEEDS):
        Es, _, _ = cb.run_setting(d, T, mm.DAY, mm.group_fn, {**mm.DEFAULT, "seed": s})
        g = Es.group.values[rl]
        ref = g if ref is None else ref
        same += adjusted_rand_score(ref, g) == 1.0
    return {"stable_runs": int(same), "n_runs": mm.N_SEEDS}


if __name__ == "__main__":
    cb.build(mm.DATA_DIR, HERE / "results", mm.DAY, mm.group_fn, mm.DEFAULT, mm.GRID, mm.describe, extra_summary=stability)
