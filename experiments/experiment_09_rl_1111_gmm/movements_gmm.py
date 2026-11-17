"""
Experiment 09, step 2: gmm on the detected events of the 11/11 recording.

Usage (from anywhere):
    python experiments/experiment_09_rl_1111_gmm/movements_gmm.py
    (reads the data of experiment 06, writes outputs/)

Same events and feature cards as experiment 06 (core/events.py); only the grouping method changes.
"""
import sys
import warnings
from pathlib import Path

from sklearn.mixture import GaussianMixture

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core import cluster_build as cb  # noqa: E402
from core.experiments import load_info  # noqa: E402

warnings.filterwarnings("ignore")

INFO = load_info(HERE)
DATA_DIR = (HERE / INFO["data_dir"]).resolve()
OUT = HERE / "outputs"
DAY = INFO["date"]

DEFAULT = {"k": 2}
GRID = {"k": [2, 3, 4]}
N_SEEDS = 20


def group_fn(X, k=2, seed=0):
    g = GaussianMixture(n_components=k, covariance_type="diag", n_init=10, reg_covar=1e-3, random_state=seed).fit(X)
    return g.predict(X), {"per_event": {"confidence": g.predict_proba(X).max(axis=1).round(3)},
                          "scalars": {"bic": round(float(g.bic(X)), 1)}}


def describe(p):
    return f"k = {p['k']}"


if __name__ == "__main__":
    cb.analyse(DATA_DIR, OUT, DAY, group_fn, DEFAULT, "gmm")
