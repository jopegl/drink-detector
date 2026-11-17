"""
Experiment 08, step 2: dbscan on the detected events of the 11/11 recording.

Usage (from anywhere):
    python experiments/experiment_08_rl_1111_dbscan/movements_dbscan.py
    (reads the data of experiment 06, writes outputs/)

Same events and feature cards as experiment 06 (core/events.py); only the grouping method changes.
"""
import sys
import warnings
from pathlib import Path

from sklearn.cluster import DBSCAN

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core import cluster_build as cb  # noqa: E402
from core.experiments import load_info  # noqa: E402

warnings.filterwarnings("ignore")

INFO = load_info(HERE)
DATA_DIR = (HERE / INFO["data_dir"]).resolve()
OUT = HERE / "outputs"
DAY = INFO["date"]

DEFAULT = {"eps": 3.0, "min_samples": 3}
GRID = {"eps": [2.0, 2.5, 3.0, 3.5, 4.0], "min_samples": [2, 3, 4]}
N_SEEDS = 20


def group_fn(X, eps=3.0, min_samples=3):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X), {}


def describe(p):
    return f"eps {p['eps']}, ms {p['min_samples']}"


if __name__ == "__main__":
    cb.analyse(DATA_DIR, OUT, DAY, group_fn, DEFAULT, "dbscan")
