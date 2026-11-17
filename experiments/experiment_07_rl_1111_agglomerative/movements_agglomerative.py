"""
Experiment 07, step 2: agglomerative on the detected events of the 11/11 recording.

Usage (from anywhere):
    python experiments/experiment_07_rl_1111_agglomerative/movements_agglomerative.py
    (reads the data of experiment 06, writes outputs/)

Same events and feature cards as experiment 06 (core/events.py); only the grouping method changes.
"""
import sys
import warnings
from pathlib import Path

from sklearn.cluster import AgglomerativeClustering

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


def group_fn(X, k=2):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X), {}


def describe(p):
    return f"k = {p['k']}"


if __name__ == "__main__":
    cb.analyse(DATA_DIR, OUT, DAY, group_fn, DEFAULT, "agglomerative")
