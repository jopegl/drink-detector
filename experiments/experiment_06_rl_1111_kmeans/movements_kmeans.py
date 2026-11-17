"""
Experiment 06, step 2: k-means on the detected events.

Usage (from anywhere):
    python experiments/experiment_06_rl_1111_kmeans/movements_kmeans.py
    (reads data/processed_log.csv, writes outputs/)

Each detected event becomes a feature card (core/events.py). k-means splits the events into K groups
WITHOUT seeing any label. Only then is each group compared with the truth (video + Andrew's notes).
"""
import sys
import warnings
from pathlib import Path

from sklearn.cluster import KMeans

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
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X), {}


def describe(p):
    return f"k = {p['k']}"


if __name__ == "__main__":
    cb.analyse(DATA_DIR, OUT, DAY, group_fn, DEFAULT, "k-means")
