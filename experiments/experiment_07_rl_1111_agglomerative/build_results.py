"""
Experiment 07, step 3: run agglomerative for every setting and save results/ for the dashboard.

Usage (from anywhere):
    python experiments/experiment_07_rl_1111_agglomerative/build_results.py

Settings tested: k = 2, 3, 4  x  detection threshold 40, 55, 70.
"""
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import movements_agglomerative as mm  # noqa: E402
from core import cluster_build as cb  # noqa: E402
from core import events as ev  # noqa: E402,F401

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    cb.build(mm.DATA_DIR, HERE / "results", mm.DAY, mm.group_fn, mm.DEFAULT, mm.GRID, mm.describe)
