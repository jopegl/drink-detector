"""Find experiment folders and read their experiment.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def load_info(folder):
    with open(Path(folder) / "experiment.json") as f:
        info = json.load(f)
    info["folder"] = Path(folder)
    info["code"] = f"{info['number']:02d}"
    return info


def list_experiments():
    """Every experiments/experiment_XX_*/ folder that has an experiment.json, sorted by number."""
    found = [load_info(p) for p in sorted(EXPERIMENTS_DIR.glob("experiment_*"))
             if (p / "experiment.json").exists() and (p / "dashboard.py").exists()]
    return sorted(found, key=lambda e: e["number"])
