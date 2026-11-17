# Experiment 08 · Right / left 11-11 (DBSCAN)

Same 11/11 recording as experiment 06 (right / left arm movements among natural ones), grouped with dbscan instead of k-means. Truth (video times + Andrew's notes) is used only to evaluate.

**Method:** DBSCAN (eps = 3.0, min_samples = 3) on one feature card per detected event; noise = unusual.

Uses the data of experiment 06 (`../experiment_06_rl_1111_kmeans/data/`): run its `prepare.py` first.

Shared steps (core/events.py): yaw drift removed (the BMI160 has no magnetometer, ~70°/min drift), events detected by a
rest/active state machine on the movement intensity, one feature card per event with angles relative to the 3 s
before it, and the ground truth (`data/annotations.csv` in experiment 06: times from the video, R/L from Andrew's
notes) attached only to evaluate.

## Run (from the repository root)

```
python experiments/experiment_08_rl_1111_dbscan/movements_dbscan.py
python experiments/experiment_08_rl_1111_dbscan/build_results.py
```

## Result

6/19 R/L correct at the default (most events become noise); best setting 10/19: too few events for DBSCAN.
