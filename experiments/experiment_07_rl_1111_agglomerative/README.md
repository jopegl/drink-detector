# Experiment 07 · Right / left 11-11 (agglomerative)

Same 11/11 recording as experiment 06 (right / left arm movements among natural ones), grouped with agglomerative instead of k-means. Truth (video times + Andrew's notes) is used only to evaluate.

**Method:** Agglomerative (Ward) clustering, k = 2, on one feature card per detected event.

Uses the data of experiment 06 (`../experiment_06_rl_1111_kmeans/data/`): run its `prepare.py` first.

Shared steps (core/events.py): yaw drift removed (the BMI160 has no magnetometer, ~70°/min drift), events detected by a
rest/active state machine on the movement intensity, one feature card per event with angles relative to the 3 s
before it, and the ground truth (`data/annotations.csv` in experiment 06: times from the video, R/L from Andrew's
notes) attached only to evaluate.

## Run (from the repository root)

```
python experiments/experiment_07_rl_1111_agglomerative/movements_agglomerative.py
python experiments/experiment_07_rl_1111_agglomerative/build_results.py
```

## Result

11/19 R/L correct with k = 2 (ARI 0.00); best setting 15/19 (k = 4).
