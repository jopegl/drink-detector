# Experiment 09 · Right / left 11-11 (GMM)

Same 11/11 recording as experiment 06 (right / left arm movements among natural ones), grouped with a Gaussian Mixture Model instead of k-means. Truth (video times + Andrew's notes) is used only to evaluate.

**Method:** Gaussian Mixture Model (k = 2, diagonal covariance) on one feature card per detected event.

Uses the data of experiment 06 (`../experiment_06_rl_1111_kmeans/data/`): run its `prepare.py` first.

Shared steps (core/events.py): yaw drift removed (the BMI160 has no magnetometer, ~70°/min drift), events detected by a
rest/active state machine on the movement intensity, one feature card per event with angles relative to the 3 s
before it, and the ground truth (`data/annotations.csv` in experiment 06: times from the video, R/L from Andrew's
notes) attached only to evaluate.

## Run (from the repository root)

```
python experiments/experiment_09_rl_1111_gmm/movements_gmm.py
python experiments/experiment_09_rl_1111_gmm/build_results.py
```

## Result

15/19 R/L correct with k = 2 (ARI 0.30); best setting 17/19 (k = 4, threshold 70).
