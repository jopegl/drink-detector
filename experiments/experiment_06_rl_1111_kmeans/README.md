# Experiment 06 · Right / left 11-11 (k-means)

48-minute recording with ~10 right and ~9 left arm movements among natural ones (writing, scratching, stretching), plus a video. Events are detected automatically and grouped with k-means; times from the video and R/L from Andrew's notes are used only to evaluate.

**Method:** k-means (k = 2) on one feature card per detected event, yaw drift removed, angles relative to the 3 s before each event.

Shared steps (core/events.py): yaw drift removed (the BMI160 has no magnetometer, ~70°/min drift), events detected by a
rest/active state machine on the movement intensity, one feature card per event with angles relative to the 3 s
before it, and the ground truth (`data/annotations.csv` in experiment 06: times from the video, R/L from Andrew's
notes) attached only to evaluate.

## Run (from the repository root)

```
python experiments/experiment_06_rl_1111_kmeans/prepare.py
python experiments/experiment_06_rl_1111_kmeans/movements_kmeans.py
python experiments/experiment_06_rl_1111_kmeans/build_results.py
```

## Result

15/19 R/L correct with k = 2 (ARI 0.30); all 23 annotated movements detected, 2 extra events.
