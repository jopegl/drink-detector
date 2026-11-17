# Experiment 12 · Right / left 11-11 (DTW)

Same 11/11 recording as experiment 06 (right / left arm movements among natural ones). This experiment compares the full shape of each movement with Dynamic Time Warping (template matching), allowing movements done faster or slower. Truth (video times + Andrew's notes) is used only to evaluate.

**Method:** Dynamic Time Warping between the events' angle curves (relative to the 3 s before each event): 1-nearest-neighbour template matching (leave-one-out) and hierarchical clustering.

Uses the data of experiment 06 (`../experiment_06_rl_1111_kmeans/data/`): run its `prepare.py` first.

Shared steps (core/events.py): yaw drift removed (the BMI160 has no magnetometer, ~70°/min drift), events detected by a
rest/active state machine on the movement intensity, one feature card per event with angles relative to the 3 s
before it, and the ground truth (`data/annotations.csv` in experiment 06: times from the video, R/L from Andrew's
notes) attached only to evaluate.

## Run (from the repository root)

```
python experiments/experiment_12_rl_1111_dtw/dtw.py
python experiments/experiment_12_rl_1111_dtw/build_results.py
```

## Result

Template matching with DTW: 19/19 in leave-one-out (18-19/19 in every setting); unsupervised clustering on DTW distances fails (10/19).
