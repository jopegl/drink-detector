# Experiment 10 · Right / left 11-11 (Matrix Profile)

Same 11/11 recording as experiment 06 (right / left arm movements among natural ones). This experiment finds repeated stretches (motifs) and unusual ones (discords) in the continuous signal with the Matrix Profile. Truth (video times + Andrew's notes) is used only to evaluate.

**Method:** Matrix Profile on the continuous signal (drift-free angles, 4 s windows): nearest-neighbour search for repeated stretches (motifs) and unusual ones (discords).

Uses the data of experiment 06 (`../experiment_06_rl_1111_kmeans/data/`): run its `prepare.py` first.

Shared steps (core/events.py): yaw drift removed (the BMI160 has no magnetometer, ~70°/min drift), events detected by a
rest/active state machine on the movement intensity, one feature card per event with angles relative to the 3 s
before it, and the ground truth (`data/annotations.csv` in experiment 06: times from the video, R/L from Andrew's
notes) attached only to evaluate.

## Run (from the repository root)

```
python experiments/experiment_10_rl_1111_matrix_profile/matrix_profile.py
python experiments/experiment_10_rl_1111_matrix_profile/build_results.py   # ~2 minutes
```

## Result

With 4 s windows, when a movement's most similar stretch is another movement, it is of the same type in 8/9 cases (11/12 with 6 s windows).
