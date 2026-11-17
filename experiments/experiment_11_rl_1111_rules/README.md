# Experiment 11 · Right / left 11-11 (rules)

Same 11/11 recording as experiment 06 (right / left arm movements among natural ones). This experiment uses simple rules: a rest/active state machine to find events and a single-threshold rule to tell R from L, as the baseline other methods must beat. Truth (video times + Andrew's notes) is used only to evaluate.

**Method:** State machine for detection + one-threshold rule (OneR) for R/L, tested with leave-one-out; also tests the rule from experiment 02.

Uses the data of experiment 06 (`../experiment_06_rl_1111_kmeans/data/`): run its `prepare.py` first.

Shared steps (core/events.py): yaw drift removed (the BMI160 has no magnetometer, ~70°/min drift), events detected by a
rest/active state machine on the movement intensity, one feature card per event with angles relative to the 3 s
before it, and the ground truth (`data/annotations.csv` in experiment 06: times from the video, R/L from Andrew's
notes) attached only to evaluate.

## Run (from the repository root)

```
python experiments/experiment_11_rl_1111_rules/rules.py
python experiments/experiment_11_rl_1111_rules/build_results.py
```

## Result

One-threshold rule 'R if pitch_max >= 51' gets 16/19 in leave-one-out; the rule from experiment 02 gets 11/19 (does not transfer).
