# Experiment 01 · Water sip detection

Andrew wore the wrist sensor for ~1 h 45 min (16:52–18:37, 4 Nov 2026) and noted when he drank water.
Goal: find the drinking events from the sensor data alone.

## Data

Raw log line: `{time: H,M,S | MM/DD/YYYY, accel: x, y, z, gyro: roll, pitch, yaw}`

- First group → **tran_x, tran_y, tran_z** (translation, per Steve's naming)
- Second group → **gyro_roll, gyro_pitch, gyro_yaw** (rotation, degrees)
- ~14 readings per second, no milliseconds, no gaps, no malformed rows (88,104 rows).

Notes (hour:minute): water at 16:53, 16:59, 17:19, 17:25, 17:49, 18:10, 18:20, 18:30, 18:37;
red herring at 18:35 and right after 17:49.

## Pipeline

```
python prepare.py           # data/accel_gryo_log_*.txt -> data/processed_log.csv
python kmeans_gestures.py   # k-means on the 13 gestures            -> outputs/
python evaluate_kmeans.py   # robustness checks + PCA map           -> outputs/
python kmeans_windows.py    # main detector (sliding windows)       -> outputs/
python build_results.py     # everything the dashboard shows        -> results/
```

No script uses Andrew's notes to train. They are only compared with the results at the end.

## Main result (4 s windows, k = 3)

| | |
|---|---|
| Annotated sips found | **9 / 9** |
| Red herring (18:35) flagged as sip | **0 / 1** |
| Detected sips with no note | **1** (17:29) |

k-means found three groups on its own: **idle**, **moving** (any movement, including the red herring and the
lift/lower phases of a sip) and **sip** (wrist strongly tilted back, mean pitch_min ≈ −59°).
k = 3 works for every window length from 2 to 8 s; k = 2 flags the red herring and k ≥ 4 loses sips.
A second method (k-means on the 13 whole gestures) agrees: the wrist rotation identifies a sip, translation alone does not.

## Caveats

- Only 1 confirmed red herring: proof of concept, not a validated model.
- 17:49 is the weakest sip: Andrew noted a red herring right after it, in the same movement segment.
- The 18:10 note matches a sip at 18:08:39 (clock or note offset).
- 17:29 looks like an unannotated sip (all methods agree).
- Remaining manual choices: features, k = 3, and the rule "most tilted group = sip".
- The estimated hand path is a rough double integration of `tran` (arbitrary units).
