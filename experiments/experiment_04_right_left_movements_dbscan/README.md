# Experiment 02 · Right / left movements

5-minute recording (11:47:10–11:51:51, 8 Nov 2026) in which Andrew made right, left and up-left-down movements.
Goal: tell the movement types apart from the sensor data alone.

## Data

Raw log line: `{time: H,M,S | MM/DD/YYYY, trans: x, y, z, rot: roll, pitch, yaw}`
(same structure as experiment 01, with Steve's labels). 3,866 rows, ~13.7 readings per second, no gaps.

Notes (with seconds): right at 11:47:20, 11:47:57, 11:49:27, 11:51:39; left at 11:50:16, 11:50:47;
up-left-down at 11:48:23.

## Pipeline

```
python prepare.py            # data/gyro_log_*.txt -> data/processed_movements.csv
python movements_kmeans.py   # main analysis + robustness -> outputs/
python build_results.py      # everything the dashboard shows -> results/
```

## Method

1. Angles are measured **relative to the hand's resting position** before each movement
   (last moment the hand stayed still for 1.5 s). This gives the direction of a turn and uses no label.
2. The log is cut into 4 s windows (50% overlap); each window gets a feature card.
3. k-means splits the windows into groups without labels.
4. Groups are named from their centres (no labels): idle / yaw down (> 45°) / yaw up (> 45°) / small movement.
5. Neighbouring non-idle windows become one detected movement; only then are they compared with the notes.

## Result (4 s windows, k = 4)

| Noted movement | Classified as | Correct |
|---|---|---|
| right (×4) | yaw down (wrist turns ~120° and back) | 4 / 4 |
| left (×2) | small movement (no large turn) | 2 / 2 |
| up-left-down (×1) | yaw up (turns ~100° the other way) | 1 / 1 |

Identical in 20/20 k-means runs. The two small unannotated gestures were treated as idle (no false alarms).
Right is robust across settings; left and up-left-down depend more on the window length and k.

## Caveats

- **Left has no signature of its own**: it is "a movement without a large turn", so any small movement would be
  called left. At 11:50:16 the yaw even goes down ~50°, the same direction as right.
- **tran follows the rotation rate** (correlation 0.80–0.91): the model separates the movements by the wrist turn,
  not by the hand's displacement, which may be why left looks weak.
- Only 7 movements: a pattern, not a validated model.
- The 1.5 s resting rule was chosen after seeing that a short pause flipped the up-left-down movement.

Next data that would help: ~10 repetitions of each movement, some red herrings, whether up-left-down is a third
movement or a red herring, and the sensor model / units of the first three values.
