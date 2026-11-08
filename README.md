# Water sip detection from wrist IMU data

Andrew wore a wrist sensor for ~1 h 45 min (16:52–18:37, 4 Nov 2026) and noted when he drank water.
The goal is to find the drinking events from the sensor data alone.

## Data

Raw log line: `{time: H,M,S | MM/DD/YYYY, accel: x, y, z, gyro: roll, pitch, yaw}`

- First group (`accel`) → **tran_x, tran_y, tran_z** (translation / acceleration, per Steve's naming)
- Second group (`gyro`) → **gyro_roll, gyro_pitch, gyro_yaw** (rotation, degrees)
- ~14 readings per second, no milliseconds, no gaps, no malformed rows (88,104 rows).

## Pipeline

```
pip install -r requirements.txt

python prepare_log.py accel_gryo_log_20261104_183756.txt   # -> processed_log.csv
python kmeans_gestures.py processed_log.csv                # k-means on the 13 gestures
python evaluate_kmeans.py processed_log.csv                # robustness checks + kmeans_pca.png
python kmeans_windows.py processed_log.csv                 # main detector (sliding windows)
python build_results.py processed_log.csv                  # pre-compute everything -> results/
streamlit run app.py                                       # interactive interface
```

| Script | What it does |
|---|---|
| `prepare_log.py` | Parses the raw log, estimates sub-second timestamps, unwraps yaw, computes `tran_mag`, splits the log into gestures (movement segments) and attaches Andrew's notes as `label` (for checking only). |
| `kmeans_gestures.py` | One feature card per gesture → k-means (k=2) with three feature sets: all, rotation only, translation only. |
| `evaluate_kmeans.py` | Leave-one-out, feature-removal robustness, margins, k=3, PCA map. |
| `kmeans_windows.py` | **Main detector.** Cuts the whole log into 4 s windows (50% overlap), k-means with k=3, picks the "sip" group as the one with the wrist tilted back the most, merges neighbouring sip windows into events, compares with the notes, and repeats for 2–8 s windows. |
| `build_results.py` | Runs every analysis once (all window lengths 2–8 s × k = 2–6, gesture k-means, robustness checks) and saves it in `results/`. |
| `app.py` + `gui_core.py` | Streamlit interface. **Dashboard** tab: timeline with detected sips, 3D hand-rotation animation, estimated hand path, per-gesture signals, result tables. **Analysis** tab: method, robustness grid, gesture-level cross-check, robustness checks, findings and caveats. Nothing is computed when the page opens: it reads `processed_log.csv` and `results/`. |

No script uses Andrew's notes to train. They are only compared with the results at the end.

## Main result (`kmeans_windows.py`, 4 s windows, k=3)

| | |
|---|---|
| Annotated sips found | **9 / 9** |
| Red herring (18:35) flagged as sip | **0 / 1** |
| Detected sips with no note | **1** (17:29) |

k-means found three groups on its own: **idle** (hand still), **moving** (any movement, including the
red herring and the lift/lower phases of a sip) and **sip** (wrist strongly tilted back, mean pitch_min ≈ −59°).
The result holds for window lengths of 2–8 s, and the detected sips are the same in 18 of 20 k-means runs.
k = 3 is the only number of groups that works at every window length: k = 2 flags the red herring
(it only separates still vs moving) and k ≥ 4 splits the sip group and loses sips.

## Sharing the interface (Streamlit Community Cloud)

1. Push this folder (including `processed_log.csv` and `results/`) to a GitHub repository.
2. Go to share.streamlit.io, sign in with GitHub, pick the repository and `app.py`.
3. The page opens with every analysis ready; no upload needed.

## Caveats

- Only 1 confirmed red herring: the detector is a proof of concept, not a validated model.
  A new recording with more non-drinking gestures (scratching the face, fixing hair, phone) is needed.
- The 17:49 sip is the weakest detection: Andrew noted a red herring right after it, and both seem to fall
  in the same movement segment.
- The 18:10 note matches a sip that actually happens at 18:08:39 (clock or note offset).
- 17:29 looks like an unannotated sip (all methods agree). Worth checking with Andrew.
- Remaining manual choices: features, k=3, and the rule "most tilted group = sip".
- The estimated hand path in the interface is a rough double integration of `tran` (arbitrary units).
