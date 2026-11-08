# Wrist sensor experiments

Andrew wears a wrist sensor (3 translation values + roll, pitch, yaw) and notes what he does.
Each experiment tries to recognise his movements **from the sensor data alone**, with unsupervised k-means.
His notes are only used at the end, to check the result.

| # | Experiment | Date | Result |
|---|---|---|---|
| 01 | [Water sip detection](experiments/experiment_01_water_sips/) | 2026-11-04 | 9/9 sips found, red herring not flagged, 1 likely unannotated sip |
| 02 | [Right / left movements](experiments/experiment_02_right_left_movements/) | 2026-11-08 | 7/7 movements classified (4 s windows, k = 4); proof of concept |

## Dashboard

```
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens on an overview page, with one page per experiment in the sidebar.
Each experiment page reads that experiment's `data/` and `results/` folders, so nothing is computed when it opens.

## Structure

```
app.py                     dashboard entry point (menu built from the experiment folders)
home.py                    overview page
core/
  log_io.py                read and clean a raw sensor log, split it into gestures
  viz.py                   shared dashboard pieces (timeline, 3D hand, estimated path, tables)
  experiments.py           finds the experiment folders
experiments/
  experiment_01_water_sips/
  experiment_02_right_left_movements/
    experiment.json        number, title, date, summary, result, notes
    README.md              what was done, results, caveats
    prepare.py             raw log -> data/processed_*.csv
    ... analysis scripts
    build_results.py       pre-computes everything the dashboard shows -> results/
    dashboard.py           the experiment's dashboard page
    data/                  raw log (git-ignored) and processed log
    results/               pre-computed results read by the dashboard
    outputs/               plots and CSVs from the analysis scripts
```

Every script can be run from anywhere and uses its own experiment folder by default, e.g.
`python experiments/experiment_02_right_left_movements/build_results.py`.

## Adding a new experiment

1. Create `experiments/experiment_03_<short_title>/` with `data/`, `results/` and `outputs/`.
2. Put the raw log in `data/` and write `experiment.json` (copy one from another experiment and edit it).
3. Write `prepare.py` (use `core.log_io`), the analysis, `build_results.py` and `dashboard.py`
   (copy the closest experiment and adapt it).
4. Run `prepare.py` and `build_results.py`.
5. The new experiment appears in the dashboard menu automatically, ordered by its number.

## Deploying (Streamlit Community Cloud)

Main file: `app.py`. Push `data/processed_*.csv` and `results/` of every experiment: the dashboard needs them.
