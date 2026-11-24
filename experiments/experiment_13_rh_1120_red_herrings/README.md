# Experiment 13 · Red herrings 11-20

New 11/20 recording (about 12 minutes) with 24 notes from Andrew, to the second: 9 right, 7 left and 8 red
herrings (random movements). Goal: tell the red herrings apart from the right / left movements.

**Shared steps:** yaw drift removed (~38°/min), events detected with the state machine of experiments 06-12 but
peaks merged only within 6 s (movements here can be 10 s apart). Each note goes to the event whose peak is closest
to *note + 4 s* (Andrew's notes come ~4 s before the peak). Notes are used only to evaluate, with leave-one-out
for every method that learns from them.

**Approaches**
1. One rule (OneR): best single measure + threshold for red herring vs movement.
2. Logistic regression on the feature card (R / L / RH).
3. DTW template matching (R / L / RH).
4. DTW unusualness: distance to the most similar R/L movement / typical one; above 1.5 = red herring.
   Uses NO red herring examples.
5. Transfer: approach 4 with the 11/11 R/L movements (experiment 06) as the only templates.

## Run (from the repository root)

```
python experiments/experiment_13_rh_1120_red_herrings/prepare.py         # data/processed_log.csv
python experiments/experiment_13_rh_1120_red_herrings/red_herrings.py    # quick look, outputs/
python experiments/experiment_13_rh_1120_red_herrings/build_results.py   # results/ for the dashboard (~1 min)
```

Approach 5 reads `experiments/experiment_06_rl_1111_kmeans/data/processed_log.csv`.

## Result

| approach | correct | red herrings caught | false alarms on R/L |
|---|---|---|---|
| 1. One rule | 17/24 | 2/8 | 1/16 |
| 2. Logistic regression | 21/24 | 6/8 | 1/16 |
| 3. DTW template matching | 21/24 | 5/8 | 0/16 |
| 4. DTW unusualness (no RH examples) | 22/24 | **8/8** | 2/16 |
| 5. Unusualness with 11/11 templates | 19/24 | 8/8 | 5/16 |

AUC of the unusualness: 0.98 (0.97-0.98 in every setting with angles; ~0.75 with rotation rates).
Right vs left within this recording: all 16 correct with DTW and logistic regression; with the 11/11 templates
only 10/16, so the R/L shapes changed between sessions while the red herrings still stand out.
