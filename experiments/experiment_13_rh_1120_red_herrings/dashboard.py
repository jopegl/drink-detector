"""Dashboard page for experiment 13 (red herrings in the 11/20 recording)."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import events_viz as evz
from core import viz
from core.experiments import load_info

HERE = Path(__file__).resolve().parent
info = load_info(HERE)
d, E, S, R = evz.load_page(HERE, info)
res, det, thr = S["results"], S["detection"], S["threshold"]
COLORS = {**evz.LABEL_COLORS, "move": "#9e9ac8"}
LABEL_NAMES = {"R": "right", "L": "left", "RH": "red herring", "—": "no note"}


def color_of(e):
    return evz.LABEL_COLORS.get(e.truth_label, "#bdbdbd")


viz.header(info)
st.sidebar.markdown(f"**Experiment {info['code']}**")
tab_dash, tab_analysis = st.tabs(["Dashboard", "Analysis"])
u = res["unusual"]

with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Red herrings caught (no RH examples)", u["rh_caught"],
              help=f"DTW unusualness above {thr}: unlike every known right / left movement.")
    c2.metric("False alarms on R/L movements", u["false_alarms"])
    c3.metric("AUC (unusualness)", f"{u['auc']:.2f}",
              help="Chance that a random red herring is more unusual than a random R/L movement. 0.5 = chance, 1 = perfect.")
    c4.metric("Notes matched to detected events", f"{det['notes_matched']}/{det['notes']}",
              help=f"Andrew's notes come a median {det['note_lag_median_s']:.1f} s before the movement's peak.")

    st.subheader("1. Timeline (coloured by Andrew's notes)")
    evz.timeline(d, E, color_of, [("right", evz.LABEL_COLORS["R"], "band"), ("left", evz.LABEL_COLORS["L"], "band"),
                                  ("red herring", evz.LABEL_COLORS["RH"], "band"),
                                  ("no note", evz.LABEL_COLORS["—"], "band")])

    st.subheader("2. How each approach did")
    rows = [
        ["1. One rule", "RH vs movement", "notes (leave-one-out)", f"{res['rule']['correct']}/{res['rule']['n']}",
         res["rule"]["rh_caught"], res["rule"]["false_alarms"]],
        ["2. Logistic regression", "R / L / RH", "notes (leave-one-out)", f"{res['logreg']['correct']}/{res['logreg']['n']}",
         res["logreg_binary"]["rh_caught"], res["logreg_binary"]["false_alarms"]],
        ["3. DTW template matching", "R / L / RH", "notes (leave-one-out)", f"{res['dtw']['correct']}/{res['dtw']['n']}",
         res["dtw_binary"]["rh_caught"], res["dtw_binary"]["false_alarms"]],
        ["4. DTW unusualness", "RH = unlike any R/L", "R/L notes only", f"{u['correct']}/{u['n']}",
         u["rh_caught"], u["false_alarms"]],
        ["5. Unusualness, 11/11 templates", "RH = unlike any 11/11 R/L", "only the 11/11 recording",
         f"{res['transfer_unusual']['correct']}/{res['transfer_unusual']['n']}",
         res["transfer_unusual"]["rh_caught"], res["transfer_unusual"]["false_alarms"]],
    ]
    viz.table(pd.DataFrame(rows, columns=["approach", "question", "learns from", "correct",
                                          "red herrings caught", "false alarms on R/L"]))
    st.caption(f"Right vs left with the 11/11 templates (approach 5): {res['transfer']['rl_correct']}/"
               f"{res['transfer']['rl_n']}, close to chance, so the R/L shapes changed between recordings, "
               "but the red herrings still stand out (AUC "
               f"{res['transfer']['auc']:.2f}).")

    st.subheader("3. How unusual is each event?")
    fig = go.Figure()
    order = ["R", "L", "RH", "—"]
    rng = np.random.default_rng(0)
    for k, lab in enumerate(order):
        x = E[E.truth_label == lab]
        fig.add_trace(go.Scatter(x=x.unusualness, y=k + rng.uniform(-0.15, 0.15, len(x)), mode="markers",
                                 marker=dict(size=11, color=evz.LABEL_COLORS[lab], line=dict(width=1, color="#333")),
                                 name=LABEL_NAMES[lab], text=x.start.dt.strftime("%H:%M:%S"),
                                 hovertemplate="%{text}<br>unusualness %{x:.2f}<extra></extra>"))
    fig.add_vline(x=thr, line_dash="dash", line_color="#d62728",
                  annotation_text=f"threshold {thr}", annotation_position="top")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      xaxis_title="unusualness (distance to the most similar R/L movement / typical distance)",
                      yaxis=dict(tickvals=list(range(4)), ticktext=[LABEL_NAMES[l] for l in order]))
    viz.show(fig)
    st.caption("Each dot is one detected event. Right of the dashed line = unlike every known right / left movement "
               "= flagged as a red herring. This uses no red herring examples at all.")

    st.subheader("4. Events")
    evz.events_table(E, ["note_lag_s", "peak", "rule_pred", "logreg_pred", "dtw_pred", "unusualness",
                         "transfer_unusualness"])
    st.caption("note_lag_s: seconds from Andrew's note to the movement's peak. *_pred: each approach's answer "
               "(leave-one-out). Events with no note ('—') are scored by nobody; they show what the methods think "
               "of the unannotated movements.")

    st.subheader("5. Inspect an event")
    evz.inspector(d, E, info, color_of,
                  lambda e: f"{e.start:%H:%M:%S}  ·  note: {LABEL_NAMES.get(e.truth_label, e.truth_label)}  ·  "
                            f"unusualness {e.unusualness:.2f}", f"e{info['code']}")

with tab_analysis:
    st.header("How it works")
    st.markdown(
        "**Data:** 11/20 recording, about 12 minutes, 24 notes from Andrew to the second: 9 right, 7 left and "
        "8 red herrings (random movements).\n\n"
        "**Shared steps:** yaw drift removed (~38°/min here), events detected with the same state machine as "
        "experiments 06–12, but two peaks only merge if they are **less than 6 s apart** (movements here can be "
        "10 s apart). Each note goes to one event: the one whose peak is closest to *note + 4 s* (Andrew's notes "
        f"come a median {det['note_lag_median_s']:.1f} s before the peak), within 3 s before to 12 s after the note.\n\n"
        "**Approaches** (every method that learns from the notes is scored with leave-one-out):\n"
        f"1. **One rule:** the single measure + threshold that best splits red herrings from R/L. On all the data "
        f"it picks *{res['rule_found']['feature']} {res['rule_found']['direction']} {res['rule_found']['threshold']}*.\n"
        "2. **Logistic regression** on the feature card (R / L / RH).\n"
        "3. **DTW template matching:** each event gets the label of its most similar other event (R / L / RH).\n"
        "4. **DTW unusualness:** distance to the most similar right / left movement, divided by the typical one. "
        f"Above **{thr}** (threshold fixed in experiment 12, before this data) = red herring. "
        "**It needs examples of the real movements only, never of red herrings**, which is what a real system "
        "would have: you can't list every possible random movement in advance.\n"
        "5. **Transfer:** the same unusualness, but the templates are the R/L movements of the 11/11 recording.")
    st.header("Does the result depend on the settings?")
    st.markdown(f"**{S['robustness']['settings']} settings** (threshold × minimum gap × DTW channels). AUC of the "
                f"unusualness: min **{S['robustness']['auc_min']:.2f}**, median **{S['robustness']['auc_median']:.2f}**, "
                f"max **{S['robustness']['auc_max']:.2f}**.")
    for ch in ["angles", "rates"]:
        st.markdown(f"**DTW on {ch}**")
        evz.heatmap(R[R.channels == ch].assign(ok=lambda x: x.unusual_auc.clip(0.5, 1).sub(0.5).mul(2),
                                               gap=lambda x: x.min_gap_s.map(lambda g: f"{g:g} s")),
                    "high", "gap", "ok", "text", "threshold", "gap")
    st.caption("Angles (wrist orientation) separate the red herrings in every setting; rotation rates do not.")
    st.header("Caveats")
    st.markdown(
        "- **Small sample:** 8 red herrings and 16 R/L movements from one person in one session.\n"
        "- **No video for this recording:** the labels come only from Andrew's notes, matched by time. The ~4 s "
        "lag is the same in almost every note, so the matching is consistent, but two notes 10 s apart could be "
        "swapped.\n"
        "- **One unannotated event is very unusual** (15:25:42, just before the red herring noted at 15:25:48): it "
        "may be part of that red herring.\n"
        "- **The false alarms** are one right (15:27:36) and one left (15:33:23) done differently from the others.\n"
        "- **Right vs left within this recording is easy** (DTW and logistic regression get all 16 R/L notes right; "
        "their errors are all red herrings), but the 11/11 templates do not transfer for R/L: the way the movements "
        "were done changed between sessions.")
