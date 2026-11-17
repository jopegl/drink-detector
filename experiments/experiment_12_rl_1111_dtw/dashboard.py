"""Dashboard page for experiment 12 (DTW on the 11/11 events)."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import events_viz as evz
from core import viz
from core.experiments import load_info

HERE = Path(__file__).resolve().parent
info = load_info(HERE)
d, E, S, R = evz.load_page(HERE, info)
M = pd.read_csv(HERE / "results" / "dtw_matrix.csv", index_col=0)
res, det = S["dtw"], S["detection"]


def color_of(e):
    return evz.LABEL_COLORS.get(e.nn_label, "#bdbdbd")


viz.header(info)
st.sidebar.markdown(f"**Experiment {info['code']}**")
tab_dash, tab_analysis = st.tabs(["Dashboard", "Analysis"])

with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Template matching (leave-one-out)", f"{res['correct_1nn']}/{res['n']}",
              help="Each R/L movement gets the label of its most similar OTHER labeled movement.")
    c2.metric("Clustering on DTW distances (k = 2)", f"{res['cluster']['correct']}/{res['cluster']['n']}",
              help="No labels used to group; groups named by majority (evaluation only).")
    c3.metric("Annotated movements detected", f"{det['truth_found']}/{det['truth_total']}")
    c4.metric("Detected events with no annotation", det["events_without_truth"])

    st.subheader("1. Timeline (coloured by the label of the most similar movement)")
    evz.timeline(d, E, color_of, [("most similar is R", evz.LABEL_COLORS["R"], "band"),
                                  ("most similar is L", evz.LABEL_COLORS["L"], "band"),
                                  ("truth R", evz.LABEL_COLORS["R"], "line"), ("truth L", evz.LABEL_COLORS["L"], "line")])

    st.subheader("2. DTW distances between events")
    order = E.sort_values(["truth_label", "start"]).event.astype(str).tolist()
    lab = dict(zip(E.event.astype(str), E.truth_label))
    names = [f"{o} ({lab[o]})" for o in order]
    Mo = M.loc[[int(o) for o in order], order]
    fig = go.Figure(go.Heatmap(z=Mo.values, x=names, y=names, colorscale="Viridis_r",
                               colorbar=dict(title="DTW"), hovertemplate="%{y} vs %{x}: %{z:.1f}<extra></extra>"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    viz.show(fig)
    st.caption("Events sorted by truth (L, R, then others). Dark = similar shape. If R and L have different shapes, "
               "the L-L and R-R blocks are darker than the L-R blocks.")
    evz.events_table(E, ["nn_event", "nn_label", "nn_dist", "unusualness", "group", "named_as"])
    st.caption("unusualness: distance to the most similar labeled movement divided by the typical one "
               "(above ~1.5 = unlike any R/L movement).")
    if S.get("others"):
        st.markdown("**Unlabeled movements and possible red herrings:** " + "; ".join(
            f"{k} ({v['truth']}) → most similar to {v['nn_label']}, unusualness {v['unusualness']}"
            for k, v in S["others"].items()))
    st.subheader("3. Inspect an event")
    evz.inspector(d, E, info, color_of,
                  lambda e: f"{e.start:%H:%M:%S}  ·  truth: {e.truth_label}  ·  most similar: {e.nn_label}",
                  f"e{info['code']}")

with tab_analysis:
    st.header("How it works")
    st.markdown(evz.PIPELINE_TEXT)
    st.markdown(
        "**Dynamic Time Warping (DTW):** compares two curves by stretching or squeezing time so their shapes line "
        "up, then measures what is left. Two movements with the same shape done at different speeds get a small "
        "distance. Curves: yaw / roll / pitch relative to the 3 s before the event, from 1 s before to 1 s after it.\n\n"
        "- **Template matching (1-nearest neighbour):** each movement is labeled like its most similar other "
        "labeled movement. This **uses the labels of the other movements**, like the rule in experiment 11, so it "
        "is scored with leave-one-out.\n"
        "- **Clustering:** average-linkage hierarchical clustering on the DTW distances, without labels.")
    st.header("Does the result depend on the settings?")
    evz.heatmap(R.assign(ok=R.acc_1nn), "high", "setting", "ok", "text", "threshold", "")
    st.caption("Channels: angles or rotation rates; raw or z-normalised (shape only). Each cell: template matching "
               "and clustering scores.")
    st.header("Caveats")
    st.markdown(evz.COMMON_CAVEATS + "\n- **Template matching needs labeled examples.** It answers \"is this new "
                "movement like a known R or a known L?\", not \"find the types on your own\".\n"
                "- **Clustering on DTW distances does not separate R from L** in most settings: average linkage "
                "isolates the single most different event and leaves everything else in one group.")
