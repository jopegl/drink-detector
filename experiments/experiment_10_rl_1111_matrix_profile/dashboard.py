"""Dashboard page for experiment 10 (Matrix Profile on the 11/11 recording)."""
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
MP = pd.read_csv(HERE / "results" / "matrix_profile.csv", parse_dates=["time", "nn_time"])
DISC = pd.read_csv(HERE / "results" / "discords.csv", parse_dates=["time"])


def color_of(e):
    return evz.LABEL_COLORS.get(e.truth_label, "#bdbdbd")


viz.header(info)
st.sidebar.markdown(f"**Experiment {info['code']}**")
tab_dash, tab_analysis = st.tabs(["Dashboard", "Analysis"])
nn, det = S["nn"], S["detection"]

with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nearest neighbour is a movement of the same type", f"{nn['nn_same_label']}/{nn['nn_on_rl']}",
              help="Among R/L movements whose most similar stretch is another R/L movement.")
    c2.metric("Nearest neighbour is another movement", f"{nn['nn_on_event']}/{nn['n_rl']}")
    c3.metric("Annotated movements detected", f"{det['truth_found']}/{det['truth_total']}")
    c4.metric("Top discords that are detected events", f"{S['discords_on_events']}/{S['n_discords']}")

    st.subheader("1. Matrix Profile over time")
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=MP.time, y=MP.mp, mode="lines", line=dict(width=1, color="#555"),
                               name="distance to the most similar other stretch"))
    viz.add_spans(fig, [(e.start, e.end, color_of(e)) for e in E.itertuples()], opacity=0.35)
    fig.add_trace(go.Scatter(x=DISC.time, y=DISC.mp, mode="markers+text", text=[f"D{r}" for r in DISC["rank"]],
                             textposition="top center", marker=dict(color="red", size=8), name="top discords"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="matrix profile",
                      legend=dict(orientation="h", y=1.15), hovermode="x unified")
    viz.show(fig)
    st.caption("Low = this stretch repeats somewhere else (motif). High = it looks like nothing else (discord). "
               "Bands: detected events coloured by truth (blue R, orange L, purple unlabeled, green possible red herring).")

    st.subheader("2. Each event and its most similar stretch")
    evz.events_table(E, ["mp", "mp_percentile", "nn_event", "nn_label"])
    st.caption("mp_percentile: share of the whole recording that is more repetitive than this event (low = typical). "
               "nn_event / nn_label: the detected event where the most similar stretch lies (0 = not inside any event).")
    if S.get("others"):
        st.markdown("**Unlabeled movements and possible red herrings:** " + "; ".join(
            f"{k} ({v['truth']}): more unusual than {v['mp_percentile']:.0f}% of the recording, nearest neighbour {v['nn']}"
            for k, v in S["others"].items()))
    st.subheader("3. Top discords (most unusual stretches)")
    viz.table(DISC.assign(time=DISC.time.dt.strftime("%H:%M:%S")))
    st.subheader("4. Inspect an event")
    evz.inspector(d, E, info, color_of,
                  lambda e: f"{e.start:%H:%M:%S}  ·  truth: {e.truth_label}  ·  nearest neighbour: {e.nn_label}",
                  f"e{info['code']}")

with tab_analysis:
    st.header("How it works")
    st.markdown(evz.PIPELINE_TEXT)
    st.markdown(
        "**Matrix Profile:** works on the **continuous** signal, not on the events. For every window of *M* seconds "
        "(drift-free yaw, roll and pitch, averaged to 4 readings/s), it finds the most similar other window anywhere "
        "in the recording, comparing the **shape** of the curves (each window is z-normalised, so only the shape "
        "matters, not the level). It needs no labels and no number of groups.\n\n"
        "**How it is scored:** for each detected event, where is its most similar stretch? If it is another movement "
        "of the same type, the method found the repetition on its own (like template matching with no template). "
        "The **discords** are the most unusual stretches: candidates for red herrings.")
    st.header("Does the result depend on the settings?")
    evz.heatmap(R.assign(m=R.m_s.astype(str) + " s"), "signal", "m", "share_same", "text", "signal", "window")
    st.caption("Signal: drift-free angles or rotation rates. Each cell: nearest neighbour of the same type / landing on "
               "an R/L movement, and how many of the 19 R/L movements have another movement as nearest neighbour.")
    st.header("Caveats")
    st.markdown(evz.COMMON_CAVEATS + "\n- **The recording is mostly writing**, so many stretches repeat (writing looks "
                "like writing). A movement's nearest neighbour is often a writing stretch, not another movement.\n"
                "- **The top discords are not the red herrings:** with the default settings, the most unusual "
                "stretches fall outside the detected events, in parts of the video that were not annotated, so it "
                "is not yet known what they are.")
