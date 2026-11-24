"""
Dashboard pieces shared by the event-based experiments (06 on).
Every experiment page reads its own results/ folder and the shared data folder.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import events as ev
from core import viz

LABEL_COLORS = {"R": "#2b8cbe", "L": "#e6550d", "?": "#9e9ac8", "other?": "#31a354", "RH": "#31a354", "—": "#bdbdbd"}
NAMED_COLORS = {"R": "#2b8cbe", "L": "#e6550d", "tie": "#9e9ac8", "no R/L": "#31a354",
                "unusual": "#636363", "not grouped": "#bdbdbd"}


def data_dir(here, info):
    return (Path(here) / info.get("data_dir", "data")).resolve()


@st.cache_data(show_spinner="Reading the log...")
def _load_log(path):
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_page(here, info):
    """Processed log, events (with the method's groups), summary and robustness table."""
    results = Path(here) / "results"
    if not (results / "summary.json").exists():
        st.error("Missing results. Run build_results.py in this experiment's folder.")
        st.stop()
    d = _load_log(str(data_dir(here, info) / info["processed_log"]))
    E = ev.load_events(results / "events.csv")
    with open(results / "summary.json") as f:
        S = json.load(f)
    R = pd.read_csv(results / "robustness.csv") if (results / "robustness.csv").exists() else None
    return d, E, S, R


def metrics(S):
    det, ev_ = S["detection"], S["evaluation"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R/L movements classified correctly", f"{ev_['correct']}/{ev_['n']}",
              help="Each group is named after the majority of its members (evaluation only).")
    c2.metric("Adjusted Rand index", f"{ev_['ari']:.2f}",
              help="Agreement between the groups and the R/L truth: 1 = perfect, 0 = chance.")
    c3.metric("Annotated movements detected", f"{det['truth_found']}/{det['truth_total']}")
    c4.metric("Detected events with no annotation", det["events_without_truth"])


def timeline(d, E, color_of, legend, title_note=""):
    """Movement intensity over time, events as coloured bands, truth as dotted lines."""
    s = pd.Series(ev.intensity(d), index=d.timestamp)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=s.index, y=s.values, name="intensity (1 s mean of |tran|)", mode="lines",
                               line=dict(width=1, color="#555")))
    viz.add_spans(fig, [(e.start, e.end, color_of(e)) for e in E.itertuples()], opacity=0.45)
    notes = [(e.peak_time, LABEL_COLORS.get(e.truth_label, "#bdbdbd")) for e in E.itertuples() if e.truth_label != "—"]
    viz.add_note_lines(fig, notes)
    viz.add_legend_entries(fig, legend)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="intensity",
                      legend=dict(orientation="h", y=1.15), hovermode="x unified")
    viz.show(fig)
    st.caption("Grey line: movement intensity. Coloured bands: detected events, coloured by the method's result. "
               "Dotted lines: ground truth from the video and Andrew's notes (blue R, orange L, purple unlabeled, "
               "green possible red herring)." + title_note)


def events_table(E, extra_cols):
    cols = ["event", "start", "end", "truth_id", "truth_label"] + extra_cols
    viz.table(E.assign(start=E.start.dt.strftime("%H:%M:%S"), end=E.end.dt.strftime("%H:%M:%S"))[cols]
              .rename(columns={"truth_id": "video id", "truth_label": "truth"}))


def inspector(d, E, info, color_of, describe, key):
    G = E.rename(columns={})[["start", "end"]].copy()
    G["description"] = [describe(e) for e in E.itertuples()]
    spans = [(e.start, e.end, color_of(e)) for e in E.itertuples()]
    fps = st.sidebar.slider("Animation speed (frames/s)", 2, 20, 8, key=f"{key}_fps")
    viz.gesture_inspector(d, G, spans, fps, info["rate_hz"], "EVENT", key=key)


def heatmap(R, index, columns, ok_col, text_col, index_label, columns_label, height=None):
    piv_ok = R.pivot(index=index, columns=columns, values=ok_col).astype(float)
    piv_t = R.pivot(index=index, columns=columns, values=text_col)
    fig = go.Figure(go.Heatmap(
        z=piv_ok.values, x=[f"{columns_label} {c}" for c in piv_ok.columns],
        y=[f"{index_label} {i}" for i in piv_ok.index], text=piv_t.values, texttemplate="%{text}",
        colorscale=[[0, "#f4c7c3"], [0.5, "#fbe3a1"], [1, "#b7e1b0"]], zmin=0, zmax=1, showscale=False,
        xgap=3, ygap=3, hovertemplate="%{y}, %{x}<br>%{text}<extra></extra>"))
    fig.update_layout(height=height or 110 + 70 * len(piv_ok), margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"))
    viz.show(fig)


PIPELINE_TEXT = (
    "**Shared steps (experiments 06–12):**\n"
    "1. **Yaw drift removed.** The BMI160 sensor has no magnetometer, so the yaw drifts at an almost constant "
    "rate (~70°/min in this recording). A straight line is fitted and subtracted.\n"
    "2. **Events detected by a small state machine** on the movement intensity: it switches from *rest* to "
    "*active* above a high threshold and back below a low one. Each active stretch around a strong peak is one event.\n"
    "3. **Feature card per event:** how far yaw / roll / pitch moved from the median of the 3 s *before* the event "
    "(local baseline, since the hand is rarely still), the min/max rotation rates and the intensity.\n"
    "4. **Ground truth** (times from the video, R/L from Andrew's notes) is attached to the events **only to "
    "evaluate**. Each group is named after the majority of its members, and the adjusted Rand index measures the "
    "agreement without any naming.")

COMMON_CAVEATS = (
    "- **19 labeled movements** (10 R, 9 L): much better than before, but still small.\n"
    "- **R/L labels come from Andrew's notes.** The video confirms the timing of each movement, but the arm leaves "
    "the camera frame, so it cannot confirm the direction.\n"
    "- **Only 2 possible red herrings** were seen in the video (15:00:04 and 15:13:04); Andrew did not note the "
    "scratching / stretching movements.\n"
    "- **The signature from experiment 02 does not transfer:** there, *right* was a ~120° yaw turn; here R and L "
    "change the yaw by less than 50°, and they differ mostly in pitch and rotation rate.")


def _color_of_named(e):
    return NAMED_COLORS.get(getattr(e, "named_as", ""), "#bdbdbd")


def cluster_page(here, method_title, method_text, grid_cols, grid_label, extra_dashboard=None, extra_analysis=None,
                 extra_cols=()):
    """
    Standard page for the clustering experiments (06-09).
    grid_cols : column of robustness.csv used as heatmap rows (the method's parameter)
    """
    from core.experiments import load_info
    info = load_info(here)
    key = f"e{info['code']}"
    d, E, S, R = load_page(here, info)
    viz.header(info)
    st.sidebar.markdown(f"**Experiment {info['code']}**")
    tab_dash, tab_analysis = st.tabs(["Dashboard", "Analysis"])

    with tab_dash:
        metrics(S)
        st.subheader("1. Timeline")
        timeline(d, E, _color_of_named,
                 [(f"group named {k}", c, "band") for k, c in NAMED_COLORS.items() if k in set(E.named_as)]
                 + [("truth R", LABEL_COLORS["R"], "line"), ("truth L", LABEL_COLORS["L"], "line")])
        st.subheader("2. Events and groups")
        st.markdown("Groups are named after the majority of their R/L members (evaluation only): "
                    + ", ".join(f"group {g} → **{n}**" for g, n in S["evaluation"]["mapping"].items()) + ".")
        viz.table(pd.crosstab(E.truth_label, E.group).reset_index().rename(columns={"truth_label": "truth \\ group"}))
        events_table(E, ["group", "named_as"] + list(extra_cols) + ["peak", "pitch_max", "rate_y_min", "yaw_min", "yaw_max"])
        if S.get("others"):
            st.markdown("**Unlabeled movements and possible red herrings:** "
                        + "; ".join(f"{k} ({v['truth']}) → group {v['group']} ({v['named_as']})"
                                    for k, v in S["others"].items()))
        if extra_dashboard:
            extra_dashboard(d, E, S, R)
        st.subheader("3. Inspect an event")
        inspector(d, E, info, _color_of_named,
                  lambda e: f"{e.start:%H:%M:%S}  ·  truth: {e.truth_label}  ·  group {e.group} ({e.named_as})", key)

    with tab_analysis:
        st.header("How it works")
        st.markdown(PIPELINE_TEXT)
        st.markdown(method_text)
        st.header("Does the result depend on the settings?")
        rob = S["robustness"]
        st.markdown(f"**{rob['settings']} settings** tested ({grid_label} × detection threshold). R/L accuracy: "
                    f"min **{rob['accuracy_min']:.0%}**, median **{rob['accuracy_median']:.0%}**, "
                    f"max **{rob['accuracy_max']:.0%}**. Each cell: R/L correct and adjusted Rand index.")
        heatmap(R, "high", grid_cols, "accuracy", "text", "threshold", "")
        if extra_analysis:
            extra_analysis(d, E, S, R)
        st.header("Caveats")
        st.markdown(COMMON_CAVEATS)
