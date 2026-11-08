"""
Dashboard page for experiment 02 (right / left movements).
Opened through app.py. Nothing is computed here: it reads data/ and results/
(created by prepare.py and build_results.py).
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import viz
from core.experiments import load_info

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
info = load_info(HERE)
LOG = HERE / "data" / info["processed_log"]
DATE = info["date"]
NOTES = info["notes"]

TYPE_COLORS = {"yaw down": "#2b8cbe", "yaw up": "#756bb1", "small movement": "#e6550d"}
NOTE_COLORS = {"right": "#2b8cbe", "left": "#e6550d", "up-left-down": "#756bb1"}
EXPECTED = {"right": "yaw down", "left": "small movement", "up-left-down": "yaw up"}

if not LOG.exists() or not (RESULTS / "summary.json").exists():
    st.error("Missing data or results. Run prepare.py and build_results.py in this experiment's folder.")
    st.stop()


@st.cache_data(show_spinner="Reading the log...")
def load_log(path, rel_path):
    d = viz.load_log(path)
    rel = pd.read_csv(rel_path, parse_dates=["timestamp"])
    for c in ["yaw_rel", "roll_rel", "pitch_rel"]:
        d[c] = rel[c].values
    return d


@st.cache_data
def load_detection(results_dir, window_s, k):
    tag = f"w{int(window_s)}_k{int(k)}"
    M = pd.read_csv(f"{results_dir}/movements_{tag}.csv", parse_dates=["start", "end"])
    crosstab = pd.read_csv(f"{results_dir}/crosstab_{tag}.csv")
    means = pd.read_csv(f"{results_dir}/group_means_{tag}.csv")
    return M, crosstab, means


@st.cache_data
def load_analysis(results_dir):
    with open(f"{results_dir}/summary.json") as f:
        summary = json.load(f)
    return summary, pd.read_csv(f"{results_dir}/robustness.csv")


d = load_log(str(LOG), str(RESULTS / "relative_angles.csv"))
S, rob = load_analysis(str(RESULTS))

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"**Experiment {info['code']} settings**")
window_s = st.sidebar.select_slider("Window length (s)", options=[2, 3, 4, 5, 6], value=4, key="e02_w")
k = st.sidebar.select_slider("Number of groups (k-means)", options=[3, 4, 5, 6], value=4, key="e02_k")
fps = st.sidebar.slider("Animation speed (frames/s)", 2, 20, 8, key="e02_fps")
st.sidebar.caption("Default: 4 s windows, 4 groups. Every setting was pre-computed. "
                   "The detector does not use Andrew's notes; they are shown only for comparison.")

M, crosstab, means = load_detection(str(RESULTS), window_s, k)
spans = [(m.start, m.end, TYPE_COLORS[m.type]) for m in M.itertuples()]
noted = M[M.andrew_note != "—"].copy()
noted["note"] = noted.andrew_note.str.split(" ").str[0]
noted["correct"] = [EXPECTED[n] == t for n, t in zip(noted.note, noted.type)]
total = pd.Series(list(NOTES.values())).value_counts()


def score(kind):
    x = noted[noted.note == kind]
    return f"{int(x.correct.sum())}/{int(total[kind])}"


viz.header(info)
tab_dash, tab_analysis = st.tabs(["Dashboard", "Analysis"])

# ===========================================================================
# DASHBOARD
# ===========================================================================
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Noted movements classified correctly", f"{int(noted.correct.sum())}/{len(NOTES)}")
    c2.metric("right → yaw down", score("right"))
    c3.metric("left → small movement", score("left"))
    c4.metric("up-left-down → yaw up", score("up-left-down"))
    if (window_s, k) != (4, 4):
        st.info(f"Showing {window_s} s windows and {k} groups. Default is 4 s and 4 groups.")

    st.subheader("1. Timeline")
    notes = [(pd.Timestamp(f"{DATE} {t}"), NOTE_COLORS[lab]) for t, lab in NOTES.items()]
    legend = ([(f"detected: {t}", c, "band") for t, c in TYPE_COLORS.items()]
              + [(f"note: {n}", c, "line") for n, c in NOTE_COLORS.items()])
    viz.show(viz.timeline(
        d, [("yaw_rel", "yaw (relative to rest)", viz.C_YAW), ("roll_rel", "roll (relative to rest)", viz.C_ROLL),
            ("pitch_rel", "pitch (relative to rest)", viz.C_PITCH)], spans, notes, legend))
    st.caption("Angles are measured from the hand's resting position before each movement. Coloured bands: "
               "movements detected by k-means. Dotted lines: Andrew's notes.")

    st.subheader("2. Inspect a gesture")
    G = viz.gestures(d)

    def detected_type(g):
        x = M[(M.start <= g.end) & (M.end >= g.start)]
        return x.type.iloc[0] if len(x) else "not detected"

    G["detected"] = [detected_type(g) for g in G.itertuples()]
    G["description"] = [f"{g.start:%H:%M:%S}  ·  note: {g.label}  ·  model: {g.detected}" for g in G.itertuples()]
    viz.gesture_inspector(d, G, spans, fps, info["rate_hz"], "DETECTED", key="e02_gesture")

    st.subheader("3. Detector results")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Detected movements**")
        viz.table(M.assign(start=M.start.dt.strftime("%H:%M:%S"), end=M.end.dt.strftime("%H:%M:%S"))
                  .rename(columns={"n_windows": "windows", "window_types": "window classes",
                                   "andrew_note": "Andrew's note"}))
    with t2:
        st.markdown("**Windows by note × k-means class**")
        viz.table(crosstab.rename(columns={"label": "note"}))

# ===========================================================================
# ANALYSIS
# ===========================================================================
with tab_analysis:
    st.header("How the detector works")
    st.markdown(
        "1. For each reading, roll, pitch and yaw are measured **relative to the hand's resting position** "
        "before the movement (the last moment the hand stayed still for 1.5 s). This gives the **direction** "
        "of a turn and uses no label.\n"
        "2. The whole log is cut into **windows** (default 4 s, 50% overlap). Each window gets a feature card: "
        "how far yaw / roll / pitch went up or down, how agitated, how intense.\n"
        "3. **k-means** splits the windows into groups **without seeing any label**.\n"
        "4. Each group is named from its centre, again without labels: least movement = *idle*; yaw goes down "
        "by more than 45° = *yaw down*; yaw goes up by more than 45° = *yaw up*; anything else = *small movement*.\n"
        "5. Neighbouring non-idle windows become one detected movement.\n"
        "6. Only at the end are the movements compared with Andrew's notes.")

    st.header("A. Groups k-means found (default settings)")
    d_means = pd.read_csv(RESULTS / "group_means_w4_k4.csv")
    viz.table(d_means.rename(columns={"group": "group", "name": "named as"}))
    st.markdown("With 4 groups, k-means separates **idle**, a **large turn one way** (yaw ≈ −120°, only the "
                "*right* movements), a **large turn the other way** (yaw ≈ +100°, only *up-left-down*) and "
                "**small movements** (the *left* movements plus the start/end of the others). "
                f"The default classification was identical in **{S['stable_runs']}/{S['n_runs']}** k-means runs.")

    st.header("B. Does the result depend on the settings?")
    rob = rob.copy()
    rob["all_ok"] = ((rob.right_as_yaw_down == rob.n_right) & (rob.left_as_small == rob.n_left)
                     & (rob.uld_as_yaw_up == rob.n_uld))
    piv_ok = rob.pivot(index="k", columns="window_s", values="all_ok")
    text = rob.assign(t=lambda x: "R " + x.right_as_yaw_down.astype(str) + "/" + x.n_right.astype(str)
                      + "<br>L " + x.left_as_small.astype(str) + "/" + x.n_left.astype(str)
                      + "<br>U " + x.uld_as_yaw_up.astype(str) + "/" + x.n_uld.astype(str)
                      ).pivot(index="k", columns="window_s", values="t")
    figh = go.Figure(go.Heatmap(
        z=piv_ok.astype(int).values, x=[f"{w} s" for w in piv_ok.columns], y=[f"k = {kk}" for kk in piv_ok.index],
        text=text.values, texttemplate="%{text}", colorscale=[[0, "#f4c7c3"], [1, "#b7e1b0"]], zmin=0, zmax=1,
        showscale=False, xgap=3, ygap=3, hovertemplate="%{y}, window %{x}<br>%{text}<extra></extra>"))
    figh.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="window length",
                       yaxis=dict(autorange="reversed"))
    viz.show(figh)
    st.markdown(
        "R = right classified as *yaw down*, L = left as *small movement*, U = up-left-down as *yaw up*. "
        "Green = all 7 correct.\n\n"
        f"- **Right** is the most robust: 4/4 in {int((rob.right_as_yaw_down == 4).sum())} of {len(rob)} settings "
        "(it only fails with 6 s windows, where two movements merge).\n"
        f"- **Left**: 2/2 in {int((rob.left_as_small == 2).sum())} of {len(rob)} settings.\n"
        f"- **Up-left-down**: 1/1 in {int((rob.uld_as_yaw_up == 1).sum())} of {len(rob)} settings. With few "
        "groups it does not get a group of its own, because it is a single example.\n"
        "- **4 groups with 4–5 s windows** gets all 7 right; 3 s windows are unstable.")

    st.header("C. Findings and caveats")
    st.markdown(
        "**Findings**\n"
        "- *Right* has a clear signature: the wrist turns ~120° (yaw) and comes back.\n"
        "- *Up-left-down* turns the other way (yaw ≈ +100°) and tilts down at the end.\n"
        "- The two small unannotated gestures (11:51:10 and 11:51:32) were treated as idle: no false alarms.\n\n"
        "**Caveats**\n"
        "- ***Left* has no signature of its own.** It is recognised as \"a movement without a large turn\", so "
        "any other small movement would also be called *left*. At 11:50:16 the yaw even goes **down** ~50°, the "
        "same direction as *right*. *Left* is not the mirror image of *right*.\n"
        "- **The tran values follow the rotation rate again** (correlation 0.80–0.91 with the change in angles). "
        "Right and left are displacements, but the model tells them apart by the **wrist turn**, not by the hand's "
        "displacement. This may be why *left* looks so weak.\n"
        "- **Only 7 movements** (4 right, 2 left, 1 up-left-down): this shows a pattern but cannot validate a model.\n"
        "- **One adjustment was made after looking at the plots:** the resting position now requires 1.5 s of "
        "stillness, because a short pause mid-gesture was being taken as rest and flipped the up-left-down "
        "movement. It is a logic fix, but it was chosen after seeing the result.\n\n"
        "**Next data that would help**: ~10 repetitions of each movement, a few red herrings, confirmation of "
        "whether *up-left-down* is a third movement or a red herring, and the sensor model / units of the first "
        "three values.")
