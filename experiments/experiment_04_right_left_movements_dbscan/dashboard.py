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

TYPE_COLORS = {"yaw down": "#2b8cbe", "yaw up": "#756bb1", "small movement": "#e6550d",
               "unusual": "#8c8c8c"}
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


def make_tag(window_s, eps, ms):
    # must match the file names written by build_results.py: w4_eps1.5_ms5
    return f"w{int(window_s)}_eps{eps}_ms{int(ms)}"


@st.cache_data
def load_detection(results_dir, window_s, eps, ms):
    tag = make_tag(window_s, eps, ms)
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
S, rob_all = load_analysis(str(RESULTS))

# Settings available = the ones build_results.py pre-computed; defaults come from summary.json
WINDOW_OPTIONS = sorted(rob_all.window_s.unique().tolist())
EPS_OPTIONS = sorted(rob_all.eps.unique().tolist())
MS_OPTIONS = sorted(rob_all.min_samples.unique().tolist())
DEF_W, DEF_EPS, DEF_MS = S["default_window_s"], S["default_eps"], S["default_min_samples"]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"**Experiment {info['code']} settings**")
window_s = st.sidebar.select_slider("Window length (s)", options=WINDOW_OPTIONS,
                                    value=DEF_W if DEF_W in WINDOW_OPTIONS else WINDOW_OPTIONS[0], key="e04_w")
eps = st.sidebar.select_slider("eps (neighbourhood size)", options=EPS_OPTIONS,
                               value=DEF_EPS if DEF_EPS in EPS_OPTIONS else EPS_OPTIONS[0], key="e04_eps")
ms = st.sidebar.select_slider("min_samples (windows to form a group)", options=MS_OPTIONS,
                              value=DEF_MS if DEF_MS in MS_OPTIONS else MS_OPTIONS[0], key="e04_ms")
fps = st.sidebar.slider("Animation speed (frames/s)", 2, 20, 8, key="e04_fps")
st.sidebar.caption(f"Default: {DEF_W:g} s windows, eps = {DEF_EPS}, min_samples = {DEF_MS}. "
                   "Every setting was pre-computed. The detector does not use Andrew's notes; "
                   "they are shown only for comparison.")

M, crosstab, means = load_detection(str(RESULTS), window_s, eps, ms)
spans = [(m.start, m.end, TYPE_COLORS[m.type]) for m in M.itertuples()]
noted = M[M.andrew_note != "—"].copy()
noted["note"] = noted.andrew_note.str.split(" ").str[0]
noted["correct"] = [EXPECTED[n] == t for n, t in zip(noted.note, noted.type)]
total = pd.Series(list(NOTES.values())).value_counts()
current = rob_all[(rob_all.window_s == window_s) & (rob_all.eps == eps) & (rob_all.min_samples == ms)].iloc[0]


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
    c5, c6, _, _ = st.columns(4)
    c5.metric("Groups found by DBSCAN", int(current.n_groups), help="Noise not counted")
    c6.metric("Windows marked as noise", int(current.n_noise), help="Shown as 'unusual'")
    if (window_s, eps, ms) != (DEF_W, DEF_EPS, DEF_MS):
        st.info(f"Showing {window_s} s windows, eps = {eps}, min_samples = {ms}. "
                f"Default is {DEF_W:g} s, eps = {DEF_EPS}, min_samples = {DEF_MS}.")

    st.subheader("1. Timeline")
    notes = [(pd.Timestamp(f"{DATE} {t}"), NOTE_COLORS[lab]) for t, lab in NOTES.items()]
    legend = ([(f"detected: {t}", c, "band") for t, c in TYPE_COLORS.items()]
              + [(f"note: {n}", c, "line") for n, c in NOTE_COLORS.items()])
    viz.show(viz.timeline(
        d, [("yaw_rel", "yaw (relative to rest)", viz.C_YAW), ("roll_rel", "roll (relative to rest)", viz.C_ROLL),
            ("pitch_rel", "pitch (relative to rest)", viz.C_PITCH)], spans, notes, legend))
    st.caption("Angles are measured from the hand's resting position before each movement. Coloured bands: "
               "movements detected by DBSCAN (grey = 'unusual', made only of noise windows). "
               "Dotted lines: Andrew's notes.")

    st.subheader("2. Inspect a gesture")
    G = viz.gestures(d)

    def detected_type(g):
        x = M[(M.start <= g.end) & (M.end >= g.start)]
        return x.type.iloc[0] if len(x) else "not detected"

    G["detected"] = [detected_type(g) for g in G.itertuples()]
    G["description"] = [f"{g.start:%H:%M:%S}  ·  note: {g.label}  ·  model: {g.detected}" for g in G.itertuples()]
    viz.gesture_inspector(d, G, spans, fps, info["rate_hz"], "DETECTED", key="e04_gesture")

    st.subheader("3. Detector results")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Detected movements**")
        viz.table(M.assign(start=M.start.dt.strftime("%H:%M:%S"), end=M.end.dt.strftime("%H:%M:%S"))
                  .rename(columns={"n_windows": "windows", "window_types": "window classes",
                                   "andrew_note": "Andrew's note"}))
    with t2:
        st.markdown("**Windows by note × DBSCAN class**")
        viz.table(crosstab.rename(columns={"label": "note"}))
        st.markdown("**Groups found** (group -1 = noise)")
        viz.table(means.rename(columns={"name": "named as"}))

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
        "3. **DBSCAN** groups windows that sit in dense regions, **without seeing any label**. Unlike k-means, "
        "it does not need the number of groups: it uses **eps** (how close two windows must be to count as "
        "neighbours) and **min_samples** (how many neighbours are needed to form a group). Windows that fit no "
        "group are marked as **noise** and shown as *unusual*.\n"
        "4. Each group is named from its centre, again without labels: least movement = *idle*; yaw goes down "
        "by more than 45° = *yaw down*; yaw goes up by more than 45° = *yaw up*; anything else = *small movement*.\n"
        "5. Neighbouring non-idle windows become one detected movement.\n"
        "6. Only at the end are the movements compared with Andrew's notes.")

    st.header("A. Groups DBSCAN found (default settings)")
    d_means = pd.read_csv(RESULTS / f"group_means_{make_tag(DEF_W, DEF_EPS, DEF_MS)}.csv")
    viz.table(d_means.rename(columns={"name": "named as"}))
    st.markdown(f"Group **-1** is noise (*unusual*). DBSCAN is deterministic: the same settings always give the "
                f"same result, so there is no run-to-run variation to check. All 7 movements were classified "
                f"correctly in **{S['all_correct']} of {S['n_settings']}** settings tested.")

    st.header("B. Does the result depend on the settings?")
    ms_view = st.select_slider("min_samples shown in the grid", options=MS_OPTIONS, value=ms, key="e04_ms_grid")
    rob = rob_all[rob_all.min_samples == ms_view].copy()
    rob["all_ok"] = ((rob.right_as_yaw_down == rob.n_right) & (rob.left_as_small == rob.n_left)
                     & (rob.uld_as_yaw_up == rob.n_uld))
    piv_ok = rob.pivot(index="eps", columns="window_s", values="all_ok")
    text = rob.assign(t=lambda x: "R " + x.right_as_yaw_down.astype(str) + "/" + x.n_right.astype(str)
                      + "<br>L " + x.left_as_small.astype(str) + "/" + x.n_left.astype(str)
                      + "<br>U " + x.uld_as_yaw_up.astype(str) + "/" + x.n_uld.astype(str)
                      + "<br>" + x.n_groups.astype(str) + " gr · " + x.n_noise.astype(str) + " noise"
                      ).pivot(index="eps", columns="window_s", values="t")
    figh = go.Figure(go.Heatmap(
        z=piv_ok.astype(int).values, x=[f"{w:g} s" for w in piv_ok.columns],
        y=[f"eps = {e}" for e in piv_ok.index],
        text=text.values, texttemplate="%{text}", colorscale=[[0, "#f4c7c3"], [1, "#b7e1b0"]], zmin=0, zmax=1,
        showscale=False, xgap=3, ygap=3, hovertemplate="%{y}, window %{x}<br>%{text}<extra></extra>"))
    figh.update_layout(height=120 + 75 * len(piv_ok), margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="window length", yaxis=dict(autorange="reversed"))
    viz.show(figh)
    st.markdown(
        "R = right classified as *yaw down*, L = left as *small movement*, U = up-left-down as *yaw up*; "
        "below: groups found and windows marked as noise. Green = all 7 correct.\n\n"
        f"With min_samples = {ms_view}:\n"
        f"- **Right** 4/4 in {int((rob.right_as_yaw_down == rob.n_right).sum())} of {len(rob)} settings.\n"
        f"- **Left** 2/2 in {int((rob.left_as_small == rob.n_left).sum())} of {len(rob)} settings.\n"
        f"- **Up-left-down** 1/1 in {int((rob.uld_as_yaw_up == rob.n_uld).sum())} of {len(rob)} settings.\n"
        "- **Small eps** → many windows become noise (*unusual*). **Large eps** → everything merges into few "
        "groups and the movement types are no longer told apart.")

    st.header("C. Findings and caveats")
    st.markdown(
        "**What DBSCAN adds compared with k-means**\n"
        "- It does not need the number of groups.\n"
        "- It can say *\"this does not look like anything\"* (noise). This is useful for red herrings: an odd "
        "movement can be marked *unusual* instead of being forced into a group.\n\n"
        "**Caveats**\n"
        "- **eps and min_samples are manual choices.** min_samples must stay below the number of windows of the "
        "smallest movement (up-left-down has only ~5–7), or that movement can never form a group.\n"
        "- ***Left* has no signature of its own.** It is recognised as \"a movement without a large turn\", so any "
        "other small movement would also be called *left*.\n"
        "- **The tran values follow the rotation rate** (correlation 0.80–0.91 with the change in angles): the "
        "model separates movements by the wrist turn, not by the hand's displacement.\n"
        "- **Only 7 movements** (4 right, 2 left, 1 up-left-down): this shows a pattern but cannot validate a model.\n\n"
        "**Next data that would help**: ~10 repetitions of each movement, a few red herrings, confirmation of "
        "whether *up-left-down* is a third movement or a red herring, and the sensor model / units of the first "
        "three values.")