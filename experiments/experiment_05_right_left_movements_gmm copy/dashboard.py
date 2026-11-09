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


def make_tag(window_s, k):
    return f"w{int(window_s)}_k{int(k)}"


@st.cache_data
def load_detection(results_dir, window_s, k):
    tag = make_tag(window_s, k)
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
WINDOW_OPTIONS = sorted(int(w) for w in rob_all.window_s.unique())
K_OPTIONS = sorted(int(k) for k in rob_all.k.unique())
DEF_W, DEF_K = int(S["default_window_s"]), int(S["default_k"])

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"**Experiment {info['code']} settings**")
window_s = st.sidebar.select_slider("Window length (s)", options=WINDOW_OPTIONS,
                                    value=DEF_W if DEF_W in WINDOW_OPTIONS else WINDOW_OPTIONS[0], key="e05_w")
k = st.sidebar.select_slider("Number of components (GMM)", options=K_OPTIONS,
                             value=DEF_K if DEF_K in K_OPTIONS else K_OPTIONS[0], key="e05_k")
fps = st.sidebar.slider("Animation speed (frames/s)", 2, 20, 8, key="e05_fps")
st.sidebar.caption(f"Default: {DEF_W} s windows, {DEF_K} components. Every setting was pre-computed. "
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
    if (window_s, k) != (DEF_W, DEF_K):
        st.info(f"Showing {window_s} s windows and {k} components. Default is {DEF_W} s and {DEF_K} components.")

    st.subheader("1. Timeline")
    notes = [(pd.Timestamp(f"{DATE} {t}"), NOTE_COLORS[lab]) for t, lab in NOTES.items()]
    legend = ([(f"detected: {t}", c, "band") for t, c in TYPE_COLORS.items()]
              + [(f"note: {n}", c, "line") for n, c in NOTE_COLORS.items()])
    viz.show(viz.timeline(
        d, [("yaw_rel", "yaw (relative to rest)", viz.C_YAW), ("roll_rel", "roll (relative to rest)", viz.C_ROLL),
            ("pitch_rel", "pitch (relative to rest)", viz.C_PITCH)], spans, notes, legend))
    st.caption("Angles are measured from the hand's resting position before each movement. Coloured bands: "
               "movements detected by the GMM. Dotted lines: Andrew's notes.")

    st.subheader("2. Inspect a gesture")
    G = viz.gestures(d)

    def detected_type(g):
        x = M[(M.start <= g.end) & (M.end >= g.start)]
        return x.type.iloc[0] if len(x) else "not detected"

    G["detected"] = [detected_type(g) for g in G.itertuples()]
    G["description"] = [f"{g.start:%H:%M:%S}  ·  note: {g.label}  ·  model: {g.detected}" for g in G.itertuples()]
    viz.gesture_inspector(d, G, spans, fps, info["rate_hz"], "DETECTED", key="e05_gesture")

    st.subheader("3. Detector results")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Detected movements**")
        viz.table(M.assign(start=M.start.dt.strftime("%H:%M:%S"), end=M.end.dt.strftime("%H:%M:%S"))
                  .rename(columns={"n_windows": "windows", "window_types": "window classes",
                                   "andrew_note": "Andrew's note"}))
    with t2:
        st.markdown("**Windows by note × GMM class**")
        viz.table(crosstab.rename(columns={"label": "note"}))
        st.markdown("**Components found**")
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
        "3. A **Gaussian Mixture Model (GMM)** splits the windows into groups (*components*) **without seeing any "
        "label**. It is similar to k-means, but it describes each group as a cloud of points with its own spread, "
        "and gives each window a probability of belonging to each group instead of a hard yes/no.\n"
        "4. Each group is named from its centre, again without labels: least movement = *idle*; yaw goes down "
        "by more than 45° = *yaw down*; yaw goes up by more than 45° = *yaw up*; anything else = *small movement*.\n"
        "5. Neighbouring non-idle windows become one detected movement.\n"
        "6. Only at the end are the movements compared with Andrew's notes.")

    st.header("A. Groups the GMM found (default settings)")
    d_means = pd.read_csv(RESULTS / f"group_means_{make_tag(DEF_W, DEF_K)}.csv")
    viz.table(d_means.rename(columns={"name": "named as"}))
    st.markdown(f"Like k-means, the GMM starts from a random guess, so it was run with 20 different random starts. "
                f"The default classification was identical in **{S['stable_runs']}/{S['n_runs']}** runs.")

    st.header("B. Does the result depend on the settings?")
    rob = rob_all.copy()
    rob["all_ok"] = ((rob.right_as_yaw_down == rob.n_right) & (rob.left_as_small == rob.n_left)
                     & (rob.uld_as_yaw_up == rob.n_uld))
    piv_ok = rob.pivot(index="k", columns="window_s", values="all_ok")
    text = rob.assign(t=lambda x: "R " + x.right_as_yaw_down.astype(str) + "/" + x.n_right.astype(str)
                      + "<br>L " + x.left_as_small.astype(str) + "/" + x.n_left.astype(str)
                      + "<br>U " + x.uld_as_yaw_up.astype(str) + "/" + x.n_uld.astype(str)
                      ).pivot(index="k", columns="window_s", values="t")
    figh = go.Figure(go.Heatmap(
        z=piv_ok.astype(int).values, x=[f"{w:g} s" for w in piv_ok.columns], y=[f"k = {kk}" for kk in piv_ok.index],
        text=text.values, texttemplate="%{text}", colorscale=[[0, "#f4c7c3"], [1, "#b7e1b0"]], zmin=0, zmax=1,
        showscale=False, xgap=3, ygap=3, hovertemplate="%{y}, window %{x}<br>%{text}<extra></extra>"))
    figh.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="window length",
                       yaxis=dict(autorange="reversed"))
    viz.show(figh)
    ok_settings = ", ".join(f"k = {r.k}, {r.window_s:g} s" for r in rob[rob.all_ok].itertuples()) or "none"
    st.markdown(
        "R = right classified as *yaw down*, L = left as *small movement*, U = up-left-down as *yaw up*. "
        "Green = all 7 correct.\n\n"
        f"- **All 7 correct** in **{int(rob.all_ok.sum())} of {len(rob)}** settings: {ok_settings}.\n"
        f"- **Right** 4/4 in {int((rob.right_as_yaw_down == rob.n_right).sum())} of {len(rob)} settings.\n"
        f"- **Left** 2/2 in {int((rob.left_as_small == rob.n_left).sum())} of {len(rob)} settings.\n"
        f"- **Up-left-down** 1/1 in {int((rob.uld_as_yaw_up == rob.n_uld).sum())} of {len(rob)} settings "
        "(a single example, so it only gets a group of its own when there are enough components).")

    st.header("C. How many groups does the data itself suggest? (BIC)")
    if "bic" in rob.columns:
        fig_bic = go.Figure()
        for ws, x in rob.groupby("window_s"):
            x = x.sort_values("k")
            fig_bic.add_trace(go.Scatter(x=x.k, y=x.bic, mode="lines+markers", name=f"{ws:g} s windows"))
        fig_bic.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis=dict(title="number of components (k)", dtick=1),
                              yaxis_title="BIC (lower = better)", legend=dict(orientation="h", y=1.15))
        viz.show(fig_bic)
        best = S.get("best_k_by_bic", {})
        best_default = best.get(str(DEF_W))
        agree = (" They agree: the number of groups is supported by the data itself."
                 if best_default == DEF_K else
                 " They disagree: the BIC prefers a different number of groups than the one that matches the "
                 "notes (it describes all windows, mostly idle ones, not just the movements).")
        st.markdown(
            "The **BIC** scores each number of groups using only the sensor data (not Andrew's notes): it rewards "
            "explaining the data well and penalises extra groups. **Lower is better.** Compare BIC values only "
            "within the same window length.\n\n"
            "- Best k by BIC: " + ", ".join(f"{w} s → k = {kk}" for w, kk in best.items()) + ".\n"
            f"- With the default {DEF_W} s windows, the BIC suggests **k = {best_default}**; the dashboard uses "
            f"**k = {DEF_K}**." + agree)
    else:
        st.info("No BIC in results/robustness.csv. Run build_results.py to add it.")

    st.header("D. Findings and caveats")
    st.markdown(
        "**What the GMM adds compared with k-means**\n"
        "- Each window gets a **probability** of belonging to each group, not just a hard label, so uncertain "
        "cases can be identified.\n"
        "- The **BIC** gives a data-driven way to choose the number of groups, without using Andrew's notes.\n\n"
        "**Caveats**\n"
        "- ***Left* has no signature of its own.** It is recognised as \"a movement without a large turn\", so "
        "any other small movement would also be called *left*. At 11:50:16 the yaw even goes **down** ~50°, the "
        "same direction as *right*. *Left* is not the mirror image of *right*.\n"
        "- **The tran values follow the rotation rate** (correlation 0.80–0.91 with the change in angles). "
        "Right and left are displacements, but the model tells them apart by the **wrist turn**, not by the hand's "
        "displacement. This may be why *left* looks so weak.\n"
        "- **Only 7 movements** (4 right, 2 left, 1 up-left-down): this shows a pattern but cannot validate a model.\n"
        "- **Manual choices:** the covariance type of the GMM (diagonal, chosen because there are few movement "
        "windows), the 45° naming rule and the 1.5 s resting rule.\n\n"
        "**Next data that would help**: ~10 repetitions of each movement, a few red herrings, confirmation of "
        "whether *up-left-down* is a third movement or a red herring, and the sensor model / units of the first "
        "three values.")