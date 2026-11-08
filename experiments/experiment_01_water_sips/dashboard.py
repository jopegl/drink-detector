"""
Dashboard page for experiment 01 (water sip detection).
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
WATER_NOTES = info["water_notes"]
RH_NOTES = info["red_herring_notes"]

C_SIP, C_WATER, C_RH, C_UNKNOWN = "#2ca02c", "#2b8cbe", "#e34a33", "#888888"
LABEL_COLORS = {"water": C_WATER, "water (+red herring?)": C_WATER, "red_herring": C_RH, "unknown": C_UNKNOWN}

if not LOG.exists() or not (RESULTS / "summary.json").exists():
    st.error("Missing data or results. Run prepare.py and build_results.py in this experiment's folder.")
    st.stop()


@st.cache_data(show_spinner="Reading the log...")
def load_log(path):
    return viz.load_log(path)


@st.cache_data
def load_detection(results_dir, window_s, k):
    tag = f"w{int(window_s)}_k{int(k)}"
    E = pd.read_csv(f"{results_dir}/events_{tag}.csv", parse_dates=["start", "end"])
    crosstab = pd.read_csv(f"{results_dir}/crosstab_{tag}.csv")
    means = pd.read_csv(f"{results_dir}/group_means_{tag}.csv")
    return E, crosstab, means


@st.cache_data
def load_analysis(results_dir):
    with open(f"{results_dir}/summary.json") as f:
        summary = json.load(f)
    return {"summary": summary,
            "robustness": pd.read_csv(f"{results_dir}/window_robustness.csv"),
            "gesture_summary": pd.read_csv(f"{results_dir}/gesture_kmeans_summary.csv"),
            "gestures": pd.read_csv(f"{results_dir}/gestures.csv"),
            "removal": pd.read_csv(f"{results_dir}/feature_removal.csv")}


d = load_log(str(LOG))
A = load_analysis(str(RESULTS))
day = d.timestamp.iloc[0].strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"**Experiment {info['code']} settings**")
window_s = st.sidebar.select_slider("Window length (s)", options=[2, 3, 4, 5, 6, 7, 8], value=4, key="e01_w")
k = st.sidebar.select_slider("Number of groups (k-means)", options=[2, 3, 4, 5, 6], value=3, key="e01_k")
fps = st.sidebar.slider("Animation speed (frames/s)", 2, 20, 8, key="e01_fps")
st.sidebar.caption("Default: 4 s windows, 3 groups. Every setting was pre-computed. "
                   "The detector does not use Andrew's notes; they are shown only for comparison.")

E, crosstab, means = load_detection(str(RESULTS), window_s, k)
found = E[E.andrew_note.str.startswith("water")]
rh = E[E.andrew_note.str.startswith("red_herring")]
extras = E[E.andrew_note == "—"]
spans = [(e.start, e.end, C_SIP) for e in E.itertuples()]

viz.header(info)
tab_dash, tab_analysis = st.tabs(["Dashboard", "Analysis"])

# ===========================================================================
# DASHBOARD
# ===========================================================================
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annotated sips found", f"{len(found)}/{len(WATER_NOTES)}")
    c2.metric("Red herring flagged as sip", f"{len(rh)}/{len(RH_NOTES)}")
    c3.metric("Total sips detected", len(E))
    c4.metric("Detected with no note", len(extras),
              help=", ".join(t.strftime("%H:%M:%S") for t in extras.start) or None)
    if (window_s, k) != (4, 3):
        st.info(f"Showing {window_s} s windows and {k} groups. Default is 4 s and 3 groups.")

    st.subheader("1. Timeline")
    notes = ([(pd.Timestamp(f"{day} {n}:30"), C_WATER) for n in WATER_NOTES]
             + [(pd.Timestamp(f"{day} {n}:30"), C_RH) for n in RH_NOTES])
    viz.show(viz.timeline(
        d, [("gyro_roll", "roll", viz.C_ROLL), ("gyro_pitch", "pitch", viz.C_PITCH)], spans, notes,
        [("detected sip", C_SIP, "band"), ("note: water", C_WATER, "line"), ("note: red herring", C_RH, "line")]))
    st.caption("Green bands: sips detected by k-means. Dotted lines: Andrew's notes. Drag to zoom.")

    st.subheader("2. Inspect a gesture")
    G = viz.gestures(d)
    G["detected"] = [((E.start <= g.end) & (E.end >= g.start)).any() for g in G.itertuples()]
    G["description"] = [f"{g.start:%H:%M:%S}  ·  note: {g.label}  ·  model: "
                        f"{'DETECTED AS SIP' if g.detected else 'not a sip'}" for g in G.itertuples()]
    viz.gesture_inspector(d, G, spans, fps, info["rate_hz"], "SIP", key="e01_gesture")

    st.subheader("3. Detector results")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Detected sips**")
        viz.table(E.assign(start=E.start.dt.strftime("%H:%M:%S"), end=E.end.dt.strftime("%H:%M:%S"))
                  .rename(columns={"n_windows": "windows", "andrew_note": "Andrew's note"}))
    with t2:
        st.markdown("**Windows by note × k-means group**")
        viz.table(crosstab.rename(columns={"label": "note"}))
        st.markdown("**Mean of each group**")
        viz.table(means)

# ===========================================================================
# ANALYSIS
# ===========================================================================
with tab_analysis:
    S = A["summary"]
    rob = A["robustness"]

    st.header("How the detector works")
    st.markdown(
        "1. The whole log is cut into short **windows** (default 4 s, 50% overlap).\n"
        "2. Each window becomes a feature card: wrist tilt and twist (pitch, roll) and movement intensity (tran).\n"
        "3. **k-means** splits the windows into groups **without seeing any label**.\n"
        "4. The **sip** group is the one where the wrist is tilted back the most (lowest pitch). "
        "This rule does not use the labels either.\n"
        "5. Neighbouring sip windows become one detected event.\n"
        "6. Only at the end are the events compared with Andrew's notes.")

    st.header("A. Does the result depend on the settings?")
    piv_found = rob.pivot(index="k", columns="window_s", values="sips_found")
    piv_rh = rob.pivot(index="k", columns="window_s", values="red_herring_as_sip")
    ok = (piv_found == len(WATER_NOTES)) & (piv_rh == 0)
    text = [[f"{piv_found.loc[kk, ws]}/{len(WATER_NOTES)} sips<br>RH {piv_rh.loc[kk, ws]}/1"
             for ws in piv_found.columns] for kk in piv_found.index]
    figh = go.Figure(go.Heatmap(
        z=ok.astype(int).values, x=[f"{w} s" for w in piv_found.columns],
        y=[f"k = {kk}" for kk in piv_found.index], text=text, texttemplate="%{text}",
        colorscale=[[0, "#f4c7c3"], [1, "#b7e1b0"]], zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
        hovertemplate="%{y}, window %{x}<br>%{text}<extra></extra>"))
    figh.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="window length", yaxis=dict(autorange="reversed"))
    viz.show(figh)
    st.markdown(
        "Green = all 9 sips found **and** the red herring (RH) not flagged.\n\n"
        "- **k = 3 works for every window length** (2 to 8 s).\n"
        "- **k = 2** is too coarse: it only separates *still* from *moving*, so the red herring is flagged.\n"
        "- **k ≥ 4** is too fine: the sip group splits into sub-types and some sips are lost.")
    st.markdown("**Extra detections (no note) with k = 3**")
    viz.table(rob[rob.k == 3][["window_s", "sips_found", "red_herring_as_sip", "extra_events", "extra_times"]]
              .rename(columns={"window_s": "window (s)", "sips_found": "sips found (of 9)",
                               "red_herring_as_sip": "red herring as sip", "extra_events": "extra events",
                               "extra_times": "extra at"}))
    st.caption("17:29 appears at every window length: it is very likely an unannotated sip. With 2–3 s "
               "windows one sip is sometimes split in two (16:53 or 16:58), which shows up as an extra event.")

    st.header("B. Cross-check: k-means on whole gestures")
    st.markdown("A second, independent approach: the log is split into the 13 movement segments "
                "(gestures) and k-means (k = 2) groups the gestures themselves.")
    viz.table(A["gesture_summary"].rename(columns={
        "feature_set": "features used", "n_features": "# features", "same_split_runs": "stable runs",
        "water_group": "group with the sips", "other_group": "other group"}))
    st.markdown("- With **rotation** features, all 9 annotated sips plus 17:29 end up in one group, and the "
                "red herring, 16:54 and 17:26 in the other.\n"
                "- With **translation (tran) only**, it fails: the red herring moves as much as a sip. "
                "**What identifies a sip is the wrist rotation.**")
    F = A["gestures"]
    figpca = go.Figure()
    for lab, color in LABEL_COLORS.items():
        for in_w, symbol, grp in [(True, "circle", "water group"), (False, "square", "other group")]:
            x = F[(F.label == lab) & (F.in_water_group == in_w)]
            if x.empty:
                continue
            figpca.add_trace(go.Scatter(
                x=x.pca_1, y=x.pca_2, mode="markers+text",
                text=x.start.str[:5] + x.margin.map(lambda m: f" ({m:+.1f})"), textposition="top right",
                marker=dict(size=16, color=color, symbol=symbol, line=dict(color="black", width=1)),
                name=f"{lab} · {grp}"))
    ev = S["pca_variance"]
    figpca.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title=f"component 1 ({ev[0] * 100:.0f}% of variance)",
                         yaxis_title=f"component 2 ({ev[1] * 100:.0f}%)", legend=dict(orientation="h", y=-0.15))
    viz.show(figpca)
    st.caption("Each point is a gesture; close points move alike. Colour = Andrew's note, circle = 'sip' group, "
               "square = other group. Number = margin (positive = sip side, close to 0 = uncertain).")

    st.header("C. Robustness checks (gesture level)")
    rem = A["removal"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy on annotated gestures", f"{S['accuracy'] * 100:.0f}%",
              help=f"{S['n_annotated']} gestures with a note")
    m2.metric("Leave-one-out (unseen gesture)", f"{S['loo_correct']}/{S['n_annotated']}")
    r1, r2 = rem[rem.n_removed == 1], rem[rem.n_removed == 2]
    m3.metric("Removing 1 feature", f"{(r1.accuracy == 1).sum()}/{len(r1)} at 100%",
              help=f"worst: {r1.accuracy.min() * 100:.0f}%")
    m4.metric("Removing 2 features", f"{(r2.accuracy == 1).sum()}/{len(r2)} at 100%",
              help=f"worst: {r2.accuracy.min() * 100:.0f}%")
    st.markdown(
        "- **Leave-one-out:** k-means is fitted without one gesture, then that gesture is classified as if it were "
        "new. Repeated for every gesture.\n"
        "- **Removing features:** the result only breaks when **pitch_min** (how far the wrist tilts back) is "
        "removed, which matches the drinking motion.\n"
        f"- **k = 3:** the sips stay together: {', '.join(S['k3_groups'][0])}.")
    st.markdown("**Cases where removing features causes errors**")
    viz.table(rem[rem.accuracy < 1]
              .assign(accuracy=lambda x: (x.accuracy * 100).round(0).astype(int).astype(str) + "%")
              .rename(columns={"n_removed": "# removed", "removed": "features removed",
                               "wrong": "gestures misclassified"}))
    st.markdown("**Per-gesture result**")
    cols = ["start", "label", "kmeans_says", "leave_one_out_says", "margin", "duration_s",
            "pitch_min", "largest_angle_range", "time_pitch_below_-60_s"]
    viz.table(F[cols].rename(columns={"label": "Andrew's note", "kmeans_says": "k-means",
                                      "leave_one_out_says": "leave-one-out"}))

    st.header("D. Findings and caveats")
    st.markdown(
        "**Findings**\n"
        "- All 9 annotated sips are found and the red herring at 18:35 is not flagged, without the model ever "
        "seeing the notes.\n"
        "- **17:29 looks like an unannotated sip**: every method puts it with the sips. Worth checking with Andrew.\n"
        "- The 18:10 note matches a sip that happens at **18:08:39** (clock or note offset).\n\n"
        "**Caveats**\n"
        "- Only **one** confirmed red herring: this is a proof of concept, not a validated model. A new recording "
        "with more non-drinking gestures (scratching the face, fixing hair, phone) is needed.\n"
        "- **17:49** is the weakest sip: Andrew noted a red herring right after it, and both seem to be in the "
        "same movement segment.\n"
        "- Remaining manual choices: the features, k = 3, and the rule \"most tilted group = sip\".\n"
        "- The estimated hand path is a rough double integration of tran, in arbitrary units.")
