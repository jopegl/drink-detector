"""
Interface to view Andrew's hand movement and the detected water sips.

Usage:
    pip install -r requirements.txt
    streamlit run app.py

Needs, in the same folder: gui_core.py, kmeans_windows.py, processed_log.csv and
the results/ folder (created by: python build_results.py processed_log.csv).
Nothing is computed when the page opens: every analysis is read from results/.
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import gui_core as gc
import kmeans_windows as kw

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "processed_log.csv")
RESULTS = os.path.join(HERE, "results")

st.set_page_config(page_title="Water sip detector", layout="wide")

C_SIP = "#2ca02c"
C_WATER = "#2b8cbe"
C_RH = "#e34a33"
C_UNKNOWN = "#888888"
C_ROLL, C_PITCH, C_YAW = "#1f77b4", "#d62728", "#9467bd"
LABEL_COLORS = {"water": C_WATER, "water (+red herring?)": C_WATER,
                "red_herring": C_RH, "unknown": C_UNKNOWN}


def show(fig):
    """Show a chart at full width (works with both new and old Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.plotly_chart(fig, use_container_width=True)


def table(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)


if not os.path.exists(LOG) or not os.path.exists(os.path.join(RESULTS, "summary.json")):
    st.error("Missing processed_log.csv or the results/ folder. "
             "Run: python build_results.py processed_log.csv")
    st.stop()


@st.cache_data(show_spinner="Reading the log...")
def load_log():
    return gc.load(LOG)


@st.cache_data
def load_detection(window_s, k):
    return gc.load_detection(RESULTS, window_s, k)


@st.cache_data
def load_analysis():
    return gc.load_analysis(RESULTS)


d = load_log()
A = load_analysis()
day = d.timestamp.iloc[0].strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Settings")
window_s = st.sidebar.select_slider("Window length (s)", options=[2, 3, 4, 5, 6, 7, 8], value=4)
k = st.sidebar.select_slider("Number of groups (k-means)", options=[2, 3, 4, 5, 6], value=3)
fps = st.sidebar.slider("Animation speed (frames/s)", 2, 20, 8)
st.sidebar.caption("Default: 4 s windows, 3 groups. Every setting was pre-computed.\n\n"
                   "The detector does not use Andrew's notes. They are shown only for comparison.")

E, crosstab, means, found, rh, extras = load_detection(window_s, k)
G = gc.flag_detected_gestures(gc.gestures(d), E)

st.title("Water sip detector")
st.caption(f"Wrist sensor log of {day}, {d.timestamp.iloc[0]:%H:%M}–{d.timestamp.iloc[-1]:%H:%M} · "
           f"{len(d):,} readings · unsupervised k-means")

tab_dash, tab_analysis = st.tabs(["Dashboard", "Analysis"])

# ===========================================================================
# DASHBOARD
# ===========================================================================
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annotated sips found", f"{len(found)}/{len(kw.WATER_NOTES)}")
    c2.metric("Red herring flagged as sip", f"{len(rh)}/{len(kw.RED_HERRING_NOTES)}")
    c3.metric("Total sips detected", len(E))
    c4.metric("Detected with no note", len(extras),
              help=", ".join(t.strftime("%H:%M:%S") for t in extras.start) or None)
    if (window_s, k) != (4, 3):
        st.info(f"Showing {window_s} s windows and {k} groups. Default is 4 s and 3 groups.")

    # 1. Timeline ------------------------------------------------------------
    st.subheader("1. Timeline")
    fig = go.Figure()
    for col, name, color in [("gyro_roll", "roll", C_ROLL), ("gyro_pitch", "pitch", C_PITCH)]:
        fig.add_trace(go.Scattergl(x=d.timestamp, y=d[col], name=name, mode="lines",
                                   line=dict(width=1, color=color)))
    for _, e in E.iterrows():
        fig.add_vrect(x0=e.start, x1=e.end, fillcolor=C_SIP, opacity=0.3, line_width=0)
    for notes, color in [(kw.WATER_NOTES, C_WATER), (kw.RED_HERRING_NOTES, C_RH)]:
        for n in notes:
            t = pd.Timestamp(f"{day} {n}:30")
            fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper",
                          line=dict(color=color, width=2, dash="dot"))
    for name, color, style in [("detected sip", C_SIP, "solid"),
                               ("note: water", C_WATER, "dot"),
                               ("note: red herring", C_RH, "dot")]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name,
                                 line=dict(color=color, width=6 if style == "solid" else 2, dash=style)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="angle (°)",
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    show(fig)
    st.caption("Green bands: sips detected by k-means. Dotted lines: Andrew's notes. "
               "Drag with the mouse to zoom.")

    # 2. One gesture -----------------------------------------------------------
    st.subheader("2. Inspect a gesture")

    def describe(g):
        det = "DETECTED AS SIP" if g.detected_as_sip else "not a sip"
        return f"{g.start:%H:%M:%S}  ·  note: {g.label}  ·  model: {det}"

    options = {describe(g): g for _, g in G.iterrows()}
    g = options[st.selectbox("Gesture", list(options.keys()))]
    pad = pd.Timedelta(seconds=2)
    segment = d[(d.timestamp >= g.start - pad) & (d.timestamp <= g.end + pad)].reset_index(drop=True)

    col_a, col_b = st.columns(2)

    with col_a:  # hand rotation
        st.markdown("**Hand rotation** (roll, pitch, yaw)")
        step = max(1, int(round(kw.RATE_HZ / fps)))
        sample = segment.iloc[::step].reset_index(drop=True)
        R = gc.rotation_matrices(sample.gyro_roll, sample.gyro_pitch, sample.gyro_yaw)
        I, J, K = gc.TRIANGLES
        is_sip = [((E.start <= t) & (E.end >= t)).any() for t in sample.timestamp]

        def traces(i):
            palm, finger, axes = gc.hand_geometry(R[i])
            data = [go.Mesh3d(x=palm[:, 0], y=palm[:, 1], z=palm[:, 2], i=I, j=J, k=K,
                              color=C_SIP if is_sip[i] else "#999999", opacity=0.85,
                              flatshading=True, name="hand"),
                    go.Scatter3d(x=finger[:, 0], y=finger[:, 1], z=finger[:, 2], mode="lines",
                                 line=dict(color="black", width=10), name="forward")]
            for j, (color, name) in enumerate([(C_ROLL, "x"), (C_PITCH, "y"), (C_YAW, "z")]):
                data.append(go.Scatter3d(x=[0, axes[j, 0]], y=[0, axes[j, 1]], z=[0, axes[j, 2]],
                                         mode="lines", line=dict(color=color, width=4),
                                         name=f"{name} axis"))
            return data

        def title(i):
            r = sample.iloc[i]
            return (f"{r.timestamp:%H:%M:%S}  roll {r.gyro_roll:.0f}°  pitch {r.gyro_pitch:.0f}°  "
                    f"yaw {r.gyro_yaw:.0f}°" + ("  ·  SIP" if is_sip[i] else ""))

        frames = [go.Frame(data=traces(i), name=str(i), layout=go.Layout(title_text=title(i)))
                  for i in range(len(sample))]
        axis = dict(range=[-1.3, 1.3], showticklabels=False, title="")
        fig3d = go.Figure(data=traces(0), frames=frames)
        fig3d.update_layout(
            title_text=title(0), height=520, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
            scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube",
                       camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))),
            updatemenus=[dict(type="buttons", x=0, y=0, xanchor="left", yanchor="top", direction="left",
                              buttons=[dict(label="▶ Play", method="animate",
                                            args=[None, {"frame": {"duration": int(1000 / fps), "redraw": True},
                                                         "fromcurrent": True, "transition": {"duration": 0}}]),
                                       dict(label="⏸ Pause", method="animate",
                                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                           "mode": "immediate"}])])],
            sliders=[dict(x=0.15, len=0.85, y=0, currentvalue=dict(visible=False),
                          steps=[dict(method="animate", label="",
                                      args=[[str(i)], {"mode": "immediate",
                                                       "frame": {"duration": 0, "redraw": True},
                                                       "transition": {"duration": 0}}])
                                 for i in range(len(sample))])])
        show(fig3d)
        st.caption("The box turns green at the moments the model classified as a sip. "
                   "The black rod points to the front of the sensor.")

    with col_b:  # estimated path
        st.markdown("**Estimated hand path** (from tran)")
        gesture = d[(d.timestamp >= g.start) & (d.timestamp <= g.end)].reset_index(drop=True)
        P = gc.estimated_path(gesture)
        scale = np.abs(P[["x", "y", "z"]].values).max() or 1.0
        P[["x", "y", "z"]] = P[["x", "y", "z"]] / scale
        figp = go.Figure()
        figp.add_trace(go.Scatter3d(x=P.x, y=P.y, z=P.z, mode="lines",
                                    line=dict(color=P.t_s, colorscale="Viridis", width=6,
                                              colorbar=dict(title="s", thickness=12)), name="path"))
        figp.add_trace(go.Scatter3d(x=[P.x.iloc[0]], y=[P.y.iloc[0]], z=[P.z.iloc[0]], mode="markers",
                                    marker=dict(size=6, color="black"), name="start"))
        figp.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                           title_text="colour = time since the start of the gesture",
                           scene=dict(aspectmode="cube", xaxis_title="x", yaxis_title="y",
                                      zaxis_title="z"))
        show(figp)
        st.warning("Rough estimate, in arbitrary units. The acceleration is integrated twice "
                   "and errors build up. Velocity was forced to zero at the start and end of the gesture.")

    figs = go.Figure()
    for col, name, color in [("gyro_roll", "roll", C_ROLL), ("gyro_pitch", "pitch", C_PITCH),
                             ("gyro_yaw", "yaw", C_YAW)]:
        figs.add_trace(go.Scatter(x=segment.timestamp, y=segment[col], name=name, line=dict(color=color)))
    figs.add_trace(go.Scatter(x=segment.timestamp, y=segment.tran_mag, name="|tran|", yaxis="y2",
                              line=dict(color="#888", width=1), fill="tozeroy", opacity=0.3))
    for _, e in E.iterrows():
        if e.end >= segment.timestamp.iloc[0] and e.start <= segment.timestamp.iloc[-1]:
            figs.add_vrect(x0=e.start, x1=e.end, fillcolor=C_SIP, opacity=0.25, line_width=0)
    figs.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="angle (°)",
                       yaxis2=dict(overlaying="y", side="right", title="|tran|", showgrid=False),
                       legend=dict(orientation="h", y=1.15), hovermode="x unified")
    show(figs)

    # 3. Tables ----------------------------------------------------------------
    st.subheader("3. Detector results")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Detected sips**")
        table(E.assign(start=E.start.dt.strftime("%H:%M:%S"), end=E.end.dt.strftime("%H:%M:%S"))
              .rename(columns={"n_windows": "windows", "andrew_note": "Andrew's note"}))
    with t2:
        st.markdown("**Windows by note × k-means group**")
        table(crosstab.rename(columns={"label": "note"}))
        st.markdown("**Mean of each group**")
        table(means)

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

    # A. Robustness grid -------------------------------------------------------
    st.header("A. Does the result depend on the settings?")
    st.markdown("Every combination of window length and number of groups was tested.")
    piv_found = rob.pivot(index="k", columns="window_s", values="sips_found")
    piv_rh = rob.pivot(index="k", columns="window_s", values="red_herring_as_sip")
    ok = (piv_found == len(kw.WATER_NOTES)) & (piv_rh == 0)
    text = [[f"{piv_found.loc[kk, ws]}/9 sips<br>RH {piv_rh.loc[kk, ws]}/1"
             for ws in piv_found.columns] for kk in piv_found.index]
    figh = go.Figure(go.Heatmap(
        z=ok.astype(int).values, x=[f"{w} s" for w in piv_found.columns],
        y=[f"k = {kk}" for kk in piv_found.index], text=text, texttemplate="%{text}",
        colorscale=[[0, "#f4c7c3"], [1, "#b7e1b0"]], showscale=False, xgap=3, ygap=3,
        hovertemplate="%{y}, window %{x}<br>%{text}<extra></extra>"))
    figh.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="window length", yaxis=dict(autorange="reversed"))
    show(figh)
    st.markdown(
        "Green = all 9 sips found **and** the red herring (RH) not flagged.\n\n"
        "- **k = 3 works for every window length** (2 to 8 s).\n"
        "- **k = 2** is too coarse: it only separates *still* from *moving*, so the red herring is flagged as a sip.\n"
        "- **k ≥ 4** is too fine: the sip group splits into sub-types and some sips are lost.")
    st.markdown("**Extra detections (no note) with k = 3**")
    table(rob[rob.k == 3][["window_s", "sips_found", "red_herring_as_sip", "extra_events", "extra_times"]]
          .rename(columns={"window_s": "window (s)", "sips_found": "sips found (of 9)",
                           "red_herring_as_sip": "red herring as sip", "extra_events": "extra events",
                           "extra_times": "extra at"}))
    st.caption("17:29 appears at every window length: it is very likely an unannotated sip. "
               "With 2–3 s windows, one sip is sometimes split in two (16:53 or 16:58), "
               "which shows up as an extra event.")

    # B. Gesture-level k-means -------------------------------------------------
    st.header("B. Cross-check: k-means on whole gestures")
    st.markdown("A second, independent approach: the log is split into the 13 movement segments "
                "(gestures) and k-means (k = 2) groups the gestures themselves.")
    table(A["gesture_summary"].rename(columns={
        "feature_set": "features used", "n_features": "# features", "same_split_runs": "stable runs",
        "water_group": "group with the sips", "other_group": "other group"}))
    st.markdown("- With **rotation** features, k-means puts all 9 annotated sips plus 17:29 in one group, "
                "and the red herring, 16:54 and 17:26 in the other.\n"
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
                x=x.pca_1, y=x.pca_2, mode="markers+text", text=x.start.str[:5] +
                x.margin.map(lambda m: f" ({m:+.1f})"), textposition="top right",
                marker=dict(size=16, color=color, symbol=symbol, line=dict(color="black", width=1)),
                name=f"{lab} · {grp}"))
    ev = S["pca_variance"]
    figpca.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title=f"component 1 ({ev[0] * 100:.0f}% of variance)",
                         yaxis_title=f"component 2 ({ev[1] * 100:.0f}%)",
                         legend=dict(orientation="h", y=-0.15))
    show(figpca)
    st.caption("Each point is a gesture; close points move alike. Colour = Andrew's note, "
               "circle = 'sip' group, square = other group. Number = margin (positive = sip side, "
               "close to 0 = uncertain).")

    # C. Robustness of the gesture-level k-means -------------------------------
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
        "- **Leave-one-out:** k-means is fitted without one gesture, then that gesture is classified "
        "as if it were new. Repeated for every gesture.\n"
        "- **Removing features:** the result only breaks when **pitch_min** (how far the wrist tilts back) "
        "is removed, which matches the drinking motion.\n"
        f"- **k = 3:** the sips stay together: {', '.join(S['k3_groups'][0])}.")
    st.markdown("**Cases where removing features causes errors**")
    table(rem[rem.accuracy < 1].assign(accuracy=lambda x: (x.accuracy * 100).round(0).astype(int).astype(str) + "%")
          .rename(columns={"n_removed": "# removed", "removed": "features removed",
                           "wrong": "gestures misclassified"}))
    st.markdown("**Per-gesture result**")
    cols = ["start", "label", "kmeans_says", "leave_one_out_says", "margin", "duration_s",
            "pitch_min", "largest_angle_range", "time_pitch_below_-60_s"]
    table(F[cols].rename(columns={"label": "Andrew's note", "kmeans_says": "k-means",
                                  "leave_one_out_says": "leave-one-out"}))

    # D. Caveats --------------------------------------------------------------
    st.header("D. Findings and caveats")
    st.markdown(
        "**Findings**\n"
        "- All 9 annotated sips are found and the red herring at 18:35 is not flagged, "
        "without the model ever seeing the notes.\n"
        "- **17:29 looks like an unannotated sip**: every method puts it with the sips. Worth checking with Andrew.\n"
        "- The 18:10 note matches a sip that happens at **18:08:39** (clock or note offset).\n\n"
        "**Caveats**\n"
        "- Only **one** confirmed red herring: this is a proof of concept, not a validated model. "
        "A new recording with more non-drinking gestures (scratching the face, fixing hair, phone) is needed.\n"
        "- **17:49** is the weakest sip: Andrew noted a red herring right after it, and both seem to be "
        "in the same movement segment.\n"
        "- Remaining manual choices: the features, k = 3, and the rule \"most tilted group = sip\".\n"
        "- The estimated hand path is a rough double integration of tran, in arbitrary units.")
