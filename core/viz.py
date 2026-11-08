"""
Dashboard building blocks shared by every experiment (Streamlit + Plotly).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

C_ROLL, C_PITCH, C_YAW = "#1f77b4", "#d62728", "#9467bd"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
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


def header(info):
    st.title(f"Experiment {info['code']} · {info['title']}")
    st.caption(f"{info['date']} · {info['summary']}")


def load_log(path):
    return pd.read_csv(path, parse_dates=["timestamp"])


def gestures(d):
    """The movement segments from the 'event' column, with start, end and label."""
    return (d[d.event > 0]
            .groupby("event")
            .agg(start=("timestamp", "first"), end=("timestamp", "last"), label=("label", "first"))
            .reset_index())


# ---------------------------------------------------------------------------
# Rotation: roll, pitch, yaw (degrees) -> rotation matrix (Z-Y-X convention)
# ---------------------------------------------------------------------------
def rotation_matrices(roll, pitch, yaw):
    r, p, y = (np.deg2rad(np.asarray(v, dtype=float)) for v in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty((len(r), 3, 3))
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


# A simple "hand": flat palm (box) + a finger pointing forward (+x)
_PALM = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.35, 0.35) for z in (-0.06, 0.06)])
_TRI = ([0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1],
        [1, 3, 5, 7, 1, 5, 3, 7, 2, 6, 3, 7],
        [3, 2, 7, 6, 5, 4, 7, 6, 6, 4, 7, 5])
_FINGER = np.array([[0.5, 0.0, 0.0], [1.1, 0.0, 0.0]])
_AXES = np.eye(3) * 0.9


def hand_geometry(R):
    return _PALM @ R.T, _FINGER @ R.T, _AXES @ R.T


def estimated_path(segment):
    """
    Rough hand path over one gesture: rotate tran into the world frame, integrate to velocity,
    force zero velocity at start and end (linear drift removal), integrate to position.
    Arbitrary units: errors build up with each integration.
    """
    t = (segment.timestamp - segment.timestamp.iloc[0]).dt.total_seconds().values
    dt = np.diff(t, prepend=t[0])
    R = rotation_matrices(segment.gyro_roll, segment.gyro_pitch, segment.gyro_yaw)
    a_world = np.einsum("nij,nj->ni", R, segment[["tran_x", "tran_y", "tran_z"]].values)
    v = np.cumsum(a_world * dt[:, None], axis=0)
    if t[-1] > 0:
        v = v - np.outer(t / t[-1], v[-1])
    p = np.cumsum(v * dt[:, None], axis=0)
    return pd.DataFrame({"t_s": t, "x": p[:, 0], "y": p[:, 1], "z": p[:, 2]})


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def add_spans(fig, spans, opacity=0.3):
    """spans: list of (start, end, colour)."""
    for start, end, color in spans:
        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=opacity, line_width=0)


def add_note_lines(fig, notes):
    """notes: list of (timestamp, colour)."""
    for t, color in notes:
        fig.add_shape(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper",
                      line=dict(color=color, width=2, dash="dot"))


def add_legend_entries(fig, entries):
    """entries: list of (name, colour, 'band' | 'line')."""
    for name, color, kind in entries:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name,
                                 line=dict(color=color, width=6 if kind == "band" else 2,
                                           dash="solid" if kind == "band" else "dot")))


def timeline(d, series, spans, notes, legend, height=380):
    """series: list of (column, name, colour)."""
    fig = go.Figure()
    for col, name, color in series:
        fig.add_trace(go.Scattergl(x=d.timestamp, y=d[col], name=name, mode="lines",
                                   line=dict(width=1, color=color)))
    add_spans(fig, spans)
    add_note_lines(fig, notes)
    add_legend_entries(fig, legend)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="angle (°)",
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    return fig


def hand_animation(sample, highlight, fps, highlight_color, highlight_name):
    """3D hand that turns like Andrew's wrist. highlight: list of bools, one per sample row."""
    R = rotation_matrices(sample.gyro_roll, sample.gyro_pitch, sample.gyro_yaw)
    I, J, K = _TRI

    def traces(i):
        palm, finger, axes = hand_geometry(R[i])
        data = [go.Mesh3d(x=palm[:, 0], y=palm[:, 1], z=palm[:, 2], i=I, j=J, k=K,
                          color=highlight_color if highlight[i] else "#999999", opacity=0.85,
                          flatshading=True, name="hand"),
                go.Scatter3d(x=finger[:, 0], y=finger[:, 1], z=finger[:, 2], mode="lines",
                             line=dict(color="black", width=10), name="forward")]
        for j, (color, name) in enumerate([(C_ROLL, "x"), (C_PITCH, "y"), (C_YAW, "z")]):
            data.append(go.Scatter3d(x=[0, axes[j, 0]], y=[0, axes[j, 1]], z=[0, axes[j, 2]],
                                     mode="lines", line=dict(color=color, width=4), name=f"{name} axis"))
        return data

    def title(i):
        r = sample.iloc[i]
        return (f"{r.timestamp:%H:%M:%S}  roll {r.gyro_roll:.0f}°  pitch {r.gyro_pitch:.0f}°  "
                f"yaw {r.gyro_yaw:.0f}°" + (f"  ·  {highlight_name}" if highlight[i] else ""))

    frames = [go.Frame(data=traces(i), name=str(i), layout=go.Layout(title_text=title(i)))
              for i in range(len(sample))]
    axis = dict(range=[-1.3, 1.3], showticklabels=False, title="")
    fig = go.Figure(data=traces(0), frames=frames)
    fig.update_layout(
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
                                  args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                                   "transition": {"duration": 0}}])
                             for i in range(len(sample))])])
    return fig


def path_figure(gesture):
    P = estimated_path(gesture)
    scale = np.abs(P[["x", "y", "z"]].values).max() or 1.0
    P[["x", "y", "z"]] = P[["x", "y", "z"]] / scale
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=P.x, y=P.y, z=P.z, mode="lines",
                               line=dict(color=P.t_s, colorscale="Viridis", width=6,
                                         colorbar=dict(title="s", thickness=12)), name="path"))
    fig.add_trace(go.Scatter3d(x=[P.x.iloc[0]], y=[P.y.iloc[0]], z=[P.z.iloc[0]], mode="markers",
                               marker=dict(size=6, color="black"), name="start"))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                      title_text="colour = time since the start of the gesture",
                      scene=dict(aspectmode="cube", xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    return fig


def signals_figure(segment, spans, series=None):
    series = series or [("gyro_roll", "roll", C_ROLL), ("gyro_pitch", "pitch", C_PITCH),
                        ("gyro_yaw_unwrapped", "yaw", C_YAW)]
    fig = go.Figure()
    for col, name, color in series:
        fig.add_trace(go.Scatter(x=segment.timestamp, y=segment[col], name=name, line=dict(color=color)))
    fig.add_trace(go.Scatter(x=segment.timestamp, y=segment.tran_mag, name="|tran|", yaxis="y2",
                             line=dict(color="#888", width=1), fill="tozeroy", opacity=0.3))
    t0, t1 = segment.timestamp.iloc[0], segment.timestamp.iloc[-1]
    add_spans(fig, [(s, e, c) for s, e, c in spans if e >= t0 and s <= t1], opacity=0.25)
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="angle (°)",
                      yaxis2=dict(overlaying="y", side="right", title="|tran|", showgrid=False),
                      legend=dict(orientation="h", y=1.15), hovermode="x unified")
    return fig


def gesture_inspector(d, G, spans, fps, rate_hz, highlight_name, key):
    """
    Pick a gesture and see its 3D hand rotation, estimated path and signals.
    G: one row per gesture with columns start, end, description.
    spans: list of (start, end, colour) of detected events (used to colour the hand).
    """
    options = {r.description: r for r in G.itertuples()}
    g = options[st.selectbox("Gesture", list(options.keys()), key=key)]
    pad = pd.Timedelta(seconds=2)
    segment = d[(d.timestamp >= g.start - pad) & (d.timestamp <= g.end + pad)].reset_index(drop=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Hand rotation** (roll, pitch, yaw)")
        step = max(1, int(round(rate_hz / fps)))
        sample = segment.iloc[::step].reset_index(drop=True)
        color_at = []
        for t in sample.timestamp:
            c = next((c for s, e, c in spans if s <= t <= e), None)
            color_at.append(c)
        # the hand takes the colour of the detected event it is in
        main_color = next((c for c in color_at if c), "#2ca02c")
        show(hand_animation(sample, [c is not None for c in color_at], fps, main_color, highlight_name))
        st.caption("The box takes the colour of the detected event while it is inside one. "
                   "The black rod points to the front of the sensor.")
    with col_b:
        st.markdown("**Estimated hand path** (from tran)")
        gesture = d[(d.timestamp >= g.start) & (d.timestamp <= g.end)].reset_index(drop=True)
        show(path_figure(gesture))
        st.warning("Rough estimate, in arbitrary units. The acceleration is integrated twice "
                   "and errors build up. Velocity was forced to zero at the start and end of the gesture.")
    show(signals_figure(segment, spans))
