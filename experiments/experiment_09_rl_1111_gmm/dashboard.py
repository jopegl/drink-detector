"""Dashboard page for experiment 09 (Gaussian Mixture Model on the 11/11 events)."""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from core import events_viz as evz
from core import viz

HERE = Path(__file__).resolve().parent


def bic(d, E, S, R):
    st.subheader("How many groups does the data suggest? (BIC)")
    fig = go.Figure()
    for h, x in R.groupby("high"):
        x = x.sort_values("k")
        fig.add_trace(go.Scatter(x=x.k, y=x.bic, mode="lines+markers", name=f"threshold {h}"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="k", dtick=1),
                      yaxis_title="BIC (lower = better)", legend=dict(orientation="h", y=1.15))
    viz.show(fig)
    st.caption("The BIC uses only the sensor data. Compare values only within the same detection threshold, "
               "since the number of events changes with it.")
    if S.get("stable_runs") is not None:
        st.markdown(f"The default grouping of the R/L movements was identical in **{S['stable_runs']}/{S['n_runs']}** "
                    "random starts.")


evz.cluster_page(
    HERE, "GMM",
    "**Gaussian Mixture Model (GMM):** like k-means, but each group is a cloud with its own spread (diagonal "
    "covariance), and each event gets a **probability** of belonging to each group, shown as *confidence*.",
    grid_cols="setting", grid_label="k = 2, 3, 4", extra_analysis=bic, extra_cols=("confidence",))
