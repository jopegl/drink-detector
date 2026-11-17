"""Dashboard page for experiment 07 (agglomerative clustering on the 11/11 events)."""
from pathlib import Path

import plotly.figure_factory as ff
import streamlit as st
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler

from core import events as ev
from core import events_viz as evz
from core import viz

HERE = Path(__file__).resolve().parent


def dendrogram(d, E, S, R):
    st.subheader("Dendrogram")
    X = StandardScaler().fit_transform(E[ev.FEATURES])
    labels = [f"{e.start:%H:%M:%S} {e.truth_label}" for e in E.itertuples()]
    fig = ff.create_dendrogram(X, labels=labels, linkagefun=lambda x: linkage(x, "ward"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=120))
    viz.show(fig)
    st.caption("Each leaf is an event (start time and truth). Events joined lower down are more similar. "
               "Cutting the tree into k branches gives the k groups.")


evz.cluster_page(
    HERE, "agglomerative clustering",
    "**Agglomerative (hierarchical) clustering:** starts with every event on its own and repeatedly joins the two "
    "most similar groups (Ward linkage) until *k* groups are left. No random start: same settings, same result.",
    grid_cols="setting", grid_label="k = 2, 3, 4", extra_analysis=dendrogram)
