"""Dashboard page for experiment 08 (DBSCAN on the 11/11 events)."""
from pathlib import Path

import streamlit as st

from core import events_viz as evz
from core import viz

HERE = Path(__file__).resolve().parent


def noise(d, E, S, R):
    st.subheader("Groups and noise per setting")
    viz.table(R[["high", "eps", "min_samples", "n_groups", "n_noise", "correct", "n", "ari"]]
              .rename(columns={"high": "threshold", "n_groups": "groups", "n_noise": "noise events",
                               "correct": "R/L correct"}))
    st.caption("Small eps: most events become noise. Large eps: everything merges into one group. "
               "DBSCAN struggles here because ~25 events in 14 dimensions are too sparse to form dense regions.")


evz.cluster_page(
    HERE, "DBSCAN",
    "**DBSCAN:** groups events that sit in dense regions of the feature space. No number of groups, but two "
    "settings: **eps** (how close two events must be to count as neighbours) and **min_samples** (how many "
    "neighbours form a group). Events that fit no group are noise, shown as *unusual*.",
    grid_cols="setting", grid_label="eps × min_samples", extra_analysis=noise)
