"""Dashboard page for experiment 06 (k-means on the 11/11 events). Reads results/ (build_results.py)."""
from pathlib import Path

import streamlit as st

from core import events_viz as evz

HERE = Path(__file__).resolve().parent


def stability(d, E, S, R):
    sc = S.get("stable_runs")
    if sc is not None:
        st.markdown(f"k-means starts from a random guess: the default grouping of the R/L movements was identical in "
                    f"**{sc}/{S['n_runs']}** random starts.")


evz.cluster_page(
    HERE, "k-means",
    "**k-means:** splits the events into *k* groups so that each event is as close as possible to the centre of its "
    "group. It needs the number of groups in advance (default k = 2: right and left).",
    grid_cols="setting", grid_label="k = 2, 3, 4", extra_analysis=stability)
