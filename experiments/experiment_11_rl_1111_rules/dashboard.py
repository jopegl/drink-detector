"""Dashboard page for experiment 11 (rules on the 11/11 recording)."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import events_viz as evz
from core import viz
from core.experiments import load_info

HERE = Path(__file__).resolve().parent
info = load_info(HERE)
d, E, S, R = evz.load_page(HERE, info)
rules, det = S["rules"], S["detection"]
rule = rules["rule"]
other = "L" if rule["side"] == "R" else "R"


def color_of(e):
    return evz.LABEL_COLORS.get(e.one_rule, "#bdbdbd")


viz.header(info)
st.sidebar.markdown(f"**Experiment {info['code']}**")
tab_dash, tab_analysis = st.tabs(["Dashboard", "Analysis"])

with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("One-threshold rule (leave-one-out)", f"{rules['loo_correct']}/{rules['n']}",
              help="For each movement, the rule is chosen without it, then applied to it.")
    c2.metric("Rule from experiment 02", f"{rules['exp02_correct']}/{rules['n']}",
              help="'Large yaw turn down = right', applied as is.")
    c3.metric("Annotated movements detected", f"{det['truth_found']}/{det['truth_total']}")
    c4.metric("Detected events with no annotation", det["events_without_truth"])
    st.info(f"**Rule found:** {rule['side']} if **{rule['feature']} ≥ {rule['threshold']:.1f}**, otherwise {other}. "
            f"In leave-one-out it picked the same feature in "
            f"{max(rules['loo_features'].values())} of {rules['n']} rounds.")

    st.subheader("1. Timeline (coloured by the one-threshold rule)")
    evz.timeline(d, E, color_of, [("rule says R", evz.LABEL_COLORS["R"], "band"),
                                  ("rule says L", evz.LABEL_COLORS["L"], "band"),
                                  ("truth R", evz.LABEL_COLORS["R"], "line"), ("truth L", evz.LABEL_COLORS["L"], "line")])

    st.subheader("2. The rule's feature for every event")
    x = E.copy()
    fig = go.Figure()
    for lab, c in evz.LABEL_COLORS.items():
        s = x[x.truth_label == lab]
        if len(s):
            fig.add_trace(go.Scatter(x=s.start, y=s[rule["feature"]], mode="markers", name=f"truth {lab}",
                                     marker=dict(size=12, color=c, line=dict(color="black", width=1))))
    fig.add_hline(y=rule["threshold"], line_dash="dash", line_color="black")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title=rule["feature"],
                      legend=dict(orientation="h", y=1.15))
    viz.show(fig)
    st.caption(f"Dashed line: threshold ({rule['threshold']:.1f}). Above it the rule says {rule['side']}, below it {other}.")
    evz.events_table(E, ["one_rule", "loo_pred", "exp02_rule", rule["feature"], "yaw_min", "yaw_max"])
    if S.get("others"):
        st.markdown("**Unlabeled movements and possible red herrings:** "
                    + "; ".join(f"{k} ({v['truth']}) → rule says {v['one_rule']}" for k, v in S["others"].items()))

    st.subheader("3. Inspect an event")
    evz.inspector(d, E, info, color_of,
                  lambda e: f"{e.start:%H:%M:%S}  ·  truth: {e.truth_label}  ·  rule: {e.one_rule}", f"e{info['code']}")

with tab_analysis:
    st.header("How it works")
    st.markdown(evz.PIPELINE_TEXT)
    st.markdown(
        "**Rules, the baseline:**\n"
        "1. **Detection by a state machine:** *rest* → *active* when the intensity goes above HIGH, back to *rest* "
        "below LOW.\n"
        "2. **Transfer test:** the rule from experiment 02 (*large yaw turn down = right*) applied as is.\n"
        "3. **One-threshold rule:** the single feature and threshold that best separate R from L. Because it is "
        "learned from the labels, it is scored with **leave-one-out**: each movement is classified by a rule "
        "chosen without it. Any other method should beat this score to be worth its complexity.")
    st.header("Does detection depend on the thresholds?")
    evz.heatmap(R.assign(ok=R.truth_found / R.truth_total), "high", "low", "ok", "text", "HIGH", "LOW")
    st.caption("Each cell: annotated movements found, extra events, and the rule's leave-one-out score.")
    st.header("Caveats")
    st.markdown(evz.COMMON_CAVEATS + "\n- **The one-threshold rule was learned from the labels.** Leave-one-out "
                "keeps the test honest, but with 19 movements one or two errors change the score a lot.")
