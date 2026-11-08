"""Home page: one card per experiment."""
import streamlit as st

from core.experiments import ROOT, list_experiments

st.title("Wrist sensor experiments")
st.markdown(
    "Andrew wears a wrist sensor (3 translation values + roll, pitch, yaw) and notes what he does. "
    "Each experiment tries to recognise his movements **from the sensor data alone**, using unsupervised "
    "k-means. His notes are only used at the end, to check the result.")

for e in list_experiments():
    with st.container(border=True):
        st.subheader(f"Experiment {e['code']} · {e['title']}")
        st.caption(f"{e['date']} · method: {e['method']}")
        st.markdown(e["summary"])
        st.markdown(f"**Result:** {e['result']}")
        st.page_link(str(e["folder"].relative_to(ROOT) / "dashboard.py"), label="Open dashboard →")
