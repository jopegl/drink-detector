"""
Entry point of the dashboard: a home page plus one page per experiment.

Usage:
    pip install -r requirements.txt
    streamlit run app.py

Every folder experiments/experiment_XX_<title>/ that has an experiment.json and a dashboard.py
appears automatically in the menu, ordered by its number.
"""
import streamlit as st

from core.experiments import ROOT, list_experiments

st.set_page_config(page_title="Wrist sensor experiments", layout="wide")

home = st.Page("home.py", title="Overview", default=True)
pages = [st.Page(str(e["folder"].relative_to(ROOT) / "dashboard.py"),
                 title=f"{e['code']} · {e['title']}",
                 url_path=f"experiment-{e['code']}")
         for e in list_experiments()]

st.navigation({"Home": [home], "Experiments": pages}).run()
