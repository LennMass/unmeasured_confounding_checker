"""
Streamlit dashboard for unmeasured_confounding_checker.

Run with:  streamlit run dashboard/app.py
Talks to the FastAPI backend (default http://localhost:8000).
"""

import os

import requests
import streamlit as st
import plotly.graph_objects as go

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Sensitivity to Unmeasured Confounding", layout="wide")
st.title("Sensitivity to Unmeasured Confounding")
st.caption("Robustness Analysis as a Service for ATE estimation")

# --- Sidebar inputs ---
with st.sidebar:
    st.header("Configuration")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    treatment = st.text_input("Treatment column", "treatment")
    outcome = st.text_input("Outcome column", "outcome")
    estimator = st.selectbox("Estimator", ["plr", "irm"])
    n_placebo = st.slider("Placebo permutations", 10, 500, 100, step=10)
    confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, step=0.01)
    run = st.button("Run Analysis", type="primary")


def call_api(file, params):
    files = {"file": (file.name, file.getvalue(), "text/csv")}
    resp = requests.post(f"{API_URL}/sensitivity", files=files, params=params, timeout=300)
    resp.raise_for_status()
    return resp.json()


if run and uploaded:
    with st.spinner("Running sensitivity analysis..."):
        try:
            params = {
                "treatment_col": treatment,
                "outcome_col": outcome,
                "estimator": estimator,
                "n_placebo": n_placebo,
                "confidence_level": confidence,
            }
            report = call_api(uploaded, params)
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    est = report["estimation"]

    # --- Top metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("ATE", f"{est['coefficient']:.3f}")
    col2.metric("p-value", f"{est['p_value']:.4f}")
    col3.metric("Significant", "Yes" if est["significant"] else "No")

    st.divider()

    # --- E-value ---
    st.subheader("E-Value (Unmeasured Confounding)")
    ev = report["evalue"]
    c1, c2 = st.columns(2)
    c1.metric("E-value (point)", f"{ev['evalue_point']:.2f}")
    c2.metric("E-value (CI)", f"{ev['evalue_ci']:.2f}")
    st.info(ev["interpretation"])

    st.divider()

    # --- Learner robustness ---
    st.subheader("Learner Robustness")
    learners = report["learner_robustness"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[l["learner"] for l in learners],
        y=[l["ate"] for l in learners],
        error_y=dict(
            type="data",
            array=[l["ci_upper"] - l["ate"] for l in learners],
            arrayminus=[l["ate"] - l["ci_lower"] for l in learners],
        ),
        marker_color="steelblue",
    ))
    fig.update_layout(yaxis_title="ATE", height=350)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Placebo test ---
    st.subheader("Placebo Test")
    pl = report["placebo"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Real ATE", f"{pl['real_ate']:.3f}")
    c2.metric("Placebo mean", f"{pl['placebo_mean']:.3f}")
    c3.metric("Placebo p-value", f"{pl['placebo_p_value']:.3f}")
    st.info(pl["interpretation"])

elif run and not uploaded:
    st.warning("Please upload a CSV file first.")
else:
    st.info("Configure your analysis in the sidebar and click Run.")