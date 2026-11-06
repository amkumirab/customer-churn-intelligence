"""Interactive Streamlit dashboard for customer churn prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.campaign_view import render_campaign_workspace  # noqa: E402
from app.customer_view import render_customer_review  # noqa: E402
from app.evaluation_view import render_model_evaluation  # noqa: E402
from src.data import FEATURE_COLUMNS, KAGGLE_FILENAME, load_training_data  # noqa: E402
from src.model import (  # noqa: E402
    ARTIFACT_VERSION,
    load_artifact,
    train_and_evaluate,
)
from src.threshold import recommended_threshold, threshold_scenario  # noqa: E402

MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / KAGGLE_FILENAME


@st.cache_resource
def get_artifact() -> dict:
    artifact = load_artifact(MODEL_PATH) if MODEL_PATH.exists() else None
    saved_features = artifact.get("report", {}).get("feature_columns") if artifact else None
    saved_version = artifact.get("artifact_version") if artifact else None
    if saved_features != FEATURE_COLUMNS or saved_version != ARTIFACT_VERSION:
        data = load_training_data(DATA_PATH)
        train_and_evaluate(data, MODEL_PATH, METRICS_PATH)
        artifact = load_artifact(MODEL_PATH)
    return artifact


st.set_page_config(page_title="Churn Intelligence", page_icon="📉", layout="wide")
st.title("Customer Churn Intelligence")
st.caption("Risk scoring trained on the Kaggle IBM Telco Customer Churn dataset.")

artifact = get_artifact()
report = artifact["report"]
best_metrics = report["models"][report["best_model"]]

st.sidebar.header("Decision policy")
st.sidebar.caption(
    "Compare retention outreach cost with the cost of missing a customer who churns."
)
intervention_cost = st.sidebar.number_input(
    "Intervention cost per contacted customer ($)",
    min_value=0.0,
    value=25.0,
    step=5.0,
)
missed_churn_cost = st.sidebar.number_input(
    "Cost per missed churner ($)",
    min_value=0.0,
    value=300.0,
    step=25.0,
)
recommendation = recommended_threshold(
    report["threshold_counts"], intervention_cost, missed_churn_cost
)
if "high_risk_threshold" not in st.session_state:
    st.session_state["high_risk_threshold"] = 0.70
if st.sidebar.button("Use recommended threshold"):
    st.session_state["high_risk_threshold"] = float(recommendation["threshold"])
high_risk_threshold = st.sidebar.slider(
    "High-risk threshold",
    min_value=0.10,
    max_value=0.90,
    step=0.05,
    key="high_risk_threshold",
)
medium_risk_threshold = round(max(0.05, high_risk_threshold - 0.30), 2)
selected_scenario = threshold_scenario(
    report["threshold_counts"],
    high_risk_threshold,
    intervention_cost,
    missed_churn_cost,
)
st.sidebar.metric("Recommended threshold", f"{recommendation['threshold']:.0%}")
st.sidebar.metric(
    "Estimated validation cost",
    f"${selected_scenario['estimated_cost']:,.0f}",
    delta=(
        f"${selected_scenario['estimated_cost'] - recommendation['estimated_cost']:,.0f} "
        "vs. recommended"
    ),
    delta_color="inverse",
)
st.sidebar.caption(
    f"Validation policy: {selected_scenario['precision']:.1%} precision, "
    f"{selected_scenario['recall']:.1%} recall, "
    f"{int(selected_scenario['customers_contacted']):,} contacts. "
    f"medium risk starts at {medium_risk_threshold:.0%}."
)
test_scenario = threshold_scenario(
    report["test_threshold_counts"], high_risk_threshold, intervention_cost, missed_churn_cost
)

metric_columns = st.columns(4)
metric_columns[0].metric("Best model", report["best_model"].replace("_", " ").title())
metric_columns[1].metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
metric_columns[2].metric("Test recall (selected policy)", f"{test_scenario['recall']:.3f}")
metric_columns[3].metric("Dataset churn rate", f"{report['churn_rate']:.1%}")


customer_tab, campaign_tab, evaluation_tab = st.tabs(
    ["Customer review", "Campaign workspace", "Model evaluation"]
)
with customer_tab:
    render_customer_review(artifact, high_risk_threshold, medium_risk_threshold)
with campaign_tab:
    render_campaign_workspace(
        artifact, ROOT, high_risk_threshold, medium_risk_threshold, intervention_cost
    )
with evaluation_tab:
    render_model_evaluation(
        report, high_risk_threshold, intervention_cost, missed_churn_cost
    )

st.divider()
st.caption(
    "Portfolio demonstration using the public IBM Telco dataset from Kaggle. "
    "No outreach is executed. Predictions and scenario values are not suitable "
    "for real business decisions."
)
