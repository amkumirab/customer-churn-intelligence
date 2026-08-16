"""Interactive Streamlit dashboard for customer churn prediction."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.batch import (  # noqa: E402
    BatchValidationError,
    batch_summary,
    predict_batch,
)
from src.data import FEATURE_COLUMNS, KAGGLE_FILENAME, load_training_data  # noqa: E402
from src.explain import explain_customer  # noqa: E402
from src.model import (  # noqa: E402
    ARTIFACT_VERSION,
    load_artifact,
    predict_customer,
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
high_risk_threshold = st.sidebar.slider(
    "High-risk threshold",
    min_value=0.20,
    max_value=0.90,
    value=0.70,
    step=0.05,
)
medium_risk_threshold = round(max(0.10, high_risk_threshold - 0.30), 2)
selected_scenario = threshold_scenario(
    report["threshold_counts"],
    high_risk_threshold,
    intervention_cost,
    missed_churn_cost,
)
st.sidebar.metric("Recommended threshold", f"{recommendation['threshold']:.0%}")
st.sidebar.metric(
    "Estimated evaluation cost",
    f"${selected_scenario['estimated_cost']:,.0f}",
    delta=(
        f"${selected_scenario['estimated_cost'] - recommendation['estimated_cost']:,.0f} "
        "vs. recommended"
    ),
    delta_color="inverse",
)
st.sidebar.caption(
    f"Selected policy: {selected_scenario['precision']:.1%} precision, "
    f"{selected_scenario['recall']:.1%} recall, "
    f"medium risk starts at {medium_risk_threshold:.0%}."
)

metric_columns = st.columns(4)
metric_columns[0].metric("Best model", report["best_model"].replace("_", " ").title())
metric_columns[1].metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
metric_columns[2].metric("Recall", f"{best_metrics['recall']:.3f}")
metric_columns[3].metric("Dataset churn rate", f"{report['churn_rate']:.1%}")

st.divider()
left, right = st.columns([1.1, 0.9])
with left:
    st.subheader("Customer profile")
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        tenure = c1.slider("Tenure (months)", 0, 72, 12)
        monthly = c2.number_input("Monthly charges ($)", 18.0, 125.0, 79.5, step=1.0)
        total = c3.number_input(
            "Total charges ($)", 0.0, 10_000.0, float(round(monthly * tenure, 2)), step=25.0
        )
        contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = c2.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment = c3.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
                "Mailed check",
            ],
        )
        paperless = c1.selectbox("Paperless billing", ["Yes", "No"])
        senior = c2.selectbox("Senior citizen", ["No", "Yes"])
        gender = c3.selectbox("Gender", ["Female", "Male"])
        partner = c1.selectbox("Has a partner", ["No", "Yes"])
        dependents = c2.selectbox("Has dependents", ["No", "Yes"])
        phone = c3.selectbox("Phone service", ["Yes", "No"])

        with st.expander("Service subscriptions"):
            s1, s2, s3 = st.columns(3)
            multiple_lines = s1.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            online_security = s2.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            online_backup = s3.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = s1.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
            tech_support = s2.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = s3.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = s1.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )
        submitted = st.form_submit_button("Calculate churn risk", width="stretch")

with right:
    st.subheader("Risk assessment")
    if submitted:
        customer = {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "gender": gender,
            "senior_citizen": senior,
            "has_partner": partner,
            "has_dependents": dependents,
            "phone_service": phone,
            "multiple_lines": multiple_lines,
            "internet_service": internet,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract_type": contract,
            "paperless_billing": paperless,
            "payment_method": payment,
        }
        result = predict_customer(
            artifact["pipeline"],
            customer,
            high_risk_threshold=high_risk_threshold,
            medium_risk_threshold=medium_risk_threshold,
        )
        probability = result["churn_probability"]
        st.metric("Predicted churn probability", f"{probability:.1%}")
        st.progress(probability)
        if result["risk_level"] == "High":
            st.error(f"High risk — {result['action']}")
        elif result["risk_level"] == "Medium":
            st.warning(f"Medium risk — {result['action']}")
        else:
            st.success(f"Low risk — {result['action']}")

        local_explanation = explain_customer(
            artifact["pipeline"],
            customer,
            artifact["explanation_background"],
        )
        st.markdown("**Why this prediction?**")
        st.caption(
            "Positive SHAP values increase predicted churn risk; negative values reduce it."
        )
        st.bar_chart(local_explanation.set_index("feature")[["impact"]])
        st.dataframe(
            local_explanation,
            hide_index=True,
            width="stretch",
            column_config={
                "feature": "Factor",
                "value": "Customer value",
                "impact": st.column_config.NumberColumn("SHAP impact", format="%.4f"),
                "direction": "Effect",
            },
        )

        st.markdown("**Submitted profile**")
        st.dataframe(pd.DataFrame([customer]), hide_index=True, width="stretch")
    else:
        st.info("Complete the profile and calculate its risk to see a prediction.")

st.divider()
st.subheader("What generally drives churn?")
st.caption(
    "Mean absolute SHAP values rank the features with the strongest overall influence."
)
global_importance = pd.DataFrame(report["global_feature_importance"])
st.bar_chart(global_importance.set_index("feature")[["importance"]])

st.divider()
st.subheader("Batch churn prediction")
st.caption(
    "Upload up to 10,000 customers, review the highest-risk accounts, and export the results."
)

template = pd.read_csv(ROOT / "data" / "example_customers.csv")[FEATURE_COLUMNS]
upload_column, template_column = st.columns([3, 1])
with upload_column:
    uploaded_file = st.file_uploader("Upload a customer CSV", type=["csv"])
with template_column:
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="customer_batch_template.csv",
        mime="text/csv",
        width="stretch",
    )

if uploaded_file is None:
    st.info(
        "Use the template for project-style columns. "
        "Original Kaggle Telco columns are also accepted."
    )
else:
    try:
        uploaded_data = pd.read_csv(uploaded_file)
        batch_results = predict_batch(
            artifact["pipeline"],
            uploaded_data,
            high_risk_threshold=high_risk_threshold,
            medium_risk_threshold=medium_risk_threshold,
        )
        summary = batch_summary(batch_results)

        summary_columns = st.columns(4)
        summary_columns[0].metric("Customers scored", f"{summary['customers']:,}")
        summary_columns[1].metric("High risk", f"{summary['high_risk']:,}")
        summary_columns[2].metric("Medium risk", f"{summary['medium_risk']:,}")
        summary_columns[3].metric(
            "Average churn probability", f"{summary['average_probability']:.1%}"
        )

        st.dataframe(
            batch_results,
            hide_index=True,
            width="stretch",
            column_config={
                "customer_id": "Customer ID",
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn probability", min_value=0.0, max_value=1.0, format="%.1%%"
                ),
                "risk_level": "Risk level",
                "recommended_action": "Recommended action",
            },
        )
        st.download_button(
            "Download scored customers",
            batch_results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
            width="stretch",
        )
    except (BatchValidationError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"The CSV could not be processed: {error}")

st.divider()
st.caption(
    "Portfolio demonstration using the public IBM Telco dataset from Kaggle. "
    "Predictions are not suitable for real business decisions."
)
