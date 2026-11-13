"""Transparent model comparison and validation-policy review."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.calibration_view import render_calibration_comparison
from src.threshold import cost_analysis, threshold_scenario


def render_model_evaluation(
    report: dict, threshold: float, intervention_cost: float, missed_churn_cost: float
) -> None:
    st.subheader("Model comparison")
    st.caption(
        f"Training: {report['train_rows']:,} rows. Validation: {report['validation_rows']:,}. "
        f"Final test: {report['test_rows']:,}. Selected model: "
        f"{report['best_model'].replace('_', ' ').title()}. Artifact schema: "
        f"v{report['artifact_version']}."
    )
    rows = [
        {"model": name, "split": split, "selected": name == report["best_model"], **metrics}
        for split, models in (
            ("Validation", report["validation_models"]), ("Final test", report["models"])
        )
        for name, metrics in models.items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info(
        "Validation ROC-AUC selects the model. Final-test metrics above use a fixed 50% cutoff; "
        "the confusion matrix below follows the active sidebar policy. Final-test labels are "
        "never used to choose the model or recommend a threshold."
    )
    st.subheader("Final-test confusion matrix")
    scenario = threshold_scenario(
        report["test_threshold_counts"], threshold, intervention_cost, missed_churn_cost
    )
    columns = st.columns(2)
    columns[0].metric("Test precision (selected policy)", f"{scenario['precision']:.3f}")
    columns[1].metric("Test recall at active threshold", f"{scenario['recall']:.3f}")
    confusion = pd.DataFrame(
        {
            "Predicted stay": [scenario["true_negative"], scenario["false_negative"]],
            "Predicted churn": [scenario["false_positive"], scenario["true_positive"]],
        }, index=["Actual stay", "Actual churn"],
    ).astype(int)
    st.table(confusion)
    st.caption(
        f"Active threshold: {threshold:.0%}. Counts are from the independent final test split."
    )

    st.subheader("Validation threshold trade-offs")
    validation = cost_analysis(
        report["threshold_counts"], intervention_cost, missed_churn_cost
    ).sort_values("threshold")
    st.line_chart(validation.set_index("threshold")[["precision", "recall"]])
    st.line_chart(validation.set_index("threshold")[["estimated_cost"]])
    with st.expander("Validation policy counts and costs"):
        st.dataframe(validation, hide_index=True, width="stretch")
    st.caption(
        "Cost = contacted customers x intervention cost + missed churners x missed-churn cost. "
        "This classification-policy comparison is separate from campaign revenue scenarios."
    )
    render_calibration_comparison(report)
    st.subheader("What generally drives churn?")
    st.caption("Mean absolute SHAP values on a training-only sample; associations are not causes.")
    importance = pd.DataFrame(report["global_feature_importance"])
    st.bar_chart(importance.set_index("feature")[["importance"]], horizontal=True)
    with st.expander("Feature importance values"):
        st.dataframe(importance, hide_index=True, width="stretch")
    st.warning(
        "Model scores are not calibrated probabilities. The fictional Telco sample does not "
        "establish future performance, retention impact, or fairness across customer groups. "
        "Do not use this demonstration to make real customer decisions."
    )
