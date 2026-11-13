"""Read-only probability-quality comparison for the selected churn model."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st


def reliability_chart_spec() -> dict:
    """Use observed points, not interpolated gaps, with common [0, 1] axes."""
    x_axis = {
        "field": "mean_probability", "type": "quantitative",
        "title": "Mean predicted probability", "scale": {"domain": [0, 1]},
    }
    y_axis = {
        "field": "observed_rate", "type": "quantitative",
        "title": "Observed churn rate", "scale": {"domain": [0, 1]},
    }
    return {
        "description": (
            "Reliability diagram comparing original and sigmoid-calibrated scores. "
            "The dashed diagonal indicates perfect calibration. Each point is a populated bin; "
            "empty bins are omitted. Exact values and counts are in the adjacent data table."
        ),
        "layer": [
            {
                "data": {"values": [
                    {"mean_probability": 0, "observed_rate": 0},
                    {"mean_probability": 1, "observed_rate": 1},
                ]},
                "mark": {"type": "line", "strokeDash": [6, 4]},
                "encoding": {"x": x_axis, "y": y_axis},
            },
            {
                "mark": {"type": "point", "size": 90},
                "encoding": {
                    "x": x_axis, "y": y_axis,
                    "color": {
                        "field": "variant", "type": "nominal", "title": "Variant",
                        "legend": {"orient": "bottom", "columns": 1},
                    },
                    "shape": {
                        "field": "variant", "type": "nominal", "title": "Variant",
                        "legend": {"orient": "bottom", "columns": 1},
                    },
                    "tooltip": [
                        {"field": "variant", "type": "nominal"},
                        {"field": "count", "type": "quantitative"},
                        {"field": "mean_probability", "type": "quantitative", "format": ".3f"},
                        {"field": "observed_rate", "type": "quantitative", "format": ".3f"},
                    ],
                },
            },
        ],
    }


def render_calibration_comparison(report: dict) -> None:
    st.subheader("Probability quality and calibration")
    calibration = report.get("calibration")
    if calibration is None:
        st.info("Retrain the local model to generate probability-quality diagnostics.")
        return
    st.info(
        "Diagnostic comparison only: the sigmoid candidate does not replace the active original "
        "model, policy thresholds, campaign scores, or SHAP explanations."
    )
    split_label = st.selectbox(
        "Calibration evaluation split", ["Validation", "Final test"], key="calibration_split",
    )
    split = "validation" if split_label == "Validation" else "test"
    labels = {"original": "Original (active)", "sigmoid": "Sigmoid (comparison only)"}
    qualities = calibration[split]
    comparison = pd.DataFrame([
        {"variant": labels[name], **{key: value for key, value in quality.items() if key != "bins"}}
        for name, quality in qualities.items()
    ])
    comparison = comparison[["variant", "brier_score", "log_loss", "samples", "positive_rate"]]
    st.dataframe(
        comparison, hide_index=True, width="stretch",
        column_config={
            "brier_score": st.column_config.NumberColumn("Brier score", format="%.5f"),
            "log_loss": st.column_config.NumberColumn("Log loss", format="%.5f"),
            "positive_rate": st.column_config.NumberColumn("Observed prevalence", format="%.4f"),
        },
    )
    st.caption(
        f"{split_label} comparison. Candidate fitting: {calibration['folds']}-fold, training-only "
        f"out-of-fold sigmoid calibration on {calibration['training_rows']:,} rows. "
        "Validation and final-test labels never fit the calibrator."
    )
    bins = pd.DataFrame([
        {"variant": labels[name], **bin_row}
        for name, quality in qualities.items() for bin_row in quality["bins"]
    ])
    st.vega_lite_chart(
        bins.loc[bins["count"] > 0], reliability_chart_spec(), width="stretch",
    )
    st.caption(
        "The dashed diagonal is perfect calibration; markers show populated bins only. "
        "Ten equal-width bins use [lower, upper), with 1 included in the last bin. "
        "Small bins are noisy; no uncertainty intervals or causal effects are estimated."
    )
    with st.expander("Reliability bin values and sample counts"):
        st.dataframe(bins, hide_index=True, width="stretch")
    st.warning(
        "Lower Brier score and log loss mean better overall probability quality, not necessarily "
        "better calibration alone. Compare the diagram and sample counts too. Calibration on "
        "this fictional dataset does not establish future accuracy or retention impact."
    )
    st.download_button(
        "Download probability-quality report",
        json.dumps(calibration, indent=2, allow_nan=False).encode("utf-8"),
        file_name="calibration_report.json", mime="application/json",
        key="download_calibration_report",
    )
