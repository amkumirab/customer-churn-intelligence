"""Display how an uploaded batch differs from its model's training data."""

import pandas as pd
import streamlit as st

from src.drift import compare_drift

_LABELS = {
    "tenure_months": "Tenure (months)",
    "monthly_charges": "Monthly charges ($)",
    "contract_type": "Contract type",
}
_SOURCES = {
    "synthetic": "synthetic training data",
    "kaggle_telco": "IBM Telco training data",
    "custom_csv": "custom CSV training data",
}


def render_drift_report(cohort: pd.DataFrame, report: dict) -> None:
    with st.expander("Batch data drift"):
        reference = report.get("drift_reference")
        if reference is None:
            st.info(
                "This saved model has no training reference. Run python -m src.train "
                "and restart the app to enable drift diagnostics."
            )
            return

        source = _SOURCES.get(report.get("training_data_source"), "training data (source unknown)")
        st.caption(
            f"Reference: {source} ({reference['rows']:,} rows). "
            f"Uploaded batch: {len(cohort):,} rows before campaign filters."
        )
        st.caption(
            "Distance ranges from 0 (same bucket shares) to 1 (no shared distribution). "
            "Small batches can vary substantially. This report does not measure model accuracy."
        )
        summary, details = compare_drift(reference, cohort)
        st.dataframe(
            summary.assign(feature=summary["feature"].map(_LABELS)),
            hide_index=True,
            column_config={
                "feature": "Feature",
                "distance": st.column_config.NumberColumn("Distribution distance", format="%.3f"),
            },
        )
        field = st.selectbox(
            "Inspect feature distribution", list(_LABELS), format_func=_LABELS.__getitem__,
            key="drift_feature",
        )
        st.dataframe(
            details.loc[details["feature"].eq(field)].drop(columns="feature"),
            hide_index=True,
            column_config={
                "bucket": "Bucket",
                "reference_count": "Training rows",
                "batch_count": "Batch rows",
                "reference_share": st.column_config.NumberColumn(
                    "Training share", format="percent"
                ),
                "batch_share": st.column_config.NumberColumn("Batch share", format="percent"),
                "change_pp": st.column_config.NumberColumn("Change (pp)", format="%+.1f"),
            },
        )
