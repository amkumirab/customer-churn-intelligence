"""Identity-safe cohort scoring and operational segment summaries."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.pipeline import Pipeline

from .batch import BatchValidationError, predict_batch, validate_batch

SEGMENT_FIELDS = ("contract_type", "internet_service", "payment_method", "tenure_band")
COHORT_FEATURES = [
    "tenure_months", "monthly_charges", "contract_type", "internet_service", "payment_method",
]


def score_cohort(
    pipeline: Pipeline,
    data: pd.DataFrame,
    high_risk_threshold: float = 0.70,
    medium_risk_threshold: float = 0.40,
) -> pd.DataFrame:
    """Score a cohort and join operational features by unique literal ID, not row order.

    Unlike the general batch scorer, campaign preparation rejects duplicate IDs:
    repeated identities could result in repeated outreach. Demographic features and
    uploaded labels are used neither for segmentation nor exported cohort profiles.
    """
    features, identifiers = validate_batch(data)
    if identifiers.duplicated().any():
        raise BatchValidationError("Campaign customer identifiers must be unique.")
    scores = predict_batch(pipeline, data, high_risk_threshold, medium_risk_threshold)
    operational = features[COHORT_FEATURES].reset_index(drop=True)
    operational.insert(0, "customer_id", identifiers.to_numpy())
    operational["imputed_total_charges"] = features["total_charges"].isna().to_numpy()
    cohort = scores.merge(operational, on="customer_id", validate="one_to_one")
    cohort["tenure_band"] = pd.cut(
        cohort["tenure_months"], bins=[-1, 12, 24, float("inf")],
        labels=["0-12 months", "13-24 months", "25+ months"],
    ).astype(str)
    cohort["monthly_revenue_exposure"] = cohort["churn_probability"] * cohort["monthly_charges"]
    return cohort.sort_values(
        ["churn_probability", "customer_id"], ascending=[False, True], ignore_index=True,
    )


def filter_cohort(
    cohort: pd.DataFrame, risk_levels: Sequence[str], contracts: Sequence[str]
) -> pd.DataFrame:
    """Intersect explicit selections; an empty selection means no matching customers."""
    return cohort.loc[
        cohort["risk_level"].isin(risk_levels) & cohort["contract_type"].isin(contracts)
    ].copy().reset_index(drop=True)


def segment_summary(cohort: pd.DataFrame, field: str) -> pd.DataFrame:
    """Summarize model scores, not observed churn, for an allowlisted operational field."""
    if field not in SEGMENT_FIELDS:
        raise ValueError(f"Unsupported segment field: {field}")
    work = cohort.assign(high_risk=cohort["risk_level"].eq("High").astype(int))
    summary = work.groupby(field, observed=True).agg(
        customers=("customer_id", "size"),
        average_probability=("churn_probability", "mean"),
        high_risk_customers=("high_risk", "sum"),
        monthly_revenue_exposure=("monthly_revenue_exposure", "sum"),
    ).reset_index().rename(columns={field: "segment"})
    summary["high_risk_rate"] = summary["high_risk_customers"] / summary["customers"]
    return summary.sort_values(
        ["average_probability", "segment"], ascending=[False, True], ignore_index=True,
    )
