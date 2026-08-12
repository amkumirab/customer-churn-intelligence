"""Validation and batch scoring for uploaded customer CSV files."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import FEATURE_COLUMNS, KAGGLE_COLUMNS, TARGET
from .model import risk_details

MAX_BATCH_ROWS = 10_000
KAGGLE_FEATURE_COLUMNS = {
    source: destination
    for source, destination in KAGGLE_COLUMNS.items()
    if destination != TARGET
}


class BatchValidationError(ValueError):
    """Raised when an uploaded batch cannot be safely scored."""


def _standardize_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if set(FEATURE_COLUMNS).issubset(data.columns):
        identifier = (
            data["customer_id"].astype(str)
            if "customer_id" in data.columns
            else pd.Series(data.index + 1, index=data.index).astype(str)
        )
        return data[FEATURE_COLUMNS].copy(), identifier

    missing_kaggle = set(KAGGLE_FEATURE_COLUMNS) - set(data.columns)
    if not missing_kaggle:
        identifier = (
            data["customerID"].astype(str)
            if "customerID" in data.columns
            else pd.Series(data.index + 1, index=data.index).astype(str)
        )
        standardized = data[list(KAGGLE_FEATURE_COLUMNS)].rename(
            columns=KAGGLE_FEATURE_COLUMNS
        )
        standardized["senior_citizen"] = standardized["senior_citizen"].replace(
            {1: "Yes", 0: "No", "1": "Yes", "0": "No"}
        )
        return standardized[FEATURE_COLUMNS], identifier

    missing_standard = set(FEATURE_COLUMNS) - set(data.columns)
    missing = sorted(missing_standard, key=FEATURE_COLUMNS.index)
    preview = ", ".join(missing[:6])
    suffix = " ..." if len(missing) > 6 else ""
    raise BatchValidationError(f"Missing required columns: {preview}{suffix}")


def validate_batch(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and standardize an uploaded batch without mutating the input."""
    if data.empty:
        raise BatchValidationError("The uploaded CSV does not contain any customer rows.")
    if len(data) > MAX_BATCH_ROWS:
        raise BatchValidationError(
            f"The upload contains {len(data):,} rows; the limit is {MAX_BATCH_ROWS:,}."
        )

    standardized, identifier = _standardize_columns(data)
    for column in ["tenure_months", "monthly_charges", "total_charges"]:
        original = standardized[column]
        standardized[column] = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & original.astype(str).str.strip().ne("") & standardized[
            column
        ].isna()
        if invalid.any():
            raise BatchValidationError(
                f"Column '{column}' contains {int(invalid.sum())} non-numeric value(s)."
            )

    if standardized[["tenure_months", "monthly_charges"]].isna().any().any():
        raise BatchValidationError("Tenure and monthly charges cannot be blank.")
    if (standardized["tenure_months"] < 0).any():
        raise BatchValidationError("Tenure cannot be negative.")
    if (standardized["monthly_charges"] < 0).any():
        raise BatchValidationError("Monthly charges cannot be negative.")

    empty_categories = standardized.drop(
        columns=["tenure_months", "monthly_charges", "total_charges"]
    ).isna()
    if empty_categories.any().any():
        columns = ", ".join(empty_categories.columns[empty_categories.any()].tolist())
        raise BatchValidationError(f"Categorical values cannot be blank: {columns}")

    return standardized, identifier


def predict_batch(pipeline: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Score a validated customer batch and return a risk-ranked result table."""
    customers, identifiers = validate_batch(data)
    probabilities = pipeline.predict_proba(customers)[:, 1]
    details = [risk_details(float(probability)) for probability in probabilities]

    result = pd.DataFrame(
        {
            "customer_id": identifiers.values,
            "churn_probability": np.round(probabilities, 4),
            "risk_level": [detail[0] for detail in details],
            "recommended_action": [detail[1] for detail in details],
        }
    )
    return result.sort_values(
        "churn_probability", ascending=False, ignore_index=True
    )


def batch_summary(results: pd.DataFrame) -> dict[str, Any]:
    """Return compact dashboard metrics for scored batch results."""
    return {
        "customers": len(results),
        "high_risk": int((results["risk_level"] == "High").sum()),
        "medium_risk": int((results["risk_level"] == "Medium").sum()),
        "average_probability": float(results["churn_probability"].mean()),
    }
