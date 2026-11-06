"""Validation and batch scoring for uploaded customer CSV files."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .data import FEATURE_COLUMNS, KAGGLE_COLUMNS, TARGET
from .model import risk_details

MAX_BATCH_ROWS = 10_000
CATEGORY_VALUES = {
    "gender": {"Female", "Male"},
    "senior_citizen": {"Yes", "No"},
    "has_partner": {"Yes", "No"},
    "has_dependents": {"Yes", "No"},
    "phone_service": {"Yes", "No"},
    "multiple_lines": {"Yes", "No", "No phone service"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "contract_type": {"Month-to-month", "One year", "Two year"},
    "paperless_billing": {"Yes", "No"},
    "payment_method": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
    **{
        feature: {"Yes", "No", "No internet service"}
        for feature in (
            "online_security", "online_backup", "device_protection",
            "tech_support", "streaming_tv", "streaming_movies",
        )
    },
}
KAGGLE_FEATURE_COLUMNS = {
    source: destination
    for source, destination in KAGGLE_COLUMNS.items()
    if destination != TARGET
}


class BatchValidationError(ValueError):
    """Raised when an uploaded batch cannot be safely scored."""


def read_customer_csv(source: Any) -> pd.DataFrame:
    """Read a bounded UTF-8 CSV while preserving literal customer identifiers."""
    try:
        data = pd.read_csv(
            source,
            dtype={"customer_id": "string", "customerID": "string"},
            keep_default_na=False,
            nrows=MAX_BATCH_ROWS + 1,
        )
    except pd.errors.EmptyDataError as error:
        raise BatchValidationError("The uploaded CSV is empty.") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise BatchValidationError("Upload a valid UTF-8 CSV with a consistent header.") from error
    if data.empty:
        raise BatchValidationError("The uploaded CSV is empty.")
    if len(data) > MAX_BATCH_ROWS:
        raise BatchValidationError(f"The CSV exceeds the {MAX_BATCH_ROWS:,}-customer limit.")
    return data


def export_predictions_csv(results: pd.DataFrame) -> bytes:
    """Encode results for spreadsheets without allowing formula-like text cells."""
    safe = results.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].map(
            lambda value: "'" + value
            if isinstance(value, str)
            and (value.lstrip().startswith(("=", "+", "-", "@"))
                 or value.startswith(("\t", "\r", "\n")))
            else value
        )
    return safe.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _standardize_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if set(FEATURE_COLUMNS).issubset(data.columns):
        identifier = (
            data["customer_id"].astype("string")
            if "customer_id" in data.columns
            else pd.Series(np.arange(1, len(data) + 1), index=data.index).astype(str)
        )
        return data[FEATURE_COLUMNS].copy(), identifier

    missing_kaggle = set(KAGGLE_FEATURE_COLUMNS) - set(data.columns)
    if not missing_kaggle:
        identifier = (
            data["customerID"].astype("string")
            if "customerID" in data.columns
            else pd.Series(np.arange(1, len(data) + 1), index=data.index).astype(str)
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
    if data.columns.has_duplicates:
        raise BatchValidationError("CSV column names must be unique.")

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
        if not np.isfinite(standardized[column].dropna()).all():
            raise BatchValidationError(f"Column '{column}' must contain finite numbers.")
        if (standardized[column] < 0).any():
            raise BatchValidationError(f"Column '{column}' cannot contain negative values.")

    if standardized[["tenure_months", "monthly_charges"]].isna().any().any():
        raise BatchValidationError("Tenure and monthly charges cannot be blank.")
    for column, allowed in CATEGORY_VALUES.items():
        standardized[column] = standardized[column].astype("string").str.strip()
        if not standardized[column].isin(allowed).all():
            raise BatchValidationError(f"Column '{column}' contains blank or unsupported values.")
    if identifier.isna().any() or identifier.astype(str).str.strip().eq("").any():
        raise BatchValidationError("Customer identifiers cannot be blank.")

    return standardized, identifier


def predict_batch(
    pipeline: Pipeline,
    data: pd.DataFrame,
    high_risk_threshold: float = 0.70,
    medium_risk_threshold: float = 0.40,
) -> pd.DataFrame:
    """Score a validated batch, preserving decision precision in the ranked result table."""
    customers, identifiers = validate_batch(data)
    probabilities = pipeline.predict_proba(customers)[:, 1]
    details = [
        risk_details(
            float(probability),
            high_risk_threshold=high_risk_threshold,
            medium_risk_threshold=medium_risk_threshold,
        )
        for probability in probabilities
    ]

    result = pd.DataFrame(
        {
            "customer_id": identifiers.values,
            # Keep threshold decisions and downstream campaign eligibility consistent.
            "churn_probability": probabilities,
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
