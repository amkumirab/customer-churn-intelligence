import pandas as pd
import pytest

from src.batch import BatchValidationError, batch_summary, predict_batch, validate_batch
from src.data import FEATURE_COLUMNS, generate_customers
from src.model import train_and_evaluate


@pytest.fixture(scope="module")
def trained_pipeline():
    data = generate_customers(n_rows=500, random_state=23)
    pipeline, _ = train_and_evaluate(data, random_state=23)
    return pipeline


def test_batch_predictions_are_ranked_and_summarized(trained_pipeline) -> None:
    customers = generate_customers(n_rows=100, random_state=29).head(12)
    uploaded = customers[FEATURE_COLUMNS].copy()
    uploaded.insert(0, "customer_id", [f"C-{index:03d}" for index in range(12)])

    results = predict_batch(trained_pipeline, uploaded)
    summary = batch_summary(results)

    assert len(results) == 12
    assert results["churn_probability"].is_monotonic_decreasing
    assert results["customer_id"].str.startswith("C-").all()
    assert results["risk_level"].isin(["Low", "Medium", "High"]).all()
    assert summary["customers"] == 12
    assert summary["high_risk"] + summary["medium_risk"] <= 12

    lower_threshold = predict_batch(
        trained_pipeline,
        uploaded,
        high_risk_threshold=0.30,
        medium_risk_threshold=0.10,
    )
    assert (lower_threshold["risk_level"] == "High").sum() >= summary["high_risk"]


def test_original_kaggle_columns_are_accepted() -> None:
    raw = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", nrows=3)
    standardized, identifiers = validate_batch(raw)

    assert list(standardized.columns) == FEATURE_COLUMNS
    assert identifiers.tolist() == raw["customerID"].tolist()
    assert standardized["senior_citizen"].isin(["Yes", "No"]).all()


def test_missing_columns_return_a_readable_error() -> None:
    with pytest.raises(BatchValidationError, match="Missing required columns"):
        validate_batch(pd.DataFrame({"tenure_months": [12]}))


def test_invalid_numeric_values_are_rejected() -> None:
    uploaded = generate_customers(n_rows=100, random_state=31)[FEATURE_COLUMNS].head(2)
    uploaded = uploaded.astype({"monthly_charges": "object"})
    uploaded.loc[uploaded.index[0], "monthly_charges"] = "not-a-number"

    with pytest.raises(BatchValidationError, match="non-numeric"):
        validate_batch(uploaded)
