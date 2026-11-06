from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from src.batch import BatchValidationError, validate_batch


@pytest.fixture
def customers():
    return pd.read_csv("data/example_customers.csv").drop(columns="churn")


@pytest.mark.parametrize("value", [np.inf, -np.inf, -100.0])
def test_rejects_non_finite_or_negative_total_charges(customers, value):
    customers["total_charges"] = value
    with pytest.raises(BatchValidationError):
        validate_batch(customers)


@pytest.mark.parametrize("value", ["invalid-contract", "   "])
def test_rejects_invalid_contracts(customers, value):
    customers["contract_type"] = value
    with pytest.raises(BatchValidationError, match="contract_type"):
        validate_batch(customers)


def test_missing_total_charges_are_still_imputed(customers):
    customers["total_charges"] = " "
    cleaned, _ = validate_batch(customers)
    assert cleaned["total_charges"].isna().all()


def test_csv_reader_preserves_identifiers_and_accepts_kaggle(customers):
    from src.batch import read_customer_csv

    customers["customer_id"] = ["000123", "NA", "003"]
    parsed = read_customer_csv(BytesIO(customers.to_csv(index=False).encode()))
    _, identifiers = validate_batch(parsed)
    assert identifiers.tolist() == ["000123", "NA", "003"]
    raw = read_customer_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    assert validate_batch(raw)[0].shape[0] == 7043


def test_empty_csv_returns_a_validation_error():
    from src.batch import read_customer_csv

    with pytest.raises(BatchValidationError, match="empty"):
        read_customer_csv(BytesIO(b""))


@pytest.mark.parametrize("identifier", [None, pd.NA, np.nan])
def test_missing_identifiers_are_rejected_before_string_conversion(customers, identifier):
    customers["customer_id"] = [identifier, "002", "003"]
    with pytest.raises(BatchValidationError, match="identifiers"):
        validate_batch(customers)


@pytest.mark.parametrize("identifier", ["=1+1", "+1+1", "-1+1", "@SUM(1)", "  =1+1"])
def test_export_neutralizes_formula_cells_without_mutating_results(identifier):
    from src.batch import export_predictions_csv

    results = pd.DataFrame({"customer_id": [identifier], "churn_probability": [0.75]})
    payload = export_predictions_csv(results)
    exported = pd.read_csv(BytesIO(payload), dtype={"customer_id": str})
    assert exported.customer_id.iloc[0] == "'" + identifier
    assert results.customer_id.iloc[0] == identifier
    assert exported.churn_probability.iloc[0] == 0.75


def test_export_uses_platform_independent_newlines_for_cohort_fingerprints():
    from src.batch import export_predictions_csv

    payload = export_predictions_csv(pd.DataFrame({"customer_id": ["001", "002"]}))
    assert payload == b"\xef\xbb\xbfcustomer_id\n001\n002\n"
