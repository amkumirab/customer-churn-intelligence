import json

import pandas as pd
import pytest

from src.drift import build_drift_reference, compare_drift


def test_identical_distributions_have_zero_distance():
    rows = pd.DataFrame({
        "tenure_months": [0, 12, 24, 72],
        "monthly_charges": [10, 50, 100, 125],
        "contract_type": ["Month-to-month", "One year", "Two year", "Two year"],
    })
    reference = build_drift_reference(rows)
    json.dumps(reference, allow_nan=False)
    summary, details = compare_drift(reference, pd.concat([rows, rows], ignore_index=True))

    assert reference["rows"] == 4
    assert summary["distance"].eq(0).all()
    assert details.groupby("feature")["batch_count"].sum().eq(8).all()
    assert "customer_id" not in json.dumps(reference)


def test_shifted_categories_and_charges_preserve_outliers_and_missing_values():
    reference = build_drift_reference(pd.DataFrame({
        "tenure_months": [10, 10],
        "monthly_charges": [20, 20],
        "contract_type": ["One year", "One year"],
    }))
    batch = pd.DataFrame({
        "tenure_months": [10, 10],
        "monthly_charges": [10000, 10000],
        "contract_type": ["Month-to-month", None],
    })
    summary, details = compare_drift(reference, batch)

    assert summary.set_index("feature")["distance"].to_dict() == {
        "tenure_months": 0.0, "monthly_charges": 1.0, "contract_type": 1.0,
    }
    contracts = details.loc[details["feature"].eq("contract_type")].set_index("bucket")
    assert contracts.loc["Missing", "batch_share"] == 0.5
    assert contracts.loc["One year", "change_pp"] == -100
    assert details.groupby("feature")["batch_share"].sum().eq(1).all()


def test_empty_batch_does_not_create_misleading_distances():
    rows = pd.DataFrame({
        "tenure_months": [12], "monthly_charges": [50], "contract_type": ["One year"],
    })
    with pytest.raises(ValueError, match="at least one"):
        compare_drift(build_drift_reference(rows), rows.iloc[:0])
