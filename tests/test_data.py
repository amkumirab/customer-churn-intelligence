from pathlib import Path

import pandas as pd
import pytest

from src.data import FEATURE_COLUMNS, TARGET, generate_customers, load_kaggle_telco


def test_generated_data_has_expected_shape_and_columns() -> None:
    data = generate_customers(n_rows=250, random_state=7)
    assert data.shape == (250, len(FEATURE_COLUMNS) + 1)
    assert set(FEATURE_COLUMNS + [TARGET]) == set(data.columns)
    assert set(data[TARGET].unique()).issubset({0, 1})


def test_generation_is_reproducible() -> None:
    first = generate_customers(n_rows=100, random_state=11)
    second = generate_customers(n_rows=100, random_state=11)
    pd.testing.assert_frame_equal(first, second)


def test_too_few_rows_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        generate_customers(n_rows=99)


def test_kaggle_dataset_loads_and_is_standardized() -> None:
    path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data = load_kaggle_telco(path)

    assert data.shape == (7_043, len(FEATURE_COLUMNS) + 1)
    assert data[TARGET].isin([0, 1]).all()
    assert data["total_charges"].isna().sum() == 11
