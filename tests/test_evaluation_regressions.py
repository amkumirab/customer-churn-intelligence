import numpy as np
import pytest

from src.data import generate_customers
from src.threshold import cost_analysis, threshold_counts


def test_training_validation_and_test_rows_are_disjoint():
    from src.model import split_training_data

    data = generate_customers(500)
    train, validation, test = split_training_data(data, random_state=42)
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)
    assert len(train) + len(validation) + len(test) == len(data)
    assert len(test) == 125


def test_holdout_labels_cannot_affect_selection_or_threshold_tuning(monkeypatch):
    from src import model

    data = generate_customers(500, random_state=17)
    splits = model.split_training_data(data, random_state=42)
    monkeypatch.setattr(model, "split_training_data", lambda *args, **kwargs: splits)
    _, first = model.train_and_evaluate(data)
    train, validation, test = splits
    changed_test = test.copy()
    changed_test["churn"] = 1 - changed_test["churn"]
    monkeypatch.setattr(
        model, "split_training_data", lambda *args, **kwargs: (train, validation, changed_test)
    )
    _, second = model.train_and_evaluate(data)
    assert first["best_model"] == second["best_model"]
    assert first["validation_models"] == second["validation_models"]
    assert first["threshold_counts"] == second["threshold_counts"]
    assert first["models"] != second["models"]
    assert first["threshold_data_source"] == "validation"
    assert sum(first["test_threshold_counts"][0][key] for key in (
        "true_positive", "false_positive", "true_negative", "false_negative"
    )) == first["test_rows"]


def test_non_binary_labels_are_not_silently_truncated():
    with pytest.raises(ValueError, match="Labels"):
        threshold_counts([0.5, 1], [0.20, 0.80])


def test_non_finite_threshold_inputs_are_rejected():
    with pytest.raises(ValueError, match="Probabilities"):
        threshold_counts([0, 1], [0.20, np.nan])
    with pytest.raises(ValueError, match="finite"):
        cost_analysis(threshold_counts([0, 1], [0.2, 0.8]), np.inf, 100)
