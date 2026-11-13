import json

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.calibration import fit_sigmoid_calibrator, probability_quality
from src.data import FEATURE_COLUMNS, generate_customers
from src.model import build_pipeline


def test_quality_metrics_and_fixed_bin_boundaries_match_known_values():
    quality = probability_quality([0, 0, 1, 1], [0, 0.25, 0.75, 1], n_bins=4)
    assert quality["brier_score"] == pytest.approx(0.03125)
    assert quality["log_loss"] == pytest.approx(log_loss([0, 0, 1, 1], [0, 0.25, 0.75, 1]))
    assert quality["samples"] == 4
    assert quality["positive_rate"] == 0.5
    bins = quality["bins"]
    assert [row["count"] for row in bins] == [1, 1, 0, 2]
    assert bins[2]["mean_probability"] is None
    assert bins[2]["observed_rate"] is None
    assert bins[3]["mean_probability"] == 0.875
    assert bins[3]["observed_rate"] == 1
    json.dumps(quality, allow_nan=False)


def test_single_class_and_exact_zero_one_scores_are_valid():
    quality = probability_quality([0, 0], [0, 1])
    assert quality["brier_score"] == 0.5
    assert np.isfinite(quality["log_loss"])
    assert sum(row["count"] for row in quality["bins"]) == 2


@pytest.mark.parametrize("labels,scores", [
    ([], []), ([0, 1], [0.5]), ([0.5, 1], [0.1, 0.9]),
    ([None, 1], [0.1, 0.9]), ([0, 1], [np.nan, 0.5]),
    ([0, 1], [np.inf, 0.5]), ([0, 1], [-0.01, 0.5]),
    ([0, 1], [0.1, 1.01]), ([[0], [1]], [0.1, 0.9]),
    ([0, 1], [[0.1], [0.9]]),
    ([pd.NA, 1], [0.1, 0.9]),
])
def test_invalid_probability_inputs_are_rejected(labels, scores):
    with pytest.raises(ValueError):
        probability_quality(labels, scores)


@pytest.mark.parametrize("n_bins", [0, 1, 51, True, 2.5])
def test_invalid_bin_counts_are_rejected(n_bins):
    with pytest.raises(ValueError, match="bins"):
        probability_quality([0, 1], [0.1, 0.9], n_bins=n_bins)


def test_cv_refits_preprocessing_only_inside_training_population(monkeypatch):
    training = generate_customers(100, random_state=23)
    training.index = np.arange(1000, 1100)
    fitted_populations = []
    original = ColumnTransformer.fit_transform

    def record_population(self, features, *args, **kwargs):
        fitted_populations.append(set(features.index))
        return original(self, features, *args, **kwargs)

    monkeypatch.setattr(ColumnTransformer, "fit_transform", record_population)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    candidate = fit_sigmoid_calibrator(pipeline, training[FEATURE_COLUMNS], training["churn"])
    assert candidate.cv.n_splits == 5
    assert candidate.ensemble is False
    assert "transformers_" not in vars(pipeline.named_steps["preprocessor"])
    assert all(population <= set(training.index) for population in fitted_populations)
    assert sum(population < set(training.index) for population in fitted_populations) == 5
    assert set(training.index) in fitted_populations
    assert np.isfinite(candidate.predict_proba(training[FEATURE_COLUMNS])).all()


def test_fold_count_adapts_to_small_minority_class():
    training = generate_customers(100).head(10).copy()
    training["churn"] = [1, 1] + [0] * 8
    candidate = fit_sigmoid_calibrator(
        build_pipeline(LogisticRegression(max_iter=1000)),
        training[FEATURE_COLUMNS], training["churn"],
    )
    assert candidate.cv.n_splits == 2
    training["churn"] = [1] + [0] * 9
    with pytest.raises(ValueError, match="two"):
        fit_sigmoid_calibrator(
            build_pipeline(LogisticRegression()), training[FEATURE_COLUMNS], training["churn"]
        )


def test_metrics_are_invariant_to_row_order():
    labels = pd.Series([0, 1, 0, 1])
    scores = pd.Series([0.1, 0.9, 0.4, 0.6])
    assert probability_quality(labels, scores) == probability_quality(labels[::-1], scores[::-1])
