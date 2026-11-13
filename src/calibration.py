"""Training-only sigmoid calibration and aggregate probability-quality diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def probability_quality(
    labels: Iterable[int], probabilities: Iterable[float], n_bins: int = 10
) -> dict[str, Any]:
    """Return Brier/log loss and fixed-width reliability bins, including empty bins.

    Bins are left-inclusive/right-exclusive, with 1 included in the final bin.
    Empty bins have null statistics, never an invented observed zero. These losses
    assess overall probability quality, not calibration independently of discrimination.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or not 2 <= n_bins <= 50:
        raise ValueError("The number of bins must be an integer between 2 and 50.")
    observed = np.asarray(list(labels))
    try:
        scores = np.asarray(list(probabilities), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Probabilities must be numeric values between 0 and 1.") from error
    if (
        observed.ndim != 1 or scores.ndim != 1
        or observed.size == 0 or observed.size != scores.size
    ):
        raise ValueError("Labels and scores must be one-dimensional with equal non-zero length.")
    if pd.isna(observed).any() or not np.isin(observed, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite numbers between 0 and 1.")
    edges = np.linspace(0, 1, n_bins + 1)
    memberships = np.searchsorted(edges[1:-1], scores, side="right")
    bins = []
    for index in range(n_bins):
        selected = memberships == index
        count = int(selected.sum())
        bins.append({
            "lower_bound": float(edges[index]),
            "upper_bound": float(edges[index + 1]),
            "count": count,
            "mean_probability": float(scores[selected].mean()) if count else None,
            "observed_rate": float(observed[selected].mean()) if count else None,
        })
    return {
        "samples": int(observed.size),
        "positive_rate": float(observed.mean()),
        "brier_score": float(brier_score_loss(observed, scores)),
        "log_loss": float(log_loss(observed, scores, labels=[0, 1])),
        "bins": bins,
    }


def fit_sigmoid_calibrator(
    pipeline: Pipeline, training_features: pd.DataFrame, training_labels: pd.Series,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """Fit a separate candidate from training-only out-of-fold predictions.

    Clone the entire preprocessing pipeline so every CV fold fits its own
    imputers, scaler, and encoder. The original fitted estimator is untouched.
    Do not pass validation or final-test rows into this function.
    """
    labels = np.asarray(training_labels)
    if (
        labels.ndim != 1 or len(labels) != len(training_features)
        or pd.isna(labels).any() or not np.isin(labels, [0, 1]).all()
        or len(np.unique(labels)) != 2
    ):
        raise ValueError(
            "Training labels must align with features and contain both binary classes."
        )
    minority_count = int(np.bincount(labels.astype(int), minlength=2).min())
    if minority_count < 2:
        raise ValueError("Calibration requires at least two training rows per class.")
    folds = StratifiedKFold(
        n_splits=min(5, minority_count), shuffle=True, random_state=random_state,
    )
    candidate = CalibratedClassifierCV(
        estimator=clone(pipeline), method="sigmoid", cv=folds, ensemble=False,
    )
    candidate.fit(training_features, training_labels)
    return candidate
