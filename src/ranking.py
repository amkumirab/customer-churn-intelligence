"""Evaluate churn ranking at fixed outreach capacities."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

CAPACITY_PERCENTAGES = (5, 10, 20)


def ranking_at_capacity(
    labels: Iterable[int], scores: Iterable[float],
) -> list[dict[str, int | float | None]]:
    """Compare top-ranked selection with uniform random selection at the same size.

    Round slots down. If a cutoff splits a score tie, average over uniformly
    choosing the remaining slots from that tie rather than relying on row order.
    Capture and lift are undefined when the split has no churners. Lift is also
    undefined when there are no available slots.
    """
    observed = np.asarray(list(labels))
    probabilities = np.asarray(list(scores), dtype=float)
    if (
        observed.ndim != 1 or probabilities.ndim != 1
        or observed.size == 0 or observed.size != probabilities.size
    ):
        raise ValueError("Labels and scores must be nonempty, aligned one-dimensional inputs.")
    if pd.isna(observed).any() or not np.isin(observed, [0, 1]).all():
        raise ValueError("Labels must be binary.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Scores must be finite probabilities between 0 and 1.")

    size = len(observed)
    total_churners = int(np.count_nonzero(observed))
    ranked_scores = np.sort(probabilities)[::-1]
    rows = []
    for capacity in CAPACITY_PERCENTAGES:
        slots = size * capacity // 100
        captured = 0.0
        if slots:
            cutoff = ranked_scores[slots - 1]
            above = probabilities > cutoff
            tied = probabilities == cutoff
            remaining = slots - int(above.sum())
            captured = float(
                np.count_nonzero(observed[above])
                + remaining * np.count_nonzero(observed[tied]) / tied.sum()
            )

        random_capture = slots / size if total_churners else None
        capture = captured / total_churners if total_churners else None
        rows.append({
            "capacity_pct": capacity,
            "rows": size,
            "selected": slots,
            "total_churners": total_churners,
            "captured_churners": captured,
            "churn_capture": capture,
            "random_capture": random_capture,
            "lift": capture / random_capture if capture is not None and random_capture else None,
        })
    return rows
