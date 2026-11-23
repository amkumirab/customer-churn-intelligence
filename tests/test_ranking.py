import json

import numpy as np
import pytest

from src.ranking import ranking_at_capacity


def test_fixed_capacities_use_floor_and_actual_selection_rate_for_lift():
    rows = ranking_at_capacity([1, 0, 1] + [0] * 20, np.linspace(1, 0, 23))
    assert [row["capacity_pct"] for row in rows] == [5, 10, 20]
    assert [row["selected"] for row in rows] == [1, 2, 4]
    assert rows[0]["churn_capture"] == 0.5
    assert rows[0]["random_capture"] == pytest.approx(1 / 23)
    assert rows[0]["lift"] == pytest.approx(11.5)
    assert rows[-1]["churn_capture"] == 1


def test_ties_at_the_cutoff_are_averaged_and_independent_of_row_order():
    labels = [1, 0, 1, 0] * 5
    scores = [0.5] * 20
    rows = ranking_at_capacity(labels, scores)
    assert rows == ranking_at_capacity(labels[::-1], scores)
    assert rows[0]["captured_churners"] == 0.5
    assert all(row["lift"] == pytest.approx(1) for row in rows)


def test_zero_slots_and_zero_churn_have_explicit_undefined_metrics():
    small = ranking_at_capacity([0, 1], [0.1, 0.9])
    assert small[0]["selected"] == 0
    assert small[0]["churn_capture"] == 0
    assert small[0]["lift"] is None
    no_churn = ranking_at_capacity([0] * 20, [0.1] * 20)
    assert no_churn[0]["churn_capture"] is None
    assert no_churn[0]["lift"] is None
    json.dumps([small, no_churn], allow_nan=False)


@pytest.mark.parametrize(("labels", "scores"), [
    ([], []), ([0, 1], [0.1]), ([0.5], [0.1]), ([0], [float("nan")]), ([1], [1.2]),
])
def test_invalid_ranking_inputs_are_rejected(labels, scores):
    with pytest.raises(ValueError):
        ranking_at_capacity(labels, scores)
