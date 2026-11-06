import numpy as np
import pandas as pd
import pytest

from src.batch import BatchValidationError
from src.campaign import CampaignConfig, plan_campaign
from src.cohort import filter_cohort, score_cohort, segment_summary
from src.data import FEATURE_COLUMNS, generate_customers


class ChargeBasedPipeline:
    """A deterministic score source that makes feature/score misalignment visible."""

    def predict_proba(self, customers):
        positive = customers["monthly_charges"].to_numpy() / 200
        return np.column_stack([1 - positive, positive])


@pytest.fixture
def customers():
    rows = generate_customers(100, random_state=10)[FEATURE_COLUMNS].head(4).copy()
    rows.index = [20, 10, 40, 30]
    rows.insert(0, "customer_id", ["003", "001", "004", "002"])
    rows["monthly_charges"] = [30.0, 100.0, 50.0, 80.0]
    rows["tenure_months"] = [0, 12, 24, 25]
    rows["contract_type"] = ["Month-to-month", "Two year", "One year", "Month-to-month"]
    return rows


def test_scores_stay_aligned_to_features_after_ranking(customers):
    before = customers.copy(deep=True)
    cohort = score_cohort(ChargeBasedPipeline(), customers, 0.45, 0.25)
    assert cohort["customer_id"].tolist() == ["001", "002", "004", "003"]
    assert cohort["monthly_charges"].tolist() == [100, 80, 50, 30]
    assert cohort["churn_probability"].tolist() == [0.5, 0.4, 0.25, 0.15]
    assert cohort["tenure_band"].tolist() == [
        "0-12 months", "25+ months", "13-24 months", "0-12 months",
    ]
    assert cohort["monthly_revenue_exposure"].sum() == pytest.approx(99)
    assert not {"gender", "senior_citizen", "has_partner"}.intersection(cohort.columns)
    pd.testing.assert_frame_equal(customers, before)


def test_segment_totals_reconcile_with_cohort(customers):
    cohort = score_cohort(ChargeBasedPipeline(), customers, 0.45, 0.25)
    segments = segment_summary(cohort, "contract_type")
    assert segments["customers"].sum() == 4
    assert segments["high_risk_customers"].sum() == 1
    assert segments["monthly_revenue_exposure"].sum() == pytest.approx(99)
    month = segments.set_index("segment").loc["Month-to-month"]
    assert month["average_probability"] == pytest.approx(0.275)
    assert month["high_risk_rate"] == 0


def test_filters_combine_without_changing_the_original(customers):
    cohort = score_cohort(ChargeBasedPipeline(), customers, 0.45, 0.25)
    filtered = filter_cohort(cohort, ["Medium"], ["Month-to-month"])
    assert filtered["customer_id"].tolist() == ["002"]
    assert len(cohort) == 4
    assert filter_cohort(cohort, [], ["Month-to-month"]).empty
    assert segment_summary(cohort.iloc[:0], "tenure_band").empty
    with pytest.raises(ValueError, match="segment"):
        segment_summary(cohort, "gender")


def test_kaggle_schema_can_be_used_for_campaigns():
    raw = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", nrows=5)
    cohort = score_cohort(ChargeBasedPipeline(), raw)
    assert set(cohort["customer_id"]) == set(raw["customerID"])
    assert len(cohort) == 5


def test_duplicate_identifiers_are_rejected_before_scoring(customers):
    customers.loc[10, "customer_id"] = "003"
    with pytest.raises(BatchValidationError, match="unique"):
        score_cohort(ChargeBasedPipeline(), customers)


def test_rounding_cannot_promote_a_below_threshold_customer_into_a_campaign(customers):
    class BoundaryPipeline:
        def predict_proba(self, rows):
            positive = np.full(len(rows), 0.69996)
            return np.column_stack([1 - positive, positive])

    cohort = score_cohort(BoundaryPipeline(), customers, 0.7, 0.4)
    assert cohort["risk_level"].eq("Medium").all()
    assert (cohort["churn_probability"] < 0.7).all()
    assert plan_campaign(cohort, CampaignConfig(minimum_probability=0.7)).empty
