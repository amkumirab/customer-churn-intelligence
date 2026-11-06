from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.campaign import CampaignConfig, budget_scenarios, campaign_summary, plan_campaign


@pytest.fixture
def cohort():
    return pd.DataFrame({
        "customer_id": ["003", "001", "002", "004"],
        "churn_probability": [0.8, 0.8, 0.9, 0.2],
        "monthly_charges": [100.0, 100.0, 20.0, 50.0],
        "risk_level": ["High", "High", "High", "Low"],
    })


def test_budget_capacity_and_risk_ranking(cohort):
    config = CampaignConfig(budget=50, contact_cost=25, capacity=3, minimum_probability=0.7)
    plan = plan_campaign(cohort, config)
    assert plan["customer_id"].tolist() == ["002", "001"]
    assert plan["priority"].tolist() == [1, 2]
    summary = campaign_summary(plan, config)
    assert summary["contacts"] == 2
    assert summary["spend"] == 50
    assert summary["remaining_budget"] == 0
    assert summary["scenario_retained_customers"] == pytest.approx(0.34)


def test_value_ranking_and_positive_net_filter_are_explicit(cohort):
    config = CampaignConfig(
        budget=25, contact_cost=25, capacity=10, minimum_probability=0.7,
        ranking="net_value", positive_only=True,
    )
    plan = plan_campaign(cohort, config)
    assert plan["customer_id"].tolist() == ["001"]
    assert plan.iloc[0]["scenario_net_value"] == pytest.approx(71)
    no_benefit = plan_campaign(cohort, replace(config, success_rate=0))
    assert no_benefit.empty
    assert campaign_summary(no_benefit, config)["spend"] == 0


def test_zero_cost_is_still_capacity_limited(cohort):
    config = CampaignConfig(budget=0, contact_cost=0, capacity=1, minimum_probability=0)
    assert len(plan_campaign(cohort, config)) == 1
    assert plan_campaign(cohort, replace(config, capacity=0)).empty


def test_budget_uses_cents_and_never_rounds_up_capacity(cohort):
    config = CampaignConfig(budget=0.30, contact_cost=0.10, capacity=4, minimum_probability=0)
    assert len(plan_campaign(cohort, config)) == 3
    assert len(plan_campaign(cohort, replace(config, budget=0.29))) == 2
    with pytest.raises(ValueError, match="cent"):
        plan_campaign(cohort, replace(config, contact_cost=0.001))


def test_ties_are_stable_across_row_order_and_input_is_not_mutated(cohort):
    original = cohort.copy(deep=True)
    config = CampaignConfig(minimum_probability=0)
    first = plan_campaign(cohort, config)
    second = plan_campaign(cohort.sample(frac=1, random_state=4), config)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(cohort, original)


@pytest.mark.parametrize("changes", [
    {"budget": -1}, {"budget": np.inf}, {"contact_cost": np.nan},
    {"capacity": 1.5}, {"capacity": -1}, {"capacity": True},
    {"success_rate": 1.1}, {"minimum_probability": -0.1},
    {"horizon_months": 0}, {"horizon_months": 2.5}, {"ranking": "unknown"},
])
def test_invalid_assumptions_are_rejected(cohort, changes):
    with pytest.raises(ValueError):
        plan_campaign(cohort, replace(CampaignConfig(), **changes))


@pytest.mark.parametrize("column,value", [
    ("customer_id", "003"), ("customer_id", None),
    ("churn_probability", np.nan), ("churn_probability", 1.1),
    ("monthly_charges", -10), ("monthly_charges", np.inf),
])
def test_malformed_cohorts_are_rejected(cohort, column, value):
    cohort.loc[1, column] = value
    with pytest.raises(ValueError):
        plan_campaign(cohort, CampaignConfig())


def test_empty_and_ineligible_cohorts_produce_exportable_plans(cohort):
    config = CampaignConfig(minimum_probability=1)
    for data in [cohort, cohort.iloc[:0]]:
        plan = plan_campaign(data, config)
        assert plan.empty
        assert "scenario_net_value" in plan.columns
        assert campaign_summary(plan, config)["remaining_budget"] == config.budget


def test_sensitivity_scenarios_reallocate_without_exceeding_any_budget(cohort):
    config = CampaignConfig(contact_cost=25, minimum_probability=0)
    scenarios = budget_scenarios(cohort, config, [0, 24.99, 25, 50, 100])
    assert scenarios["contacts"].tolist() == [0, 0, 1, 2, 4]
    assert (scenarios["spend"] <= scenarios["budget"]).all()


def test_reusing_a_previously_exported_plan_does_not_duplicate_priority(cohort):
    config = CampaignConfig(minimum_probability=0)
    first = plan_campaign(cohort, config)
    second = plan_campaign(first, replace(config, capacity=1))
    assert second["priority"].tolist() == [1]
    assert second["customer_id"].tolist() == ["002"]


def test_randomized_allocations_preserve_budget_and_eligibility_invariants():
    rng = np.random.default_rng(2026)
    cohort = pd.DataFrame({
        "customer_id": [f"C-{number:04d}" for number in range(100)],
        "churn_probability": rng.random(100),
        "monthly_charges": rng.uniform(20, 150, 100),
    })
    for _ in range(40):
        config = CampaignConfig(
            budget=int(rng.integers(0, 100_000)) / 100,
            contact_cost=int(rng.integers(0, 5_000)) / 100,
            capacity=int(rng.integers(0, 120)), minimum_probability=float(rng.random()),
            ranking=rng.choice(["risk", "net_value"]), positive_only=bool(rng.integers(0, 2)),
        )
        plan = plan_campaign(cohort, config)
        summary = campaign_summary(plan, config)
        assert summary["spend"] <= config.budget
        assert summary["remaining_budget"] >= 0
        assert len(plan) <= config.capacity
        assert (plan["churn_probability"] >= config.minimum_probability).all()
        assert plan["customer_id"].is_unique
        if config.positive_only:
            assert (plan["scenario_net_value"] > 0).all()
