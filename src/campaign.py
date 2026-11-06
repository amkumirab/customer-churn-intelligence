"""Budget-constrained outreach scenarios; no campaigns are executed here."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
from math import isfinite
from numbers import Integral

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CampaignConfig:
    """Uniform contact cost, eligibility, and hypothetical retention assumptions.

    Monetary inputs must be finite, nonnegative whole-cent amounts. Success rate
    means the assumed fraction of otherwise-churning contacts retained, not a
    treatment effect learned by the model. Revenue excludes margins and discounting.
    """

    budget: float = 2_500.0
    contact_cost: float = 25.0
    capacity: int = 100
    minimum_probability: float = 0.70
    success_rate: float = 0.20
    horizon_months: int = 6
    ranking: str = "risk"
    positive_only: bool = False


def _cents(value: float) -> int:
    if not isfinite(value) or value < 0:
        raise ValueError("Money amounts must be finite and nonnegative.")
    cents = Decimal(str(value)) * 100
    if cents != cents.to_integral_value():
        raise ValueError("Money amounts must use whole cents.")
    return int(cents)


def _validate_config(config: CampaignConfig) -> None:
    _cents(config.budget)
    _cents(config.contact_cost)
    for name in ("capacity", "horizon_months"):
        value = getattr(config, name)
        minimum = 1 if name == "horizon_months" else 0
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")
    if config.horizon_months > 120:
        raise ValueError("The scenario horizon cannot exceed 120 months.")
    for value in (config.minimum_probability, config.success_rate):
        if not isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Risk and success assumptions must be finite and between 0 and 1.")
    if config.ranking not in {"risk", "net_value"}:
        raise ValueError("Ranking must be 'risk' or 'net_value'.")
    if not isinstance(config.positive_only, bool):
        raise ValueError("positive_only must be a boolean.")


def plan_campaign(cohort: pd.DataFrame, config: CampaignConfig) -> pd.DataFrame:
    """Return a ranked contact list within budget/capacity, without mutating inputs.

    Required columns: customer_id (unique literal strings), churn_probability,
    monthly_charges. Constant outreach cost makes sorted top-N allocation exact
    for the selected ranking objective; customer ID is a deterministic tie-breaker.
    """
    _validate_config(config)
    required = {"customer_id", "churn_probability", "monthly_charges"}
    if cohort.columns.has_duplicates or not required.issubset(cohort.columns):
        raise ValueError("The cohort needs unique columns with IDs, scores, and monthly charges.")
    ranked = cohort.drop(columns="priority", errors="ignore").copy()
    identifiers = ranked["customer_id"]
    if (
        not identifiers.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
        or identifiers.duplicated().any()
    ):
        raise ValueError("Customer identifiers must be unique, nonblank strings.")
    for column in ("churn_probability", "monthly_charges"):
        values = pd.to_numeric(ranked[column], errors="coerce")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative numbers.")
        if column == "churn_probability" and (values > 1).any():
            raise ValueError("Churn probability cannot exceed 1.")
        ranked[column] = values

    ranked["scenario_retained_customers"] = ranked["churn_probability"] * config.success_rate
    ranked["scenario_retained_revenue"] = (
        ranked["scenario_retained_customers"] * ranked["monthly_charges"] * config.horizon_months
    )
    ranked["scenario_net_value"] = ranked["scenario_retained_revenue"] - config.contact_cost
    if not np.isfinite(ranked[["scenario_retained_revenue", "scenario_net_value"]]).all().all():
        raise ValueError("The scenario produces non-finite financial values.")
    eligible = ranked["churn_probability"] >= config.minimum_probability
    if config.positive_only:
        eligible &= ranked["scenario_net_value"] > 0
    ranked = ranked.loc[eligible]
    sort_columns = (
        ["scenario_net_value", "churn_probability", "customer_id"]
        if config.ranking == "net_value" else ["churn_probability", "customer_id"]
    )
    ranked = ranked.sort_values(sort_columns, ascending=[False] * (len(sort_columns) - 1) + [True])
    contact_cents = _cents(config.contact_cost)
    slots = min(config.capacity, len(ranked))
    if contact_cents:
        slots = min(slots, _cents(config.budget) // contact_cents)
    plan = ranked.head(slots).reset_index(drop=True)
    plan.insert(0, "priority", np.arange(1, len(plan) + 1))
    plan["contact_cost"] = config.contact_cost
    return plan


def campaign_summary(plan: pd.DataFrame, config: CampaignConfig) -> dict[str, int | float | None]:
    """Summarize a plan produced using the same configuration, including unused budget."""
    _validate_config(config)
    spend_cents = len(plan) * _cents(config.contact_cost)
    revenue = float(plan["scenario_retained_revenue"].sum())
    spend = spend_cents / 100
    return {
        "contacts": len(plan),
        "spend": spend,
        "remaining_budget": (_cents(config.budget) - spend_cents) / 100,
        "scenario_retained_customers": float(plan["scenario_retained_customers"].sum()),
        "scenario_retained_revenue": round(revenue, 2),
        "scenario_net_value": round(revenue - spend, 2),
        "scenario_roi": (revenue - spend) / spend if spend else None,
    }


def budget_scenarios(
    cohort: pd.DataFrame, config: CampaignConfig, budgets: Iterable[float]
) -> pd.DataFrame:
    """Reallocate independently for each budget, holding all other assumptions fixed."""
    rows = []
    for budget in budgets:
        scenario = replace(config, budget=budget)
        plan = plan_campaign(cohort, scenario)
        rows.append({"budget": float(budget), **campaign_summary(plan, scenario)})
    return pd.DataFrame(rows)
