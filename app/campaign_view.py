"""Cohort exploration and budget-aware retention planning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.batch import BatchValidationError, batch_summary, export_predictions_csv, read_customer_csv
from src.campaign import CampaignConfig, budget_scenarios, campaign_summary, plan_campaign
from src.cohort import SEGMENT_FIELDS, filter_cohort, score_cohort, segment_summary
from src.data import FEATURE_COLUMNS
from src.reporting import COHORT_EXPORT_COLUMNS, PLAN_EXPORT_COLUMNS, build_decision_bundle


def _render_cohort_analysis(cohort: pd.DataFrame) -> pd.DataFrame:
    summary = batch_summary(cohort)
    columns = st.columns(4)
    columns[0].metric("Customers scored", f"{summary['customers']:,}")
    columns[1].metric("High risk", f"{summary['high_risk']:,}")
    columns[2].metric("Medium risk", f"{summary['medium_risk']:,}")
    columns[3].metric("Average churn probability", f"{summary['average_probability']:.1%}")
    imputed = int(cohort["imputed_total_charges"].sum())
    if imputed:
        st.warning(f"{imputed:,} blank total-charge value(s) were imputed by the trained pipeline.")

    with st.expander("Cohort filters", expanded=True):
        risk_levels = st.multiselect(
            "Risk levels", ["High", "Medium", "Low"], default=["High", "Medium", "Low"],
            key="cohort_risk_filter",
        )
        contracts = st.multiselect(
            "Contracts", ["Month-to-month", "One year", "Two year"],
            default=["Month-to-month", "One year", "Two year"], key="cohort_contract_filter",
        )
    filtered = filter_cohort(cohort, risk_levels, contracts)
    columns = st.columns(2)
    columns[0].metric("Matching customers", f"{len(filtered):,}")
    columns[1].metric(
        "Score-weighted monthly exposure", f"${filtered['monthly_revenue_exposure'].sum():,.2f}"
    )
    st.caption(
        "Filters affect the segment tables, campaign, and decision bundle below. "
        "Exposure is score x monthly charges, not observed lost revenue."
    )
    if filtered.empty:
        st.info("No customers match these filters. Select risk levels and contracts to continue.")
    else:
        segment_field = st.selectbox(
            "Analyze segments by", SEGMENT_FIELDS,
            format_func=lambda value: value.replace("_", " ").title(), key="cohort_segment_field",
        )
        segments = segment_summary(filtered, segment_field)
        st.bar_chart(segments.set_index("segment")[["average_probability"]], horizontal=True)
        st.dataframe(segments, hide_index=True, width="stretch")
    with st.expander("Scored customers"):
        st.dataframe(
            filtered[COHORT_EXPORT_COLUMNS], hide_index=True, width="stretch",
            column_config={
                "churn_probability": st.column_config.NumberColumn("Churn score", format="%.4f"),
                "monthly_charges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
                "monthly_revenue_exposure": st.column_config.NumberColumn(
                    "Score-weighted monthly exposure", format="$%.2f"
                ),
            },
        )
        st.download_button(
            "Download scored customers", export_predictions_csv(filtered[COHORT_EXPORT_COLUMNS]),
            file_name="churn_predictions.csv", mime="text/csv", key="download_cohort",
        )
    return filtered


def _render_campaign_plan(
    cohort: pd.DataFrame, artifact: dict, high: float, medium: float, contact_cost: float
) -> None:
    st.subheader("Plan an outreach campaign")
    st.caption(
        f"Only matching customers with scores of at least {high:.0%} are eligible. "
        f"Outreach cost is ${contact_cost:,.2f} per contact, set in the decision policy sidebar."
    )
    with st.container(border=True):
        left, right = st.columns(2)
        budget = left.number_input(
            "Campaign budget ($)", min_value=0.0, max_value=1_000_000_000.0,
            value=2_500.0, step=100.0, format="%.2f", key="campaign_budget",
        )
        capacity = right.number_input(
            "Maximum contacts", min_value=0, max_value=10_000, value=100,
            step=10, key="campaign_capacity",
        )
        ranking = left.selectbox(
            "Contact priority", ["Highest churn risk", "Highest scenario net value"],
            key="campaign_ranking",
        )
        horizon = right.number_input(
            "Revenue horizon (months)", min_value=1, max_value=24, value=6, step=1,
            key="campaign_horizon",
        )
        success_rate = st.slider(
            "Assumed retention success rate", min_value=0.0, max_value=1.0, value=0.20,
            step=0.05, format="%.2f", key="campaign_success_rate",
            help="Assumed fraction of otherwise-churning contacted customers retained.",
        )
        positive_only = st.checkbox(
            "Exclude contacts with non-positive scenario net value", key="campaign_positive_only",
        )
    st.warning(
        "Scenario only: model scores are not calibrated forecasts and retention success is an "
        "assumption, not a measured treatment effect. Revenue is not profit or lifetime value."
    )
    config = CampaignConfig(
        budget=budget, contact_cost=contact_cost, capacity=capacity,
        minimum_probability=high, success_rate=success_rate, horizon_months=horizon,
        ranking="risk" if ranking == "Highest churn risk" else "net_value",
        positive_only=positive_only,
    )
    try:
        plan = plan_campaign(cohort, config)
        summary = campaign_summary(plan, config)
    except ValueError as error:
        st.error(f"The campaign could not be prepared: {error}")
        return
    columns = st.columns(4)
    columns[0].metric("Planned contacts", f"{summary['contacts']:,}")
    columns[1].metric("Outreach spend", f"${summary['spend']:,.2f}")
    columns[2].metric("Remaining budget", f"${summary['remaining_budget']:,.2f}")
    columns[3].metric("Scenario net value", f"${summary['scenario_net_value']:,.2f}")
    st.caption(
        f"Scenario retained customers: {summary['scenario_retained_customers']:.2f}. "
        f"Scenario retained revenue: ${summary['scenario_retained_revenue']:,.2f}. "
        "Net value subtracts outreach spend only. No messages or offers will be sent."
    )
    if plan.empty:
        st.info(
            "No customers selected. Check the filters, risk threshold, budget, capacity, "
            "and positive-net-value option."
        )
    else:
        st.dataframe(
            plan[[
                "priority", "customer_id", "churn_probability", "risk_level",
                "monthly_charges", "scenario_retained_revenue", "scenario_net_value",
            ]], hide_index=True, width="stretch",
            column_config={
                "churn_probability": st.column_config.NumberColumn("Churn score", format="%.4f"),
                "monthly_charges": st.column_config.NumberColumn("Monthly charges", format="$%.2f"),
                "scenario_retained_revenue": st.column_config.NumberColumn(
                    "Scenario retained revenue", format="$%.2f"
                ),
                "scenario_net_value": st.column_config.NumberColumn(
                    "Scenario net value", format="$%.2f"
                ),
            },
        )
    with st.expander("Budget sensitivity"):
        budgets = sorted({0.0, round(budget / 2, 2), budget})
        scenarios = budget_scenarios(cohort, config, budgets)
        st.bar_chart(scenarios.set_index("budget")[["contacts"]])
        st.dataframe(scenarios, hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.download_button(
        "Download contact list", export_predictions_csv(plan[PLAN_EXPORT_COLUMNS]),
        file_name="retention_contact_list.csv", mime="text/csv", width="stretch",
    )
    right.download_button(
        "Download decision bundle",
        build_decision_bundle(
            cohort, config, artifact["report"], high, medium,
            population_filters={
                "risk_levels": list(st.session_state["cohort_risk_filter"]),
                "contracts": list(st.session_state["cohort_contract_filter"]),
            },
        ),
        file_name="retention_decision_bundle.zip", mime="application/zip", width="stretch",
    )
    st.caption(
        "The ZIP includes the filtered scored cohort, contact list, contract segments, budget "
        "scenarios, and a JSON manifest with assumptions and fingerprints. Uploaded records "
        "are not saved to disk. Downloads contain customer IDs; store them carefully."
    )


def render_campaign_workspace(
    artifact: dict, root: Path, high: float, medium: float, contact_cost: float
) -> None:
    st.subheader("Batch churn prediction")
    st.caption(
        "Upload up to 10,000 customers or try the sample. Customer IDs must be unique. "
        "Original Kaggle Telco columns and the project template are both accepted."
    )
    template = pd.read_csv(root / "data" / "example_customers.csv")[FEATURE_COLUMNS]
    uploaded = st.file_uploader("Upload a customer CSV", type=["csv"])
    left, right = st.columns(2)
    if left.button("Use sample customers", width="stretch"):
        st.session_state["use_sample_cohort"] = True
    right.download_button(
        "Download CSV template", template.to_csv(index=False).encode("utf-8"),
        file_name="customer_batch_template.csv", mime="text/csv", width="stretch",
    )
    if uploaded is None and not st.session_state.get("use_sample_cohort", False):
        st.info("Upload a customer CSV or choose 'Use sample customers' to explore the workspace.")
        return
    try:
        if uploaded is not None:
            customers = read_customer_csv(uploaded)
            st.caption("Source: uploaded CSV. Labels, if present, are ignored for scoring.")
        else:
            customers = template.copy()
            st.caption("Source: three illustrative sample customers, not an evaluation dataset.")
        cohort = score_cohort(artifact["pipeline"], customers, high, medium)
    except BatchValidationError as error:
        st.error(f"The CSV could not be processed: {error}")
        return
    filtered = _render_cohort_analysis(cohort)
    st.divider()
    _render_campaign_plan(filtered, artifact, high, medium, contact_cost)
