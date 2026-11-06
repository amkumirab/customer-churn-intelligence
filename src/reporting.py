"""In-memory decision bundles with explicit assumptions and bounded, fixed exports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from io import BytesIO
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from .batch import export_predictions_csv
from .campaign import CampaignConfig, budget_scenarios, campaign_summary, plan_campaign
from .cohort import segment_summary
from .model import risk_details

COHORT_EXPORT_COLUMNS = [
    "customer_id", "churn_probability", "risk_level", "recommended_action",
    "tenure_months", "monthly_charges", "contract_type", "internet_service",
    "payment_method", "tenure_band", "monthly_revenue_exposure", "imputed_total_charges",
]
PLAN_EXPORT_COLUMNS = [
    "priority", *COHORT_EXPORT_COLUMNS, "scenario_retained_customers",
    "scenario_retained_revenue", "scenario_net_value", "contact_cost",
]
BUNDLE_NOTES = """Retention scenario decision bundle

This is a portfolio demonstration, not an executed outreach campaign.
Model scores are not calibrated forecasts. Balanced class weights can distort probabilities.
Success rate is an assumed fraction of otherwise-churning contacted customers retained.
Scenario retained revenue = score * success rate * monthly charges * horizon months.
Scenario net value = retained revenue - outreach cost. This is not profit or lifetime value.
No causal treatment effect, consent, contactability, or fairness assessment is modeled.
Budget is allocated at a uniform contact cost; eligibility and capacity are enforced.
CSV text is protected against spreadsheet formulas. IDs may have an apostrophe prefix.
The fingerprint hashes the protected scored-cohort CSV sorted by literal customer ID.
It identifies this exported cohort, not the original uploaded file or model weights.
Demographic profiles and uploaded churn labels are excluded from these exports.
Downloaded bundles still contain customer identifiers; store and share them carefully.
"""


def build_decision_bundle(
    cohort: pd.DataFrame,
    config: CampaignConfig,
    report: dict[str, Any],
    high_risk_threshold: float,
    medium_risk_threshold: float,
    *,
    population_filters: dict[str, list[str]] | None = None,
) -> bytes:
    """Build fixed-name ZIP members in memory; never write customer records to disk.

    The provided cohort is the filtered population. The plan is recomputed from
    that exact population/config to prevent stale contact lists and metadata.
    Only the exported operational fields are retained, even if extra columns exist.
    """
    risk_details(0, high_risk_threshold, medium_risk_threshold)
    exported_cohort = cohort[COHORT_EXPORT_COLUMNS].copy()
    plan = plan_campaign(exported_cohort, config)
    canonical = export_predictions_csv(
        exported_cohort.sort_values("customer_id").reset_index(drop=True)
    )
    report_bytes = json.dumps(report, sort_keys=True, allow_nan=False).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "cohort_rows": len(exported_cohort),
        "population_filters": population_filters,
        "cohort_sha256": hashlib.sha256(canonical).hexdigest(),
        "model": {
            "name": report.get("best_model"),
            "artifact_version": report.get("artifact_version"),
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        },
        "policy": {
            "high_risk_threshold": high_risk_threshold,
            "medium_risk_threshold": medium_risk_threshold,
        },
        "assumptions": asdict(config),
        "summary": campaign_summary(plan, config),
        "limitations": BUNDLE_NOTES.splitlines()[2:],
    }
    scenarios = budget_scenarios(
        exported_cohort, config, sorted({0.0, round(config.budget / 2, 2), config.budget})
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as bundle:
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2, allow_nan=False))
        bundle.writestr("scored_customers.csv", export_predictions_csv(exported_cohort))
        bundle.writestr("contact_list.csv", export_predictions_csv(plan[PLAN_EXPORT_COLUMNS]))
        bundle.writestr(
            "contract_segments.csv",
            export_predictions_csv(segment_summary(exported_cohort, "contract_type")),
        )
        bundle.writestr("budget_scenarios.csv", export_predictions_csv(scenarios))
        bundle.writestr("README.txt", BUNDLE_NOTES)
    return buffer.getvalue()
