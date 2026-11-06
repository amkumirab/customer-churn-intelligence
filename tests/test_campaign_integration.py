import json
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from src.batch import BatchValidationError
from src.campaign import CampaignConfig, campaign_summary, plan_campaign
from src.cohort import score_cohort
from src.data import FEATURE_COLUMNS, generate_customers
from src.model import train_and_evaluate
from src.reporting import build_decision_bundle


def test_trained_pipeline_to_campaign_to_bundle_keeps_identities_and_totals():
    pipeline, report = train_and_evaluate(generate_customers(500, random_state=93))
    customers = generate_customers(100, random_state=96)[FEATURE_COLUMNS].head(20).copy()
    customers.insert(0, "customer_id", [f"00{i:03d}" for i in range(20)])
    cohort = score_cohort(pipeline, customers, 0.3, 0.1)
    config = CampaignConfig(
        budget=75, contact_cost=25, capacity=10, minimum_probability=0.3,
        ranking="net_value",
    )
    plan = plan_campaign(cohort, config)
    assert len(plan) == 3
    by_id = customers.set_index("customer_id")
    for row in plan.itertuples():
        assert row.monthly_charges == by_id.loc[row.customer_id, "monthly_charges"]
        assert row.scenario_retained_revenue == pytest.approx(
            row.churn_probability * 0.2 * row.monthly_charges * 6
        )
    with ZipFile(BytesIO(build_decision_bundle(cohort, config, report, 0.3, 0.1))) as bundle:
        exported = pd.read_csv(BytesIO(bundle.read("contact_list.csv")), dtype={"customer_id": str})
        assert exported["customer_id"].tolist() == plan["customer_id"].tolist()
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["summary"] == campaign_summary(plan, config)
        assert manifest["model"]["artifact_version"] == report["artifact_version"]


def test_upload_limit_is_enforced_before_model_scoring():
    customers = generate_customers(100)[FEATURE_COLUMNS]
    oversized = pd.concat([customers] * 101, ignore_index=True)
    with pytest.raises(BatchValidationError, match="limit"):
        score_cohort(None, oversized)
