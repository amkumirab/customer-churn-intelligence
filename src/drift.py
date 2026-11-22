"""Compare batch feature distributions with aggregate training counts."""

import pandas as pd

_NUMERIC_BUCKETS = {
    "tenure_months": ([12, 24, 48, 72], ["<12", "12–<24", "24–<48", "48–<72", "72+"]),
    "monthly_charges": ([25, 50, 75, 100], ["<25", "25–<50", "50–<75", "75–<100", "100+"]),
}
_CONTRACTS = ["Month-to-month", "One year", "Two year"]


def build_drift_reference(features: pd.DataFrame) -> dict:
    """Return only row and bucket counts, using the same buckets for every cohort."""
    if features.empty:
        raise ValueError("A drift comparison needs at least one row per cohort.")
    counts = {}
    for field, (edges, labels) in _NUMERIC_BUCKETS.items():
        buckets = pd.cut(
            features[field], [-float("inf"), *edges, float("inf")], labels=labels, right=False,
        ).astype("object").fillna("Missing")
        frequency = buckets.value_counts()
        counts[field] = {label: int(frequency.get(label, 0)) for label in [*labels, "Missing"]}

    contracts = features["contract_type"]
    buckets = contracts.where(contracts.isin(_CONTRACTS), "Other")
    frequency = buckets.mask(contracts.isna(), "Missing").value_counts()
    counts["contract_type"] = {
        label: int(frequency.get(label, 0)) for label in [*_CONTRACTS, "Other", "Missing"]
    }
    return {"rows": len(features), "counts": counts}


def compare_drift(reference: dict, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return total variation distances and per-bucket changes in percentage points."""
    batch = build_drift_reference(features)
    rows = []
    for feature, buckets in batch["counts"].items():
        for bucket, batch_count in buckets.items():
            reference_count = reference["counts"][feature][bucket]
            reference_share = reference_count / reference["rows"]
            batch_share = batch_count / batch["rows"]
            rows.append({
                "feature": feature, "bucket": bucket,
                "reference_count": reference_count, "batch_count": batch_count,
                "reference_share": reference_share, "batch_share": batch_share,
                "change_pp": 100 * (batch_share - reference_share),
            })
    details = pd.DataFrame(rows)
    distances = details["change_pp"].abs().groupby(details["feature"], sort=False).sum() / 200
    return distances.rename("distance").reset_index(), details
