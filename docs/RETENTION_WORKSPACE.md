# Retention workspace design

## Local, side-effect-free planning

The dashboard prepares an illustrative contact list. It never sends offers, contacts customers,
or persists uploaded profiles. The core uses existing dependencies and has no Streamlit state:

```python
from src.campaign import CampaignConfig, plan_campaign
from src.cohort import score_cohort

cohort = score_cohort(pipeline, uploaded_dataframe, 0.70, 0.40)
config = CampaignConfig(budget=2500, contact_cost=25, capacity=100)
plan = plan_campaign(cohort, config)
```

`score_cohort` accepts the same project/Kaggle schemas as the batch reader. It rejects duplicate
IDs and joins features to ranked predictions by identity, not position. Its operational fields
are tenure, monthly charges, contract, internet service, and payment method. Demographics and
uploaded target labels are omitted from the cohort output. An omitted ID becomes a one-based
row identifier and cannot be used to identify a real account outside this upload.

## Allocation semantics

`CampaignConfig` is immutable. Money must be finite, nonnegative, and represent whole cents;
capacity must be a nonnegative integer. Success rate and minimum score are in [0, 1], and the
horizon is an integer from 1 to 120 months (the dashboard exposes 1 to 24).

Customers are eligible when their full-precision score is at least the minimum. The optional
positive-value filter is applied before allocation. Sort by descending score or descending
scenario net value, then score, then ascending literal ID. Since every contact has the same
cost, sorted top-N selection is sufficient; a variable-cost knapsack solver would add complexity
without improving this contract. Slots are bounded by floor(budget cents / contact cost cents)
and capacity. Free outreach still uses capacity. An empty plan retains its export schema.

Classification policy costs and campaign scenarios are intentionally separate. Validation
policy cost compares classification errors. Campaign net value uses hypothetical retention
success and revenue. Neither is an observed business result. The current class-balanced
models do not establish calibrated probabilities or individualized treatment effects.

## Trust boundaries and privacy

- CSV uploads are untrusted. Existing schema, category, numeric, and row-limit validation runs
  before scoring; nullable IDs stay nullable until blank checks finish.
- Joins are one-to-one and campaign IDs are unique. Filters never silently widen a population.
- ZIP entries use fixed names; customer text cannot become a path or executable instruction.
- All CSV exports use the existing spreadsheet-formula protection, including contact lists.
  UTF-8 with BOM and fixed LF line endings keep exported cohort fingerprints platform-independent.
- Bundle metadata uses strict JSON, includes every scenario setting, and identifies the model
  report and canonical filtered cohort with SHA-256. These are fingerprints, not authentication
  signatures, original-upload hashes, or serialized model-weight identities.
- Dashboard bundles also record the active risk and contract filter selections. Standalone
  calls may omit this optional provenance field, represented explicitly as JSON null.
- Only operational export fields are allowlisted. Downloads still contain IDs and financial
  information and must not be treated as anonymized data.
- There is no upload history, database, global upload cache, or external analytics integration.
  Existing model caching contains only the shared model, never customer uploads. Session data
  and downloads are held in memory; this is not a multi-user production privacy guarantee.

## Evaluation invariants

No changes are made to model fitting or split selection. The model comparison labels
validation and final test separately. Fixed-50% model metrics must not be confused with the
active-threshold confusion matrix. Threshold recommendation still uses validation counts only.
SHAP global importance still comes from training-only data and is not causal evidence.

## Implementation boundaries

- `src/campaign.py`: configuration, validation, allocation, totals, budget sensitivity.
- `src/cohort.py`: scoring-to-feature alignment, explicit filters, operational aggregation.
- `src/reporting.py`: bounded export fields, canonical hashes, fixed-name in-memory ZIP.
- `app/*_view.py`: presentation and session interaction, not business calculations.

Public core functions raise `ValueError` for malformed campaign assumptions and
`BatchValidationError` for invalid customer inputs. Empty filters and zero budgets are valid,
non-exceptional states. Tests exercise these boundaries without external services.
