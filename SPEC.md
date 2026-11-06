# Retention Campaign Workspace

## Objective

Extend the local churn demonstration into a reviewable workflow: score a customer cohort,
inspect its segments, allocate a hypothetical outreach budget, and download a decision bundle.
The existing single-customer explanations and validation-only threshold selection remain intact.

## Scope and boundaries

- Keep Python 3.12, pandas, scikit-learn, and Streamlit; add no dependencies or external services.
- Publication requires separate authorization. Local implementation was completed without Git
  operations; the user authorized publication as eight separate commits on 2026-11-06.
- Do not send messages, launch campaigns, or save uploaded customer records to disk.
- Keep source, documentation, tests, and interface text in English.
- Treat all monetary outcomes as illustrative scenarios, not causal or calibrated forecasts.
- Retain all existing tests and verify remote checks before integrating into the default branch.

## Capability map and build order

| Module | Responsibility | Depends on |
|---|---|---|
| campaign | Deterministic budget/capacity allocation and sensitivity scenarios | Existing batch scores |
| cohort | Identity-safe scoring, tenure bands, and segment summaries | Existing batch validation |
| reporting | In-memory CSV/JSON ZIP with provenance and assumptions | campaign, cohort |
| workspace | Cohort analysis, campaign controls, model/policy review | All above |

Build order: campaign -> cohort -> reporting -> workspace. Each boundary is a Python
DataFrame or immutable configuration; the core has no Streamlit dependency.

## Acceptance criteria

1. Campaign allocation never exceeds the monetary budget, contact capacity, or risk eligibility.
   Rank by risk or scenario net value; break ties by literal customer ID. Reject invalid inputs.
2. A zero-cost campaign is still capacity-limited. Budget arithmetic uses cents. Negative-net
   scenarios can be excluded; an empty eligible cohort produces a valid empty plan.
3. Joining scores to features is one-to-one by unique, nonblank customer ID, never by sorted
   row position. Original Kaggle columns and leading-zero IDs remain supported.
4. Segment summaries reconcile with the scored cohort. Risk-band and contract filters are
   explicit. Unlabeled uploaded records are never reported as observed churn outcomes.
5. The dashboard has separate Customer review, Campaign workspace, and Model evaluation
   tabs, meaningful empty/error states, a sample-data path, and persistent policy controls.
6. The evaluation screen distinguishes validation policy estimates from final-test metrics;
   it shows model comparisons, confusion counts, threshold trade-offs, and global importance.
7. The ZIP has fixed member names, protected spreadsheet strings, model/schema identity,
   a scored-cohort fingerprint, thresholds, complete scenario assumptions, and no raw profiles.
8. Existing tests, new unit/integration checks, and lint pass locally. Record actual runtime
   verification and remaining manual checks without claiming an unrun remote build.

## Code style and structure

Use typed, small functions, snake_case, immutable dataclasses, and existing error/export helpers.
Example: `plan = plan_campaign(cohort, CampaignConfig(budget=2500, capacity=100))`.
Pure logic belongs in `src/`; Streamlit rendering belongs in focused `app/*_view.py` files.
Tests live in `tests/`; design notes and release checks live in `docs/`.

## Commands and testing strategy

From the project directory in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider --tb=short
& .\.venv\Scripts\python.exe -B -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

Write failing behavior tests first. Cover budget boundaries, ranking, identity alignment,
empty cohorts, unsafe CSV identifiers, ZIP metadata, filters, and dashboard reruns. Preserve
the independent final-test split; no new fitting or threshold tuning uses uploaded labels.
