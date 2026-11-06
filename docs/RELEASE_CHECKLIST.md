# Local review and publication checklist

The user authorized publication on 2026-11-06 as eight separate commits: four existing repairs
plus four workspace commits. Preserve each commit and verify remote checks before integration.
This checklist distinguishes completed automated checks from remaining manual release checks.

## Reproducible local checks

From the project directory in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider --tb=short
& .\.venv\Scripts\python.exe -B -m ruff check . --no-cache
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

## Manual walkthrough before publication

- [ ] Customer review: submit a profile, inspect factors, then change the sidebar threshold;
  the submitted profile and prediction remain visible and the risk tier updates.
- [ ] Campaign workspace: choose the sample, check three scored customers, then upload the
  full Kaggle CSV. Confirm original IDs remain attached to the correct charges and contracts.
- [ ] Clear every risk filter: matching customers and planned contacts must both be zero.
- [ ] Set budget to zero, then cost to zero with capacity one: verify the respective zero-contact
  and capacity-limited states. Try a non-positive-value exclusion with success rate zero.
- [ ] Apply the recommended policy: campaign eligibility and evaluation confusion counts update.
- [ ] Download and open the ZIP. Compare manifest settings/totals with the displayed plan.
  Verify leading-zero and formula-like IDs are preserved safely when opened as text.
- [ ] Model evaluation: distinguish fixed-50% model metrics, active-policy test counts, and
  validation-only threshold costs. Do not report sample-upload scores as measured outcomes.
- [ ] Check layout and keyboard navigation at narrow and desktop viewport sizes. Check browser
  console and network errors. Automated AppTest is not a visual/accessibility audit.
- [ ] Remove temporary downloaded customer files from any later publication selection.

## Deferred release gates

- [ ] Human review of the local changes and documentation.
- [ ] Fresh-environment dependency install and current advisory audit.
- [ ] Docker build/run check on a machine with Docker installed.
- [x] Decide logical commit grouping and obtain publication authorization (2026-11-06).
- [ ] Run and inspect remote checks after publication; no remote success is claimed locally.
