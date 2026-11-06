# Local implementation checklist

- [x] Campaign allocation and sensitivity scenarios: budget/capacity, deterministic ties,
  finite inputs, zero cost, and empty plans. Verify: `pytest tests/test_campaign.py`.
- [x] Cohort scoring and segments: unique identities and correct feature/score alignment.
  Verify: `pytest tests/test_cohort.py`.
- [x] Decision bundle: safe CSVs and complete JSON provenance without raw profiles.
  Verify: `pytest tests/test_reporting.py`.
- [x] Dashboard integration: preserve old flows, add sample/filters/campaign/evaluation.
  Verify: `pytest tests/test_dashboard.py` plus local runtime checks. Real-browser visual
  verification is explicitly deferred after the browser connector failed to initialize.
- [x] Final checkpoint: 82 tests, Ruff, dependency consistency, local runtime review, documentation.
  Evidence and unverified release gates are recorded in `docs/LOCAL_VERIFICATION.md`.
- [x] Publication authorization received on 2026-11-06 for eight separate commits.
- [ ] Remote checks and integration: authoritative results are recorded on the publication PR
  and its associated GitHub Actions runs. Manual visual/container gates remain separate.
