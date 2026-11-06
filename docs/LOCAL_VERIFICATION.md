# Local verification record

Date: 2026-11-05. Scope: local retention workspace preparation only; no Git or GitHub
commands or operations were performed. No dependencies or remote workflows were changed.

This is the historical local-preparation record. Publication was subsequently authorized on
2026-11-06; current remote results belong to the publication PR and its Actions runs.

## Executed checks

| Check | Observed result |
|---|---|
| Baseline pytest before implementation | 36 passed |
| Final full pytest suite | 82 passed in 18.54 seconds, no skips |
| Ruff | All checks passed |
| Dependency consistency (`pip check`) | No broken requirements found |
| Local Streamlit server health | `/_stcore/health` returned `ok` |
| Actual stored Telco model AppTest | Customer submit, sample cohort, recommended threshold, zero budget passed |
| Maximum-size end-to-end smoke | 10,000 customers scored and decision ZIP prepared |

The maximum-size smoke used the actual stored Logistic Regression pipeline and Kaggle-schema
records with unique test IDs. It selected 100 contacts for exactly $2,500 under a $2,500 budget
and 100-contact cap. Scoring, allocation, and bundle creation together took approximately
0.63 seconds in one local run; this is not a portable performance guarantee. Test data was
generated in memory, and no uploaded customer files or bundles were saved to disk.

The test-only server was bound to loopback, usage telemetry was disabled, and the server was
stopped after verification. Relaunch using the command in `docs/RELEASE_CHECKLIST.md`.

## Review findings resolved during implementation

- Nullable identifiers could become literal `None`, `nan`, or `<NA>` identifiers before
  validation. Nullable string conversion now preserves missingness until the blank check.
- Reallocating an exported plan could collide with an existing priority column. Derived
  priorities are recomputed rather than duplicated.
- Four-decimal rounding could promote a below-threshold score into campaign eligibility while
  its risk label remained Medium. Batch decision scores now retain full precision.
- OS-dependent CSV newlines made a canonical cohort fingerprint vary across platforms.
  Protected CSV exports now use fixed LF line endings and UTF-8 with BOM.

Each finding has a regression test that failed before its fix. Additional tests cover
whole-cent budgets, zero-cost capacity, deterministic ties, positive-value exclusion, missing
or duplicate identities, row alignment, segment reconciliation, empty cohorts, strict bundle
metadata, formula-like IDs, randomized allocation invariants, and dashboard reruns.

## Review conclusions and remaining gates

The calculations remain separate from Streamlit rendering; all new core modules are local,
side-effect-free, and use existing dependencies. Allocations are bounded by the existing
10,000-row upload limit. Exports have fixed names and allowlisted fields. Model fitting,
validation-only model selection, and the independent final-test boundary are unchanged.

Actual browser visual, keyboard, console, and responsive verification is **not completed**.
The available browser connector failed initialization with a timeout; Chrome DevTools tools
were not configured. Streamlit AppTest exercises Python-side interaction, not rendered CSS
or browser accessibility. Docker was unavailable, so no container build/run is claimed.

Fresh dependency installation, a current advisory audit, human review, remote checks, and
publication remain deferred. The completed local checks do not stand in for those gates.
Follow `docs/RELEASE_CHECKLIST.md` before publication.
