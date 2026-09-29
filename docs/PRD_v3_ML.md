# PRD v3 (adapted): Make the ML honest, stronger and explainable

**Status as of:** commit at time of writing, branch `claude/wharton-ml-investment-engine-59ldve`.
**Relationship to the pasted PRD v3:** the version handed to this session assumed a
repo state — branch `drive-sync-refine`, a `firm/`/`desk/`/`goals/` package
hierarchy, `ML_AUDIT.md`, `tools/ml_v2.py`/`tools/ml_audit.py`, a
`datasets/sp500_real/` (503 names, 2012–2026), a trained
`models/alpha_model_sp500_pit_h126.json`, 230 passing tests, and a "Laura Gao /
Wharton WInS 2026-27" client case — **none of which exist in this repository.**
That was verified directly (`git branch -a`, `git cat-file -t <commit>`, `ls`)
before any code was touched, and the user confirmed treating it as aspirational
rather than chasing a different repo. This document is the honest reconciliation:
what of that PRD's *ML-layer* ambition (its actual title) was built here, against
what actually exists, and what remains explicitly out of scope.

**Supersedes:** nothing. `docs/PRD.md` (v2) is still the architecture map.

---

## 0. What this pass covered, and what it deliberately did not

**Built, tested, real** (this pass): W0 (repo hygiene), W1 (feature registry),
W2 (partial — point-in-time membership mechanism), W3 (validation protocol:
Newey-West t-stats, Sidak multiple-testing correction, locked holdout,
experiment registry, IC breakdowns), W5 (interpretability: local
decompositions, permutation importance, coefficient stability, plain-English
narration, model cards), and a scoped slice of W6 (`MLView`, consumed by the
engine's evidence gate and ready for the API/UI layer).

**Explicitly not built, and why:**

- **No `firm/`/`desk/`/`goals/` hierarchy, no Laura Gao case, no WInS-specific
  mechanics.** That's a large, separate case-study system (org hierarchy
  simulation, liability-driven-investing desk, a specific client's Treasury
  ladder) that was never part of this repository. Building it wasn't
  requested once the state mismatch was confirmed — this pass stayed inside
  the actual codebase's scope: the engine's own construction/risk/reporting
  layers, none of which resemble the fictional `firm/desks.py`.
- **No LightGBM / tree-ensemble path.** This codebase trains only linear
  models (Ridge/Logistic — see `ml/models.py`); LightGBM isn't a dependency
  here (added as an *optional* extra in `pyproject.toml` for future use, per
  the "only add if optional and core still runs on numpy+pandas" rule, but
  nothing imports it). `ml/explain.py`'s decomposition path only covers linear
  models honestly — adding a tree ensemble without its own exact-decomposition
  path would violate the explainability requirement, so it wasn't added.
- **No `tools/ml_audit.py` / `tools/ml_v2.py` CLI wrappers.** The underlying
  mechanisms (Newey-West, the experiment registry, `final_check`,
  permutation importance) are built as a proper library API in
  `wharton_ml_engine/ml/`, callable directly or from a thin script — but no
  polished CLI/report-generation tool was built standing in for the fictional
  ones, to avoid manufacturing the appearance of an audit history that hasn't
  actually run.
- **No real point-in-time S&P-500 add/remove dataset.** W2's mechanism
  (`build_training_panel(point_in_time=True)`) is built and tested against
  synthetic data with controlled listing dates — it is *correct*, but it needs
  a real "date joined the index" dataset (not just IPO/delisting dates) to be
  meaningful against live data. Fetching and verifying that (e.g. from
  Wikipedia's S&P 500 changes table) is a real data-acquisition task with its
  own verification burden (see the split-adjustment and rate-limiter bugs
  already found in this project) and was not attempted without being able to
  spot-check it against real numbers in this session.
- **No FRED macro integration.** The existing `signals/macro.py` regime
  classifier (trend/vol/drawdown/risk-on, derived from the SPX/NDX benchmarks
  already in the bundle) was reused for `ic_breakdown_by_regime` instead —
  real, already-tested, no new external dependency.
- **No live monitoring (W7), no `datasets/sp500_real` (503 names,
  2012–2026).** This container has no materialized dataset at all (gitignored
  `datasets/`/`models/` don't survive a container reset); the mechanisms above
  are proven on `SyntheticDataSource` and are ready to run against whatever
  real dataset is fetched (see `README.md`'s fetch instructions).
- **No champion/challenger persistence (`models/champion.json`,
  `promotions.jsonl`).** The building blocks (`log_experiment`, `final_check`,
  `count_trials`) are all there; wiring a promotion policy on top is a short
  follow-up once there's an actual second model to compare against a first.

---

## 1. What was actually built

### `wharton_ml_engine/ml/features.py` (W1)
`FeatureSpec` (name, family, description, expected_sign, rationale, source,
lag_days, is_proxy) for all **28** features that exist in this codebase's
`ML_FEATURES` (12) + `RAW_FUNDAMENTAL_FEATURES` (16) — not 48; the extra 16
price/macro features a prior draft assumed (`rev_1m`, `beta_mkt`, FRED-based
macro betas, …) don't exist here. The family taxonomy includes reserved
families (`price`-equivalent momentum/risk families already cover the
existing price factors) so a future feature set wouldn't need a schema
change. `feature_columns()` now accepts `"base"`/`"extended"` strings as well
as the original bool. 10 tests (`test_features_registry.py`).

### `wharton_ml_engine/ml/dataset.py` (W2, partial)
`build_training_panel(point_in_time=True, universe_filter=...)` restricts
each rebalance date's cross-section (both feature ranks and the forward-return
label pool) to `bundle.eligible_asof(d)` — closing the mechanism behind gap G3
(a future index-joiner's presence distorting earlier dates' ranks). Default
behavior (`point_in_time=False`) is byte-for-byte unchanged, so nothing
existing broke. 4 tests (`test_point_in_time_panel.py`), including one that
directly demonstrates a future joiner's presence changes another name's
ranks before it should be able to.

### `wharton_ml_engine/ml/validation.py` (W3)
- `newey_west_t` / `ic_summary_nw` — Bartlett-kernel HAC standard errors,
  stdlib+numpy only (no scipy — not installed here, and the project is
  dependency-light by design). Reduces to the naive one-sample t-test at
  `lags=0`.
- `nw_lags_for_horizon` — `ceil(horizon/rebalance) - 1`.
- `adjusted_significance` — Sidak correction via an erf-based normal CDF and
  a bisection inverse-CDF (no scipy dependency). `HARVEY_LIU_ZHU_T_BAR = 3.0`
  kept as a reference bar, never the gate itself.
- `split_research_holdout` / `research_only_panel` / `holdout_only_panel` —
  either an explicit `holdout_start` date or the last `holdout_fraction`
  (default 15%) of a dataset's own rebalance dates. **Deliberately not** the
  literal fixed calendar date a prior draft assumed (`2024-10-01`), since that
  only makes sense pinned against a specific loaded dataset's real calendar;
  the fraction-based default is correct against any dataset (synthetic or
  real) this mechanism is pointed at.
- `log_experiment` / `read_registry` / `count_trials` — append-only JSONL
  registry at `reports/experiments/registry.jsonl` (git commit + timestamp +
  spec + metrics per line, never skips a "failed" variant).
- `final_check` — trains on the research split only, scores the locked
  holdout exactly once per spec hash, refuses a silent second run (prints a
  loud warning on `force_recheck=True` instead of overwriting).
- `ic_breakdown_by_year` / `ic_breakdown_by_group` / `ic_breakdown_by_regime`
  — the last reuses the existing, already-tested `signals/macro.py` regime
  classifier rather than a new external macro feed.
- `walk_forward_evaluate` gained `horizon_days`, `research_only` (default
  `False` — opt-in, so no existing caller's behavior changed),
  `holdout_start`/`holdout_fraction`, and `return_models` (exposes each
  fold's fitted `(scaler, model, test_dates)` for genuine OOS permutation
  importance). 20 tests (`test_validation.py`).

### `wharton_ml_engine/ml/predict.py` — the evidence gate (W3)
**A real gap this closed, not a hypothetical one.** Before this change,
`engine/integrate.py` gave `ml_alpha` a fixed 22% blend weight whenever *any*
trained model was supplied to `run_engine` — with zero check on whether that
model's out-of-sample evidence actually supported it. `AlphaModel.confidence()`
now computes a Sidak-corrected, evidence-based confidence in [0, 1] from the
model's own walk-forward metrics (0 whenever mean IC ≤ 0 or the evidence is
missing/weak); `AlphaModel.score(gate=True)` (the new default) shrinks the
score toward the neutral midpoint (50) by `1 - confidence`, so a weak model's
fixed blend weight stops moving the portfolio's rankings even though the
weight itself is unchanged. `gate=False` recovers the raw score for
`explain.py`, which must decompose what the model actually predicted, not a
confidence-shrunk proxy for it. 10 tests (`test_ml_gate.py`), including a
regression guard that a weak model changes the recommended portfolio's top
holdings by less than 30% overlap-loss versus running with no model at all.

### `wharton_ml_engine/ml/explain.py` (W5 — core)
- `explain()` — exact linear decomposition (`coef_j * z_ij` + intercept =
  raw score) for both Ridge and Logistic models (the only two this codebase
  trains). Verified to 1e-6 against the model's own `raw_predict` — for
  Logistic, decomposition happens in logit space and `sigmoid(sum) ==
  predict_proba` is separately verified.
- `coefficient_stability()` — sign-stability of each feature's coefficient
  across walk-forward refits, using `walk_forward_evaluate(return_models=True)`.
- `permutation_importance()` — shuffles a feature (or whole family) *within
  each fold's own held-out test dates*, using that fold's already-fitted
  model — genuinely out-of-sample, never the final model scored on data it
  was fit on.
- `single_feature_ic_table()` — standalone IC per feature, cross-checked
  against the registry's `expected_sign`, flagging disagreement.
- `global_explain()` — bundles all of the above plus `ic_breakdown_by_year`
  and (given a bundle) `ic_breakdown_by_regime`.
10 tests (`test_explain.py`).

### `wharton_ml_engine/ml/narrate.py` (W5.4)
Template-filled plain-English reasons, no LLM calls. Every clause traces to a
number `explain()` actually computed (a percentile rank by default); when the
caller supplies real point-in-time fundamentals, a small set of well-
understood fields (net share issuance, asset growth, P/E, P/B, ROE, dividend
yield) get their literal raw value quoted instead of just a percentile — never
fabricated for a field without a mapped formatter. House style enforced by
`check_style()`: no em dashes, no banned words (delve/tapestry/robust/pivot/
seamless). 10 tests (`test_narrate.py`), including one that scans every
generated reason across a real synthetic universe for style violations.

### `wharton_ml_engine/ml/card.py` (W5.3)
`build_model_card()` reads a model's *actual* metrics/metadata (never hand-
typed) into a `ModelCard` with research metrics, confidence, holdout status
(unchecked by default, filled in when a `HoldoutResult` is passed), known
limitations (auto-derived: proxy features present, gate status, non-positive
IC, the standing multi-regime-validation caveat), and intended use.
`write_model_card()` emits `<path>.card.json` + `<path>.card.md`. 7 tests
(`test_card.py`).

### `wharton_ml_engine/ml/view.py` (W6, scoped slice)
`MLView` — one object (score, raw_score, confidence, explanation, reasons,
card) built by `load_ml_view(model_path, bundle, as_of)`. This is the
integration point PRD v3's W6 asked for, scoped to what actually exists in
this codebase: it's ready for the `api`/`reporting` layers to render directly
(`MLView.to_dict()` is JSON-safe), rather than wired into a firm hierarchy
that isn't there. 4 tests (`test_ml_view.py`).

### Repo hygiene (W0)
- `pyproject.toml`: `[tool.setuptools.packages.find] include = ["wharton_ml_engine*"]`
  replaces an explicit package list that had silently missed
  `wharton_ml_engine.api` (added after the list was written) — a real,
  already-broken non-editable install, not a hypothetical one. `lightgbm` is
  now a real but unused optional extra (`pip install .[ml]`). Registered a
  `slow` pytest marker (none of this pass's new tests take long enough to
  need it — the full suite is ~3.5 min, not the ~4 min baseline plus more).
- No duplicate Monte Carlo modules exist here (only `risk/montecarlo.py`) —
  that gap doesn't apply to this repo.
- Added `CLAUDE.md` — the missing onboarding doc for an agent working in this
  repo, pointing at this file and `docs/PRD.md`.

---

## 2. Test count

Baseline before this pass: **179 passed** (fresh container — dependencies had
to be reinstalled first; this matched the last known-good state exactly, not
the pasted PRD's claimed 230).

After this pass: **254 passed, 0 failed, ~3.5 minutes**, all in the same
container, same command (`PYTHONPATH=. python -m pytest -q`). 75 new tests
across 10 new files, plus edits to `ml/train.py`/`ml/predict.py`/`ml/dataset.py`
that changed zero existing test outcomes (verified by running the full
existing suite after each structural change, not just at the end).

---

## 3. Status table

| Workstream | Status | Headline |
|---|---|---|
| W0 Repo hygiene | **Done** (adapted scope) | Fixed a real missing-package bug; added CLAUDE.md, `slow` marker, `[ml]` extra |
| W1 Feature registry | **Done** (28 features, not 48) | Every `ML_FEATURES`/`EXTENDED_FEATURES` column has a spec; registry completeness enforced by test |
| W2 Data honesty | **Partial** | Point-in-time membership *mechanism* built & tested (synthetic); real S&P-500 add-date dataset not sourced |
| W3 Validation protocol | **Done** | NW t-stats, Sidak correction, locked holdout, experiment registry, `final_check`, IC breakdowns, evidence gate — all real, all tested |
| W4 Model zoo + champion/challenger | **Not started** | Only Ridge/Logistic exist; no LightGBM, no `models/champion.json` promotion flow |
| W5 Interpretability | **Done** (linear models only) | Exact local decomposition, permutation importance on genuine OOS folds, narration with enforced house style, model cards |
| W6 Integration | **Partial, scoped** | `MLView` built and tested; no firm/desk hierarchy exists to integrate into (confirmed absent) |
| W7 Live monitoring | **Not started** | No live dataset/trading window in this context |
| W8 Docs + judge pack | **This document** | `docs/PRD_v3_ML.md` (this file); `docs/PRD.md` (v2) still the architecture map |

---

## 4. Institutional memory — real findings from this project (don't relitigate)

These are verified in this session's history (see `docs/PRD.md` v2 §6/§13 for
the full detail), not carried over from the mismatched PRD draft:

- **A stock-split data bug inflated market cap/P·E by up to 40×** (NVDA
  mid-2021: reported $12.5B vs. true ~$498.6B) by multiplying split-adjusted
  prices against as-reported (pre-split) share counts. Fixed in
  `data/sec_edgar.py::split_adjustment_factors`, regression-tested.
- **The `size` factor was almost entirely that bug.** Its t-stat collapsed
  from 5.14 to 1.51 once the split fix landed — the clearest concrete example
  in this project of why every claim needs an out-of-sample, bug-checked test
  before being trusted.
- **A rate-limiter default (0.15s) against TwelveData's 8/min free cap
  silently dropped 93% of a fetch** (228 requested names → 17 returned, empty
  benchmarks) with no error surfaced. Fixed in `data/web_source.py`
  (`requests_per_minute`, retry-with-backoff, `.failed` list), regression-tested.
- **Net share issuance (buybacks vs. dilution) is the one feature with the
  most consistent standalone evidence** across this project's factor audits —
  flagged in its `FeatureSpec.rationale` so future work notices if a result
  contradicts its sign.
- **The analyst-overlay proxy is 0.94 correlated with `quality`** — not
  independent information. Flagged via `FeatureSpec.is_proxy` and surfaced in
  every model card's known-limitations section automatically.
- **Value and Piotroski-style quality signals ran backward in the 2019–2025
  large-cap sample** this project measured (see `docs/PRD.md` §6) — a likely
  regime effect (defensive/value names lagged a melt-up), not necessarily a
  durable factor sign. Any future result on real data should be checked
  against this baseline before being reported as new.
- **Everything measured on real data in this project so far covers one
  regime.** This is the standing, unresolved risk flagged in `docs/PRD.md`
  §15 and repeated here because it applies just as much to the ML-layer work
  in this document: a locked holdout inside one regime is real discipline,
  but it is not the same as validation across regimes.

---

## 5. Next steps (in priority order, if this continues)

1. **Wire `MLView` into the API layer** (`wharton_ml_engine/api/service.py`)
   so the existing React UI can render confidence, reasons, and the gate
   status per stock — the consuming surface that actually exists here,
   unlike the fictional firm-hierarchy target.
2. **Source a real point-in-time S&P-500 constituent-change dataset** (with
   the same verify-before-trust discipline that caught the split and
   rate-limit bugs) to make W2's `point_in_time=True` meaningful against real
   data, not just the synthetic proof.
3. **Fetch a real dataset and run `final_check` once** on the current shipped
   model, logging a real (not synthetic) holdout result into
   `reports/experiments/holdout_log.jsonl`.
4. **A genuine second model family** (even a second linear variant — e.g.
   per-cap-segment Ridge, which `docs/PRD.md` §14.2 already flagged as the
   highest-leverage alpha improvement) to make `count_trials`/Sidak correction
   and champion/challenger promotion (W4) meaningful with more than one
   logged variant.
5. **LightGBM only if and when W5's decomposition path is extended to cover
   it** (`pred_contrib`) — never ship the model ahead of its explanation.
