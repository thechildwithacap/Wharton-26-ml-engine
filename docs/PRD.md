# Wharton ML Investment Engine — Product Requirements Document v2.0

**From decision-support research tool to competition-winning system.**

Status as of engine `v1.21.0`. This document is the authoritative, living
specification for the project. Sections 1–11 formalize the original design
(already built — every source file's `PRD §X` comment refers to these) and
record its **actual, empirically-measured status**, not its aspirational one.
Sections 12+ are new: they define what has to be added to turn a sound research
engine into the strongest possible entry once real competition rules are known.

**Governing principle, unchanged from day one:** every claim in this system is
backed by an out-of-sample test on real data, or it is labeled a hypothesis.
Nothing here is designed to look good in one backtest — it is designed to
survive rules, regimes, and judges we haven't seen yet.

---

## 0. Executive summary

- The engine is a **multi-layer, explainable, factor-based decision-support
  system** — not a black-box predictor and not an execution bot. Layers: data
  → signals → ML alpha → risk → construction → backtest/regime → reporting,
  plus a React UI and a competition-rules adapter.
- It is **real, not simulated only**: prices from TwelveData, fundamentals from
  SEC EDGAR, point-in-time throughout, walk-forward validated, 345-name
  universe spanning micro to mega-cap.
- Two serious data bugs were found and fixed this cycle (split-adjustment
  corrupting market cap/P·E by up to 40×; a rate-limiter silently discarding
  93% of a fetch). Both are now regression-tested. **Assume any new data path
  has a bug like this until proven otherwise** — §13 makes that a standing
  process, not a one-time fix.
- The single highest-leverage gap for actually *winning* a competition is not
  more factors — it's that **construction currently optimizes an integrated
  factor score, not the contest's scoring metric.** §14 makes closing that gap
  the top priority once rules exist.
- This document's roadmap (§16) is phased so that everything before "rules
  arrive" is useful regardless of what the rules turn out to be, and everything
  after is a short, well-defined build driven by `CompetitionRules.readiness()`.

---

## 1. Product goal (PRD §1)

Give a student-run investment team a **transparent, auditable, statistically
honest** system to select and justify a long-only US-equity portfolio, under
whatever mandate a client (or a competition) imposes — and to do so in a way
that can be explained line-by-line to a judge, not just back-tested.

**Non-goals (unchanged):** this is not a trading bot, not an execution system,
not a signal-selling product, and it does not chase black-box accuracy at the
cost of explainability. Every score the engine produces must be decomposable
into named, inspectable sub-scores.

## 2. Users (PRD §2)

- **The competition team** — runs the engine, reads the decision report, may
  override, must be able to defend every holding verbally.
- **A judge / evaluator** — reads the exported report and the UI; needs the
  *reasoning* legible without reading code.
- **A future maintainer** (this document's primary audience) — needs the
  empirical status of every component so work is not repeated or contradicted.

## 3. Architecture (PRD §3)

```
DataSource (CRSP | Synthetic | TwelveData+SEC | CSV cache | merged wide universe)
   -> DataBundle (point-in-time prices, fundamentals, benchmarks, meta)
   -> Signal layer (fundamental, quant/factor, macro-regime, hybrid/analyst)
   -> ML alpha layer (walk-forward Ridge/Logistic, 12 features)
   -> Client Fit + Risk layer (stock risk, liquidity, stress, Monte Carlo)
   -> Construction (score -> weights under PortfolioConstraints)
   -> Backtest/Regime (template backtests, style-weight recommendation)
   -> Reporting (CSV/JSON export, console summary, React UI, API server)
   -> CompetitionRules adapter (maps a rulebook onto every layer above)
```

No layer is allowed to silently assume something about the layer below it
(e.g., construction never assumes the universe is survivorship-free — it asks
`DataBundle.eligible_asof`). This is why the CRSP path could be added later
without touching construction at all, and why `CompetitionRules` can retarget
the whole stack from one dataclass.

## 4. Data (PRD §4) — status

| Component | Status | Notes |
|---|---|---|
| SEC EDGAR fundamentals (point-in-time, `filed` date gated) | ✅ Built, tested | `sec_edgar.py` |
| TwelveData / FMP real prices | ✅ Built, tested | FMP key currently dead (403); TwelveData works |
| Split-adjustment correction | ✅ **Fixed this cycle** | Was corrupting market cap up to 40× — see §13 |
| Free-tier rate-limit handling | ✅ **Fixed this cycle** | Was silently dropping 93% of a fetch — see §13 |
| CRSP/WRDS survivorship-free adapter | ✅ Built, tested offline | **Not connected to a live run** — needs real WRDS creds (§15) |
| Synthetic S&P 500 point-in-time membership + modeled bias | ✅ Built | Real removals (SIVB/FRC/SBNY etc.), modeled outcome, not measured on real prices for the live universe |
| Small/mid-cap universe (S&P 400/600) | ✅ Built, fetched | 219/228 names, merged into a 345-name wide universe |
| Index registry (SPX/NDX/DJIA/SOX/sectors/SMID) | ✅ Built | Backs the UI selector |
| Bundle merging across fetch batches | ✅ Built, tested | `data/merge.py` |
| Insider transactions (Form 4) | ✅ Built, verified against live SEC | Filing-count proxy by default; `fetch_detail=True` parses true buy/sell — expensive at scale (§13) |
| International / non-US equities | ❌ Not built | Explicit non-goal unless rules require it |

## 5. Client questionnaire & mandate (PRD §5)

✅ Built and unchanged in scope: objective, risk tolerance, horizon, sector
preferences/avoidances, ESG flag, turnover tolerance, minimum income need. Maps
to `PortfolioConstraints` via `ClientProfile.apply_to_constraints`. This layer
is competition-relevant as the **"house view"/IPS** a team can present to
judges independent of whatever the contest's own rules impose.

## 6. Signal layer (PRD §6) — status and honest findings

| Signal | Status | Empirical finding (2019–2025, this universe) |
|---|---|---|
| Value (composite: valuation/balance-sheet/cash-flow) | ✅ Built | Large-cap: significantly **wrong** (t=−1.94 @252d, pre-fix). Wide universe: improves to +t=1.24, still not significant. |
| **Intrinsic value / Graham margin-of-safety** | ✅ Built this cycle | Large-cap: only ~5 names/date pass; t=1.25. Wide universe: ~23 names/date pass, t=1.67 — usable screen, not yet significant. |
| **Piotroski F-score** | ✅ Built this cycle | Runs **backward** in this sample: F≥7 minus F≤3 = −7.4%/yr (t=−4.73), consistent with the independent `quality` factor also being significantly negative. Regime effect (2019–2025 favored weak balance sheets), not a proxy artifact — though ROIC standing in for ROA is an approximation worth revisiting (§13). |
| Quality, Growth, GARP | ✅ Built | No standalone significance found in this sample. |
| Capital discipline (net issuance / asset growth / accruals) | ✅ Built this cycle | Max correlation with any other factor 0.16 — genuinely independent — but IC ≈ flat/negative in this one-regime sample. |
| Momentum, low-vol, size, factor (price-based) | ✅ Built | `size` was **entirely a split-adjustment artifact** — t collapsed from 5.14 to 1.51 once fixed. `low_vol`/`macro_tilt` are consistently, significantly negative (defensive names lagged a melt-up) — real for this regime, not necessarily durable. |
| Macro-regime factor (`macro_fit`) | ✅ Built | Regime-conditioned by design (flips sign with risk-on/off) but the sample never saw a real risk-off stretch, so it tested flat. |
| **Earnings-event / overreaction model** | ✅ Built this cycle | Mechanically verified (Fitbit-style setup); not yet run through a full significance test on real data — flagged in §13. |
| **SEC Form 4 insider signal** | ✅ Built this cycle | Verified against live SEC data; zero open-market buys found across 4 mega-caps in the sample window — insider buying appears concentrated in smaller names, untested at scale. |
| Analyst overlay | ✅ Built | Proxy only (0.94 correlated with `quality`) — not real estimate-revision data; a known, stated gap. |
| Theme/sector tilt | ✅ Built | Cosmetic/mandate layer, not alpha-tested. |

**Standing finding, unchanged since the first factor audit:** the combined
model's out-of-sample rank-IC is real (t up to ~3.5 on large caps at 126d) but
**individual factors are mostly noise in this one regime.** Treat every new
factor as noise until it clears a walk-forward significance test — this
project's biggest risk is quietly relaxing that bar.

## 7. Machine-learning alpha model (PRD §6, ML subsystem) — status

- Ridge (regression) / Logistic (classification) over 12 profile-independent
  features, walk-forward evaluated with a purge gap, numpy-only (no black box).
- **Training horizon lengthened from 21d to 126d** — materially improved the
  large-cap model (OOS t 1.92→3.53, hit-rate 54%→71%), confirming value/quality
  factors need quarters, not weeks.
- **Model does not transfer across cap segments.** Trained and evaluated on
  the 345-name wide universe: OOS t collapses to 0.77, hit-rate to 45% — worse
  than a coin flip on direction. This is the most important open finding: **one
  model across $500M–$4T of market cap is averaging two different return
  processes.** §14 makes segment-aware modeling the top alpha-improvement item.
- Extended feature set (raw SEC line items as ranks) exists but has not been
  shown to beat the composite-score feature set out-of-sample — untested claim,
  flagged.

## 8. Risk & constraints (PRD §6.2 / §8) — status

✅ Built: stock risk (vol/beta/drawdown/tail flags), liquidity model, portfolio
risk aggregation, concentration alerts, crowding model, stress testing (crisis
drawdowns + factor-beta scenarios + crisis correlation), sector-neutralization,
turnover & cost model, mandate check, data-quality report, model-robustness
scores. All wired into `run_engine`. Constraint set: min/max holdings, position
caps, sector caps, style-weight cap, turnover cap, ADV participation cap,
leverage/shorting/options flags (currently all long-only, no leverage).

**Gap:** the risk model is historical vol/beta/correlation — a single
covariance snapshot, not a proper factor-risk model, and it has only ever been
tested in one benign-vol regime. Flagged in §13.

## 9. Backtest & regime layer (PRD §6.3 / §9) — status

✅ Built: strategy-template backtest engine, regime classification
(trend/vol-state), style-weight recommendation, signal-stability scoring,
walk-forward ML-vs-rules comparison, point-in-time delisting-aware backtest,
**measured** (not modeled) survivorship-bias backtest via CRSP fixtures.

**Gap:** every backtest to date covers the same 2019–2025 window. A system
this well-instrumented has never been tested against a real bear market or a
real rate-hiking regime. This is the single biggest validity gap in the whole
project (§15).

## 10. Reporting (PRD §7.3 / §10) — status

✅ Built: full CSV/JSON export bundle, console decision summary, React UI
(index/objective/risk selector, decision card, Monte Carlo fan, factor
heatmap, portfolio risk panel, portfolio table), stdlib API server.

**Gap:** no judge-facing narrative document is generated automatically — a
judge currently has to read JSON/CSVs or the UI. §14 adds a one-command
"audit report" generator.

## 11. Transparency & auditability (PRD §11)

✅ Every score decomposes into named sub-scores (e.g. Value → valuation +
balance_sheet + cash_flow). Every decision carries `reasons: List[str]`. Every
trained model persists its walk-forward metrics alongside its coefficients (no
opaque pickle — JSON only). This principle is **not up for negotiation** in
any future work, including any ensemble/nonlinear exploration (§17).

---

## 12. Competition-rules adapter (new, built) — status

✅ Built: `CompetitionRules` dataclass (capital, universe/index, position/sector
caps, cash, instruments, cadence, contest window→ML horizon, benchmark, scoring
metric) with `.to_engine_config()` and, critically, `.readiness()` — a classifier
that reports every rulebook lever as **enforced / partial / needs_code**, and
only emits the `needs_code` items a given ruleset actually triggers.

On a plausible Wharton-style ruleset (100k, S&P 500, weekly rebalance, 70–100%
invested, total-return scored): **11 levers enforced by config, 3 partial, 2
need code** (a cash sleeve, and true objective-aligned construction). Those two
are exactly what §14 builds next.

**This is the adapter the rest of this document assumes exists.** When real
rules arrive, the workflow is: fill in `CompetitionRules`, run `.readiness()`,
build only the flagged items, re-run the full test suite.

---

## 13. Known bugs, fixed and standing (institutional memory — do not relitigate)

| Bug | Impact | Status |
|---|---|---|
| Split-adjusted prices × as-reported share counts | Market cap/P·E off by up to 40× (NVDA mid-2021: $12.5B reported vs $498.6B true); manufactured a spurious `size` factor (t 5.14→1.51 once fixed) | ✅ Fixed, regression-tested (`test_split_adjustment_factors`) |
| Rate-limiter default (0.15s) vs free-tier cap (8/min) | A 228-name fetch silently returned 17 names and empty benchmarks — looked like "data unavailable," was actually a client bug | ✅ Fixed, regression-tested (`test_rate_limit_derives_throttle_and_records_failures`, `test_failed_symbols_are_reported_not_silently_dropped`) |
| Piotroski ROA/current-ratio proxies (ROIC for ROA, interest coverage for current ratio) | F-score may be systematically miscalibrated, not just regime-affected | ⚠️ Standing — needs either better raw fields (extend SEC adapter to pull `Assets`/`AssetsCurrent`/`LiabilitiesCurrent` directly, already partially fetched for asset_growth) or explicit documentation that it's an approximation |
| Analyst overlay is a fundamentals proxy, not real estimate data | Any "analyst" signal in reports is not independent information | ⚠️ Standing, stated everywhere it's used — needs a real data source (paid) to close |
| Earnings-event/overreaction model mechanically verified but not significance-tested on real data | Unknown whether it adds real signal | ⚠️ Standing — do before relying on it (§17) |
| Insider `fetch_detail=True` at scale is slow (per-filing XML fetch) | Not yet run across the full 345-name universe | ⚠️ Standing — needs a cached bulk-fetch pass (§14) |

**Process requirement going forward:** any new data path must ship with (a) a
sanity check against at least 2 real, independently-verifiable data points
(like the NVDA market-cap check that caught the split bug), and (b) a
regression test. Two serious bugs in one project this careful is a base rate,
not bad luck — assume a third exists somewhere until proven otherwise.

---

## 14. Phase 1 roadmap — alpha & construction (do regardless of competition rules)

These are the highest-leverage, rule-agnostic improvements. Ordered by
expected impact on actually winning.

### 14.1 Objective-aligned construction (HIGHEST PRIORITY)
**Problem:** construction maximizes the integrated factor score. A contest
scored on raw total return rewards concentration; one scored on Sharpe rewards
the diversified book the engine already builds. Right now the *scoring metric*
only nudges a concentration exponent (`CompetitionRules.to_constraints`) — it
does not drive selection.
**Build:** a constrained optimizer (still numpy-only, still explainable) that
takes the contest's scoring metric as the literal objective — e.g. maximize
expected total return subject to the existing hard constraints for
`total_return` contests; maximize expected Sharpe (using the Monte Carlo
distribution's mean/std) for `sharpe` contests; minimize tracking error to
benchmark for `vs_benchmark` contests. Every output must still decompose into
"why this weight" the way scores do today.
**Definition of done:** side-by-side backtest showing objective-aligned
construction beats plain integrated-score construction on *its own* declared
metric, out-of-sample, on at least two different historical windows.

### 14.2 Segment-aware ML models
**Problem:** one model across $500M–$4T market cap degrades to near-random
(OOS t 2.66→0.77, hit-rate 67%→45%) on the wide universe.
**Build:** train separate alpha models per cap bucket (e.g. large/mid-small),
or a single model with cap-bucket as an explicit interaction feature; compare
both against the pooled model out-of-sample before choosing.
**Definition of done:** segment-aware OOS IC/hit-rate on the wide universe
recovers to within the large-cap model's current performance, measured on data
the segmentation choice was not tuned on.

### 14.3 Cash sleeve / regime de-gross
**Problem:** construction is always ~fully invested; a competition that allows
cash, or any regime where being defensive would help, has no lever today.
**Build:** an explicit cash weight, sized by `MacroRegime.risk_on` and/or the
mandate check, subtracted proportionally from the invested book (not from any
single name), fully auditable in the decision reasons.
**Definition of done:** a synthetic bear-market backtest (§15) shows the
cash-aware construction reduces max drawdown versus always-fully-invested,
without materially hurting the bull-market backtest already in hand.

### 14.4 Better Piotroski inputs
Pull `Assets`, `AssetsCurrent`, `LiabilitiesCurrent` directly from SEC (already
partially fetched for `asset_growth`) so the F-score uses true ROA and a true
current ratio instead of the ROIC/interest-coverage proxies. Re-run the
significance test after the fix — the −7.4%/yr finding may partly be proxy
noise, not pure regime effect, and that distinction matters for how much to
trust the signal going forward.

### 14.5 Bulk insider-detail backfill
Run `fetch_detail=True` across the full 345-name universe as a one-time cached
pass (not per-request), producing a point-in-time insider net-buying panel
that plugs into the ML feature set the same way `discipline` and `macro_fit`
did. Test its IC and independence before adding it as a live feature.

### 14.6 Earnings-event significance test
Run the same walk-forward IC/t-stat test already applied to every other
factor against `overreaction` and `earnings_drift`. Do not present the
Fitbit-style narrative as validated until it clears that bar.

---

## 15. Phase 2 roadmap — validity across regimes (before trusting any of this in a live competition)

**This is the single biggest risk to the whole project.** Every backtest,
every t-stat, every "this factor works/doesn't work" conclusion in this
document was measured on one continuous 2019–2025, overwhelmingly risk-on
window. A system this well-instrumented is still fundamentally unvalidated
against a bear market.

- **15.1 Extend history where the data allows it.** Push the fetch back to
  2015 or earlier (TwelveData free tier permitting) to capture the Dec 2018
  selloff and Feb–Mar 2020 COVID crash as real, not synthetic, regimes.
- **15.2 Live-wire the CRSP path.** The survivorship-free adapter exists and is
  tested offline; it has never run against a real WRDS connection. If the
  school/team has (or can get) WRDS access, this closes the biggest remaining
  "is our backtest lying to us" risk.
- **15.3 Stress-test every "this works" claim from §6–§9 against a second
  regime** before it goes into a live competition config. A factor that only
  worked in a risk-on melt-up is not a factor — it's a description of 2019–2025.
- **15.4 Keep the synthetic-delisting demo as the control.** It remains useful
  precisely because it's a clean, understood counterfactual — don't retire it
  once real data is available; use both.

---

## 16. Phase 3 roadmap — competition operations

Turning the engine into something that runs unattended and correctly during an
actual multi-week competition window.

- **16.1 `competition_day.py`** — one script: refresh data → retrain if the
  schedule says so → run engine → Monte Carlo → paper-trade step → export →
  fail loudly (not silently) if the mandate check fails or a data-quality gate
  trips.
- **16.2 Drift monitoring.** Track the live OOS IC of the deployed model
  week-over-week during the competition; alert (don't just log) if it falls
  through a threshold — the same discipline used to catch the size-factor
  artifact should run continuously, not just at build time.
- **16.3 Judge-facing audit report.** One command that renders the existing
  decision JSON + factor scorecard + Monte Carlo fan into a clean PDF/HTML
  narrative — reuse `reporting/export.py` and the UI's rendering logic rather
  than building a new template from scratch.
- **16.4 Scheduled CI against fresh data.** A recurring job (the `/loop` or
  Routine mechanism already available in this environment) that re-runs the
  full test suite plus a live small-scale data fetch, so a silent upstream API
  change (SEC schema, TwelveData format) is caught before competition day, not
  during it.
- **16.5 UI additions:** manual override controls, a data-coverage indicator
  for the `—` cells already visible in the factor heatmap, and a live chart of
  the actual paper-trading track record next to the recommended book.

---

## 17. Phase 4 (stretch) — alpha exploration, held to the same bar as everything else

- **Ensemble/nonlinear model as a side-by-side candidate, never a silent
  swap.** A small gradient-boosted-tree or similar model may be tried, but it
  ships only if it beats the linear model out-of-sample on a window it wasn't
  tuned on, **and** it comes with a feature-importance/SHAP-style explanation
  path — the explainability requirement in §11 is not negotiable.
- **Risk-parity / Kelly-influenced sizing** as an alternative to
  score-proportional weighting, compared head-to-head, not assumed better.
- **A true factor-covariance risk model** (vs. the current historical
  vol/beta/correlation snapshot) if a second regime (§15) shows the current
  risk model is unstable across environments.
- **Real analyst-estimate data / true earnings-surprise consensus** if a paid
  feed becomes available — closes the two "proxy, not real data" gaps flagged
  in §6 and §13.

---

## 18. Success metrics — what "best" actually means here

Not "highest backtested return." Specifically:

1. **Out-of-sample IC/hit-rate on a held-out final window** that is locked
   before any further tuning happens — never touched until the last check
   before a real decision.
2. **Stability across at least two distinct regimes** (§15), not just
   2019–2025 repeated with different tickers.
3. **Every reported number ships with its honest uncertainty** — the Monte
   Carlo fan, not a point return — because a judge asking "how confident are
   you" deserves the real answer, and because that's the difference between
   this project and a spreadsheet that got lucky.
4. **A pre-mortem, explicitly written down before the competition starts:**
   what would make this entry lose, and does the current design defend against
   each failure mode (concentration risk, regime change, a single bad-data
   ticker, a rule the team misread)? If §14.1 (objective-aligned construction)
   is not done, "we optimized the wrong thing" must be listed as a live risk,
   not discovered afterward.

## 19. Explicit non-goals (unchanged, restated for this version)

- No leverage, shorting, or options unless a specific ruleset explicitly
  requires and permits them (`CompetitionRules.allow_*` stays `False` by
  default).
- No chasing a backtest number without an out-of-sample test backing it.
- No silent swap from the explainable linear scoring to a black-box model —
  §17's ensemble work is additive and provable, never a replacement made on
  vibes.
- No claim in a judge-facing report that isn't traceable to a specific,
  inspectable score or test in this codebase.

---

*This document supersedes no code — it is the map of what the code in this
repository actually does, what it has actually measured, and what has to be
built next. Update it in the same commit as any change that alters a status
line above; a PRD that drifts from the code is worse than no PRD.*
