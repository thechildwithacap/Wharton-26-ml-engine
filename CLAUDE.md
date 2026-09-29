# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repo.

## What this is
A multi-layer, explainable ML investment-decision-support engine for the
Wharton Global High School Investment Competition. See `docs/PRD.md` (v2,
architecture map) and `docs/PRD_v3_ML.md` (v3, ML-layer honesty/strength/
explainability workstreams) for the full spec and current status.

## Before you touch code
1. `pip install -r requirements.txt && pip install pytest` (fresh containers
   have nothing installed — check `python -c "import numpy"` first).
2. `PYTHONPATH=. python -m pytest -q` — note the pass count. If it doesn't
   match `docs/PRD_v3_ML.md` §3's status table, stop and reconcile before
   changing anything.
3. Read the relevant section of `docs/PRD_v3_ML.md` before starting a
   workstream — it records what's already been tried and why (§ "Institutional
   memory"), so you don't relitigate settled findings.

## Non-negotiable rules
- **No look-ahead.** Every signal/feature reads only data available as-of its
  decision date (`fundamentals_asof`, `prices_upto`, `eligible_asof`).
- **No number reported as a result unless it's out-of-sample, walk-forward,
  point-in-time, with an honest test.** Label anything else `EXPLORATORY`.
- **Explainability is not optional.** Any model that ships must decompose its
  score into named, inspectable contributions per stock.
- **No black-box swap without a side-by-side, out-of-sample comparison** against
  the current linear model, logged in the experiment registry.
- **Every new data path needs a real-world sanity check + a regression test**
  before it's trusted. This project has already found a 40x split-adjustment
  bug and a rate limiter silently dropping 93% of a fetch — assume a third
  exists until proven otherwise.
- Free data sources only unless the user supplies a key: SEC EDGAR (set
  `SEC_USER_AGENT`), TwelveData/FMP (need a key), Yahoo-style feeds. No paid
  APIs, no scraping sites that block bots.
- Update the relevant PRD status table in the same commit as any change that
  moves a number.

## Commands
```bash
export SEC_USER_AGENT="you@example.com"
PYTHONPATH=. python -m pytest -q                 # full suite
PYTHONPATH=. python -m pytest -q -m "not slow"    # fast subset
python -m wharton_ml_engine.api.server --port 8000  # UI + API (build ui/ first)
```
