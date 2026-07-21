# Wharton ML Investment Engine

A **multi-model machine-learning investment engine** for the **Wharton Global
High School Investment Competition**. It is an *internal decision-support
system* — not a trading bot — that analyses the allowed US equity universe
across multiple investment styles, applies strict risk and mandate controls,
adapts its style weights to the prevailing market regime via daily backtests,
and produces **transparent, auditable** portfolio recommendations for the
Wharton Investment Simulator (WInS).

> **Ethics & integrity.** This is a tool the student team builds and operates
> itself. Every number is reproducible from stored code or documented formulas.
> Outputs are probabilistic and risk-aware — there are no guaranteed returns.

---

## Why this design

Most teams pick stocks ad-hoc. This engine instead runs a disciplined,
repeatable process organised into three layers plus an integration engine, each
mapping directly to the PRD:

```
                 ┌─────────────────────────────────────────────┐
   data ▶        │  DATA LAYER  (wharton_ml_engine/data)        │  §4
                 │  prices · fundamentals · benchmarks · meta   │
                 └─────────────────────────────────────────────┘
                 ┌───────────────┬───────────────┬─────────────┐
  signals ▶      │ SIGNAL LAYER (wharton_ml_engine/signals)     │  §6.1
                 │ fundamental   │ quant factor  │ hybrid/theme │
                 │ value/quality │ momentum/size │ analyst/alpha│
                 │ growth/GARP   │ low-vol/macro │ theme fit    │
                 │ income        │               │              │
                 └───────────────┴───────────────┴─────────────┘
                 ┌─────────────────────────────────────────────┐
  risk ▶         │ RISK & CONSTRAINTS (wharton_ml_engine/risk)  │  §6.2
                 │ stock risk · liquidity · concentration ·     │
                 │ crowding · turnover/cost · robustness ·      │
                 │ data-quality · compliance/mandate            │
                 └─────────────────────────────────────────────┘
                 ┌─────────────────────────────────────────────┐
  backtest ▶     │ BACKTEST & REGIME (wharton_ml_engine/backtest)│ §6.3
                 │ strategy templates · regime classification · │
                 │ style-weight recommendation · signal stability│
                 └─────────────────────────────────────────────┘
                 ┌─────────────────────────────────────────────┐
  ml ▶           │ ML LAYER (wharton_ml_engine/ml)              │  §6
                 │ learns to combine signals into an alpha score│
                 │ walk-forward CV · Information Coefficient    │
                 └─────────────────────────────────────────────┘
                 ┌─────────────────────────────────────────────┐
  engine ▶       │ INTEGRATION ENGINE (wharton_ml_engine/engine)│  §7
                 │ integrated score · construction · trade logic│
                 │ (blends in ml_alpha when a model is trained) │
                 └─────────────────────────────────────────────┘
                 ┌─────────────────────────────────────────────┐
  reporting ▶    │ REPORTING (wharton_ml_engine/reporting)      │ §7.3 §10
                 │ CSV exports for Excel/Sheets · IPS summaries │
                 └─────────────────────────────────────────────┘
```

---

## Quick start

```bash
pip install -r requirements.txt      # numpy + pandas

python run_demo.py                   # runs the whole pipeline, offline
```

The demo builds a deterministic synthetic universe (no API keys needed), makes
an initial portfolio, then re-decides six months later — showing the
rebalance-vs-hold logic — and exports CSVs to `engine_output/`.

Minimal programmatic use:

```python
from wharton_ml_engine import SyntheticDataSource, sample_profile, run_engine
from wharton_ml_engine.reporting import print_summary, export_report

bundle  = SyntheticDataSource(n_tickers=60, years=8).load()
profile = sample_profile()                      # from the questionnaire

report = run_engine(bundle, profile)            # one full decision run
print_summary(report)
export_report(report, "engine_output")
```

---

## The client questionnaire (PRD §5)

The fictional client is captured by a **20+ question structured questionnaire**
(`wharton_ml_engine/client/questionnaire.py`). Edit the answers each competition
year:

```python
from wharton_ml_engine.client import QUESTIONNAIRE, profile_from_answers

answers = {
    "objective": "balanced", "risk_tolerance": 3, "horizon": "long",
    "sector_avoidances": ["Energy"], "esg_required": True,
    "min_dividend_yield": 0.012, "turnover_tolerance": "medium",
    # ... see QUESTIONNAIRE for every field
}
profile = profile_from_answers(answers)
```

The profile drives:

* **Hard rules** — excluded sectors/themes, position/sector caps, turnover cap.
* **Client Fit Score (0–100)** — per-stock alignment with objectives and risk.
* **Style tilts** — nudges the regime style weights toward client preferences.

---

## Constraints enforced (PRD §8)

| Constraint | Default | Where |
|---|---|---|
| Holdings | 20–30 | `config.PortfolioConstraints` |
| Max per stock | 8% (target ≤5%) | construction water-fill |
| Max per sector | 25% (soft 20%) | construction + mandate |
| Factor diversification | no single style dominates | crowding model |
| Turnover per rebalance | ≤35% (tightened by client) | trade logic |
| Liquidity | position ≤10% of ADV | liquidity model |
| No leverage / shorting | enforced | mandate check |

Conservative clients automatically get tighter caps; the mandate check is a hard
gate that can veto any candidate book.

---

## Real data: financialdatasets.ai (PRD §4)

The engine ships with a live adapter, **`FinancialDatasetSource`**, that pulls
daily prices, historical financial metrics and company facts from
[financialdatasets.ai](https://financialdatasets.ai) and assembles them into a
`DataBundle` — identical in structure to the synthetic one, so every downstream
layer (signals, risk, backtest, **ML training**) works unchanged.

```bash
export FINANCIAL_DATASETS_API_KEY=sk-...

# 1) Fetch a real US-equity universe and cache it to CSVs
python fetch_dataset.py --tickers AAPL,MSFT,NVDA,JPM,XOM,UNH,PG --years 6 \
                        --out datasets/us_sample

# 2) Train the ML alpha model on the real dataset
python train_model.py --source datasets/us_sample

# 3) Run the engine on it
python -c "from wharton_ml_engine.data import CSVDataSource; \
           from wharton_ml_engine import run_engine, sample_profile; \
           from wharton_ml_engine.reporting import print_summary; \
           print_summary(run_engine(CSVDataSource('datasets/us_sample').load(), sample_profile()))"
```

Key properties of the adapter:

* **In-process fetch** — data is pulled by the Python process (not routed
  through any tool/chat context), so multi-year histories are no problem.
* **Point-in-time** — fundamentals are dated at `report_period + 45 days` to
  approximate public availability, and read via `fundamentals_asof` — no
  look-ahead. Earnings volatility / growth stability are derived from the
  *expanding* history available at each report date.
* **Robust field mapping** — API metric names map to the engine's fundamental
  fields; missing fields (e.g. `accruals`) degrade gracefully instead of
  nuking a signal.
* **Fully tested offline** — the HTTP layer is a single injectable
  `fetch_json`, so the whole assembly path is unit-tested with fixture payloads
  shaped exactly like the real API (`tests/test_financial_dataset.py`).

> Live pulls require an API key **and** account credits. Without them the
> adapter raises a clear error; use `SyntheticDataSource` or a cached dataset
> in the meantime.

### Any `DataSource` works

`FinancialDatasetSource`, `CSVDataSource`, and `SyntheticDataSource` all produce
a `DataBundle`. To wire up a different provider (a broker export, the Wharton
approved-list file, another API), implement one method:

```python
from wharton_ml_engine.data import DataSource, DataBundle, SecurityMeta

class MyApprovedUniverseSource(DataSource):
    def load(self) -> DataBundle:
        prices       = ...   # DataFrame: index=dates, cols=tickers (adjusted close)
        benchmarks   = ...   # DataFrame: 'SPX','NDX', sector ETFs
        fundamentals = ...   # MultiIndex (date, ticker) point-in-time panel
        meta         = ...   # {ticker: SecurityMeta(...)}
        return DataBundle(prices, benchmarks, fundamentals, meta)
```

Once you have a broad-universe bundle, restrict it to the approved list:

```python
approved = ["AAPL", "MSFT", ...]              # from Wharton
bundle = bundle.restrict_universe(approved)
```

Point-in-time integrity is built in: `fundamentals_asof(date)` and
`prices_upto(date)` never look ahead of the decision date.

---

## What a run produces (PRD §7.3, §10)

`export_report(report, outdir)` writes flat CSVs for the Excel / Google Sheets
front-end, one set per decision date:

| File | Contents |
|---|---|
| `scores_*.csv` | integrated score + component scores, ranked |
| `signals_*.csv` | every style/factor/hybrid signal per stock |
| `client_fit_*.csv` | fit score + hard exclusions |
| `stock_risk_*.csv` | vol, beta, drawdown, risk band, tail flags |
| `liquidity_*.csv` | ADV, liquidity score, max safe trade |
| `portfolio_*.csv` | proposed weights + sector + score |
| `trades_*.csv` | add / increase / trim / exit with deltas |
| `portfolio_risk_*.csv` | vol, beta, concentration, sector exposure |
| `style_weights_*.csv` | regime-driven recommended style weights |
| `templates_summary_*.csv` | backtest metrics per strategy template |
| `decision_*.json` | full auditable decision summary + reasons |

---

## Backtesting & regime adaptation (PRD §6.3, §9)

Six strategy templates (deep value, quality compounder, GARP blend, momentum
tilt, defensive low-vol, hybrid multi-factor) are backtested on a rolling
window, tracking return, volatility, Sharpe, max drawdown and worst year. The
regime classifier labels the environment (bull/bear/sideways × low/high vol)
from index trend, volatility and breadth, and the style-weight recommender
blends regime priors with the best-performing template and the client's tilts.
Signal-stability checks translate day-to-day rank noise into a **Signal
Confidence** that makes the engine trade more cautiously when signals are shaky.

---

## Machine-learning alpha model (PRD §6)

The signal scores above are hand-built. The ML layer *learns from history* how
to combine them into a single expected-return ranking — the **`ml_alpha`**
signal — and folds it into the integrated score.

```bash
python train_model.py                 # train, evaluate, save to models/
python run_demo.py --with-ml          # train + run the engine using ml_alpha
```

```python
from wharton_ml_engine import SyntheticDataSource, run_engine, sample_profile
from wharton_ml_engine.ml import train_alpha_model, AlphaModel

bundle = SyntheticDataSource().load()
trained = train_alpha_model(bundle, task="regression")   # or "classification"
print(trained.metrics)          # out-of-sample IC, t-stat, hit rate
print(trained.coefficients())   # what the model learned to weight (interpretable!)
trained.save("models/alpha_model.json")

report = run_engine(bundle, sample_profile(), alpha_model=AlphaModel(trained))
```

**How it's built (and why it's trustworthy):**

* **Features** — the profile-independent signal scores (value, quality, growth,
  GARP, income, momentum, low-vol, size, factor, macro-tilt, analyst), already
  cross-sectionally normalised to 0–100.
* **Label** — the *forward* return over a horizon (default 21 trading days),
  demeaned across the universe so the model learns **relative selection**, not
  market direction.
* **No look-ahead** — features at date `t` use only data up to `t`; labels use
  the realised `t → t+h` window; dates without a full forward window are dropped.
* **Walk-forward evaluation** — expanding-window, out-of-sample, with a purge
  gap. The headline metric is the **Information Coefficient (IC)** — the rank
  correlation between the model's score and realised returns — plus its t-stat
  and hit rate. This is the honest way to judge a stock-selection signal (R² is
  near zero for return prediction and is *not* the right yardstick).
* **Interpretable, dependency-light** — ridge / logistic regression implemented
  in numpy (no scikit-learn). You can read off exactly what the model weights,
  which is ideal for the IPS write-up. Models persist as plain JSON (no pickle).

> On the deterministic synthetic data the IC is small (and sometimes not
> statistically significant) — the tooling reports this honestly and treats
> `ml_alpha` as *one input among many*, never a standalone strategy. On real
> data with genuine cross-sectional structure, the same pipeline is where the
> learning happens.

## Project layout

```
wharton_ml_engine/
  config.py            constraints, cost model, styles, engine config
  utils.py             cross-sectional scoring & performance helpers
  data/                DataBundle + sources: Synthetic, FinancialDataset, CSV
  client/              questionnaire, profile, Client Fit
  signals/             fundamental, quant, hybrid pods
  risk/                stock, portfolio, operational/mandate checks
  backtest/            templates, regime, style weights, stability
  ml/                  dataset, models, train (walk-forward), predict, metrics
  engine/              integrate, construct, trade, pipeline
  reporting/           CSV export + console summary
run_demo.py            end-to-end demonstration (--with-ml to include the model)
train_model.py         train / evaluate / save the ML alpha model (--source for real data)
fetch_dataset.py       download & cache a real universe from financialdatasets.ai
tests/                 pytest suite (incl. sub-model & sub-sub-model tests)
```

Run the tests:

```bash
pip install pytest && pytest -q
```

---

## Scope & disclaimers

**In scope:** daily equity screening, multi-style scoring, risk/mandate
controls, template backtests, regime-aware style weighting, portfolio
construction, and reporting.

**Out of scope:** intraday/high-frequency trading, options, leverage, or
shorting (disabled by default and blocked by the mandate check). Outputs support
human decisions — the team makes the final call and enters trades in WInS.
