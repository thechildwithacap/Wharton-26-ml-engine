"""Deterministic synthetic market generator.

Produces a coherent offline universe so the entire engine runs without any
external API.  The generator is intentionally *structured*: each stock is
assigned latent "style traits" (value-ness, quality, growth, income) and both
its fundamentals and its return stream are derived from those traits.  That
gives the signal models something real to find and lets backtests differentiate
strategy templates, while regime blocks in the market factor give the regime
classifier a genuine signal to detect.

Nothing here claims to model real markets — it exists to exercise the pipeline
deterministically and to demo reporting.  Swap in a real ``DataSource`` for
live work.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import SECTORS, THEMES
from .source import FUNDAMENTAL_FIELDS, DataBundle, DataSource, SecurityMeta


# Regime archetypes for the market factor: (daily drift, daily vol).
_REGIME_ARCHETYPES = {
    "bull_low_vol": (0.00075, 0.0070),
    "bull_high_vol": (0.00060, 0.0130),
    "sideways": (0.00005, 0.0090),
    "bear": (-0.00090, 0.0170),
    "recovery": (0.00110, 0.0110),
}


def _make_tickers(n: int) -> List[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out: List[str] = []
    i = 0
    while len(out) < n:
        a = letters[i % 26]
        b = letters[(i // 26) % 26]
        c = letters[(i // (26 * 26)) % 26]
        out.append(f"{c}{b}{a}"[-3:] if i >= 26 else f"{a}{b}X")
        i += 1
    # Deduplicate while preserving order.
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:n]


class SyntheticDataSource(DataSource):
    def __init__(
        self,
        n_tickers: int = 60,
        years: float = 8.0,
        seed: int = 42,
        end: Optional[pd.Timestamp] = None,
        delist_frac: float = 0.0,
    ) -> None:
        self.n_tickers = n_tickers
        self.years = years
        self.seed = seed
        self.end = pd.Timestamp(end) if end is not None else pd.Timestamp("2025-12-31")
        # Fraction of names that delist mid-history (with pre-delisting distress),
        # so survivorship bias can be reproduced and measured.
        self.delist_frac = delist_frac

    # ------------------------------------------------------------------ load
    def load(self) -> DataBundle:
        rng = np.random.default_rng(self.seed)
        n_days = int(self.years * 252)
        dates = pd.bdate_range(end=self.end, periods=n_days)

        tickers = _make_tickers(self.n_tickers)
        meta = self._make_meta(tickers, rng)
        traits = self._make_traits(tickers, rng)

        market_ret, regime_labels = self._market_factor(dates, rng)
        sector_rets = self._sector_factors(dates, rng)

        prices = self._stock_prices(dates, tickers, traits, meta, market_ret, sector_rets, rng)
        if self.delist_frac > 0:
            self._apply_delistings(prices, meta, dates, rng)
        benchmarks = self._benchmarks(dates, market_ret, sector_rets, prices, meta)
        fundamentals = self._fundamentals(dates, tickers, traits, rng)

        return DataBundle(prices=prices, benchmarks=benchmarks,
                          fundamentals=fundamentals, meta=meta)

    # --------------------------------------------------------------- helpers
    def _make_meta(self, tickers: List[str], rng) -> Dict[str, SecurityMeta]:
        meta: Dict[str, SecurityMeta] = {}
        for t in tickers:
            sector = SECTORS[rng.integers(0, len(SECTORS))]
            n_theme = rng.integers(0, 3)
            themes = list(rng.choice(THEMES, size=n_theme, replace=False)) if n_theme else []
            pays_div = bool(rng.random() < 0.55)
            meta[t] = SecurityMeta(
                ticker=t,
                name=f"{t} Corp",
                sector=sector,
                themes=[str(x) for x in themes],
                pays_dividend=pays_div,
            )
        return meta

    def _make_traits(self, tickers: List[str], rng) -> pd.DataFrame:
        """Latent style traits in [0, 1] per ticker."""
        n = len(tickers)
        df = pd.DataFrame(
            {
                "value": rng.beta(2, 2, n),
                "quality": rng.beta(2, 2, n),
                "growth": rng.beta(2, 2, n),
                "income": rng.beta(2, 2, n),
                "size": rng.beta(2, 2, n),      # 1 = mega cap
                "beta": rng.uniform(0.6, 1.5, n),
                "momentum_persist": rng.uniform(0.0, 0.15, n),
            },
            index=tickers,
        )
        # Growth and income tend to trade off.
        df["income"] = (df["income"] * (1.0 - 0.5 * df["growth"])).clip(0, 1)
        return df

    def _market_factor(self, dates, rng):
        """Piecewise-regime market daily returns plus a per-day regime label."""
        n = len(dates)
        rets = np.zeros(n)
        labels = np.empty(n, dtype=object)
        names = list(_REGIME_ARCHETYPES.keys())
        i = 0
        while i < n:
            name = names[rng.integers(0, len(names))]
            mu, sigma = _REGIME_ARCHETYPES[name]
            block = int(rng.integers(40, 160))
            j = min(i + block, n)
            rets[i:j] = rng.normal(mu, sigma, j - i)
            labels[i:j] = name
            i = j
        return pd.Series(rets, index=dates, name="market"), pd.Series(labels, index=dates)

    def _sector_factors(self, dates, rng) -> pd.DataFrame:
        n = len(dates)
        data = {}
        for s in SECTORS:
            drift = rng.normal(0.0002, 0.0002)
            vol = rng.uniform(0.006, 0.011)
            data[s] = rng.normal(drift, vol, n)
        return pd.DataFrame(data, index=dates)

    def _stock_prices(self, dates, tickers, traits, meta, market_ret, sector_rets, rng):
        n = len(dates)
        prices = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        market = market_ret.to_numpy()
        for t in tickers:
            tr = traits.loc[t]
            sec_ret = sector_rets[meta[t].sector].to_numpy()
            beta = tr["beta"]
            sec_load = rng.uniform(0.3, 0.8)
            idio_vol = rng.uniform(0.008, 0.020)
            idio = rng.normal(0.0, idio_vol, n)
            # Small persistent drift rewarding quality & value, penalising
            # extreme growth (so templates genuinely differ).
            alpha = (0.00012 * (tr["quality"] - 0.5)
                     + 0.00010 * (tr["value"] - 0.5)
                     - 0.00006 * (tr["growth"] - 0.5))
            base = alpha + beta * market + sec_load * sec_ret + idio
            # Add mild momentum via AR(1) on the idiosyncratic part.
            ar = tr["momentum_persist"]
            r = base.copy()
            for k in range(1, n):
                r[k] += ar * (r[k - 1] - alpha)
            start_price = float(np.exp(rng.uniform(2.5, 5.0)))  # ~$12-$150
            prices[t] = start_price * np.exp(np.cumsum(r))
        return prices

    def _apply_delistings(self, prices, meta, dates, rng) -> None:
        """Mark a fraction of names as delisting mid-history with prior distress.

        Delisted names suffer a ramping drawdown into their delisting date and
        have no price afterward — the pattern that makes survivorship bias real
        (a survivor-only backtest never sees the crash).
        """
        n = len(dates)
        for t in list(prices.columns):
            if rng.random() >= self.delist_frac:
                continue
            di = int(rng.integers(int(n * 0.4), n - 5))
            k = min(120, di)
            distress = np.linspace(0.0, float(rng.uniform(0.4, 0.85)), k)  # up to -40..-85%
            col = prices[t].to_numpy(dtype=float).copy()
            col[di - k:di] = col[di - k:di] * (1.0 - distress)
            col[di:] = np.nan                                   # delisted: no price
            prices[t] = col
            meta[t].delisting_date = str(pd.Timestamp(dates[di]).date())

    def _benchmarks(self, dates, market_ret, sector_rets, prices, meta) -> pd.DataFrame:
        cols = {}
        cols["SPX"] = 4000.0 * np.exp(np.cumsum(market_ret.to_numpy()))
        # Nasdaq proxy: tech-tilted market factor.
        tech_ret = market_ret.to_numpy() + 0.4 * sector_rets["Technology"].to_numpy()
        cols["NDX"] = 14000.0 * np.exp(np.cumsum(tech_ret))
        # Sector ETFs from the sector factors (plus market beta of ~1).
        for s in sector_rets.columns:
            r = market_ret.to_numpy() + sector_rets[s].to_numpy()
            cols[f"ETF_{s[:4].upper()}"] = 100.0 * np.exp(np.cumsum(r))
        return pd.DataFrame(cols, index=dates)

    def _fundamentals(self, dates, tickers, traits, rng) -> pd.DataFrame:
        """Monthly point-in-time fundamentals derived from latent traits."""
        month_ends = pd.bdate_range(start=dates[0], end=dates[-1], freq="BME")
        if len(month_ends) == 0:
            month_ends = pd.DatetimeIndex([dates[-1]])

        # Per-ticker slow random-walk multipliers so values drift over time.
        drift_state = {t: 0.0 for t in tickers}
        rows = []
        index_tuples = []
        for dt in month_ends:
            for t in tickers:
                tr = traits.loc[t]
                drift_state[t] += rng.normal(0.0, 0.02)
                wobble = float(np.exp(np.clip(drift_state[t], -0.4, 0.4)))
                rows.append(self._fundamental_row(tr, wobble, rng))
                index_tuples.append((dt, t))

        idx = pd.MultiIndex.from_tuples(index_tuples, names=["date", "ticker"])
        return pd.DataFrame(rows, index=idx)[FUNDAMENTAL_FIELDS]

    @staticmethod
    def _fundamental_row(tr, wobble, rng) -> Dict[str, float]:
        v, q, g, inc, size = (tr["value"], tr["quality"], tr["growth"],
                              tr["income"], tr["size"])
        eps_noise = lambda scale: float(rng.normal(0.0, scale))

        market_cap = float(np.exp(20.5 + 3.0 * size)) * wobble  # ~$0.8B-$1T
        pe = max(4.0, (14.0 * np.exp(0.9 * g - 0.8 * (v - 0.5)) + eps_noise(1.5)) * wobble)
        pb = max(0.4, (2.5 * np.exp(0.8 * g - 0.9 * (v - 0.5)) + eps_noise(0.3)))
        ev_ebit = max(3.0, pe * rng.uniform(0.7, 1.1))
        fcf_yield = float(np.clip(0.02 + 0.06 * v + 0.03 * q - 0.04 * g + eps_noise(0.01), -0.02, 0.12))
        accruals = float(np.clip(0.06 - 0.05 * q + eps_noise(0.01), -0.02, 0.15))
        roe = float(np.clip(0.05 + 0.22 * q + 0.05 * g + eps_noise(0.02), -0.05, 0.45))
        roic = float(np.clip(roe * rng.uniform(0.7, 0.95), -0.05, 0.40))
        gross_margin = float(np.clip(0.25 + 0.40 * q + eps_noise(0.03), 0.05, 0.85))
        earnings_vol = float(np.clip(0.30 - 0.22 * q + 0.10 * g + eps_noise(0.03), 0.03, 0.60))
        debt_equity = float(np.clip(1.1 - 0.7 * q + 0.3 * v + eps_noise(0.15), 0.0, 3.0))
        interest_coverage = float(np.clip(2.0 + 14.0 * q + eps_noise(1.5), 0.5, 25.0))
        revenue_growth = float(np.clip(0.01 + 0.28 * g - 0.05 * v + eps_noise(0.03), -0.15, 0.55))
        eps_growth = float(np.clip(revenue_growth + 0.05 * q + eps_noise(0.04), -0.25, 0.60))
        growth_stability = float(np.clip(0.4 + 0.4 * q - 0.2 * g + eps_noise(0.05), 0.05, 0.95))
        pays = inc > 0.35
        dividend_yield = float(np.clip((0.005 + 0.05 * inc - 0.02 * g), 0.0, 0.08)) if pays else 0.0
        payout_ratio = float(np.clip(0.2 + 0.6 * inc, 0.0, 0.95)) if pays else 0.0
        adv_usd = float(market_cap * rng.uniform(0.001, 0.006))

        return {
            "market_cap": market_cap,
            "pe": pe,
            "pb": pb,
            "ev_ebit": ev_ebit,
            "fcf_yield": fcf_yield,
            "accruals": accruals,
            "roe": roe,
            "roic": roic,
            "gross_margin": gross_margin,
            "earnings_vol": earnings_vol,
            "debt_equity": debt_equity,
            "interest_coverage": interest_coverage,
            "revenue_growth": revenue_growth,
            "eps_growth": eps_growth,
            "growth_stability": growth_stability,
            "dividend_yield": dividend_yield,
            "payout_ratio": payout_ratio,
            "adv_usd": adv_usd,
        }
