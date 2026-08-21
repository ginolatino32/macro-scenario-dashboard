"""Macro Seasons 3.0 — merged final strategy (base v2 + verified research variants).

SELF-CONTAINED merge of macro_seasons_v2.py with the four verified research
variants. No imports from the h* modules; every layer's code and constants are
copied verbatim from the variant that validated it:

  H8 (research/h8_gold_realrate.py)  — real-rate gold/duration defensive
      rotation. Toggle: use_realrate. Applied post-blend, immediately after
      the liquidity overlay (exactly where h8 applies it).
      Verified dev (<= 2018-12-31): excess Sharpe 0.690 / Calmar 0.966.
  H3 (research/h3_credit_gate.py)    — credit-stress risk dimmer (trailing-
      median HY OAS flag + DBAA-DGS10 fallback). Toggle: use_credit. Applied
      post-blend after the liquidity overlay (exactly where h3 applies it);
      when combined with H8, rotation first, then dimmer (rotation is a
      defensive-sleeve reshuffle; dimmer changes the risky/defensive split).
      Verified dev: 0.752 / 0.883.
  H2 (research/h2_daily_risk_engine.py) — daily panel (stitched yfinance
      daily adjusted closes + proxies) and the 200-day MA trend gate as a
      drop-in replacement for the 10m SMA gate. Toggle: trend_mode in
      {"sma10", "daily200"}. The rejected EWMA vol engine is NOT merged.
      Verified dev: 0.724 / 0.924.
  H5 (research/h5_ensemble_erc.py)   — inverse-vol (rolling 24m, min 12,
      shift 1, normalized) ensemble across monthly walk-forward streams
      [core seasons levered via capital_efficient_variant, full-stack book
      levered the same way, TSMOM sleeve], then causally vol-targeted to 10%.
      Verified dev over the unenhanced streams: 0.804 / 0.782.

ACCEPTANCE TESTS (run before any combined configuration counts): the merged
script must mechanically reproduce each variant's dev-window numbers (T0-T4,
tolerance +/-0.003 Sharpe, +/-0.01 Calmar). Causality self-checks are rerun
for every series this script rebuilds (real-rate shift, credit-stress flag,
daily trend panel, ensemble layer) at 2 truncation dates.

PRE-REGISTERED COMBINED CONFIGURATIONS (exactly two, no constant changes):
  CONFIG A: use_realrate + use_credit + trend_mode="daily200",
            ensemble over [core levered, enhanced stack, TSMOM]
  CONFIG B: use_realrate only (trend_mode="sma10"),
            ensemble over [core levered, enhanced stack, TSMOM]

All numbers printed by this script are DEV-WINDOW ONLY (return_date <=
2018-12-31). Full-period ledgers are written to exports/research/v3_configA/
and exports/research/v3_configB/ for later holdout evaluation by someone else.

Base v2 description follows.
---------------------------------------------------------------------------

Macro Seasons 2.0 — walk-forward validated macro-season portfolios.

Design goals (vs the v1 report in season_sharpe_research.py / mmm_proxy.py):

1. Macro-logical season definitions.
   Growth / Inflation / Liquidity pillars are composites of real, point-in-time
   macro data (industrial production, payrolls, claims, CPI, breakevens, NFCI,
   HY spreads, M2, Fed net liquidity, 2y policy impulse) plus at most one
   traded-market confirmation signal per pillar. v1 used only traded ETF ratios
   (XLI/XLP, ACWI/IEF, TIP/IEF) + M2, so its "macro seasons" were mostly
   trailing risk-asset momentum labels.

2. Symmetric, canonical season logic.
   Seasons are the standard growth x inflation quadrants (the same
   CONVENTIONAL_GI_MAP already present in mmm_proxy.py):
       SPRING = G+ I-  (recovery / disinflationary expansion)
       SUMMER = G+ I+  (reflationary expansion / overheat)
       FALL   = G- I+  (stagflation)
       WINTER = G- I-  (disinflationary slowdown / bust)
   Liquidity is NOT a season re-mapper (v1 collapsed 8 G/I/L states
   asymmetrically so that SUMMER absorbed 61% of months and FALL only 10
   months). Instead liquidity is a bounded risk overlay that shifts weight
   between the risky and defensive sleeves of whatever season mix is active.

3. Probabilities, not hard labels.
   Pillar scores map to P(G+), P(I+) through a logistic; season probabilities
   are their products. The portfolio is a probability-weighted blend of the
   four season templates, so a marginal macro month moves the book gradually
   instead of flipping 100% of it on one noisy print.

4. Minimal estimation, no in-sample return optimization.
   v1 Sharpe-optimized each season's weights on that season's FULL history
   (FALL: 10 observations) and reported the same months as the backtest, with
   constraints like SUMMER min_cagr_pct=13 hard-wiring the outcome. v2 uses
   fixed, macro-logical season templates. The only estimated quantities are
   trailing volatilities (inverse-vol tilt inside each template, and a
   portfolio vol target), both strictly causal.

5. Honest walk-forward.
   Every month t: pillars use only data published on or before t (monthly
   macro series are shifted by their publication lag; weekly series by one
   week; market series unshifted), weights are formed at t, returns are taken
   in t+1, costs charged on turnover. There is no fitting step anywhere in
   the loop, so the whole curve is out-of-sample by construction.
   Significance: excess Sharpe / IR with Newey-West t-stats vs SPY, 60/40,
   the GSMIF 70/30 policy, and — most importantly — a STATIC equal-probability
   blend of the same four templates (the "no-timing" placebo), plus a
   circular-shift permutation test of the pure timing effect.

Parameter policy: all constants below are fixed ex ante on macro reasoning.
They were not searched against the backtest. Bug fixes are allowed; changing
constants after seeing results is not (that would just overfit the walk
forward, which is the failure mode this script exists to avoid).

Data note: FRED series fetched without an API key (fredgraph.csv) are
latest-revised. Revision-sensitive monthly series are used as 3-12 month
momenta with a 1-month publication-lag shift, which removes availability
lookahead but not revision drift. M2 uses the repo's ALFRED first-print file
where available. For production BL export, swap CPI / INDPRO / PAYEMS to the
same ALFRED vintage machinery already used for M2SL in update_data.py.

Usage:
    python3 macro_seasons_v2.py                 # full run (uses network once, then cache)
    python3 macro_seasons_v2.py --no-network    # local + cached data only
    python3 macro_seasons_v2.py --self-check    # causality spot-checks, then exit
    python3 macro_seasons_v2.py --no-extend     # skip pre-2008 proxy extension
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent  # script lives in research/
DATA = ROOT / "data"
CACHE = DATA / "fred_cache"
EXPORTS = ROOT / "exports"

SEASONS = ["SPRING", "SUMMER", "FALL", "WINTER"]
STRATEGY_NAME = "Macro Seasons 2.1 full stack (seasons + momentum + trend)"
CORE_NAME = "Macro Seasons core (seasons only)"
STATIC_BLEND_NAME = "Static 25/25/25/25 season blend (no timing)"

# ---------------------------------------------------------------------------
# Ex-ante design constants (fixed before any v2 backtest was run — see header)
# ---------------------------------------------------------------------------

Z_WINDOW = 120          # months, rolling standardization window for sub-signals
Z_MIN_PERIODS = 36      # months of history before a sub-signal activates
Z_CLIP = 3.0            # winsorize sub-signal z-scores
COMPOSITE_EMA_SPAN = 2  # light smoothing of pillar composites
MIN_SUBSIGNALS = 2      # pillar is defined only when >=2 sub-signals exist
LOGISTIC_TAU = 0.75     # temperature mapping pillar z -> P(state positive)

LIQ_OVERLAY_COEF = -0.125           # shift_to_defensive = clip(coef * Lz, lo, hi)
LIQ_OVERLAY_BOUNDS = (-0.15, 0.25)  # max 15% added risk, max 25% de-risking

INV_VOL_BLEND = 0.50    # 50% template weight, 50% inverse-vol-tilted template
VOL_WINDOW = 36         # months for asset trailing vol (min 12)
VOL_MIN_PERIODS = 12
PORT_VOL_WINDOW = 24    # months for portfolio vol targeting estimate (min 12)
VOL_TARGET = 0.10       # annualized
VOL_SCALE_BOUNDS = (0.50, 1.00)  # de-risk only; no leverage

COST_BPS = 10.0         # per unit turnover (0.5 * sum |dw|); stress at 25bps
COST_BPS_STRESS = 25.0

WF_MIN_TEMPLATE_ASSETS = 3   # need >=3 investable assets per season template
PERMUTATIONS = 1000          # circular-shift draws for the timing test
PERMUTATION_MIN_SHIFT = 24   # months

# Enhancement layers (canonical published parameters, fixed ex ante — not tuned):
TREND_SMA_MONTHS = 10        # Faber (2007) GTAA 10-month SMA gate; below-trend -> cash
XSMOM_LOOKBACK = 13          # 12-1 cross-sectional momentum: P(t-1)/P(t-13) - 1
XSMOM_TILT = 0.50            # rank tilt -> weight multiplier in [0.75, 1.25]

# --- H8 real-rate defensive rotation constants (fixed ex ante, not tuned) ---
REALRATE_MOM_MONTHS = 6        # 6m change of the month-end proxy real yield
REALRATE_TILT_COEF = 0.5       # sleeve fraction rotated per 1pp 6m real-yield move
REALRATE_MAX_SHIFT = 0.50      # brief's bound: <= 50% of the source sleeve
DURATION_TICKERS = ("TLT", "IEF")  # nominal-duration side of the rotation

# --- H3 credit-conditions dimmer (fixed ex ante; see header) ---------------
CREDIT_MEDIAN_MONTHS = 36    # trailing 3y median of the spread level
CREDIT_LEVEL_MULT = 1.10     # stress requires spread > 1.10x trailing median
CREDIT_WIDEN_MONTHS = 3      # widening test: spread_t - spread_{t-3m} > 0
CREDIT_DIMMER_FACTOR = 0.50  # risky sleeve multiplier while stress is ON
CREDIT_FALLBACK_SERIES = ("DBAA", "DGS10")  # Moody's Baa - 10y, pre-OAS proxy

# --- H2 daily data layer + 200d MA trend gate (EWMA vol engine NOT merged) --
DAILY_START = "1998-01-01"
TREND_DAILY_MA = 200        # 200-day MA gate (canonical 10m-SMA equivalent)

# Daily pre-inception proxies (same sources the monthly panel splices).
DAILY_PROXIES: dict[str, str] = {
    "SPY": "VFINX", "TLT": "VUSTX", "IEF": "VFITX", "SHY": "VFISX",
    "HYG": "VWEHX", "EEM": "VEIEX", "EFA": "VGTSX", "TIP": "VIPSX",
    "GLD": "GC=F",
}

# --- H5 ensemble layer (naive ERC = inverse-vol; combination only) ----------
H5_NAME = "H5 ERC ensemble (inverse-vol weighted streams, 10% vol target)"
ERC_VOL_WINDOW = 24     # trailing months for stream inverse-vol weights
ERC_MIN_PERIODS = 12
DEV_END = pd.Timestamp("2018-12-31")

RISK_FREE = "BIL"

# Season templates: fixed macro-logical weights. Missing assets renormalize.
TEMPLATES: dict[str, dict[str, float]] = {
    # Recovery: growth improving, inflation easing -> equity beta, growth/tech,
    # credit, housing, modest duration, small gold.
    "SPRING": {
        "SPY": 0.20, "QQQ": 0.15, "SMH": 0.10, "IWM": 0.10, "XHB": 0.075,
        "HYG": 0.15, "LQD": 0.075, "IEF": 0.10, "GLD": 0.05,
    },
    # Reflationary expansion / overheat: cyclicals, value, energy/materials,
    # EM, commodities, TIPS instead of nominal duration.
    "SUMMER": {
        "XLE": 0.15, "XLB": 0.10, "XLI": 0.10, "VLUE": 0.10, "IWM": 0.10,
        "EEM": 0.10, "DBC": 0.125, "CPER": 0.05, "GLD": 0.075, "TIP": 0.10,
    },
    # Stagflation: real assets + inflation defense + dollar + defensive equity,
    # short duration only.
    "FALL": {
        "GLD": 0.175, "DBC": 0.10, "XLE": 0.075, "TIP": 0.15, "UUP": 0.10,
        "XLP": 0.10, "XLU": 0.075, "XLV": 0.075, "SHY": 0.075, "BIL": 0.075,
    },
    # Disinflationary slowdown / bust: nominal duration, quality/low-vol and
    # defensive equity, gold, safe-haven FX, cash.
    "WINTER": {
        "TLT": 0.175, "IEF": 0.175, "SHY": 0.075, "GLD": 0.125, "USMV": 0.10,
        "XLP": 0.10, "XLV": 0.10, "FXY": 0.05, "FXF": 0.05, "BIL": 0.05,
    },
}

DEFENSIVE_ASSETS = {
    "BIL", "SHY", "IEF", "TLT", "TIP", "LQD", "AGG", "UUP", "FXY", "FXF",
    "GLD", "XLP", "XLU", "XLV", "USMV", "SPLV",
}

# FRED series fetched keylessly (latest-revised; see data note in header).
FRED_SERIES = {
    "INDPRO": "monthly",        # industrial production
    "PAYEMS": "monthly",        # nonfarm payrolls
    "ICSA": "weekly",           # initial claims
    "CPIAUCSL": "monthly",      # CPI
    "T5YIE": "daily",           # 5y breakeven
    "DCOILWTICO": "daily",      # WTI spot
    "NFCI": "weekly",           # Chicago Fed financial conditions
    "BAMLH0A0HYM2": "daily",    # HY OAS
    "M2SL": "monthly",          # M2 (pre-2008 splice; 2008+ uses ALFRED file)
    "DGS2": "daily",            # 2y yield
    "DGS10": "daily",           # 10y nominal yield (H8 real-yield proxy input)
    "WALCL": "weekly",          # Fed balance sheet
    "RRPONTSYD": "daily",       # reverse repo
    "WTREGEN": "weekly",        # Treasury general account
    "TB3MS": "monthly",         # 3m bill yield (synthetic cash index pre-BIL)
}

# Pre-inception price proxies for the extended (pre-2008) panel. Mutual funds
# give NAV total returns via yfinance adjusted close; gold/cash from FRED.
EXTENSION_PROXIES: dict[str, str] = {
    "SPY": "VFINX",   # Vanguard 500 (1980)
    "TLT": "VUSTX",   # Vanguard long treasury (1986)
    "IEF": "VFITX",   # Vanguard intermediate treasury (1991)
    "SHY": "VFISX",   # Vanguard short treasury (1991)
    "AGG": "VBMFX",   # Vanguard total bond (1986)
    "HYG": "VWEHX",   # Vanguard high yield (1978)
    "EEM": "VEIEX",   # Vanguard EM (1994)
    "EFA": "VGTSX",   # Vanguard total intl (1996)
    "TIP": "VIPSX",   # Vanguard TIPS (2000)
}
EXTENSION_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "SMH", "SOXX", "IWF",
    "XLK", "XLE", "XLB", "XLI", "XLY", "XLP", "XLV", "XLU",
    "TLT", "IEF", "SHY", "AGG", "LQD", "TIP", "HYG",
    "EEM", "EFA", "GLD", "DBC", "UUP", "FXY", "FXF", "XHB", "VNQ",
    "GC=F",  # gold futures: pre-GLD proxy (price-only; gold pays no income)
]
EXTENDED_START = "1990-01-01"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wide_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def fetch_fred(series_id: str, allow_network: bool) -> pd.Series:
    """Keyless FRED download with a local CSV cache."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{series_id}.csv"
    frame: pd.DataFrame | None = None
    if cache_file.exists():
        frame = pd.read_csv(cache_file)
    elif allow_network:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read().decode("utf-8")
            frame = pd.read_csv(io.StringIO(raw))
            cache_file.write_text(raw)
        except Exception as exc:
            print(f"[fred] {series_id} unavailable: {exc}")
            return pd.Series(dtype=float, name=series_id)
    if frame is None or frame.empty:
        return pd.Series(dtype=float, name=series_id)
    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    values = pd.to_numeric(frame[frame.columns[1]], errors="coerce")
    return pd.Series(values.values, index=frame[date_col], name=series_id).dropna().sort_index()


def monthly_from_lagged(series: pd.Series, freq: str, month_end: pd.DatetimeIndex) -> pd.Series:
    """Align a raw series to month-ends respecting publication timing.

    monthly  : value for reference month m becomes usable at the end of m+1
               (1-month publication-lag shift).
    weekly   : observations become usable 7 days after their stamp.
    daily    : usable same day (market or daily-published data).
    """
    if series.empty:
        return pd.Series(np.nan, index=month_end)
    s = series.copy()
    if freq == "monthly":
        s.index = s.index + pd.offsets.MonthEnd(0)
        s = s.groupby(s.index).last()
        s = s.reindex(month_end.union(s.index)).sort_index().ffill(limit=1)
        s = s.reindex(month_end)
        return s.shift(1)
    if freq == "weekly":
        s.index = s.index + pd.Timedelta(days=7)
    s = s.sort_index().resample("ME").last()
    return s.reindex(month_end).ffill(limit=1)


def download_yahoo_monthly(tickers: list[str], start: str, allow_network: bool) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "yahoo_extension.csv"
    if cache_file.exists():
        return load_wide_csv(cache_file)
    if not allow_network:
        return pd.DataFrame()
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, interval="1mo", auto_adjust=True,
                          progress=False, group_by="column")
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        closes = closes.dropna(how="all")
        closes.index = pd.DatetimeIndex(closes.index) + pd.offsets.MonthEnd(0)
        closes = closes.groupby(closes.index).last()
        out = closes.copy()
        out.index.name = "date"
        out.reset_index().to_csv(cache_file, index=False)
        return closes
    except Exception as exc:  # network / API failure -> degrade gracefully
        print(f"[extension] yahoo download failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# H2: daily data layer + 200d MA trend gate (EWMA vol engine NOT merged)
# ---------------------------------------------------------------------------

def daily_risk_tickers() -> list[str]:
    """Union of season-template assets and the TSMOM universe (28 tickers)."""
    tickers: set[str] = set()
    for template in TEMPLATES.values():
        tickers.update(template.keys())
    tickers.update(TSMOM_UNIVERSE)
    return sorted(tickers)


def download_yahoo_daily(tickers: list[str], start: str, allow_network: bool) -> pd.DataFrame:
    """Daily adjusted closes with a local CSV cache (yahoo_daily.csv)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / "yahoo_daily.csv"
    if cache_file.exists():
        return load_wide_csv(cache_file)
    if not allow_network:
        return pd.DataFrame()
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, interval="1d", auto_adjust=True,
                          progress=False, group_by="column")
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        closes = closes.dropna(how="all")
        closes.index = pd.DatetimeIndex(closes.index).tz_localize(None).normalize()
        closes = closes.groupby(closes.index).last().sort_index()
        out = closes.copy()
        out.index.name = "date"
        out.reset_index().to_csv(cache_file, index=False)
        return closes
    except Exception as exc:  # network / API failure -> degrade gracefully
        print(f"[daily] yahoo daily download failed: {exc}")
        return pd.DataFrame()


def stitched_daily_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker daily simple returns; proxy returns strictly before the ETF's
    first live daily return (mirrors the monthly panel's proxy splice, but at
    the return level so no scale factor is needed)."""
    if daily_prices.empty:
        return pd.DataFrame()
    out: dict[str, pd.Series] = {}
    for ticker in daily_risk_tickers():
        parts: list[pd.Series] = []
        etf_r = pd.Series(dtype=float)
        if ticker in daily_prices.columns:
            etf_r = daily_prices[ticker].dropna().pct_change().dropna()
        proxy = DAILY_PROXIES.get(ticker)
        if proxy and proxy in daily_prices.columns:
            prox_r = daily_prices[proxy].dropna().pct_change().dropna()
            if not etf_r.empty:
                prox_r = prox_r.loc[prox_r.index < etf_r.index.min()]
            parts.append(prox_r)
        parts.append(etf_r)
        r = pd.concat(parts).sort_index()
        r = r[~r.index.duplicated(keep="last")]
        r = r.replace([np.inf, -np.inf], np.nan)
        r = r.mask(r.abs() > 0.75)  # guard against bad NAV prints
        r = r.dropna()
        if not r.empty:
            out[ticker] = r
    return pd.DataFrame(out).sort_index()


def daily_price_index(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Continuous per-ticker daily total-return index from stitched returns
    (used only by the optional 200-day MA trend gate)."""
    if daily_returns.empty:
        return pd.DataFrame()
    filled = daily_returns.fillna(0.0)
    index = (1.0 + filled).cumprod()
    # blank out everything before each ticker's first real return
    for col in index.columns:
        first = daily_returns[col].first_valid_index()
        if first is None:
            index[col] = np.nan
        else:
            index.loc[index.index < first, col] = np.nan
    return index


def trend_causality_check(daily_returns: pd.DataFrame, month_end: pd.DatetimeIndex) -> None:
    """Recompute the 200d-MA trend signal on truncated daily data at 2 cut
    dates; the last-close / MA ratio at the cut must match the full sample."""
    full_idx = daily_price_index(daily_returns)

    def ratios(idx: pd.DataFrame, cut: pd.Timestamp) -> pd.Series:
        out: dict[str, float] = {}
        for col in idx.columns:
            d = idx.loc[idx.index <= cut, col].dropna()
            if len(d) >= TREND_DAILY_MA:
                out[col] = float(d.iloc[-1] / d.tail(TREND_DAILY_MA).mean())
        return pd.Series(out).sort_index()

    for frac in (0.60, 0.85):
        cut = month_end[int(len(month_end) * frac)]
        trunc_idx = daily_price_index(daily_returns.loc[daily_returns.index <= cut])
        a, b = ratios(full_idx, cut), ratios(trunc_idx, cut)
        ok = a.index.equals(b.index) and np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-10)
        print(f"[self-check] 200d MA trend at {cut:%Y-%m-%d}: {'OK' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("Trend-gate causality self-check failed — do not trust results.")
    print("[self-check] trend-gate causality checks passed")


def synthetic_bill_index(tb3ms: pd.Series, month_end: pd.DatetimeIndex) -> pd.Series:
    """Total-return index for 3m bills from TB3MS yields (monthly compounding)."""
    if tb3ms.empty:
        return pd.Series(dtype=float)
    y = tb3ms.copy()
    y.index = y.index + pd.offsets.MonthEnd(0)
    y = y.groupby(y.index).last().reindex(month_end).ffill()
    monthly_return = (1.0 + y / 100.0) ** (1.0 / 12.0) - 1.0
    index = (1.0 + monthly_return.fillna(0.0)).cumprod() * 100.0
    index[y.isna()] = np.nan
    return index


def build_price_panel(allow_network: bool, extend: bool) -> tuple[pd.DataFrame, dict[str, str]]:
    """Local price panel (2008+), optionally spliced backward with proxies."""
    local_path = DATA / "prices_macro_seasons_extended.csv"
    prices = load_wide_csv(local_path if local_path.exists() else DATA / "prices.csv")
    end = prices.index.max()
    notes: dict[str, str] = {}
    if not extend:
        return prices, notes

    month_end = pd.date_range(EXTENDED_START, end, freq="ME")
    ext_tickers = sorted(set(EXTENSION_ETFS) | set(EXTENSION_PROXIES.values()))
    yahoo = download_yahoo_monthly(ext_tickers, EXTENDED_START, allow_network)
    if yahoo.empty:
        notes["extension"] = "unavailable (no network/cache); core 2008+ panel only"
        return prices, notes
    yahoo = yahoo.reindex(month_end)

    tb3 = fetch_fred("TB3MS", allow_network)
    bill_index = synthetic_bill_index(tb3, month_end)

    extended = prices.reindex(prices.index.union(month_end)).sort_index()

    def splice(target: str, source: pd.Series, label: str) -> None:
        if target not in extended.columns or source.dropna().empty:
            return
        live = extended[target].dropna()
        if live.empty:
            return
        first_live = live.index.min()
        overlap = source.dropna()
        overlap = overlap.loc[overlap.index >= first_live]
        anchor_src = overlap.iloc[0] if not overlap.empty else source.dropna().iloc[-1]
        if not np.isfinite(anchor_src) or anchor_src == 0:
            return
        scale = live.iloc[0] / anchor_src
        pre = source.loc[source.index < first_live] * scale
        pre = pre.reindex(extended.index.intersection(pre.index))
        extended.loc[pre.index, target] = extended.loc[pre.index, target].fillna(pre)
        if not pre.dropna().empty:
            notes[target] = f"pre-{first_live:%Y-%m} history from {label}"

    for etf in EXTENSION_ETFS:
        if etf in yahoo.columns and etf != "GC=F":
            splice(etf, yahoo[etf], f"{etf} (Yahoo, monthly adj close)")
    for etf, fund in EXTENSION_PROXIES.items():
        if fund in yahoo.columns:
            splice(etf, yahoo[fund], f"{fund} mutual fund NAV")
    if "GC=F" in yahoo.columns:
        splice("GLD", yahoo["GC=F"], "gold futures front month (price-only)")
    splice("BIL", bill_index, "TB3MS synthetic bill index")

    return extended.dropna(how="all"), notes


# ---------------------------------------------------------------------------
# Pillar construction (all operations strictly causal)
# ---------------------------------------------------------------------------

def rolling_z(series: pd.Series) -> pd.Series:
    mean = series.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
    std = series.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std(ddof=0).replace(0.0, np.nan)
    return ((series - mean) / std).clip(-Z_CLIP, Z_CLIP)


def log_momentum(series: pd.Series, months: int) -> pd.Series:
    positive = series.where(series > 0)
    return np.log(positive).diff(months)


@dataclass
class PillarBundle:
    composites: pd.DataFrame          # G/I/L composite z-scores
    subsignals: pd.DataFrame          # every sub-signal z, for the report
    coverage: pd.DataFrame            # first valid date per sub-signal


def build_pillars(prices: pd.DataFrame, factors_pit: pd.DataFrame,
                  allow_network: bool) -> PillarBundle:
    month_end = pd.DatetimeIndex(prices.index)

    def fred_monthly(series_id: str) -> pd.Series:
        return monthly_from_lagged(fetch_fred(series_id, allow_network),
                                   FRED_SERIES[series_id], month_end)

    subs: dict[str, pd.Series] = {}

    # --- Growth ---
    indpro = fred_monthly("INDPRO")
    payems = fred_monthly("PAYEMS")
    claims = fred_monthly("ICSA")
    subs["g_indpro_6m"] = rolling_z(log_momentum(indpro, 6))
    subs["g_payrolls_3m"] = rolling_z(log_momentum(payems, 3))
    subs["g_claims_13w"] = rolling_z(-log_momentum(claims.rolling(2, min_periods=1).mean(), 3))
    if {"XLI", "XLP"}.issubset(prices.columns):
        subs["g_cyc_def_6m"] = rolling_z(log_momentum(prices["XLI"] / prices["XLP"], 6))
    if {"SPY", "IEF"}.issubset(prices.columns):
        # Risk-appetite nowcast (the dashboard's `risk` factor concept:
        # equities vs duration), daily-updatable and never revised.
        subs["g_risk_appetite_6m"] = rolling_z(log_momentum(prices["SPY"] / prices["IEF"], 6))

    # --- Inflation ---
    cpi = fred_monthly("CPIAUCSL")
    cpi_yoy = log_momentum(cpi, 12)
    subs["i_cpi_delta3"] = rolling_z(cpi_yoy.diff(3))
    subs["i_breakeven_3m"] = rolling_z(fred_monthly("T5YIE").diff(3))
    oil = fred_monthly("DCOILWTICO")
    subs["i_oil_6m"] = rolling_z(log_momentum(oil, 6))
    if {"TIP", "IEF"}.issubset(prices.columns):
        subs["i_tip_ief_6m"] = rolling_z(log_momentum(prices["TIP"] / prices["IEF"], 6))

    # --- Liquidity / financial conditions ---
    nfci = fred_monthly("NFCI")
    subs["l_nfci"] = 0.5 * rolling_z(-nfci) + 0.5 * rolling_z(-nfci.diff(3))
    subs["l_hy_oas_3m"] = rolling_z(-fred_monthly("BAMLH0A0HYM2").diff(3))
    m2_latest = fred_monthly("M2SL")
    if "liquidity" in factors_pit.columns:
        m2_pit = factors_pit["liquidity"].reindex(month_end).shift(1)
        m2 = m2_pit.combine_first(m2_latest)
    else:
        m2 = m2_latest
    subs["l_m2_6m"] = rolling_z(log_momentum(m2, 6))
    dgs2 = fred_monthly("DGS2")
    subs["l_policy_12m"] = rolling_z(-dgs2.diff(12))
    walcl = fred_monthly("WALCL")
    rrp = fred_monthly("RRPONTSYD")
    tga = fred_monthly("WTREGEN")
    if not walcl.dropna().empty:
        netliq = walcl - rrp.fillna(0.0) - tga.fillna(0.0)
        subs["l_netliq_3m"] = rolling_z(log_momentum(netliq, 3))

    subframe = pd.DataFrame(subs, index=month_end)

    def composite(prefix: str) -> pd.Series:
        cols = [c for c in subframe.columns if c.startswith(prefix)]
        block = subframe[cols]
        raw = block.mean(axis=1).where(block.notna().sum(axis=1) >= MIN_SUBSIGNALS)
        return raw.ewm(span=COMPOSITE_EMA_SPAN, adjust=False, min_periods=1).mean()

    composites = pd.DataFrame(
        {"G": composite("g_"), "I": composite("i_"), "L": composite("l_")},
        index=month_end,
    )
    coverage = pd.DataFrame(
        [{"subsignal": c, "first_valid": subframe[c].first_valid_index(),
          "last_valid": subframe[c].last_valid_index()} for c in subframe.columns]
    )
    return PillarBundle(composites=composites, subsignals=subframe, coverage=coverage)


def season_probabilities(composites: pd.DataFrame) -> pd.DataFrame:
    """Quadrant probabilities from pillar z-scores; NaN rows stay NaN."""
    g = composites["G"] / LOGISTIC_TAU
    i = composites["I"] / LOGISTIC_TAU
    p_g = 1.0 / (1.0 + np.exp(-g))
    p_i = 1.0 / (1.0 + np.exp(-i))
    probs = pd.DataFrame(
        {
            "SPRING": p_g * (1.0 - p_i),
            "SUMMER": p_g * p_i,
            "FALL": (1.0 - p_g) * p_i,
            "WINTER": (1.0 - p_g) * (1.0 - p_i),
        },
        index=composites.index,
    )
    complete = probs[SEASONS].dropna(how="any")
    probs["modal_season"] = complete.idxmax(axis=1).reindex(probs.index)
    entropy = -(probs[SEASONS] * np.log(probs[SEASONS].clip(lower=1e-12))).sum(axis=1)
    probs["confidence"] = (1.0 - entropy / math.log(4)).where(probs[SEASONS].notna().all(axis=1))
    probs["liquidity_z"] = composites["L"]
    return probs


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def template_weights_at(season: str, as_of: pd.Timestamp, returns: pd.DataFrame) -> pd.Series:
    """Fixed template, filtered to investable assets, inverse-vol tilted."""
    template = pd.Series(TEMPLATES[season], dtype=float)
    history = returns.loc[returns.index <= as_of]
    usable: dict[str, float] = {}
    vols: dict[str, float] = {}
    for ticker, weight in template.items():
        if ticker not in history.columns:
            continue
        r = history[ticker].dropna().tail(VOL_WINDOW)
        if len(r) < VOL_MIN_PERIODS or not np.isfinite(history[ticker].reindex([as_of]).iloc[0]):
            continue
        vol = float(r.std(ddof=0))
        if not np.isfinite(vol) or vol <= 0:
            continue
        usable[ticker] = float(weight)
        vols[ticker] = vol
    if len(usable) < WF_MIN_TEMPLATE_ASSETS:
        return pd.Series(dtype=float)
    base = pd.Series(usable)
    base = base / base.sum()
    inv = base / pd.Series(vols)
    inv = inv / inv.sum()
    blended = INV_VOL_BLEND * inv + (1.0 - INV_VOL_BLEND) * base
    return blended / blended.sum()


def apply_liquidity_overlay(weights: pd.Series, liquidity_z: float) -> pd.Series:
    if weights.empty or not np.isfinite(liquidity_z):
        return weights
    shift = float(np.clip(LIQ_OVERLAY_COEF * liquidity_z, *LIQ_OVERLAY_BOUNDS))
    if abs(shift) < 1e-12:
        return weights
    defensive = weights.index.isin(DEFENSIVE_ASSETS)
    w = weights.copy()
    risky_total = float(w[~defensive].sum())
    defensive_total = float(w[defensive].sum())
    if shift > 0 and risky_total > 1e-9:          # tighten: move risk -> defense
        moved = min(shift, risky_total)
        w[~defensive] *= (risky_total - moved) / risky_total
        if defensive_total > 1e-9:
            w[defensive] *= (defensive_total + moved) / defensive_total
        else:
            w["BIL"] = w.get("BIL", 0.0) + moved
    elif shift < 0 and defensive_total > 1e-9:    # loosen: move defense -> risk
        moved = min(-shift, defensive_total)
        w[defensive] *= (defensive_total - moved) / defensive_total
        if risky_total > 1e-9:
            w[~defensive] *= (risky_total + moved) / risky_total
    return w / w.sum()


# ---------------------------------------------------------------------------
# H8: real-rate-aware defensive rotation (gold vs nominal duration)
# ---------------------------------------------------------------------------

def realrate_shift_from_raw(dgs10: pd.Series, t5yie: pd.Series,
                            month_end: pd.DatetimeIndex) -> pd.Series:
    """Month-end rotation fraction from raw daily DGS10 / T5YIE series.

    proxy real yield = DGS10 - T5YIE, month-end sampled (daily market data,
    no publication-lag shift); signal = trailing 6m change. Direction rule
    (nominal = real + breakeven, so the breakeven leg says WHY real moved):
      real falling, breakevens rising  -> stagflationary easing -> GOLD
                                          (nominals flat/up: duration can't defend)
      real falling, breakevens falling -> deflationary easing   -> DURATION
      real rising                      -> DURATION (gold's opportunity cost up)
    Magnitude = clip(coef * |6m change in real yield|, 0, max); sign > 0 means
    rotate duration -> gold, sign < 0 means rotate gold -> duration.
    Strictly causal: month-end sampling uses only data <= t, diffs trailing.
    """
    if dgs10.dropna().empty or t5yie.dropna().empty:
        return pd.Series(np.nan, index=month_end, name="realrate_shift")
    nominal = monthly_from_lagged(dgs10, "daily", month_end)
    breakeven = monthly_from_lagged(t5yie, "daily", month_end)
    real_yield = nominal - breakeven
    d_real = real_yield.diff(REALRATE_MOM_MONTHS)
    d_be = breakeven.diff(REALRATE_MOM_MONTHS)
    magnitude = (REALRATE_TILT_COEF * d_real.abs()).clip(upper=REALRATE_MAX_SHIFT)
    toward_gold = (d_real < 0) & (d_be >= 0)
    sign = pd.Series(np.where(toward_gold, 1.0, -1.0), index=month_end)
    shift = (sign * magnitude).where(d_real.notna() & d_be.notna())
    return shift.rename("realrate_shift")


def build_realrate_shift(month_end: pd.DatetimeIndex,
                         allow_network: bool) -> tuple[pd.Series, pd.Series]:
    dgs10 = fetch_fred("DGS10", allow_network)
    t5yie = fetch_fred("T5YIE", allow_network)
    if dgs10.dropna().empty or t5yie.dropna().empty:
        print("[h8] WARNING: DGS10 or T5YIE unavailable — rotation degrades to "
              "no-op (weights identical to baseline).")
    shift = realrate_shift_from_raw(dgs10, t5yie, month_end)
    real_yield = (monthly_from_lagged(dgs10, "daily", month_end)
                  - monthly_from_lagged(t5yie, "daily", month_end)).rename("real_yield")
    return shift, real_yield


def realrate_causality_check(month_end: pd.DatetimeIndex, allow_network: bool) -> None:
    """Recompute the shift series on raw data truncated at 2 cut dates; the
    values at the cut must match the full-sample computation exactly."""
    dgs10 = fetch_fred("DGS10", allow_network)
    t5yie = fetch_fred("T5YIE", allow_network)
    if dgs10.dropna().empty or t5yie.dropna().empty:
        print("[h8 self-check] skipped (input series unavailable)")
        return
    full = realrate_shift_from_raw(dgs10, t5yie, month_end)
    for frac in (0.6, 0.85):
        cut_pos = int(len(month_end) * frac)
        cut_date = month_end[cut_pos]
        trunc_me = month_end[: cut_pos + 1]
        truncated = realrate_shift_from_raw(
            dgs10.loc[dgs10.index <= cut_date],
            t5yie.loc[t5yie.index <= cut_date],
            trunc_me,
        )
        a = float(full.loc[cut_date])
        b = float(truncated.loc[cut_date])
        ok = (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, atol=1e-12)
        print(f"[h8 self-check] realrate shift at {cut_date:%Y-%m-%d}: "
              f"{'OK' if ok else 'FAIL'} full={a} truncated={b}")
        if not ok:
            raise SystemExit("H8 causality self-check failed — do not trust results.")
    print("[h8 self-check] real-rate signal causality checks passed")


def apply_realrate_rotation(weights: pd.Series, shift_frac: float) -> pd.Series:
    """Rotate weight between nominal duration (TLT/IEF) and GLD, post-blend.

    shift_frac > 0: move shift_frac of the TLT+IEF weight into GLD.
    shift_frac < 0: move |shift_frac| of the GLD weight into TLT/IEF pro rata.
    Weight-preserving; touches only positions the blend already holds.
    """
    if weights.empty or not np.isfinite(shift_frac) or abs(shift_frac) < 1e-12:
        return weights
    w = weights.copy()
    duration = [t for t in DURATION_TICKERS if t in w.index and w[t] > 0]
    dur_total = float(w[duration].sum()) if duration else 0.0
    gold = float(w.get("GLD", 0.0))
    if shift_frac > 0:                      # real rates falling: duration -> gold
        if dur_total <= 1e-9 or "GLD" not in w.index:
            return weights
        moved = shift_frac * dur_total
        w[duration] *= (dur_total - moved) / dur_total
        w["GLD"] = gold + moved
    else:                                   # real rates rising: gold -> duration
        if gold <= 1e-9 or dur_total <= 1e-9:
            return weights
        moved = -shift_frac * gold
        w["GLD"] = gold - moved
        w[duration] *= (dur_total + moved) / dur_total
    return w


# ---------------------------------------------------------------------------
# H3: credit-conditions risk dimmer
# ---------------------------------------------------------------------------

def _spread_stress_flag(spread_m: pd.Series) -> tuple[pd.Series, pd.Series]:
    """(stress flag, validity mask) from a monthly spread level series.

    stress_t = spread_t > 1.10x trailing 36m median  AND
               spread_t - spread_{t-3} > 0.
    The multiplicative level margin keeps the bar scale-invariant, so it is
    equally strict for HY OAS (~4-18) and the Baa-10y fallback (~1.5-6).
    Rolling median and diff are trailing (pandas rolling/diff look backward
    only), so the flag at t is a function of data with index <= t.
    """
    med = spread_m.rolling(CREDIT_MEDIAN_MONTHS, min_periods=CREDIT_MEDIAN_MONTHS).median()
    widen = spread_m.diff(CREDIT_WIDEN_MONTHS)
    valid = med.notna() & widen.notna() & spread_m.notna()
    flag = (spread_m > CREDIT_LEVEL_MULT * med) & (widen > 0) & valid
    return flag, valid


def load_credit_spreads(allow_network: bool) -> dict[str, pd.Series]:
    """Raw daily spread sources: cached HY OAS + Moody's Baa-10y fallback."""
    oas = fetch_fred("BAMLH0A0HYM2", allow_network)
    dbaa = fetch_fred(CREDIT_FALLBACK_SERIES[0], allow_network)
    dgs10 = fetch_fred(CREDIT_FALLBACK_SERIES[1], allow_network)
    baa_spread = pd.Series(dtype=float, name="BAA_10Y")
    if not dbaa.empty and not dgs10.empty:
        joined = pd.concat([dbaa.rename("baa"), dgs10.rename("g10")], axis=1).dropna()
        baa_spread = (joined["baa"] - joined["g10"]).rename("BAA_10Y")
    return {"oas": oas, "baa": baa_spread}


def build_credit_stress(raw: dict[str, pd.Series],
                        month_end: pd.DatetimeIndex) -> pd.DataFrame:
    """Monthly credit stress flag with per-source computation (no splicing).

    Prefers the HY OAS flag whenever OAS has a complete trailing window at t;
    otherwise uses the Baa-10y flag; if neither is valid the flag is False
    (dimmer inactive). Returns a frame with the flag and diagnostics.
    """
    oas_m = monthly_from_lagged(raw["oas"], "daily", month_end)
    baa_m = monthly_from_lagged(raw["baa"], "daily", month_end)
    oas_flag, oas_valid = _spread_stress_flag(oas_m)
    baa_flag, baa_valid = _spread_stress_flag(baa_m)
    flag = pd.Series(np.where(oas_valid, oas_flag,
                              np.where(baa_valid, baa_flag, False)),
                     index=month_end, dtype=bool)
    source = pd.Series(np.where(oas_valid, "HY_OAS",
                                np.where(baa_valid, "BAA_10Y", "none")),
                       index=month_end)
    return pd.DataFrame({"credit_stress": flag, "credit_source": source,
                         "oas_level": oas_m, "baa_spread": baa_m})


def apply_credit_dimmer(weights: pd.Series, stress: bool) -> pd.Series:
    """Scale the risky sleeve by CREDIT_DIMMER_FACTOR while stress is ON."""
    if not stress or weights.empty:
        return weights
    defensive = weights.index.isin(DEFENSIVE_ASSETS)
    w = weights.copy()
    risky_total = float(w[~defensive].sum())
    if risky_total <= 1e-9:
        return weights
    moved = risky_total * (1.0 - CREDIT_DIMMER_FACTOR)
    w[~defensive] *= CREDIT_DIMMER_FACTOR
    defensive_total = float(w[defensive].sum())
    if defensive_total > 1e-9:
        w[defensive] *= (defensive_total + moved) / defensive_total
    else:
        w["BIL"] = w.get("BIL", 0.0) + moved
    return w / w.sum()


def credit_causality_check(raw: dict[str, pd.Series],
                           month_end: pd.DatetimeIndex) -> None:
    """Recompute the stress flag on truncated data at 2 cut dates.

    The ENTIRE truncated flag/source path (every month <= cut) must match the
    full-sample computation, proving every flag value is a function of data
    with index <= that month only (all operations are trailing).
    """
    full = build_credit_stress(raw, month_end)
    for frac in (0.60, 0.85):
        cut = month_end[int(len(month_end) * frac)]
        raw_cut = {k: s.loc[s.index <= cut] for k, s in raw.items()}
        idx_cut = month_end[month_end <= cut]
        trunc = build_credit_stress(raw_cut, idx_cut)
        flags_match = bool((full.loc[idx_cut, "credit_stress"]
                            == trunc["credit_stress"]).all())
        src_match = bool((full.loc[idx_cut, "credit_source"]
                          == trunc["credit_source"]).all())
        n_on = int(trunc["credit_stress"].sum())
        ok = flags_match and src_match
        print(f"[h3-causality] flag path through {cut:%Y-%m-%d} "
              f"({len(idx_cut)} months, {n_on} ON): {'OK' if ok else 'FAIL'} "
              f"(flags_match={flags_match}, source_match={src_match})")
        if not ok:
            raise SystemExit("H3 causality check failed — flag uses future data.")
    print("[h3-causality] credit stress flag passed truncation checks")


@dataclass
class WalkForwardResult:
    ledger: pd.DataFrame                 # per-month returns, probs, diagnostics
    weights: pd.DataFrame                # weight history
    template_returns: pd.DataFrame       # per-season template next-month returns
    coverage_note: str = ""


def apply_xsmom_tilt(weights: pd.Series, prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    """12-1 cross-sectional momentum tilt within the current book (ex cash)."""
    tradable = [t for t in weights.index if t != RISK_FREE and t in prices.columns]
    if len(tradable) < 4:
        return weights
    history = prices.loc[prices.index <= as_of]
    if len(history) < XSMOM_LOOKBACK + 1:
        return weights
    momentum = history[tradable].iloc[-2] / history[tradable].iloc[-XSMOM_LOOKBACK - 1] - 1.0
    momentum = momentum.replace([np.inf, -np.inf], np.nan).dropna()
    if len(momentum) < 4:
        return weights
    ranks = momentum.rank(pct=True) - 0.5
    w = weights.copy()
    w.loc[ranks.index] = w.loc[ranks.index] * (1.0 + XSMOM_TILT * ranks)
    w = w.clip(lower=0.0)
    return w / w.sum()


def apply_trend_gate(weights: pd.Series, prices: pd.DataFrame, as_of: pd.Timestamp,
                     daily_index: pd.DataFrame | None = None) -> pd.Series:
    """Faber trend gate: weight of any below-trend asset rolls to cash.

    Base: 10-month SMA on month-end prices. H2 optional drop-in: 200-day MA on
    the stitched daily index (canonical daily equivalent); falls back to the
    monthly SMA per asset when daily history is insufficient."""
    history = prices.loc[prices.index <= as_of]
    w = weights.copy()
    to_cash = 0.0
    for ticker in list(w.index):
        if ticker == RISK_FREE or ticker not in history.columns:
            continue
        below: bool | None = None
        if daily_index is not None and ticker in daily_index.columns:
            d = daily_index.loc[daily_index.index <= as_of, ticker].dropna()
            if len(d) >= TREND_DAILY_MA:
                below = float(d.iloc[-1]) < float(d.tail(TREND_DAILY_MA).mean())
        if below is None:
            series = history[ticker].dropna()
            if len(series) < TREND_SMA_MONTHS:
                continue
            below = float(series.iloc[-1]) < float(series.tail(TREND_SMA_MONTHS).mean())
        if below:
            to_cash += float(w[ticker])
            w[ticker] = 0.0
    if to_cash > 1e-12:
        w[RISK_FREE] = w.get(RISK_FREE, 0.0) + to_cash
    w = w[w > 1e-12]
    return w / w.sum()


def run_walk_forward(prices: pd.DataFrame, probs: pd.DataFrame,
                     cost_bps: float = COST_BPS,
                     use_xsmom: bool = False, use_trend: bool = False,
                     realrate_shift: pd.Series | None = None,
                     credit_stress: pd.Series | None = None,
                     daily_index: pd.DataFrame | None = None) -> WalkForwardResult:
    returns = prices.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    idx = returns.index
    usable = probs.reindex(idx)
    valid_signal = usable[SEASONS].notna().all(axis=1)

    rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    template_rows: list[dict[str, object]] = []
    prev_weights = pd.Series(dtype=float)
    prev_static = pd.Series(dtype=float)
    strat_history: list[float] = []
    static_history: list[float] = []

    for pos in range(len(idx) - 1):
        as_of = idx[pos]
        nxt = idx[pos + 1]
        if not bool(valid_signal.loc[as_of]):
            continue

        season_w = {s: template_weights_at(s, as_of, returns) for s in SEASONS}
        if any(w.empty for w in season_w.values()):
            continue

        p_row = usable.loc[as_of, SEASONS].astype(float)
        blended = pd.Series(dtype=float)
        static = pd.Series(dtype=float)
        for season in SEASONS:
            blended = blended.add(season_w[season] * float(p_row[season]), fill_value=0.0)
            static = static.add(season_w[season] * 0.25, fill_value=0.0)
        blended = apply_liquidity_overlay(blended / blended.sum(),
                                          float(usable.loc[as_of, "liquidity_z"]))
        if realrate_shift is not None:      # H8: gold vs duration rotation
            blended = apply_realrate_rotation(
                blended, float(realrate_shift.get(as_of, np.nan)))
        stress_on = bool(credit_stress.get(as_of, False)) if credit_stress is not None else False
        if stress_on:                       # H3: credit dimmer (after rotation)
            blended = apply_credit_dimmer(blended, True)
        if use_xsmom:
            blended = apply_xsmom_tilt(blended, prices, as_of)
        if use_trend:
            blended = apply_trend_gate(blended, prices, as_of, daily_index)
        static = static / static.sum()

        # Vol target from trailing returns of the CURRENT weight vector (causal).
        hist = returns.loc[returns.index <= as_of, blended.index].tail(PORT_VOL_WINDOW)
        port_hist = hist.mul(blended, axis=1).sum(axis=1, min_count=max(1, len(blended) // 2))
        realized = float(port_hist.dropna().std(ddof=0) * math.sqrt(12.0))
        scale = float(np.clip(VOL_TARGET / realized, *VOL_SCALE_BOUNDS)) if realized > 0 else 1.0
        final = blended * scale
        cash_pad = 1.0 - float(final.sum())
        if cash_pad > 1e-9:
            final["BIL"] = final.get("BIL", 0.0) + cash_pad

        next_r = returns.loc[nxt]

        def realized_net(weights: pd.Series, prev: pd.Series) -> tuple[float, float, float]:
            avail = weights[next_r.reindex(weights.index).notna()]
            if avail.empty:
                return np.nan, np.nan, np.nan
            avail = avail / avail.sum()
            gross = float(np.log1p(float(avail.dot(next_r.reindex(avail.index)))))
            union = avail.index.union(prev.index)
            turn = float((avail.reindex(union).fillna(0.0) - prev.reindex(union).fillna(0.0)).abs().sum() * 0.5)
            return gross - turn * cost_bps / 1e4, gross, turn

        net, gross, turnover = realized_net(final, prev_weights)
        static_net, _, _ = realized_net(static, prev_static)
        if not np.isfinite(net):
            continue
        prev_weights = final[next_r.reindex(final.index).notna()]
        prev_weights = prev_weights / prev_weights.sum()
        prev_static = static[next_r.reindex(static.index).notna()]
        prev_static = prev_static / prev_static.sum()
        strat_history.append(net)
        static_history.append(static_net)

        template_month: dict[str, object] = {"as_of": as_of, "return_date": nxt}
        for season in SEASONS:
            w = season_w[season]
            avail = w[next_r.reindex(w.index).notna()]
            template_month[season] = (
                float(np.log1p(float((avail / avail.sum()).dot(next_r.reindex(avail.index)))))
                if not avail.empty else np.nan
            )
        template_rows.append(template_month)

        def bench(ticker_weights: dict[str, float]) -> float:
            s = pd.Series(ticker_weights, dtype=float)
            avail = s[next_r.reindex(s.index).notna()]
            if avail.empty:
                return np.nan
            return float(np.log1p(float((avail / avail.sum()).dot(next_r.reindex(avail.index)))))

        rows.append(
            {
                "as_of": as_of,
                "return_date": nxt,
                "strategy_return": net,
                "gross_strategy_return": gross,
                "static_blend_return": static_net,
                "spy_return": bench({"SPY": 1.0}),
                "sixty_forty_return": bench({"SPY": 0.6, "AGG": 0.4}),
                "gsmif_policy_return": bench({"VT": 0.7, "BIV": 0.3}),
                "cash_return": bench({"BIL": 1.0}),
                "turnover": turnover,
                "vol_scale": scale,
                "modal_season": str(usable.loc[as_of, "modal_season"]),
                "confidence": float(usable.loc[as_of, "confidence"]),
                "p_spring": float(p_row["SPRING"]),
                "p_summer": float(p_row["SUMMER"]),
                "p_fall": float(p_row["FALL"]),
                "p_winter": float(p_row["WINTER"]),
                "liquidity_z": float(usable.loc[as_of, "liquidity_z"]),
                "credit_stress": stress_on,
                "n_assets": int(len(final)),
                "decision_before_return": bool(as_of < nxt),
            }
        )
        for ticker, weight in final.sort_values(ascending=False).items():
            weight_rows.append({"as_of": as_of, "ticker": ticker, "weight": float(weight)})

    ledger = pd.DataFrame(rows)
    return WalkForwardResult(
        ledger=ledger,
        weights=pd.DataFrame(weight_rows),
        template_returns=pd.DataFrame(template_rows),
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

CAPITAL_EFFICIENT_NAME = "Macro Seasons 2.1 vol-targeted 10% (levered variant)"
LEVERED_SCALE_BOUNDS = (0.50, 1.50)

# --- Multi-strategy extension (added in response to the Sharpe>1 target; the
# sleeve itself is canonical TSMOM per Moskowitz-Ooi-Pedersen 2012 with
# published parameters — 12m signal, inverse-vol sizing — not fitted values).
TSMOM_NAME = "TSMOM long/short sleeve (MOP 2012)"
MULTISTRAT_NAME = "Macro Seasons multi-strategy (50/50 seasons + TSMOM, 10% vol)"
TSMOM_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC",
                  "UUP", "FXY", "FXF", "HYG"]
TSMOM_LOOKBACK = 12          # months, sign of excess return over bills
TSMOM_POSITION_VOL = 0.10    # per-position vol scaling numerator
TSMOM_MAX_ABS_WEIGHT = 0.20


def run_tsmom_sleeve(prices: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Monthly time-series momentum long/short across diversified ETFs.

    At each month-end t (data <= t only): signal = sign of 12m return minus the
    12m bill return; position = signal / N * (TSMOM_POSITION_VOL / trailing
    36m vol), capped at +/-20%; whole book causally vol-targeted to 10%.
    """
    returns = prices.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    idx = returns.index
    universe = [t for t in TSMOM_UNIVERSE if t in prices.columns]
    rows: list[dict[str, object]] = []
    prev = pd.Series(dtype=float)
    book_history: list[float] = []
    for pos in range(len(idx) - 1):
        as_of, nxt = idx[pos], idx[pos + 1]
        history = prices.loc[prices.index <= as_of]
        weights: dict[str, float] = {}
        for ticker in universe:
            series = history[ticker].dropna()
            if len(series) < TSMOM_LOOKBACK + 1:
                continue
            asset_12m = series.iloc[-1] / series.iloc[-TSMOM_LOOKBACK - 1] - 1.0
            bill = history[RISK_FREE].dropna() if RISK_FREE in history else pd.Series(dtype=float)
            bill_12m = (bill.iloc[-1] / bill.iloc[-TSMOM_LOOKBACK - 1] - 1.0) if len(bill) > TSMOM_LOOKBACK else 0.0
            r = returns.loc[returns.index <= as_of, ticker].dropna().tail(VOL_WINDOW)
            if len(r) < VOL_MIN_PERIODS:
                continue
            vol = float(r.std(ddof=0) * math.sqrt(12.0))
            if vol <= 0:
                continue
            sign = 1.0 if asset_12m - bill_12m >= 0 else -1.0
            weights[ticker] = float(np.clip(sign * TSMOM_POSITION_VOL / vol / len(universe),
                                            -TSMOM_MAX_ABS_WEIGHT, TSMOM_MAX_ABS_WEIGHT))
        if len(weights) < 6:
            continue
        w = pd.Series(weights)
        if len(book_history) >= 12:
            realized = float(pd.Series(book_history[-PORT_VOL_WINDOW:]).std(ddof=0) * math.sqrt(12.0))
            if realized > 0:
                w = w * float(np.clip(VOL_TARGET / realized, *LEVERED_SCALE_BOUNDS))
        next_r = returns.loc[nxt].reindex(w.index)
        avail = w[next_r.notna()]
        if avail.empty:
            continue
        gross_simple = float(avail.dot(next_r.reindex(avail.index)))
        union = avail.index.union(prev.index)
        turnover = float((avail.reindex(union).fillna(0.0) - prev.reindex(union).fillna(0.0)).abs().sum() * 0.5)
        cash_simple = float(next_r.get(RISK_FREE, np.nan)) if RISK_FREE in next_r else np.nan
        if not np.isfinite(cash_simple):
            cash_simple = float(returns.loc[nxt].get(RISK_FREE, 0.0)) if RISK_FREE in returns.columns else 0.0
        # Long/short overlay on a cash base: unencumbered capital earns bills.
        net_simple = cash_simple + gross_simple - turnover * cost_bps / 1e4
        book_history.append(math.log1p(gross_simple) if gross_simple > -0.999 else np.nan)
        prev = avail
        rows.append({"return_date": nxt, "tsmom_return": math.log1p(net_simple),
                     "tsmom_turnover": turnover, "tsmom_gross_exposure": float(avail.abs().sum())})
    return pd.DataFrame(rows).set_index("return_date")


def combine_multistrat(seasons_levered: pd.Series, tsmom: pd.Series,
                       cash: pd.Series) -> pd.Series:
    """50/50 monthly-rebalanced combination, causally vol-targeted to 10%."""
    joined = pd.concat([seasons_levered.rename("a"), tsmom.rename("b"),
                        cash.rename("rf")], axis=1).dropna()
    simple = 0.5 * (np.exp(joined["a"]) - 1.0) + 0.5 * (np.exp(joined["b"]) - 1.0)
    combo_log = pd.Series(np.log1p(simple), index=joined.index)
    trailing = combo_log.rolling(PORT_VOL_WINDOW, min_periods=12).std(ddof=0).shift(1) * math.sqrt(12.0)
    scale = (VOL_TARGET / trailing).clip(*LEVERED_SCALE_BOUNDS).fillna(1.0)
    rf_simple = np.exp(joined["rf"].fillna(0.0)) - 1.0
    levered_simple = rf_simple + scale * (simple - rf_simple)
    return pd.Series(np.log1p(levered_simple), index=joined.index, name="multistrat")


def capital_efficient_variant(ledger: pd.DataFrame) -> pd.Series:
    """Same signal, causally vol-targeted to VOL_TARGET, financed at BIL.

    scale_t uses only strategy returns through t-1 (trailing PORT_VOL_WINDOW
    months). Levering is a linear transformation of the already-out-of-sample
    return stream — it changes the risk level, not the information content.
    Financing is charged at the BIL return on the borrowed fraction; an
    institutional spread of ~30bps on scale-1 <= 0.5 would cost ~0.15%/yr.
    """
    simple_strategy = np.exp(ledger["strategy_return"]) - 1.0
    simple_cash = np.exp(ledger["cash_return"].fillna(0.0)) - 1.0
    excess = simple_strategy - simple_cash
    trailing_vol = (
        ledger["strategy_return"].rolling(PORT_VOL_WINDOW, min_periods=12).std(ddof=0).shift(1)
        * math.sqrt(12.0)
    )
    scale = (VOL_TARGET / trailing_vol).clip(*LEVERED_SCALE_BOUNDS).fillna(1.0)
    levered_simple = simple_cash + scale * excess
    return pd.Series(np.log1p(levered_simple), index=ledger.index, name="levered_return")


# ---------------------------------------------------------------------------
# H5: Equal-risk-contribution (inverse-vol) ensemble of walk-forward streams
# ---------------------------------------------------------------------------

def levered_stream(log_returns: pd.Series, cash_log: pd.Series, name: str) -> pd.Series:
    """Vol-target an arbitrary monthly log-return stream to VOL_TARGET.

    Identical machinery to capital_efficient_variant (trailing PORT_VOL_WINDOW
    vol computed from returns through t-1, financed at BIL, scale clipped to
    LEVERED_SCALE_BOUNDS) — strictly causal.
    """
    frame = pd.DataFrame({
        "strategy_return": log_returns,
        "cash_return": cash_log.reindex(log_returns.index),
    }).dropna(subset=["strategy_return"])
    out = capital_efficient_variant(frame)
    out.name = name
    return out


def inverse_vol_weights(streams: pd.DataFrame) -> pd.DataFrame:
    """Causal inverse-vol (naive ERC) stream weights, normalized to sum 1.

    vol_i(t) is the trailing ERC_VOL_WINDOW-month std of stream i's log
    returns through t-1 only (rolling window then shift(1)). Months without
    full vol coverage across all streams fall back to equal weight.
    """
    vol = streams.rolling(ERC_VOL_WINDOW, min_periods=ERC_MIN_PERIODS).std(ddof=0).shift(1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    weights = inv.div(inv.sum(axis=1), axis=0)
    incomplete = weights.isna().any(axis=1)
    weights.loc[incomplete] = 1.0 / float(streams.shape[1])
    return weights


def combine_erc(streams: pd.DataFrame, cash_log: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """Inverse-vol weighted ensemble, causally vol-targeted to VOL_TARGET.

    Mirrors combine_multistrat exactly except the fixed 50/50 weights are
    replaced by causal inverse-vol weights across N streams.
    """
    joined = streams.dropna()
    rf_log = cash_log.reindex(joined.index).fillna(0.0)
    weights = inverse_vol_weights(joined)
    simple = np.exp(joined) - 1.0
    combo_simple = (weights * simple).sum(axis=1)
    combo_log = pd.Series(np.log1p(combo_simple), index=joined.index)
    trailing = combo_log.rolling(PORT_VOL_WINDOW, min_periods=12).std(ddof=0).shift(1) * math.sqrt(12.0)
    scale = (VOL_TARGET / trailing).clip(*LEVERED_SCALE_BOUNDS).fillna(1.0)
    rf_simple = np.exp(rf_log) - 1.0
    levered_simple = rf_simple + scale * (combo_simple - rf_simple)
    out = pd.Series(np.log1p(levered_simple), index=joined.index, name="strategy_return")
    diag = weights.add_prefix("w_")
    diag["vol_target_scale"] = scale
    return out, diag


def erc_causality_check(streams: pd.DataFrame, cash_log: pd.Series) -> None:
    """Recompute the ensemble on truncated stream history at 2 cut dates.

    The combined return at each cut must match the full-sample computation
    exactly (mimics causality_self_check, applied to the combination layer:
    inverse-vol weights + vol-target scale must use trailing data only).
    """
    full, _ = combine_erc(streams, cash_log)
    for frac in (0.60, 0.85):
        cut = full.index[int(len(full) * frac)]
        trunc, _ = combine_erc(streams.loc[streams.index <= cut],
                               cash_log.loc[cash_log.index <= cut])
        a, b = float(full.loc[cut]), float(trunc.loc[cut])
        ok = bool(np.isclose(a, b, atol=1e-12))
        print(f"[h5 self-check] ensemble at {cut:%Y-%m-%d}: {'OK' if ok else 'FAIL'} "
              f"full={a:.12f} truncated={b:.12f}")
        if not ok:
            raise SystemExit("H5 combination causality check failed — investigate.")
    print("[h5 self-check] combination layer causal at both cut dates")


def dev_stats(ledger: pd.DataFrame, label: str) -> dict[str, object]:
    """Dev-window stats computed exactly per the research brief."""
    led = ledger.set_index("return_date") if "return_date" in ledger.columns else ledger
    led = led.loc[led.index <= DEV_END]
    r = led["strategy_return"].astype(float)
    rf = led["cash_return"].astype(float).fillna(0.0)
    excess = r - rf
    sharpe = float(excess.mean() / excess.std(ddof=0) * math.sqrt(12.0))
    equity = np.exp(r.cumsum())
    n = len(r)
    cagr = float(equity.iloc[-1] ** (12.0 / n) - 1.0)
    max_dd = float((equity / equity.cummax() - 1.0).min())
    cagr_pct, max_dd_pct = cagr * 100.0, max_dd * 100.0
    calmar = float(cagr_pct / abs(max_dd_pct)) if max_dd_pct < 0 else float("nan")
    ann_vol_pct = float(r.std(ddof=0) * math.sqrt(12.0) * 100.0)
    out = {"label": label, "months": n, "excess_sharpe": sharpe, "calmar": calmar,
           "cagr_pct": cagr_pct, "max_drawdown_pct": max_dd_pct, "ann_vol_pct": ann_vol_pct}
    print(f"[dev {led.index.min():%Y-%m}..{led.index.max():%Y-%m}] {label}: "
          f"excess Sharpe {sharpe:.3f}, Calmar {calmar:.2f}, CAGR {cagr_pct:.2f}%, "
          f"maxDD {max_dd_pct:.2f}%, vol {ann_vol_pct:.2f}%, n={n}")
    return out


def bootstrap_sharpe_difference(strategy: pd.Series, benchmark: pd.Series, rf: pd.Series,
                                n_boot: int = 2000, block: int = 6, seed: int = 11) -> dict[str, object]:
    """Circular block bootstrap p-value for excess-Sharpe(strategy) > excess-Sharpe(benchmark)."""
    joined = pd.concat([strategy.rename("s"), benchmark.rename("b"), rf.rename("rf")], axis=1).dropna()
    if len(joined) < 60:
        return {"months": int(len(joined))}
    s = (joined["s"] - joined["rf"]).to_numpy()
    b = (joined["b"] - joined["rf"]).to_numpy()
    n = len(joined)

    def sharpe(x: np.ndarray) -> float:
        sd = x.std(ddof=0)
        return float(x.mean() / sd * math.sqrt(12.0)) if sd > 0 else np.nan

    observed = sharpe(s) - sharpe(b)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(block) + st) % n for st in starts])[:n]
        diffs[k] = sharpe(s[idx]) - sharpe(b[idx])
    p_value = float((np.sum(diffs <= 0.0) + 1) / (n_boot + 1))
    return {
        "months": n,
        "sharpe_difference": observed,
        "bootstrap_p_value_one_sided": p_value,
        "ci_5pct": float(np.percentile(diffs, 5)),
        "ci_95pct": float(np.percentile(diffs, 95)),
    }


def newey_west_tstat(series: pd.Series, lags: int = 6) -> float:
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 24:
        return np.nan
    mean = x.mean()
    e = x - mean
    variance = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * float(e[lag:] @ e[:-lag]) / n
    se = math.sqrt(variance / n)
    return mean / se if se > 0 else np.nan


def performance_stats(name: str, log_returns: pd.Series, rf_log: pd.Series) -> dict[str, object]:
    r = log_returns.dropna()
    if r.empty:
        return {"series": name, "months": 0}
    aligned_rf = rf_log.reindex(r.index).fillna(0.0)
    excess = r - aligned_rf
    equity = np.exp(r.cumsum())
    years = len(r) / 12.0
    drawdown = equity / equity.cummax() - 1.0
    downside = excess[excess < 0]
    ann_ex = float(excess.mean() * 12.0)
    return {
        "series": name,
        "months": int(len(r)),
        "cagr_pct": float((equity.iloc[-1] ** (1.0 / years) - 1.0) * 100.0),
        "ann_vol_pct": float(r.std(ddof=0) * math.sqrt(12.0) * 100.0),
        "excess_sharpe": float(excess.mean() / excess.std(ddof=0) * math.sqrt(12.0)) if excess.std(ddof=0) > 0 else np.nan,
        "sortino": float(ann_ex / (downside.std(ddof=0) * math.sqrt(12.0))) if len(downside) > 2 and downside.std(ddof=0) > 0 else np.nan,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "calmar": float(((equity.iloc[-1] ** (1.0 / years) - 1.0)) / abs(drawdown.min())) if drawdown.min() < 0 else np.nan,
        "hit_rate_pct": float((r > 0).mean() * 100.0),
        "worst_month_pct": float((np.exp(r.min()) - 1.0) * 100.0),
        "final_equity": float(equity.iloc[-1]),
    }


def relative_stats(strategy: pd.Series, benchmark: pd.Series, label: str) -> dict[str, object]:
    joined = pd.concat([strategy.rename("s"), benchmark.rename("b")], axis=1).dropna()
    if len(joined) < 24:
        return {"benchmark": label, "months": int(len(joined))}
    active = joined["s"] - joined["b"]
    ir = float(active.mean() / active.std(ddof=0) * math.sqrt(12.0)) if active.std(ddof=0) > 0 else np.nan
    return {
        "benchmark": label,
        "months": int(len(joined)),
        "ann_active_return_pct": float(active.mean() * 12.0 * 100.0),
        "information_ratio": ir,
        "newey_west_tstat": newey_west_tstat(active),
        "hit_rate_pct": float((active > 0).mean() * 100.0),
        "worst_relative_month_pct": float(active.min() * 100.0),
    }


def permutation_timing_test(probs: pd.DataFrame, template_returns: pd.DataFrame,
                            rf_log: pd.Series, n_perm: int = PERMUTATIONS,
                            seed: int = 7) -> dict[str, object]:
    """Circular-shift test of the pure season-timing effect.

    Statistic: excess Sharpe of sum_s p_s(t) * template_return_s(t+1) minus
    excess Sharpe of the static 25/25/25/25 blend, using the exact months the
    walk-forward traded. Circular shifts preserve the autocorrelation of both
    the probability paths and returns while destroying their alignment.
    """
    tmpl = template_returns.set_index("as_of")[SEASONS].dropna()
    p = probs.reindex(tmpl.index)[SEASONS].dropna()
    common = tmpl.index.intersection(p.index)
    tmpl = tmpl.loc[common].to_numpy()
    pmat = p.loc[common].to_numpy()
    rf = rf_log.reindex(template_returns.set_index("as_of").loc[common, "return_date"]).fillna(0.0).to_numpy()
    n = len(common)
    if n < 48:
        return {"note": "insufficient months for permutation test", "months": n}

    def excess_sharpe(returns: np.ndarray) -> float:
        excess = returns - rf
        sd = excess.std(ddof=0)
        return float(excess.mean() / sd * math.sqrt(12.0)) if sd > 0 else np.nan

    observed_timed = excess_sharpe((pmat * tmpl).sum(axis=1))
    observed_static = excess_sharpe(tmpl.mean(axis=1))
    observed_stat = observed_timed - observed_static

    rng = np.random.default_rng(seed)
    shifts = rng.integers(PERMUTATION_MIN_SHIFT, n - PERMUTATION_MIN_SHIFT, size=n_perm)
    null_stats = np.empty(n_perm)
    for k, shift in enumerate(shifts):
        rolled = np.roll(pmat, int(shift), axis=0)
        null_stats[k] = excess_sharpe((rolled * tmpl).sum(axis=1)) - observed_static
    p_value = float((np.sum(null_stats >= observed_stat) + 1) / (n_perm + 1))
    return {
        "months": n,
        "timed_excess_sharpe": observed_timed,
        "static_excess_sharpe": observed_static,
        "timing_sharpe_spread": observed_stat,
        "permutations": n_perm,
        "p_value_one_sided": p_value,
        "null_mean": float(null_stats.mean()),
        "null_p95": float(np.percentile(null_stats, 95)),
    }


# ---------------------------------------------------------------------------
# BL view bridge
# ---------------------------------------------------------------------------

BL_VIEW_BASKETS: dict[str, dict[str, dict[str, float]]] = {
    "SPRING": {
        "long": {"QQQ": 0.30, "SPY": 0.30, "SMH": 0.20, "HYG": 0.20},
        "short": {"XLP": 0.40, "XLU": 0.30, "USMV": 0.30},
    },
    "SUMMER": {
        "long": {"XLE": 0.30, "XLB": 0.20, "DBC": 0.30, "EEM": 0.20},
        "short": {"TLT": 0.60, "IEF": 0.40},
    },
    "FALL": {
        "long": {"GLD": 0.40, "DBC": 0.25, "TIP": 0.20, "UUP": 0.15},
        "short": {"SPY": 0.70, "IWM": 0.30},
    },
    "WINTER": {
        "long": {"TLT": 0.45, "IEF": 0.30, "GLD": 0.25},
        "short": {"XLE": 0.40, "XLB": 0.30, "IWM": 0.30},
    },
}
BL_HORIZON_MONTHS = 6
BL_SHRINK = 0.50   # shrink historical season-conditional spread toward zero


def build_bl_views(prices: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    """Season-probability-weighted relative views in the dashboard's q/Omega units."""
    returns = prices.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    modal = probs["modal_season"].reindex(returns.index)
    latest = probs.dropna(subset=SEASONS).iloc[-1]
    rows = []
    for season, baskets in BL_VIEW_BASKETS.items():
        long_w = pd.Series(baskets["long"], dtype=float)
        short_w = pd.Series(baskets["short"], dtype=float)
        spread = (
            returns.reindex(columns=long_w.index).mul(long_w, axis=1).sum(axis=1, min_count=1)
            - returns.reindex(columns=short_w.index).mul(short_w, axis=1).sum(axis=1, min_count=1)
        )
        conditional = spread.groupby(modal.shift(0)).agg(["mean", "std", "count"])
        if season not in conditional.index:
            continue
        mu = float(conditional.loc[season, "mean"])
        sigma = float(conditional.loc[season, "std"])
        count = int(conditional.loc[season, "count"])
        p_season = float(latest[season])
        q_6m = (1.0 - BL_SHRINK) * mu * BL_HORIZON_MONTHS * p_season
        omega = (sigma ** 2) * BL_HORIZON_MONTHS / max(p_season * float(latest["confidence"]), 0.05)
        hit = float((spread[modal == season] > 0).mean()) if count else np.nan
        rows.append(
            {
                "view_name": f"{season.title()} tilt: {'+'.join(long_w.index)} vs {'+'.join(short_w.index)}",
                "season": season,
                "season_probability": p_season,
                "regime_confidence": float(latest["confidence"]),
                "q_6m_decimal": q_6m,
                "omega_6m": omega,
                "q_over_sqrt_omega": q_6m / math.sqrt(omega) if omega > 0 else np.nan,
                "conditional_months": count,
                "conditional_hit_rate": hit,
                "long_basket": json.dumps(long_w.round(4).to_dict()),
                "short_basket": json.dumps(short_w.round(4).to_dict()),
                "status": "Candidate" if p_season >= 0.35 and count >= 24 else "Needs Review",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value: object, digits: int = 2) -> str:
    if isinstance(value, float):
        return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"
    return html.escape(str(value))


def html_table(df: pd.DataFrame, digits: int = 2) -> str:
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda v: _fmt(float(v), digits))
    return show.to_html(index=False, escape=True, classes="data-table")


def svg_equity_chart(equity: pd.DataFrame, title: str) -> str:
    width, height = 1180, 380
    ml, mr, mt, mb = 72, 26, 40, 46
    pw, ph = width - ml - mr, height - mt - mb
    data = np.log(equity.dropna(how="all"))
    y_min, y_max = float(np.nanmin(data.values)), float(np.nanmax(data.values))
    pad = (y_max - y_min) * 0.06 or 1.0
    y_min, y_max = y_min - pad, y_max + pad
    colors = {
        MULTISTRAT_NAME: "#4fe3c0", STRATEGY_NAME: "#53d6b4", CAPITAL_EFFICIENT_NAME: "#37a385",
        TSMOM_NAME: "#e08bd0", CORE_NAME: "#9adbc8", STATIC_BLEND_NAME: "#f4c44f",
        "SPY": "#8fb3ff", "60/40 SPY/AGG": "#c0c7d2", "GSMIF 70/30 VT/BIV": "#b48ef2",
        "Cash / BIL": "#777f8e",
    }
    chart_cols = [c for c in data.columns
                  if c not in {TSMOM_NAME, CORE_NAME, CAPITAL_EFFICIENT_NAME}]
    data = data[chart_cols]
    n = max(len(data) - 1, 1)
    parts, legends = [], []
    for tick in np.linspace(y_min, y_max, 5):
        y = mt + ph * (1 - (tick - y_min) / (y_max - y_min))
        parts.append(f"<line x1='{ml}' y1='{y:.1f}' x2='{width-mr}' y2='{y:.1f}' stroke='#293040'/>")
        parts.append(f"<text x='{ml-8}' y='{y+4:.1f}' text-anchor='end' fill='#9aa4b2' font-size='11'>{math.exp(tick):.1f}x</text>")
    for k, col in enumerate(data.columns):
        series = data[col]
        pts = [f"{ml + pw*i/n:.1f},{mt + ph*(1-(v-y_min)/(y_max-y_min)):.1f}"
               for i, v in enumerate(series) if np.isfinite(v)]
        if len(pts) < 2:
            continue
        color = colors.get(str(col), "#ffffff")
        stroke = 3 if col in {STRATEGY_NAME, MULTISTRAT_NAME} else 1.6
        parts.append(f"<polyline points='{' '.join(pts)}' fill='none' stroke='{color}' stroke-width='{stroke}'/>")
        lx, ly = ml + (k % 3) * 330, height - 24 + (k // 3) * 16
        legends.append(f"<line x1='{lx}' y1='{ly-4}' x2='{lx+20}' y2='{ly-4}' stroke='{color}' stroke-width='{stroke}'/>"
                       f"<text x='{lx+26}' y='{ly}' fill='#cfd5df' font-size='11'>{html.escape(str(col))}</text>")
    x0 = equity.index.min().strftime("%Y-%m")
    x1 = equity.index.max().strftime("%Y-%m")
    return (f"<div class='chart-wrap'><h3>{html.escape(title)}</h3>"
            f"<svg viewBox='0 0 {width} {height + 18}'>"
            f"<rect width='{width}' height='{height + 18}' fill='#171a21' rx='8'/>"
            f"<text x='{ml}' y='24' fill='#f2f4f8' font-size='15' font-weight='700'>Growth of $1 (log scale)</text>"
            f"{''.join(parts)}"
            f"<text x='{ml}' y='{height-30}' fill='#9aa4b2' font-size='11'>{x0}</text>"
            f"<text x='{width-mr}' y='{height-30}' text-anchor='end' fill='#9aa4b2' font-size='11'>{x1}</text>"
            f"{''.join(legends)}</svg></div>")


def season_strip(probs: pd.DataFrame) -> str:
    colors = {"SPRING": "#53d6b4", "SUMMER": "#f4c44f", "FALL": "#f27f5b", "WINTER": "#8fb3ff"}
    rows = probs.dropna(subset=["modal_season"])
    if rows.empty:
        return ""
    width, height, ml = 1180, 96, 72
    n = len(rows)
    bar_w = (width - ml - 26) / n
    cells = [
        f"<rect x='{ml + i*bar_w:.2f}' y='34' width='{bar_w + 0.5:.2f}' height='30' fill='{colors.get(str(row.modal_season), '#666')}' opacity='{0.35 + 0.65*float(row.confidence):.2f}'/>"
        for i, row in enumerate(rows.itertuples())
    ]
    legends = "".join(
        f"<rect x='{ml + k*150}' y='74' width='12' height='12' fill='{c}'/>"
        f"<text x='{ml + k*150 + 18}' y='84' fill='#cfd5df' font-size='11'>{s}</text>"
        for k, (s, c) in enumerate(colors.items())
    )
    x0 = rows.index.min().strftime("%Y-%m")
    x1 = rows.index.max().strftime("%Y-%m")
    return (f"<div class='chart-wrap'><h3>Modal season timeline (opacity = regime confidence)</h3>"
            f"<svg viewBox='0 0 {width} {height}'><rect width='{width}' height='{height}' fill='#171a21' rx='8'/>"
            f"{''.join(cells)}{legends}"
            f"<text x='{ml}' y='24' fill='#9aa4b2' font-size='11'>{x0}</text>"
            f"<text x='{width-26}' y='24' text-anchor='end' fill='#9aa4b2' font-size='11'>{x1}</text></svg></div>")


HTML_STYLE = """
:root { --bg:#0f1115; --panel:#171a21; --text:#f2f4f8; --muted:#9aa4b2; --line:#303746; --accent:#53d6b4; }
body { margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; line-height:1.45; }
main { max-width:1280px; margin:0 auto; padding:36px 28px 72px; }
h1 { font-size:32px; margin:0 0 8px; } h2 { font-size:21px; margin:32px 0 12px; } h3 { font-size:16px; margin:0 0 12px; }
p, li { color:var(--muted); }
.kicker { color:var(--accent); text-transform:uppercase; font-size:12px; letter-spacing:.08em; font-weight:700; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; margin:14px 0; overflow-x:auto; }
.chart-wrap { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; margin:14px 0; overflow-x:auto; }
.chart-wrap svg { display:block; min-width:860px; width:100%; height:auto; }
.data-table { width:100%; border-collapse:collapse; font-size:13px; }
.data-table th { text-align:left; color:var(--muted); border-bottom:1px solid var(--line); padding:7px 9px; white-space:nowrap; }
.data-table td { border-bottom:1px solid #252b38; padding:7px 9px; }
.note { background:#151922; border-left:4px solid var(--accent); padding:12px 14px; border-radius:6px; }
.warn { border-left-color:#f27f5b; }
"""


def build_html_report(summary: pd.DataFrame, rel: pd.DataFrame, perm: dict[str, object],
                      sub: pd.DataFrame, season_diag: pd.DataFrame, equity: pd.DataFrame,
                      probs: pd.DataFrame, bl_views: pd.DataFrame, coverage: pd.DataFrame,
                      extension_notes: dict[str, str], ledger: pd.DataFrame,
                      cost_summary: pd.DataFrame, sharpe_tests: pd.DataFrame,
                      ablation: pd.DataFrame) -> str:
    perm_line = (
        f"Pure timing effect: probability-weighted template blend adds "
        f"{_fmt(perm.get('timing_sharpe_spread', float('nan')), 3)} excess-Sharpe over the static blend "
        f"(permutation p = {_fmt(perm.get('p_value_one_sided', float('nan')), 4)}, "
        f"{perm.get('permutations', 0)} circular shifts, {perm.get('months', 0)} months)."
        if "p_value_one_sided" in perm else str(perm.get("note", "n/a"))
    )
    ext_rows = "".join(f"<li><code>{html.escape(k)}</code>: {html.escape(v)}</li>" for k, v in extension_notes.items())
    first = ledger["return_date"].min()
    last = ledger["return_date"].max()
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Macro Seasons 2.0 — Walk-Forward Research</title><style>{HTML_STYLE}</style></head>
<body><main>
<div class="kicker">Standalone research report — every strategy number is out-of-sample</div>
<h1>Macro Seasons 2.0: Walk-Forward Validated Season Portfolios</h1>
<p class="note">Seasons are canonical growth&times;inflation quadrants built from point-in-time macro composites
(activity, inflation, liquidity), mapped to probabilities, and expressed through fixed macro-logical season
templates with an inverse-vol tilt, a bounded liquidity overlay, and a 10% vol target. There is no return
optimization anywhere, so the full history below is walk-forward by construction. Design constants were fixed
ex ante and not tuned to these results. Walk-forward months: {pd.Timestamp(first):%b %Y} &rarr; {pd.Timestamp(last):%b %Y}.</p>

<h2>Headline: strategy vs benchmarks (identical months, net of {COST_BPS:.0f}bps turnover cost)</h2>
{svg_equity_chart(equity, "Walk-forward equity curves")}
<section class="panel">{html_table(summary)}</section>

<h2>Outperformance evidence</h2>
<section class="panel"><h3>Active return vs each benchmark (unlevered and vol-targeted variant)</h3>{html_table(rel)}
<p>{html.escape(perm_line)}</p>
<p>The vol-targeted variant is the same walk-forward return stream scaled by a causal trailing-vol
estimate toward the {VOL_TARGET:.0%} target (financed at BIL, scale capped at {LEVERED_SCALE_BOUNDS[1]:.2f});
it changes the risk level, not the information content, and makes raw-return comparisons vol-matched.</p></section>
<section class="panel"><h3>Bootstrap Sharpe-difference tests (excess Sharpe, circular block bootstrap)</h3>
{html_table(sharpe_tests, digits=4)}</section>
<section class="panel"><h3>Layer ablation — what each canonical layer adds (all walk-forward, net of costs)</h3>
{html_table(ablation)}
<p>The momentum tilt (12-1, Jegadeesh-Titman) and trend gate (10-month SMA, Faber 2007) use their
published parameter values, fixed ex ante — they are stacked, documented return sources, not fitted knobs.</p></section>
<section class="panel"><h3>Transaction-cost stress</h3>{html_table(cost_summary)}</section>
<section class="panel"><h3>Subperiod stability</h3>{html_table(sub)}</section>

<h2>Season diagnostics</h2>
{season_strip(probs)}
<section class="panel"><h3>Realized behavior by modal season</h3>{html_table(season_diag)}
<p>The stagflation (FALL) and bust (WINTER) rows are the reason this framework exists: the test is whether
the blended book defends when the modal season turns hostile, not whether it beats SPY in a melt-up.</p></section>

<h2>Black-Litterman handoff (latest month)</h2>
<section class="panel">{html_table(bl_views, digits=4)}
<p>q is the 6-month probability- and shrinkage-scaled season-conditional spread return; Omega scales the
conditional variance by regime confidence. Wire these baskets through <code>config/view_pairs.csv</code> /
benchmark baskets so they enter the existing <code>P/q/Omega</code> exports with the production gates intact.</p></section>

<h2>Methodology and data honesty</h2>
<section class="panel">
<ul>
<li>Pillar sub-signals, first/last valid dates below. Monthly macro series are shifted one month for publication lag;
weekly series one week. Keyless FRED data is latest-revised — momenta damp revision noise, but production BL export
should reuse the repo's ALFRED vintage fetch (as already done for M2SL).</li>
<li>Pre-2008 prices are proxy-spliced (mutual-fund NAVs, London gold fix, TB3MS bill index): {ext_rows or "no extension applied"}</li>
<li>Costs: {COST_BPS:.0f}bps on 0.5&Sigma;|&Delta;w| turnover; stress table at {COST_BPS_STRESS:.0f}bps. No leverage; vol target only de-risks.</li>
<li>Design constants fixed ex ante (see script header). Bug fixes allowed; result-driven parameter changes are not.</li>
</ul>
{html_table(coverage)}
</section>
</main></body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def causality_self_check(prices: pd.DataFrame, factors_pit: pd.DataFrame,
                         allow_network: bool) -> None:
    """Recompute pillars on truncated data at 3 cut dates; values must match."""
    full = build_pillars(prices, factors_pit, allow_network).composites
    cuts = [int(len(prices) * f) for f in (0.55, 0.75, 0.9)]
    for cut in cuts:
        cut_date = prices.index[cut]
        truncated = build_pillars(prices.iloc[: cut + 1],
                                  factors_pit.loc[factors_pit.index <= cut_date],
                                  allow_network=False).composites
        a = full.loc[cut_date].to_numpy(dtype=float)
        b = truncated.loc[cut_date].to_numpy(dtype=float)
        ok = np.allclose(a, b, atol=1e-9, equal_nan=True)
        print(f"[self-check] pillars at {cut_date:%Y-%m-%d}: {'OK' if ok else 'FAIL'} full={a} truncated={b}")
        if not ok:
            raise SystemExit("Causality self-check failed — investigate before trusting results.")
    print("[self-check] all causality checks passed")


def _unused_base_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--no-extend", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output-dir", default=str(EXPORTS))
    args = parser.parse_args()
    allow_network = not args.no_network

    prices, extension_notes = build_price_panel(allow_network, extend=not args.no_extend)
    pit_path = DATA / "factors_point_in_time.csv"
    factors_pit = load_wide_csv(pit_path) if pit_path.exists() else pd.DataFrame()
    print(f"[data] price panel: {prices.shape[0]} months x {prices.shape[1]} assets "
          f"({prices.index.min():%Y-%m} → {prices.index.max():%Y-%m})")
    for note in list(extension_notes.items())[:8]:
        print(f"[data] splice {note[0]}: {note[1]}")

    if args.self_check:
        causality_self_check(prices, factors_pit, allow_network)
        return

    pillars = build_pillars(prices, factors_pit, allow_network)
    probs = season_probabilities(pillars.composites)

    ablation_configs = [
        (CORE_NAME, dict(use_xsmom=False, use_trend=False)),
        ("Core + 12-1 xs momentum tilt", dict(use_xsmom=True, use_trend=False)),
        ("Core + 10m SMA trend gate", dict(use_xsmom=False, use_trend=True)),
        (STRATEGY_NAME, dict(use_xsmom=True, use_trend=True)),
    ]
    wf_by_config = {label: run_walk_forward(prices, probs, cost_bps=COST_BPS, **kwargs)
                    for label, kwargs in ablation_configs}
    wf = wf_by_config[STRATEGY_NAME]
    wf_core = wf_by_config[CORE_NAME]
    ledger = wf.ledger
    if ledger.empty:
        raise SystemExit("Walk-forward produced no months — check data inputs.")
    ledger = ledger.set_index("return_date")
    core_ledger = wf_core.ledger.set_index("return_date")
    rf_log = ledger["cash_return"]

    ablation_rows = []
    for label, _ in ablation_configs:
        config_ledger = wf_by_config[label].ledger.set_index("return_date")
        stats = performance_stats(label, config_ledger["strategy_return"], config_ledger["cash_return"])
        stats["avg_turnover_pct"] = float(config_ledger["turnover"].mean() * 100.0)
        ablation_rows.append(stats)
    ablation = pd.DataFrame(ablation_rows)[
        ["series", "months", "cagr_pct", "ann_vol_pct", "excess_sharpe",
         "sortino", "max_drawdown_pct", "calmar", "avg_turnover_pct"]
    ]

    levered = capital_efficient_variant(ledger)
    tsmom = run_tsmom_sleeve(prices)
    multistrat = combine_multistrat(levered, tsmom["tsmom_return"], ledger["cash_return"])
    sleeve_corr = float(pd.concat([levered.rename("a"),
                                   tsmom["tsmom_return"].rename("b")], axis=1).dropna().corr().iloc[0, 1])
    series_map = {
        MULTISTRAT_NAME: multistrat,
        STRATEGY_NAME: ledger["strategy_return"],
        CAPITAL_EFFICIENT_NAME: levered,
        TSMOM_NAME: tsmom["tsmom_return"],
        CORE_NAME: core_ledger["strategy_return"],
        STATIC_BLEND_NAME: ledger["static_blend_return"],
        "SPY": ledger["spy_return"],
        "60/40 SPY/AGG": ledger["sixty_forty_return"],
        "GSMIF 70/30 VT/BIV": ledger["gsmif_policy_return"],
        "Cash / BIL": ledger["cash_return"],
    }
    summary = pd.DataFrame([performance_stats(k, v, rf_log) for k, v in series_map.items()])
    summary["avg_turnover_pct"] = np.nan
    summary.loc[summary["series"] == STRATEGY_NAME, "avg_turnover_pct"] = float(ledger["turnover"].mean() * 100)

    benchmarks = [
        ("static_blend_return", STATIC_BLEND_NAME),
        ("spy_return", "SPY"),
        ("sixty_forty_return", "60/40 SPY/AGG"),
        ("gsmif_policy_return", "GSMIF 70/30 VT/BIV"),
    ]
    rel_rows = []
    for col, label in benchmarks:
        base = relative_stats(ledger["strategy_return"], ledger[col], label)
        base["strategy"] = "unlevered"
        rel_rows.append(base)
        lev = relative_stats(levered, ledger[col], label)
        lev["strategy"] = "vol-targeted 10%"
        rel_rows.append(lev)
        multi = relative_stats(multistrat, ledger[col], label)
        multi["strategy"] = "multi-strategy"
        rel_rows.append(multi)
    rel = pd.DataFrame(rel_rows)[
        ["strategy", "benchmark", "months", "ann_active_return_pct", "information_ratio",
         "newey_west_tstat", "hit_rate_pct", "worst_relative_month_pct"]
    ]

    sharpe_tests = pd.DataFrame(
        [{"comparison": f"{STRATEGY_NAME} vs {label}",
          **bootstrap_sharpe_difference(ledger["strategy_return"], ledger[col], rf_log)}
         for col, label in benchmarks]
        + [{"comparison": f"{STRATEGY_NAME} vs {CORE_NAME}",
            **bootstrap_sharpe_difference(ledger["strategy_return"],
                                          core_ledger["strategy_return"], rf_log)}]
        + [{"comparison": f"{MULTISTRAT_NAME} vs {label}",
            **bootstrap_sharpe_difference(multistrat, ledger[col], rf_log)}
           for col, label in benchmarks]
    )
    print(f"[multistrat] sleeve correlation (seasons vs TSMOM): {sleeve_corr:.3f}")

    perm = permutation_timing_test(probs, wf.template_returns, ledger["cash_return"].copy())

    wf_stress = run_walk_forward(prices, probs, cost_bps=COST_BPS_STRESS,
                                 use_xsmom=True, use_trend=True)
    stress_ledger = wf_stress.ledger.set_index("return_date")
    cost_summary = pd.DataFrame([
        performance_stats(f"{STRATEGY_NAME} @ {COST_BPS:.0f}bps", ledger["strategy_return"], rf_log),
        performance_stats(f"{STRATEGY_NAME} @ {COST_BPS_STRESS:.0f}bps",
                          stress_ledger["strategy_return"], stress_ledger["cash_return"]),
    ])[["series", "months", "cagr_pct", "excess_sharpe", "max_drawdown_pct"]]

    bounds = [("2001-2007", "2001-01-01", "2007-12-31"), ("2008-2012", "2008-01-01", "2012-12-31"),
              ("2013-2019", "2013-01-01", "2019-12-31"), ("2020-2022", "2020-01-01", "2022-12-31"),
              ("2023-2026", "2023-01-01", "2026-12-31")]
    sub_rows = []
    for label, lo, hi in bounds:
        mask = (ledger.index >= lo) & (ledger.index <= hi)
        if mask.sum() < 12:
            continue
        stats = performance_stats(label, ledger.loc[mask, "strategy_return"], rf_log)
        spy = performance_stats(label, ledger.loc[mask, "spy_return"], rf_log)
        sixty = performance_stats(label, ledger.loc[mask, "sixty_forty_return"], rf_log)
        sub_rows.append({
            "period": label, "months": stats["months"],
            "strategy_cagr_pct": stats["cagr_pct"], "spy_cagr_pct": spy["cagr_pct"],
            "sixty_forty_cagr_pct": sixty["cagr_pct"],
            "strategy_excess_sharpe": stats["excess_sharpe"], "spy_excess_sharpe": spy["excess_sharpe"],
            "strategy_max_dd_pct": stats["max_drawdown_pct"], "spy_max_dd_pct": spy["max_drawdown_pct"],
        })
    sub = pd.DataFrame(sub_rows)

    season_rows = []
    for season, group in ledger.groupby("modal_season"):
        season_rows.append({
            "modal_season": season, "months": len(group),
            "strategy_avg_mo_pct": float((np.exp(group["strategy_return"].mean()) - 1) * 100),
            "static_avg_mo_pct": float((np.exp(group["static_blend_return"].mean()) - 1) * 100),
            "spy_avg_mo_pct": float((np.exp(group["spy_return"].mean()) - 1) * 100),
            "sixty_forty_avg_mo_pct": float((np.exp(group["sixty_forty_return"].mean()) - 1) * 100),
            "strategy_hit_rate_pct": float((group["strategy_return"] > 0).mean() * 100),
            "avg_confidence": float(group["confidence"].mean()),
        })
    season_diag = pd.DataFrame(season_rows).sort_values("months", ascending=False)

    equity = pd.DataFrame({k: np.exp(v.fillna(0.0).cumsum()) for k, v in series_map.items()},
                          index=ledger.index)

    bl_views = build_bl_views(prices, probs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "macro_seasons_v2_summary.csv", index=False)
    rel.to_csv(out_dir / "macro_seasons_v2_significance.csv", index=False)
    sharpe_tests.to_csv(out_dir / "macro_seasons_v2_sharpe_tests.csv", index=False)
    ablation.to_csv(out_dir / "macro_seasons_v2_ablation.csv", index=False)
    pd.DataFrame([perm]).to_csv(out_dir / "macro_seasons_v2_permutation.csv", index=False)
    sub.to_csv(out_dir / "macro_seasons_v2_subperiods.csv", index=False)
    season_diag.to_csv(out_dir / "macro_seasons_v2_season_diagnostics.csv", index=False)
    equity.to_csv(out_dir / "macro_seasons_v2_equity_curves.csv", index_label="date")
    probs.to_csv(out_dir / "macro_seasons_v2_season_timeline.csv", index_label="date")
    wf.weights.to_csv(out_dir / "macro_seasons_v2_weights.csv", index=False)
    ledger.reset_index().to_csv(out_dir / "macro_seasons_v2_ledger.csv", index=False)
    bl_views.to_csv(out_dir / "macro_seasons_v2_bl_views.csv", index=False)
    pillars.coverage.to_csv(out_dir / "macro_seasons_v2_signal_coverage.csv", index=False)

    report = build_html_report(summary, rel, perm, sub, season_diag, equity, probs,
                               bl_views, pillars.coverage, extension_notes,
                               ledger.reset_index(), cost_summary, sharpe_tests, ablation)
    (out_dir / "macro_seasons_v2_report.html").write_text(report, encoding="utf-8")

    print("\n=== Walk-forward summary ===")
    print(summary[["series", "months", "cagr_pct", "ann_vol_pct", "excess_sharpe",
                   "max_drawdown_pct", "calmar"]].to_string(index=False))
    print("\n=== Layer ablation (all walk-forward, net of costs) ===")
    print(ablation.to_string(index=False))
    print("\n=== Active vs benchmarks ===")
    print(rel.to_string(index=False))
    print("\n=== Bootstrap Sharpe-difference tests ===")
    print(sharpe_tests.to_string(index=False))
    print("\n=== Timing permutation test ===")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in perm.items()}, indent=2))
    print(f"\nWrote report: {out_dir / 'macro_seasons_v2_report.html'}")


# ---------------------------------------------------------------------------
# V3 driver: acceptance tests, then the two pre-registered combined configs
# ---------------------------------------------------------------------------

ACCEPTANCE_TESTS: dict[str, tuple[str, float, float]] = {
    "T0": ("base full stack (toggles off, sma10, xsmom+trend on)", 0.700, 0.834),
    "T1": ("use_realrate only (H8)", 0.690, 0.966),
    "T2": ("use_credit only (H3)", 0.752, 0.883),
    "T3": ("trend_mode=daily200 only (H2)", 0.724, 0.924),
    "T4": ("H5 ensemble over unenhanced streams (base full stack book)", 0.804, 0.782),
}
TOL_SHARPE = 0.003
TOL_CALMAR = 0.01


def acceptance_check(test_id: str, stats: dict[str, object]) -> bool:
    name, exp_sharpe, exp_calmar = ACCEPTANCE_TESTS[test_id]
    d_sharpe = float(stats["excess_sharpe"]) - exp_sharpe
    d_calmar = float(stats["calmar"]) - exp_calmar
    ok = abs(d_sharpe) <= TOL_SHARPE and abs(d_calmar) <= TOL_CALMAR
    print(f"[acceptance {test_id}] {'PASS' if ok else 'FAIL'} — {name}: "
          f"Sharpe {float(stats['excess_sharpe']):.4f} vs {exp_sharpe:.3f} "
          f"(diff {d_sharpe:+.4f}, tol {TOL_SHARPE}), "
          f"Calmar {float(stats['calmar']):.4f} vs {exp_calmar:.3f} "
          f"(diff {d_calmar:+.4f}, tol {TOL_CALMAR})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro Seasons 3.0 merged final strategy")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--no-extend", action="store_true")
    parser.add_argument("--output-dir", default=str(EXPORTS / "research"))
    parser.add_argument("--skip-configs", action="store_true",
                        help="run causality checks + acceptance tests only, then exit")
    args = parser.parse_args()
    allow_network = not args.no_network

    prices, extension_notes = build_price_panel(allow_network, extend=not args.no_extend)
    pit_path = DATA / "factors_point_in_time.csv"
    factors_pit = load_wide_csv(pit_path) if pit_path.exists() else pd.DataFrame()
    print(f"[data] price panel: {prices.shape[0]} months x {prices.shape[1]} assets "
          f"({prices.index.min():%Y-%m} -> {prices.index.max():%Y-%m})")

    pillars = build_pillars(prices, factors_pit, allow_network)
    probs = season_probabilities(pillars.composites)
    month_end = pd.DatetimeIndex(prices.index)

    # --- rebuilt series + causality self-checks (2 truncation dates each) ---
    realrate_causality_check(month_end, allow_network)
    shift, _real_yield = build_realrate_shift(month_end, allow_network)

    raw_spreads = load_credit_spreads(allow_network)
    credit_causality_check(raw_spreads, month_end)
    credit = build_credit_stress(raw_spreads, month_end)
    stress = credit["credit_stress"]

    tickers = sorted(set(daily_risk_tickers()) | set(DAILY_PROXIES.values()))
    daily_prices = download_yahoo_daily(tickers, DAILY_START, allow_network)
    daily_ret = stitched_daily_returns(daily_prices)
    if daily_ret.empty:
        raise SystemExit("[daily] no daily data available — cannot run trend_mode=daily200.")
    trend_causality_check(daily_ret, month_end)
    daily_idx = daily_price_index(daily_ret)

    # --- shared sleeves/streams ---
    tsmom = run_tsmom_sleeve(prices)
    tsmom_stream = tsmom["tsmom_return"].rename("tsmom")

    def run_stack(realrate: bool, credit_on: bool, trend_mode: str) -> WalkForwardResult:
        if trend_mode not in ("sma10", "daily200"):
            raise ValueError(f"unknown trend_mode {trend_mode!r}")
        return run_walk_forward(
            prices, probs, cost_bps=COST_BPS, use_xsmom=True, use_trend=True,
            realrate_shift=shift if realrate else None,
            credit_stress=stress if credit_on else None,
            daily_index=daily_idx if trend_mode == "daily200" else None,
        )

    # --- acceptance tests: the merge must reproduce each variant's numbers ---
    print("\n=== ACCEPTANCE TESTS (dev window, return_date <= 2018-12-31) ===")
    wf_t0 = run_stack(realrate=False, credit_on=False, trend_mode="sma10")
    if wf_t0.ledger.empty:
        raise SystemExit("Walk-forward produced no months — check data inputs.")
    results = {"T0": acceptance_check("T0", dev_stats(wf_t0.ledger, "T0 base full stack"))}

    wf_t1 = run_stack(realrate=True, credit_on=False, trend_mode="sma10")
    results["T1"] = acceptance_check("T1", dev_stats(wf_t1.ledger, "T1 use_realrate only"))

    wf_t2 = run_stack(realrate=False, credit_on=True, trend_mode="sma10")
    results["T2"] = acceptance_check("T2", dev_stats(wf_t2.ledger, "T2 use_credit only"))

    wf_t3 = run_stack(realrate=False, credit_on=False, trend_mode="daily200")
    results["T3"] = acceptance_check("T3", dev_stats(wf_t3.ledger, "T3 trend_mode=daily200 only"))

    # T4: h5's ensemble layer over the UNENHANCED streams (base full stack book)
    wf_core = run_walk_forward(prices, probs, cost_bps=COST_BPS,
                               use_xsmom=False, use_trend=False)
    core_ledger = wf_core.ledger.set_index("return_date")
    full_ledger_t0 = wf_t0.ledger.set_index("return_date")
    cash = full_ledger_t0["cash_return"]
    core_lev = levered_stream(core_ledger["strategy_return"], cash, "seasons_core_lev")
    full_lev_t0 = levered_stream(full_ledger_t0["strategy_return"], cash, "fullstack_lev")
    streams_t4 = pd.concat([core_lev, full_lev_t0, tsmom_stream], axis=1)
    erc_causality_check(streams_t4, cash)
    combo_t4, _ = combine_erc(streams_t4, cash)
    stats_t4 = dev_stats(pd.DataFrame({"strategy_return": combo_t4,
                                       "cash_return": cash.reindex(combo_t4.index)}),
                         "T4 H5 ensemble (unenhanced streams)")
    results["T4"] = acceptance_check("T4", stats_t4)

    if not all(results.values()):
        raise SystemExit("Acceptance tests FAILED: "
                         f"{[k for k, v in results.items() if not v]} — "
                         "fix the merge; no combined run counts until all pass.")
    print("[acceptance] all five tests passed — combined configurations may run")

    if args.skip_configs:
        return

    # --- the two pre-registered combined configurations (no others) ---------
    out_root = Path(args.output_dir)
    n_extra_trials = 0
    config_specs = [
        ("A", dict(realrate=True, credit_on=True, trend_mode="daily200"), "v3_configA"),
        ("B", dict(realrate=True, credit_on=False, trend_mode="sma10"), "v3_configB"),
    ]
    for label, spec, subdir in config_specs:
        print(f"\n=== CONFIG {label}: use_realrate={spec['realrate']}, "
              f"use_credit={spec['credit_on']}, trend_mode={spec['trend_mode']}, "
              f"ensemble=[core levered, enhanced stack, TSMOM] ===")
        if label == "B":
            wf_cfg = wf_t1  # identical configuration to T1 (use_realrate, sma10): reuse
        else:
            wf_cfg = run_stack(**spec)
        n_extra_trials += 1
        ledger_cfg = wf_cfg.ledger.set_index("return_date")
        stats_book = dev_stats(wf_cfg.ledger, f"config {label} enhanced in-book stack")
        enh_lev = levered_stream(ledger_cfg["strategy_return"], cash, "fullstack_lev")
        streams_cfg = pd.concat([core_lev, enh_lev, tsmom_stream], axis=1)
        erc_causality_check(streams_cfg, cash)
        combo_cfg, diag_cfg = combine_erc(streams_cfg, cash)
        ens_ledger = pd.DataFrame({"strategy_return": combo_cfg,
                                   "cash_return": cash.reindex(combo_cfg.index)})
        stats_ens = dev_stats(ens_ledger, f"config {label} final ensemble stream")

        out_dir = out_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        wf_cfg.ledger.to_csv(out_dir / "macro_seasons_v2_ledger.csv", index=False)
        full_ens = ens_ledger.join(diag_cfg)
        full_ens.index.name = "return_date"
        full_ens.to_csv(out_dir / "v3_ensemble_ledger.csv", index_label="return_date")
        streams_cfg.to_csv(out_dir / "v3_streams.csv", index_label="return_date")
        pd.DataFrame([stats_book, stats_ens]).to_csv(out_dir / "v3_dev_summary.csv", index=False)
        print(f"[config {label}] wrote in-book ledger ({len(ledger_cfg)} months) and "
              f"ensemble ledger ({len(full_ens)} months) to {out_dir}")

    print(f"\n[trials] n_extra_trials (combined configurations evaluated) = {n_extra_trials}")
    print("NOTE: ledgers contain the full walk-forward history for holdout "
          "evaluation; only dev-window stats are printed/reported here.")


if __name__ == "__main__":
    main()
