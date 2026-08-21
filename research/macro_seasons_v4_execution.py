"""Executable IBKR overlay for the frozen Macro Seasons V4 signal model.

The V4 signal and portfolio rules remain unchanged. This module reconstructs
the actual positions implied by the three-stream ensemble, nets duplicate ETF
exposures, applies physical cash accounting, enforces exposure limits, and
estimates IBKR Pro trading and financing costs at the final portfolio level.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

import macro_seasons_v4 as m


MODEL_VERSION = "macro_seasons_v4_ibkr_execution_v2"
EXECUTION_BACKTEST_START = pd.Timestamp("2007-01-31")
SETTINGS_FILE = m.ROOT / "config" / "ibkr_execution_settings.csv"
MARGIN_TIERS_FILE = m.ROOT / "config" / "ibkr_margin_tiers.csv"
SHORT_BORROW_FILE = m.ROOT / "config" / "ibkr_short_borrow_assumptions.csv"
SHORT_PROCEEDS_TIERS_FILE = m.ROOT / "config" / "ibkr_short_proceeds_tiers.csv"


@dataclass(frozen=True)
class ExecutionSettings:
    account_nav_usd: float
    commission_per_share_usd: float
    commission_min_order_usd: float
    commission_max_trade_fraction: float
    sec_transaction_fee_rate: float
    finra_taf_per_share_usd: float
    finra_taf_max_trade_usd: float
    finra_cat_per_share_usd: float
    slippage_bps: float
    cash_interest_spread_bps: float
    cash_interest_free_usd: float
    default_short_borrow_bps: float
    short_collateral_factor: float
    gross_exposure_limit: float
    net_exposure_min: float
    net_exposure_max: float
    short_gross_limit: float
    per_asset_abs_limit: float
    bil_abs_limit: float
    max_borrow_fraction: float
    regt_reference_initial_margin_fraction: float
    portfolio_margin_account_flag: float
    portfolio_margin_minimum_nlv_usd: float
    portfolio_margin_minimum_cushion_fraction: float
    default_futures_roll_bps: float
    annual_day_count: float


@dataclass
class ExecutionArtifacts:
    current_tsmom_positions: pd.DataFrame
    current_positions: pd.DataFrame
    current_orders: pd.DataFrame
    pm_pretrade_check: pd.DataFrame
    position_history: pd.DataFrame
    execution_ledger: pd.DataFrame
    summary: pd.DataFrame
    cost_summary: pd.DataFrame
    assumptions: pd.DataFrame
    manifest: dict[str, object]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_execution_settings(path: Path = SETTINGS_FILE) -> ExecutionSettings:
    table = pd.read_csv(path)
    values = table.set_index("setting")["value"].astype(float).to_dict()
    missing = set(ExecutionSettings.__dataclass_fields__) - set(values)
    if missing:
        raise ValueError(f"Missing IBKR execution settings: {sorted(missing)}")
    settings = ExecutionSettings(**{name: float(values[name]) for name in ExecutionSettings.__dataclass_fields__})
    validate_settings(settings)
    return settings


def validate_settings(settings: ExecutionSettings) -> None:
    if settings.account_nav_usd <= 0:
        raise ValueError("account_nav_usd must be positive")
    if not 0 < settings.gross_exposure_limit <= 2.0:
        raise ValueError("gross_exposure_limit must be in (0, 2]")
    if settings.net_exposure_min > settings.net_exposure_max:
        raise ValueError("net exposure bounds are inverted")
    if settings.max_borrow_fraction > 0.5:
        raise ValueError("max_borrow_fraction cannot exceed the Reg T 50% design cap")
    if settings.per_asset_abs_limit <= 0 or settings.short_gross_limit < 0:
        raise ValueError("position limits must be positive")
    if settings.short_collateral_factor < 1.0:
        raise ValueError("short_collateral_factor cannot be below market value")
    if settings.portfolio_margin_account_flag != 1.0:
        raise ValueError("This overlay is configured for an IBKR Portfolio Margin account")
    if settings.account_nav_usd < settings.portfolio_margin_minimum_nlv_usd:
        raise ValueError("Reference NAV is below IBKR Portfolio Margin eligibility")
    if not 0 < settings.portfolio_margin_minimum_cushion_fraction < 1:
        raise ValueError("Portfolio Margin cushion must be between zero and one")


def load_margin_tiers(path: Path = MARGIN_TIERS_FILE) -> pd.DataFrame:
    tiers = pd.read_csv(path)
    for column in ["lower_usd", "upper_usd", "spread_bps"]:
        tiers[column] = pd.to_numeric(tiers[column], errors="coerce")
    tiers["upper_usd"] = tiers["upper_usd"].fillna(np.inf)
    tiers = tiers.sort_values("lower_usd").reset_index(drop=True)
    if tiers.empty or float(tiers.iloc[0]["lower_usd"]) != 0.0:
        raise ValueError("IBKR margin tiers must begin at zero")
    return tiers


def load_short_borrow_rates(path: Path = SHORT_BORROW_FILE) -> pd.Series:
    table = pd.read_csv(path)
    rates = pd.to_numeric(table["annual_borrow_bps"], errors="coerce")
    return pd.Series(rates.to_numpy(), index=table["ticker"].astype(str), name="annual_borrow_bps")


def load_short_proceeds_tiers(path: Path = SHORT_PROCEEDS_TIERS_FILE) -> pd.DataFrame:
    tiers = pd.read_csv(path)
    for column in ["lower_usd", "upper_usd", "benchmark_spread_bps"]:
        tiers[column] = pd.to_numeric(tiers[column], errors="coerce")
    tiers["upper_usd"] = tiers["upper_usd"].fillna(np.inf)
    tiers = tiers.sort_values("lower_usd").reset_index(drop=True)
    if tiers.empty or float(tiers.iloc[0]["lower_usd"]) != 0.0:
        raise ValueError("IBKR short-proceeds tiers must begin at zero")
    allowed = {"fixed_zero", "benchmark_minus"}
    if not set(tiers["rate_type"]).issubset(allowed):
        raise ValueError("Unsupported short-proceeds rate type")
    return tiers


def ibkr_fixed_commission(shares: float, trade_value_usd: float,
                          settings: ExecutionSettings) -> float:
    if shares <= 0 or trade_value_usd <= 0:
        return 0.0
    calculated = max(settings.commission_min_order_usd, shares * settings.commission_per_share_usd)
    return float(min(calculated, trade_value_usd * settings.commission_max_trade_fraction))


def ibkr_regulatory_fees(shares: float, trade_value_usd: float, is_sell: bool,
                         settings: ExecutionSettings) -> float:
    """Current US regulatory fees passed through under IBKR Pro Fixed."""
    if shares <= 0 or trade_value_usd <= 0:
        return 0.0
    cat = shares * settings.finra_cat_per_share_usd
    if not is_sell:
        return float(cat)
    sec = trade_value_usd * settings.sec_transaction_fee_rate
    taf = min(shares * settings.finra_taf_per_share_usd, settings.finra_taf_max_trade_usd)
    return float(sec + taf + cat)


def blended_margin_spread_bps(borrow_usd: float, tiers: pd.DataFrame) -> float:
    if borrow_usd <= 0:
        return 0.0
    remaining = float(borrow_usd)
    spread_dollars = 0.0
    for row in tiers.itertuples(index=False):
        width = float(row.upper_usd - row.lower_usd)
        amount = min(remaining, width)
        if amount <= 0:
            continue
        spread_dollars += amount * float(row.spread_bps)
        remaining -= amount
        if remaining <= 1e-9:
            break
    if remaining > 1e-6:
        raise ValueError("Margin tiers do not cover requested borrowing")
    return spread_dollars / borrow_usd


def blended_short_proceeds_rate(short_collateral_usd: float, benchmark_annual: float,
                                tiers: pd.DataFrame) -> float:
    """Balance-weighted IBKR Pro rate paid on segregated USD short collateral."""
    if short_collateral_usd <= 0:
        return 0.0
    remaining = float(short_collateral_usd)
    interest_dollars = 0.0
    for row in tiers.itertuples(index=False):
        width = float(row.upper_usd - row.lower_usd)
        amount = min(remaining, width)
        if amount <= 0:
            continue
        if row.rate_type == "fixed_zero":
            rate = 0.0
        else:
            rate = max(float(benchmark_annual) - float(row.benchmark_spread_bps) / 1e4, 0.0)
        interest_dollars += amount * rate
        remaining -= amount
        if remaining <= 1e-9:
            break
    if remaining > 1e-6:
        raise ValueError("Short-proceeds tiers do not cover requested collateral")
    return interest_dollars / short_collateral_usd


def _annualized_benchmark(monthly_simple: float) -> float:
    if not np.isfinite(monthly_simple) or monthly_simple <= -1.0:
        return 0.0
    return max(float((1.0 + monthly_simple) ** 12.0 - 1.0), 0.0)


def _raw_tsmom_weights(prices: pd.DataFrame, returns: pd.DataFrame,
                        as_of: pd.Timestamp) -> pd.Series:
    history = prices.loc[prices.index <= as_of]
    universe = [ticker for ticker in m.TSMOM_UNIVERSE if ticker in prices.columns]
    weights: dict[str, float] = {}
    for ticker in universe:
        series = history[ticker].dropna()
        if len(series) < m.TSMOM_LOOKBACK + 1:
            continue
        asset_12m = float(series.iloc[-1] / series.iloc[-m.TSMOM_LOOKBACK - 1] - 1.0)
        bill = history[m.RISK_FREE].dropna() if m.RISK_FREE in history else pd.Series(dtype=float)
        bill_12m = (
            float(bill.iloc[-1] / bill.iloc[-m.TSMOM_LOOKBACK - 1] - 1.0)
            if len(bill) > m.TSMOM_LOOKBACK else 0.0
        )
        realized = returns.loc[returns.index <= as_of, ticker].dropna().tail(m.VOL_WINDOW)
        if len(realized) < m.VOL_MIN_PERIODS:
            continue
        annual_vol = float(realized.std(ddof=0) * math.sqrt(12.0))
        if not np.isfinite(annual_vol) or annual_vol <= 0:
            continue
        direction = 1.0 if asset_12m - bill_12m >= 0 else -1.0
        weights[ticker] = float(
            np.clip(
                direction * m.TSMOM_POSITION_VOL / annual_vol / len(universe),
                -m.TSMOM_MAX_ABS_WEIGHT,
                m.TSMOM_MAX_ABS_WEIGHT,
            )
        )
    return pd.Series(weights, dtype=float)


def build_tsmom_position_path(prices: pd.DataFrame, cost_bps: float = m.COST_BPS
                              ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Reproduce frozen TSMOM and retain the position vectors omitted by V3."""
    returns = prices.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    idx = returns.index
    previous = pd.Series(dtype=float)
    book_history: list[float] = []
    position_rows: list[dict[str, object]] = []
    ledger_rows: list[dict[str, object]] = []

    for position in range(len(idx) - 1):
        as_of, next_date = pd.Timestamp(idx[position]), pd.Timestamp(idx[position + 1])
        weights = _raw_tsmom_weights(prices, returns, as_of)
        if len(weights) < 6:
            continue
        if len(book_history) >= 12:
            realized = float(pd.Series(book_history[-m.PORT_VOL_WINDOW:]).std(ddof=0) * math.sqrt(12.0))
            if realized > 0:
                weights *= float(np.clip(m.VOL_TARGET / realized, *m.LEVERED_SCALE_BOUNDS))

        next_returns = returns.loc[next_date].reindex(weights.index)
        available = weights[next_returns.notna()]
        if available.empty:
            continue
        gross_simple = float(available.dot(next_returns.reindex(available.index)))
        union = available.index.union(previous.index)
        turnover = float(
            (available.reindex(union).fillna(0.0) - previous.reindex(union).fillna(0.0)).abs().sum() * 0.5
        )
        cash_simple = float(returns.loc[next_date].get(m.RISK_FREE, 0.0))
        net_simple = cash_simple + gross_simple - turnover * cost_bps / 1e4
        book_history.append(math.log1p(gross_simple) if gross_simple > -0.999 else np.nan)
        previous = available
        ledger_rows.append(
            {
                "as_of": as_of,
                "return_date": next_date,
                "tsmom_return": math.log1p(net_simple),
                "tsmom_turnover": turnover,
                "tsmom_gross_exposure": float(available.abs().sum()),
            }
        )
        for ticker, weight in available.items():
            position_rows.append(
                {"as_of": as_of, "return_date": next_date, "ticker": ticker, "weight": float(weight)}
            )

    current_as_of = pd.Timestamp(idx[-1])
    current = _raw_tsmom_weights(prices, returns, current_as_of)
    if len(book_history) >= 12:
        realized = float(pd.Series(book_history[-m.PORT_VOL_WINDOW:]).std(ddof=0) * math.sqrt(12.0))
        if realized > 0:
            current *= float(np.clip(m.VOL_TARGET / realized, *m.LEVERED_SCALE_BOUNDS))
    return pd.DataFrame(position_rows), pd.DataFrame(ledger_rows), current.sort_index()


def assert_tsmom_parity(prices: pd.DataFrame, reconstructed: pd.DataFrame) -> None:
    frozen = m.run_tsmom_sleeve(prices)["tsmom_return"]
    rebuilt = reconstructed.set_index("return_date")["tsmom_return"]
    comparison = pd.concat([frozen.rename("frozen"), rebuilt.rename("rebuilt")], axis=1).dropna()
    if comparison.empty or not np.allclose(comparison["frozen"], comparison["rebuilt"], atol=1e-12):
        raise RuntimeError("TSMOM position reconstruction does not match frozen V4 returns")


def current_tsmom_position_table(prices: pd.DataFrame, positions: pd.Series,
                                  short_rates: pd.Series,
                                  settings: ExecutionSettings) -> pd.DataFrame:
    as_of = pd.Timestamp(prices.index.max())
    history = prices.loc[prices.index <= as_of]
    returns = prices.sort_index().pct_change(fill_method=None)
    bill = history[m.RISK_FREE].dropna()
    bill_12m = (
        float(bill.iloc[-1] / bill.iloc[-m.TSMOM_LOOKBACK - 1] - 1.0)
        if len(bill) > m.TSMOM_LOOKBACK else 0.0
    )
    rows: list[dict[str, object]] = []
    for ticker, weight in positions.sort_values(ascending=False).items():
        series = history[ticker].dropna()
        asset_12m = float(series.iloc[-1] / series.iloc[-m.TSMOM_LOOKBACK - 1] - 1.0)
        trailing = returns.loc[returns.index <= as_of, ticker].dropna().tail(m.VOL_WINDOW)
        annual_vol = float(trailing.std(ddof=0) * math.sqrt(12.0))
        rows.append(
            {
                "as_of": as_of,
                "ticker": ticker,
                "side": "LONG" if weight > 0 else "SHORT",
                "asset_12m_return": asset_12m,
                "bill_12m_return": bill_12m,
                "excess_12m_return": asset_12m - bill_12m,
                "trailing_36m_vol": annual_vol,
                "sleeve_weight": float(weight),
                "sleeve_weight_pct": float(weight * 100.0),
                "annual_short_borrow_bps": (
                    float(short_rates.get(ticker, settings.default_short_borrow_bps))
                    if weight < 0 else 0.0
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["gross_exposure"] = float(positions.abs().sum())
    table.attrs["net_exposure"] = float(positions.sum())
    return table


def _current_book_weights(prices: pd.DataFrame, probabilities: pd.DataFrame,
                          as_of: pd.Timestamp, enhanced: bool,
                          realrate_shift: pd.Series | None = None,
                          credit_stress: pd.Series | None = None,
                          daily_index: pd.DataFrame | None = None) -> pd.Series:
    returns = prices.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    season_weights = {season: m.template_weights_at(season, as_of, returns) for season in m.SEASONS}
    if any(weights.empty for weights in season_weights.values()):
        raise ValueError(f"Incomplete season templates at {as_of:%Y-%m-%d}")
    probabilities_row = probabilities.loc[as_of, m.SEASONS].astype(float)
    blended = pd.Series(dtype=float)
    for season in m.SEASONS:
        blended = blended.add(season_weights[season] * float(probabilities_row[season]), fill_value=0.0)
    blended = m.apply_liquidity_overlay(blended / blended.sum(), float(probabilities.loc[as_of, "liquidity_z"]))
    if enhanced:
        if realrate_shift is not None:
            blended = m.apply_realrate_rotation(blended, float(realrate_shift.get(as_of, np.nan)))
        if credit_stress is not None and bool(credit_stress.get(as_of, False)):
            blended = m.apply_credit_dimmer(blended, True)
        blended = m.apply_xsmom_tilt(blended, prices, as_of)
        blended = m.apply_trend_gate(blended, prices, as_of, daily_index)

    history = returns.loc[returns.index <= as_of, blended.index].tail(m.PORT_VOL_WINDOW)
    portfolio_history = history.mul(blended, axis=1).sum(
        axis=1, min_count=max(1, len(blended) // 2)
    )
    realized = float(portfolio_history.dropna().std(ddof=0) * math.sqrt(12.0))
    scale = float(np.clip(m.VOL_TARGET / realized, *m.VOL_SCALE_BOUNDS)) if realized > 0 else 1.0
    final = blended * scale
    if float(final.sum()) < 1.0 - 1e-12:
        final[m.RISK_FREE] = final.get(m.RISK_FREE, 0.0) + 1.0 - float(final.sum())
    return final[final.abs() > 1e-12].sort_index()


def _weights_by_return_date(run: m.WalkForwardResult) -> pd.DataFrame:
    weights = run.weights.copy()
    weights["as_of"] = pd.to_datetime(weights["as_of"])
    dates = run.ledger[["as_of", "return_date"]].copy()
    dates["as_of"] = pd.to_datetime(dates["as_of"])
    dates["return_date"] = pd.to_datetime(dates["return_date"])
    merged = weights.merge(dates, on="as_of", how="inner")
    return merged.pivot(index="return_date", columns="ticker", values="weight").fillna(0.0)


def _stream_scale_history(log_returns: pd.Series) -> pd.Series:
    trailing = (
        log_returns.rolling(m.PORT_VOL_WINDOW, min_periods=12).std(ddof=0).shift(1)
        * math.sqrt(12.0)
    )
    return (m.VOL_TARGET / trailing).clip(*m.LEVERED_SCALE_BOUNDS).fillna(1.0)


def _current_stream_scale(log_returns: pd.Series) -> float:
    realized = float(log_returns.dropna().tail(m.PORT_VOL_WINDOW).std(ddof=0) * math.sqrt(12.0))
    return float(np.clip(m.VOL_TARGET / realized, *m.LEVERED_SCALE_BOUNDS)) if realized > 0 else 1.0


def enforce_position_limits(raw_weights: pd.Series, settings: ExecutionSettings
                            ) -> tuple[pd.Series, dict[str, object]]:
    weights = raw_weights.groupby(level=0).sum().astype(float)
    weights = weights[weights.abs() > 1e-12]
    raw_gross = float(weights.abs().sum())
    raw_net = float(weights.sum())
    netted_bil = 0.0

    margin_debit = max(float(weights[weights > 0].sum()) - 1.0, 0.0)
    if margin_debit > 0 and float(weights.get(m.RISK_FREE, 0.0)) > 0:
        netted_bil = min(margin_debit, float(weights[m.RISK_FREE]))
        weights[m.RISK_FREE] -= netted_bil
        if abs(float(weights[m.RISK_FREE])) <= 1e-12:
            weights = weights.drop(m.RISK_FREE)

    clipped: list[str] = []
    for ticker in list(weights.index):
        limit = settings.bil_abs_limit if ticker == m.RISK_FREE else settings.per_asset_abs_limit
        bounded = float(np.clip(weights[ticker], -limit, limit))
        if not np.isclose(bounded, float(weights[ticker]), atol=1e-12):
            clipped.append(ticker)
            weights[ticker] = bounded

    gross = float(weights.abs().sum())
    net = float(weights.sum())
    long_gross = float(weights[weights > 0].sum())
    short_gross = float(-weights[weights < 0].sum())
    factors = [1.0]
    if gross > 0:
        factors.append(settings.gross_exposure_limit / gross)
    if short_gross > 0:
        factors.append(settings.short_gross_limit / short_gross)
    if long_gross > 0:
        factors.append((1.0 + settings.max_borrow_fraction) / long_gross)
    if net > 0:
        factors.append(settings.net_exposure_max / net)
        factors.append((1.0 + settings.max_borrow_fraction) / net)
    limit_scale = min(factors)
    if limit_scale < 1.0:
        weights *= limit_scale

    net = float(weights.sum())
    if net < settings.net_exposure_min:
        weights[m.RISK_FREE] = weights.get(m.RISK_FREE, 0.0) + settings.net_exposure_min - net

    margin_debit = max(float(weights[weights > 0].sum()) - 1.0, 0.0)
    if margin_debit > 0 and float(weights.get(m.RISK_FREE, 0.0)) > 0:
        amount = min(margin_debit, float(weights[m.RISK_FREE]))
        weights[m.RISK_FREE] -= amount
        netted_bil += amount
        if abs(float(weights[m.RISK_FREE])) <= 1e-12:
            weights = weights.drop(m.RISK_FREE)

    weights = weights[weights.abs() > 1e-12].sort_index()
    gross = float(weights.abs().sum())
    net = float(weights.sum())
    long_gross = float(weights[weights > 0].sum())
    short_gross = float(-weights[weights < 0].sum())
    cash = 1.0 - net
    unencumbered_cash = max(1.0 - long_gross, 0.0)
    margin_debit = max(long_gross - 1.0, 0.0)
    short_collateral = short_gross * settings.short_collateral_factor
    regt_reference_requirement = settings.regt_reference_initial_margin_fraction * gross
    if gross > settings.gross_exposure_limit + 1e-10:
        raise RuntimeError("Gross exposure limit failed")
    if short_gross > settings.short_gross_limit + 1e-10:
        raise RuntimeError("Short gross limit failed")
    if (
        net > settings.net_exposure_max + 1e-10
        or margin_debit > settings.max_borrow_fraction + 1e-10
    ):
        raise RuntimeError("Net exposure or borrowing limit failed")
    if any(abs(float(value)) > (settings.bil_abs_limit if ticker == m.RISK_FREE else settings.per_asset_abs_limit) + 1e-10
           for ticker, value in weights.items()):
        raise RuntimeError("Per-asset exposure limit failed")

    diagnostics = {
        "raw_gross_exposure": raw_gross,
        "raw_net_exposure": raw_net,
        "gross_exposure": gross,
        "net_exposure": net,
        "long_gross_exposure": long_gross,
        "short_gross_exposure": short_gross,
        "cash_weight": cash,
        "unencumbered_cash_weight": unencumbered_cash,
        "borrow_weight": margin_debit,
        "margin_debit_weight": margin_debit,
        "short_collateral_weight": short_collateral,
        "regt_reference_requirement": regt_reference_requirement,
        "regt_reference_buffer": 1.0 - regt_reference_requirement,
        "portfolio_margin_minimum_cushion": settings.portfolio_margin_minimum_cushion_fraction,
        "portfolio_margin_status": "LIVE_IBKR_CHECK_REQUIRED",
        "limit_scale": limit_scale,
        "netted_bil_weight": netted_bil,
        "clipped_assets": "|".join(sorted(clipped)),
    }
    return weights, diagnostics


def _aggregate_components(core_weights: pd.Series, long_weights: pd.Series,
                          tsmom_weights: pd.Series, core_scale: float,
                          long_scale: float, outer_weights: pd.Series,
                          outer_scale: float) -> tuple[dict[str, pd.Series], pd.Series]:
    components = {
        "core": core_weights * core_scale * float(outer_weights["seasons_core_lev"]) * outer_scale,
        "long_only": long_weights * long_scale * float(outer_weights["long_only_lev"]) * outer_scale,
        "tsmom": tsmom_weights * float(outer_weights["tsmom"]) * outer_scale,
    }
    total = pd.Series(dtype=float)
    for component in components.values():
        total = total.add(component, fill_value=0.0)
    return components, total.sort_index()


def _outer_current_state(streams: pd.DataFrame) -> tuple[pd.Series, float, float]:
    history = streams.dropna()
    vol = history.tail(m.ERC_VOL_WINDOW).std(ddof=0)
    inverse = 1.0 / vol.replace(0.0, np.nan)
    weights = inverse / inverse.sum()
    historical_weights = m.inverse_vol_weights(history)
    simple = np.exp(history) - 1.0
    raw_combo = pd.Series(np.log1p((historical_weights * simple).sum(axis=1)), index=history.index)
    realized = float(raw_combo.tail(m.PORT_VOL_WINDOW).std(ddof=0) * math.sqrt(12.0))
    scale = float(np.clip(m.VOL_TARGET / realized, *m.LEVERED_SCALE_BOUNDS)) if realized > 0 else 1.0
    return weights, scale, realized


def build_target_positions(prices: pd.DataFrame, probabilities: pd.DataFrame,
                           long_run: m.WalkForwardResult, core_run: m.WalkForwardResult,
                           tsmom_positions: pd.DataFrame, tsmom_ledger: pd.DataFrame,
                           current_tsmom: pd.Series, realrate_shift: pd.Series,
                           credit_stress: pd.Series, daily_index: pd.DataFrame,
                           settings: ExecutionSettings
                           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, object], pd.DataFrame]:
    long_ledger = long_run.ledger.set_index("return_date")
    core_ledger = core_run.ledger.set_index("return_date")
    cash = long_ledger["cash_return"]
    core_levered = m.levered_stream(core_ledger["strategy_return"], cash, "seasons_core_lev")
    long_levered = m.levered_stream(long_ledger["strategy_return"], cash, "long_only_lev")
    tsmom_returns = tsmom_ledger.set_index("return_date")["tsmom_return"].rename("tsmom")
    streams = pd.concat([core_levered, long_levered, tsmom_returns], axis=1)
    _ensemble, outer_diagnostics = m.combine_erc(streams, cash)

    core_wide = _weights_by_return_date(core_run)
    long_wide = _weights_by_return_date(long_run)
    tsmom_wide = tsmom_positions.pivot(index="return_date", columns="ticker", values="weight").fillna(0.0)
    core_scales = _stream_scale_history(core_ledger["strategy_return"])
    long_scales = _stream_scale_history(long_ledger["strategy_return"])
    common = core_wide.index.intersection(long_wide.index).intersection(tsmom_wide.index).intersection(outer_diagnostics.index)
    # Keep the public comparison window consistent across the long-only and L/S
    # portfolios. The extended price panel supplies the required 2007 history.
    common = common[common >= EXECUTION_BACKTEST_START]

    position_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    as_of_map = long_run.ledger.set_index("return_date")["as_of"]
    for return_date in common:
        outer_row = outer_diagnostics.loc[return_date]
        outer_weights = pd.Series(
            {
                "seasons_core_lev": float(outer_row["w_seasons_core_lev"]),
                "long_only_lev": float(outer_row["w_long_only_lev"]),
                "tsmom": float(outer_row["w_tsmom"]),
            }
        )
        components, raw = _aggregate_components(
            core_wide.loc[return_date],
            long_wide.loc[return_date],
            tsmom_wide.loc[return_date],
            float(core_scales.loc[return_date]),
            float(long_scales.loc[return_date]),
            outer_weights,
            float(outer_row["vol_target_scale"]),
        )
        constrained, limit_diagnostics = enforce_position_limits(raw, settings)
        as_of = pd.Timestamp(as_of_map.loc[return_date])
        tickers = sorted(set(raw.index) | set(constrained.index))
        for ticker in tickers:
            position_rows.append(
                {
                    "as_of": as_of,
                    "return_date": pd.Timestamp(return_date),
                    "ticker": ticker,
                    "core_component": float(components["core"].get(ticker, 0.0)),
                    "long_only_component": float(components["long_only"].get(ticker, 0.0)),
                    "tsmom_component": float(components["tsmom"].get(ticker, 0.0)),
                    "raw_weight": float(raw.get(ticker, 0.0)),
                    "target_weight": float(constrained.get(ticker, 0.0)),
                }
            )
        diagnostics_rows.append(
            {
                "as_of": as_of,
                "return_date": pd.Timestamp(return_date),
                "inner_core_scale": float(core_scales.loc[return_date]),
                "inner_long_only_scale": float(long_scales.loc[return_date]),
                "outer_weight_core": float(outer_weights["seasons_core_lev"]),
                "outer_weight_long_only": float(outer_weights["long_only_lev"]),
                "outer_weight_tsmom": float(outer_weights["tsmom"]),
                "outer_scale": float(outer_row["vol_target_scale"]),
                **limit_diagnostics,
            }
        )

    current_as_of = pd.Timestamp(prices.index.max())
    current_core = _current_book_weights(prices, probabilities, current_as_of, enhanced=False)
    current_long = _current_book_weights(
        prices,
        probabilities,
        current_as_of,
        enhanced=True,
        realrate_shift=realrate_shift,
        credit_stress=credit_stress,
        daily_index=daily_index,
    )
    outer_weights, outer_scale, raw_vol = _outer_current_state(streams)
    current_components, current_raw = _aggregate_components(
        current_core,
        current_long,
        current_tsmom,
        _current_stream_scale(core_ledger["strategy_return"]),
        _current_stream_scale(long_ledger["strategy_return"]),
        outer_weights,
        outer_scale,
    )
    current_target, current_limits = enforce_position_limits(current_raw, settings)
    current_metadata = {
        "as_of": current_as_of,
        "effective_month": (current_as_of + pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
        "inner_core_scale": _current_stream_scale(core_ledger["strategy_return"]),
        "inner_long_only_scale": _current_stream_scale(long_ledger["strategy_return"]),
        "outer_weight_core": float(outer_weights["seasons_core_lev"]),
        "outer_weight_long_only": float(outer_weights["long_only_lev"]),
        "outer_weight_tsmom": float(outer_weights["tsmom"]),
        "outer_scale": outer_scale,
        "outer_raw_vol": raw_vol,
        **current_limits,
    }
    current_component_table = pd.DataFrame(
        {
            "core_component": current_components["core"],
            "long_only_component": current_components["long_only"],
            "tsmom_component": current_components["tsmom"],
            "raw_weight": current_raw,
            "target_weight": current_target,
        }
    ).fillna(0.0)
    return (
        pd.DataFrame(position_rows),
        pd.DataFrame(diagnostics_rows),
        current_target,
        current_metadata,
        current_component_table,
    )


def _cost_for_trade_deltas(deltas: pd.Series, nav_usd: float, prices: pd.Series,
                           settings: ExecutionSettings
                           ) -> tuple[float, float, float, list[dict[str, float]]]:
    commission = 0.0
    regulatory_fees = 0.0
    traded_notional = 0.0
    details: list[dict[str, float]] = []
    for ticker, delta in deltas.items():
        trade_value = abs(float(delta)) * nav_usd
        if trade_value <= 0.01:
            continue
        price = float(prices.get(ticker, np.nan))
        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"Missing execution price for {ticker}")
        shares = trade_value / price
        order_commission = ibkr_fixed_commission(shares, trade_value, settings)
        order_regulatory = ibkr_regulatory_fees(
            shares, trade_value, is_sell=float(delta) < 0.0, settings=settings
        )
        commission += order_commission
        regulatory_fees += order_regulatory
        traded_notional += trade_value
        details.append(
            {
                "ticker": ticker,
                "delta_weight": float(delta),
                "trade_value_usd": trade_value,
                "reference_price": price,
                "estimated_shares": shares,
                "commission_usd": order_commission,
                "regulatory_fees_usd": order_regulatory,
            }
        )
    slippage = traded_notional * settings.slippage_bps / 1e4
    return commission, regulatory_fees, slippage, details


def simulate_execution(target_history: pd.DataFrame, diagnostics: pd.DataFrame,
                       prices: pd.DataFrame, cash_log_returns: pd.Series,
                       settings: ExecutionSettings, margin_tiers: pd.DataFrame,
                       short_borrow_rates: pd.Series,
                       short_proceeds_tiers: pd.DataFrame,
                       futures_roll_bps: pd.Series | None = None
                       ) -> tuple[pd.DataFrame, pd.Series, list[dict[str, float]]]:
    targets = target_history.pivot(index="return_date", columns="ticker", values="target_weight").fillna(0.0)
    as_of_map = target_history.groupby("return_date")["as_of"].first()
    simple_returns = prices.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    diagnostics_by_date = diagnostics.set_index("return_date")
    futures_roll_bps = futures_roll_bps if futures_roll_bps is not None else pd.Series(dtype=float)
    nav = settings.account_nav_usd
    previous_end_weights = pd.Series(dtype=float)
    ledger_rows: list[dict[str, object]] = []
    final_trade_details: list[dict[str, float]] = []

    for return_date in targets.index:
        as_of = pd.Timestamp(as_of_map.loc[return_date])
        target = targets.loc[return_date]
        target = target[target.abs() > 1e-12]
        union = target.index.union(previous_end_weights.index)
        deltas = target.reindex(union).fillna(0.0) - previous_end_weights.reindex(union).fillna(0.0)
        signal_prices = prices.loc[as_of].reindex(union)
        commission_usd, regulatory_fees_usd, slippage_usd, trade_details = _cost_for_trade_deltas(
            deltas, nav, signal_prices, settings
        )
        final_trade_details = trade_details

        next_returns = simple_returns.loc[return_date].reindex(target.index)
        if next_returns.isna().any():
            missing = next_returns[next_returns.isna()].index.tolist()
            raise ValueError(f"Missing realized returns at {return_date:%Y-%m-%d}: {missing}")
        asset_return = float(target.dot(next_returns))
        long_gross = float(target[target > 0].sum())
        short_gross = float(-target[target < 0].sum())
        cash_weight = 1.0 - float(target.sum())
        unencumbered_cash_weight = max(1.0 - long_gross, 0.0)
        margin_debit_weight = max(long_gross - 1.0, 0.0)
        short_collateral_weight = short_gross * settings.short_collateral_factor
        cash_simple = float(np.expm1(cash_log_returns.reindex([return_date]).fillna(0.0).iloc[0]))
        benchmark_annual = _annualized_benchmark(cash_simple)
        days = max(int((pd.Timestamp(return_date) - as_of).days), 1)

        positive_cash_interest = 0.0
        short_proceeds_interest = 0.0
        margin_benchmark_cost = 0.0
        margin_spread_cost = 0.0
        if unencumbered_cash_weight > 0:
            cash_usd = unencumbered_cash_weight * nav
            eligible_usd = max(cash_usd - settings.cash_interest_free_usd, 0.0)
            paid_rate = max(benchmark_annual - settings.cash_interest_spread_bps / 1e4, 0.0)
            positive_cash_interest = eligible_usd / nav * paid_rate * days / settings.annual_day_count
        if margin_debit_weight > 0:
            borrow_usd = margin_debit_weight * nav
            spread_bps = blended_margin_spread_bps(borrow_usd, margin_tiers)
            margin_benchmark_cost = margin_debit_weight * benchmark_annual * days / settings.annual_day_count
            margin_spread_cost = margin_debit_weight * spread_bps / 1e4 * days / settings.annual_day_count
        if short_collateral_weight > 0:
            proceeds_rate = blended_short_proceeds_rate(
                short_collateral_weight * nav, benchmark_annual, short_proceeds_tiers
            )
            short_proceeds_interest = (
                short_collateral_weight * proceeds_rate * days / settings.annual_day_count
            )

        short_borrow_cost = 0.0
        for ticker, weight in target[target < 0].items():
            annual_bps = float(short_borrow_rates.get(ticker, settings.default_short_borrow_bps))
            short_borrow_cost += (
                -float(weight)
                * settings.short_collateral_factor
                * annual_bps
                / 1e4
                * days
                / settings.annual_day_count
            )

        futures_roll_cost = 0.0
        for ticker, weight in target.items():
            annual_bps = float(futures_roll_bps.get(ticker, 0.0))
            futures_roll_cost += abs(float(weight)) * annual_bps / 1e4 * days / settings.annual_day_count

        commission_cost = commission_usd / nav
        regulatory_fee_cost = regulatory_fees_usd / nav
        slippage_cost = slippage_usd / nav
        net_simple = (
            asset_return
            + positive_cash_interest
            + short_proceeds_interest
            - margin_benchmark_cost
            - margin_spread_cost
            - short_borrow_cost
            - futures_roll_cost
            - commission_cost
            - regulatory_fee_cost
            - slippage_cost
        )
        if net_simple <= -0.999:
            raise RuntimeError(f"Execution portfolio lost more than 99.9% at {return_date}")

        asset_end = target * (1.0 + next_returns)
        cash_end = (
            cash_weight
            + positive_cash_interest
            + short_proceeds_interest
            - margin_benchmark_cost
            - margin_spread_cost
            - short_borrow_cost
            - futures_roll_cost
            - commission_cost
            - regulatory_fee_cost
            - slippage_cost
        )
        end_total = float(asset_end.sum() + cash_end)
        if not np.isclose(end_total, 1.0 + net_simple, atol=1e-10):
            raise RuntimeError("Execution cash accounting does not reconcile")
        previous_end_weights = asset_end / end_total
        nav *= end_total

        diag = diagnostics_by_date.loc[return_date]
        ledger_rows.append(
            {
                "as_of": as_of,
                "return_date": pd.Timestamp(return_date),
                "gross_return": math.log1p(
                    asset_return
                    + positive_cash_interest
                    + short_proceeds_interest
                    - margin_benchmark_cost
                ),
                "net_return": math.log1p(net_simple),
                "nav_usd": nav,
                "turnover": float(deltas.abs().sum() * 0.5),
                "traded_notional_usd": float(deltas.abs().sum() * nav / max(end_total, 1e-12)),
                "commission_usd": commission_usd,
                "commission_cost": commission_cost,
                "regulatory_fees_usd": regulatory_fees_usd,
                "regulatory_fee_cost": regulatory_fee_cost,
                "slippage_usd": slippage_usd,
                "slippage_cost": slippage_cost,
                "positive_cash_interest": positive_cash_interest,
                "short_proceeds_interest": short_proceeds_interest,
                "margin_benchmark_cost": margin_benchmark_cost,
                "margin_spread_cost": margin_spread_cost,
                "short_borrow_cost": short_borrow_cost,
                "futures_roll_cost": futures_roll_cost,
                "total_explicit_cost": (
                    commission_cost
                    + regulatory_fee_cost
                    + slippage_cost
                    + margin_spread_cost
                    + short_borrow_cost
                    + futures_roll_cost
                ),
                "cash_weight": cash_weight,
                "unencumbered_cash_weight": unencumbered_cash_weight,
                "margin_debit_weight": margin_debit_weight,
                "short_collateral_weight": short_collateral_weight,
                "long_gross_exposure": long_gross,
                "gross_exposure": float(diag["gross_exposure"]),
                "net_exposure": float(diag["net_exposure"]),
                "short_gross_exposure": float(diag["short_gross_exposure"]),
                "borrow_weight": float(diag["borrow_weight"]),
                "regt_reference_requirement": float(diag["regt_reference_requirement"]),
                "regt_reference_buffer": float(diag["regt_reference_buffer"]),
                "portfolio_margin_minimum_cushion": float(
                    diag["portfolio_margin_minimum_cushion"]
                ),
                "portfolio_margin_status": str(diag["portfolio_margin_status"]),
                "limit_scale": float(diag["limit_scale"]),
                "netted_bil_weight": float(diag["netted_bil_weight"]),
                "clipped_assets": str(diag["clipped_assets"]),
            }
        )
    return pd.DataFrame(ledger_rows), previous_end_weights, final_trade_details


def performance_row(name: str, log_returns: pd.Series, cash_log: pd.Series) -> dict[str, object]:
    returns = log_returns.dropna()
    cash = cash_log.reindex(returns.index).fillna(0.0)
    equity = np.exp(returns.cumsum())
    drawdown = float((equity / equity.cummax() - 1.0).min())
    years = len(returns) / 12.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0)
    excess = returns - cash
    return {
        "series": name,
        "months": len(returns),
        "cagr_pct": round(cagr * 100.0, 2),
        "ann_vol_pct": round(float(returns.std(ddof=0) * math.sqrt(12.0) * 100.0), 2),
        "excess_sharpe": round(float(excess.mean() / excess.std(ddof=0) * math.sqrt(12.0)), 3),
        "max_dd_pct": round(drawdown * 100.0, 2),
        "calmar": round(cagr / abs(drawdown), 3) if drawdown < 0 else np.nan,
    }


def _current_position_table(current_target: pd.Series, component_table: pd.DataFrame,
                            current_metadata: dict[str, object], prices: pd.DataFrame,
                            settings: ExecutionSettings, short_rates: pd.Series) -> pd.DataFrame:
    as_of = pd.Timestamp(current_metadata["as_of"])
    rows: list[dict[str, object]] = []
    for ticker, weight in current_target.sort_values(ascending=False).items():
        price = float(prices.loc[as_of, ticker])
        notional = float(weight) * settings.account_nav_usd
        rows.append(
            {
                "as_of": as_of,
                "effective_month": current_metadata["effective_month"],
                "ticker": ticker,
                "side": "LONG" if weight > 0 else "SHORT",
                "core_component": float(component_table.loc[ticker, "core_component"]),
                "long_only_component": float(component_table.loc[ticker, "long_only_component"]),
                "tsmom_component": float(component_table.loc[ticker, "tsmom_component"]),
                "raw_weight": float(component_table.loc[ticker, "raw_weight"]),
                "target_weight": float(weight),
                "target_weight_pct": float(weight * 100.0),
                "target_notional_usd": notional,
                "reference_price": price,
                "estimated_shares": notional / price,
                "annual_short_borrow_bps": float(short_rates.get(ticker, settings.default_short_borrow_bps)) if weight < 0 else 0.0,
                "annual_futures_roll_bps": 0.0,
            }
        )
    cash = float(current_metadata["cash_weight"])
    rows.append(
        {
            "as_of": as_of,
            "effective_month": current_metadata["effective_month"],
            "ticker": "USD_CASH",
            "side": "CASH" if cash >= 0 else "BORROW",
            "core_component": np.nan,
            "long_only_component": np.nan,
            "tsmom_component": np.nan,
            "raw_weight": np.nan,
            "target_weight": cash,
            "target_weight_pct": cash * 100.0,
            "target_notional_usd": cash * settings.account_nav_usd,
            "reference_price": 1.0,
            "estimated_shares": cash * settings.account_nav_usd,
            "annual_short_borrow_bps": 0.0,
            "annual_futures_roll_bps": 0.0,
        }
    )
    return pd.DataFrame(rows)


def _current_orders(current_positions: pd.DataFrame, previous_end: pd.Series,
                    prices: pd.DataFrame, settings: ExecutionSettings) -> pd.DataFrame:
    securities = current_positions.loc[current_positions["ticker"] != "USD_CASH"].set_index("ticker")["target_weight"]
    union = securities.index.union(previous_end.index)
    deltas = securities.reindex(union).fillna(0.0) - previous_end.reindex(union).fillna(0.0)
    as_of = pd.Timestamp(current_positions["as_of"].iloc[0])
    rows: list[dict[str, object]] = []
    for ticker, delta in deltas[deltas.abs() > 1e-8].sort_values(ascending=False).items():
        price = float(prices.loc[as_of, ticker])
        value = abs(float(delta)) * settings.account_nav_usd
        shares = value / price
        rows.append(
            {
                "as_of": as_of,
                "ticker": ticker,
                "action": "BUY" if delta > 0 else "SELL",
                "delta_weight": float(delta),
                "delta_weight_pct": float(delta * 100.0),
                "trade_notional_usd": value,
                "reference_price": price,
                "estimated_shares": shares,
                "estimated_commission_usd": ibkr_fixed_commission(shares, value, settings),
                "estimated_regulatory_fees_usd": ibkr_regulatory_fees(
                    shares, value, is_sell=float(delta) < 0.0, settings=settings
                ),
                "estimated_slippage_usd": value * settings.slippage_bps / 1e4,
            }
        )
    return pd.DataFrame(rows)


def build_execution_artifacts(prices: pd.DataFrame, probabilities: pd.DataFrame,
                              realrate_shift: pd.Series, credit_stress: pd.Series,
                              daily_index: pd.DataFrame) -> ExecutionArtifacts:
    settings = load_execution_settings()
    margin_tiers = load_margin_tiers()
    short_rates = load_short_borrow_rates()
    short_proceeds_tiers = load_short_proceeds_tiers()

    long_run = m.run_walk_forward(
        prices,
        probabilities,
        cost_bps=m.COST_BPS,
        use_xsmom=True,
        use_trend=True,
        realrate_shift=realrate_shift,
        credit_stress=credit_stress,
        daily_index=daily_index,
    )
    core_run = m.run_walk_forward(prices, probabilities, cost_bps=m.COST_BPS)
    tsmom_positions, tsmom_ledger, current_tsmom = build_tsmom_position_path(prices)
    assert_tsmom_parity(prices, tsmom_ledger)

    target_history, diagnostics, current_target, current_metadata, component_table = build_target_positions(
        prices,
        probabilities,
        long_run,
        core_run,
        tsmom_positions,
        tsmom_ledger,
        current_tsmom,
        realrate_shift,
        credit_stress,
        daily_index,
        settings,
    )
    cash_log = long_run.ledger.set_index("return_date")["cash_return"]
    execution_ledger, previous_end, _ = simulate_execution(
        target_history,
        diagnostics,
        prices,
        cash_log,
        settings,
        margin_tiers,
        short_rates,
        short_proceeds_tiers,
    )
    current_positions = _current_position_table(
        current_target, component_table, current_metadata, prices, settings, short_rates
    )
    current_tsmom_positions = current_tsmom_position_table(
        prices, current_tsmom, short_rates, settings
    )
    current_orders = _current_orders(current_positions, previous_end, prices, settings)
    pm_pretrade_check = pd.DataFrame(
        [
            {
                "as_of": current_metadata["as_of"],
                "effective_month": current_metadata["effective_month"],
                "account_type": "PORTFOLIO_MARGIN",
                "status": "LIVE_IBKR_CHECK_REQUIRED",
                "reference_nav_usd": settings.account_nav_usd,
                "minimum_eligible_nlv_usd": settings.portfolio_margin_minimum_nlv_usd,
                "minimum_required_cushion_fraction": settings.portfolio_margin_minimum_cushion_fraction,
                "gross_exposure": current_metadata["gross_exposure"],
                "net_exposure": current_metadata["net_exposure"],
                "margin_debit_weight": current_metadata["margin_debit_weight"],
                "short_collateral_weight": current_metadata["short_collateral_weight"],
                "regt_reference_requirement": current_metadata["regt_reference_requirement"],
                "ibkr_projected_initial_margin_usd": np.nan,
                "ibkr_projected_maintenance_margin_usd": np.nan,
                "ibkr_projected_excess_liquidity_usd": np.nan,
                "ibkr_projected_cushion_fraction": np.nan,
                "checked_at": "",
                "instruction": (
                    "Run the complete target through IBKR TWS Check Margin or API what-if; "
                    "do not trade unless the projected cushion is at least the configured minimum."
                ),
            }
        ]
    )

    no_explicit_cost_settings = replace(
        settings,
        commission_per_share_usd=0.0,
        commission_min_order_usd=0.0,
        sec_transaction_fee_rate=0.0,
        finra_taf_per_share_usd=0.0,
        finra_taf_max_trade_usd=0.0,
        finra_cat_per_share_usd=0.0,
        slippage_bps=0.0,
        cash_interest_spread_bps=0.0,
        cash_interest_free_usd=0.0,
        default_short_borrow_bps=0.0,
    )
    zero_short = short_rates * 0.0
    benchmark_short_proceeds = short_proceeds_tiers.copy()
    benchmark_short_proceeds["rate_type"] = "benchmark_minus"
    benchmark_short_proceeds["benchmark_spread_bps"] = 0.0
    no_cost_ledger, _, _ = simulate_execution(
        target_history,
        diagnostics,
        prices,
        cash_log,
        no_explicit_cost_settings,
        margin_tiers.assign(spread_bps=0.0),
        zero_short,
        benchmark_short_proceeds,
    )

    frozen_core = core_run.ledger.set_index("return_date")
    frozen_long = long_run.ledger.set_index("return_date")
    core_levered = m.levered_stream(frozen_core["strategy_return"], cash_log, "seasons_core_lev")
    long_levered = m.levered_stream(frozen_long["strategy_return"], cash_log, "long_only_lev")
    frozen_streams = pd.concat(
        [core_levered, long_levered, tsmom_ledger.set_index("return_date")["tsmom_return"].rename("tsmom")],
        axis=1,
    )
    frozen_ensemble, _ = m.combine_erc(frozen_streams, cash_log)
    summary = pd.DataFrame(
        [
            performance_row("Frozen return-level ensemble", frozen_ensemble, cash_log),
            performance_row(
                "Frozen return-level ensemble on executable window",
                frozen_ensemble.reindex(execution_ledger["return_date"]).dropna(),
                cash_log,
            ),
            performance_row(
                "Executable physical accounting before explicit IBKR costs",
                no_cost_ledger.set_index("return_date")["net_return"],
                cash_log,
            ),
            performance_row(
                "IBKR-costed executable ensemble",
                execution_ledger.set_index("return_date")["net_return"],
                cash_log,
            ),
        ]
    )

    years = len(execution_ledger) / 12.0
    cost_columns = [
        "commission_cost",
        "regulatory_fee_cost",
        "slippage_cost",
        "margin_benchmark_cost",
        "margin_spread_cost",
        "short_borrow_cost",
        "futures_roll_cost",
    ]
    cost_rows = [
            {
                "cost_component": column,
                "cumulative_pct_of_nav": float(execution_ledger[column].sum() * 100.0),
                "average_annual_bps": float(execution_ledger[column].sum() / years * 1e4),
            }
            for column in cost_columns
        ]
    cost_rows.append(
        {
            "cost_component": "short_proceeds_interest_credit",
            "cumulative_pct_of_nav": float(-execution_ledger["short_proceeds_interest"].sum() * 100.0),
            "average_annual_bps": float(
                -execution_ledger["short_proceeds_interest"].sum() / years * 1e4
            ),
        }
    )
    cost_summary = pd.DataFrame(cost_rows)

    assumptions = pd.read_csv(SETTINGS_FILE)
    assumptions.insert(0, "category", "execution_setting")
    assumptions = pd.concat(
        [
            assumptions,
            pd.DataFrame(
                {
                    "category": "margin_tier",
                    "setting": [f"margin_{row.lower_usd:g}_{row.upper_usd:g}" for row in margin_tiers.itertuples()],
                    "value": margin_tiers["spread_bps"],
                    "unit": "bps above benchmark",
                    "source_or_rationale": margin_tiers["source_or_rationale"],
                }
            ),
            pd.DataFrame(
                {
                    "category": "short_proceeds_tier",
                    "setting": [f"short_proceeds_{row.lower_usd:g}_{row.upper_usd:g}" for row in short_proceeds_tiers.itertuples()],
                    "value": short_proceeds_tiers["benchmark_spread_bps"],
                    "unit": short_proceeds_tiers["rate_type"],
                    "source_or_rationale": short_proceeds_tiers["source_or_rationale"],
                }
            ),
            pd.DataFrame(
                {
                    "category": "short_borrow_proxy",
                    "setting": "borrow_" + short_rates.index.astype(str),
                    "value": short_rates.to_numpy(),
                    "unit": "annual bps",
                    "source_or_rationale": "Replace with same-day IBKR SLB rate before trading",
                }
            ),
        ],
        ignore_index=True,
    )
    assumptions["model_version"] = MODEL_VERSION
    manifest = {
        "model_version": MODEL_VERSION,
        "as_of": str(pd.Timestamp(prices.index.max()).date()),
        "execution_backtest_start": str(EXECUTION_BACKTEST_START.date()),
        "current_metadata": {
            key: (str(value.date()) if isinstance(value, pd.Timestamp) else value)
            for key, value in current_metadata.items()
        },
        "input_hashes": {
            str(path.relative_to(m.ROOT)): _sha256(path)
            for path in [
                SETTINGS_FILE,
                MARGIN_TIERS_FILE,
                SHORT_BORROW_FILE,
                SHORT_PROCEEDS_TIERS_FILE,
                m.ROOT / "research" / "macro_seasons_v4.py",
                m.ROOT / "research" / "macro_seasons_v4_execution.py",
                m.DATA / "prices_macro_seasons_extended.csv",
                m.DATA / "factors_point_in_time.csv",
            ]
            if path.exists()
        },
        "broker_assumptions_reviewed_on": "2026-08-20",
        "margin_policy": {
            "account_type": "PORTFOLIO_MARGIN",
            "exact_requirement_source": "LIVE_IBKR_CHECK_MARGIN",
            "minimum_projected_cushion_fraction": settings.portfolio_margin_minimum_cushion_fraction,
            "regt_result_is_reference_only": True,
        },
        "sources": {
            "commissions": "https://www.interactivebrokers.com/en/pricing/commissions-stocks.php",
            "margin_rates": "https://www.interactivebrokers.com/en/trading/pricing-margin-rates.php",
            "cash_interest": "https://www.interactivebrokers.com/en/accounts/fees/pricing-interest-rates.php",
            "short_sale_cost": "https://www.interactivebrokers.com/en/pricing/short-sale-cost.php",
            "margin_requirements": "https://www.interactivebrokers.com/en/trading/margin-stocks.php",
            "portfolio_margin_monitoring": "https://www.ibkrguides.com/traderworkstation/margin-monitoring.htm",
            "portfolio_margin_eligibility": "https://www.ibkrguides.com/clientportal/accounttype.htm",
            "short_collateral": "https://www.interactivebrokers.com/campus/glossary-terms/collateral-short-sale/",
            "short_proceeds_examples": "https://www.interactivebrokers.com/en/accounts/fees/interestPaid_Example2.php",
        },
    }
    return ExecutionArtifacts(
        current_tsmom_positions=current_tsmom_positions,
        current_positions=current_positions,
        current_orders=current_orders,
        pm_pretrade_check=pm_pretrade_check,
        position_history=target_history.merge(diagnostics, on=["as_of", "return_date"], how="left"),
        execution_ledger=execution_ledger,
        summary=summary,
        cost_summary=cost_summary,
        assumptions=assumptions,
        manifest=manifest,
    )


def write_execution_artifacts(artifacts: ExecutionArtifacts, output_dir: Path = m.EXPORTS) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.current_tsmom_positions.to_csv(
        output_dir / "macro_seasons_v4_execution_current_tsmom.csv", index=False
    )
    artifacts.current_positions.to_csv(output_dir / "macro_seasons_v4_execution_current_positions.csv", index=False)
    artifacts.current_orders.to_csv(output_dir / "macro_seasons_v4_execution_current_orders.csv", index=False)
    artifacts.pm_pretrade_check.to_csv(
        output_dir / "macro_seasons_v4_execution_pm_pretrade_check.csv", index=False
    )
    artifacts.position_history.to_csv(output_dir / "macro_seasons_v4_execution_position_history.csv", index=False)
    artifacts.execution_ledger.to_csv(output_dir / "macro_seasons_v4_execution_ledger.csv", index=False)
    artifacts.summary.to_csv(output_dir / "macro_seasons_v4_execution_summary.csv", index=False)
    artifacts.cost_summary.to_csv(output_dir / "macro_seasons_v4_execution_cost_summary.csv", index=False)
    artifacts.assumptions.to_csv(output_dir / "macro_seasons_v4_execution_assumptions.csv", index=False)
    manifest_path = output_dir / "macro_seasons_v4_execution_manifest.json"
    manifest_path.write_text(json.dumps(artifacts.manifest, indent=2, sort_keys=True) + "\n")
