"""Macro Seasons v4 production runner: PIT macro data and frozen v3 rules."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research"))

import macro_seasons_v4 as m  # noqa: E402


FREEZE_DATE = pd.Timestamp("2026-08-20")
FREEZE_DATA_END = pd.Timestamp("2026-07-31")
FIRST_LIVE_AS_OF = pd.Timestamp("2026-08-31")
FIRST_LIVE_RETURN_DATE = pd.Timestamp("2026-09-30")
MONITOR_MIN_SHARPE_12M = -0.5
MONITOR_MAX_DD_BOOK = -0.08
MONITOR_MAX_DD_ENSEMBLE = -0.15
EXPORTS = ROOT / "exports"


def stats_row(name: str, returns: pd.Series, cash: pd.Series) -> dict[str, object]:
    returns = returns.dropna()
    cash = cash.reindex(returns.index).fillna(0.0)
    excess = returns - cash
    equity = np.exp(returns.cumsum())
    years = len(returns) / 12.0
    drawdown = float((equity / equity.cummax() - 1.0).min())
    cagr = float((equity.iloc[-1] ** (1.0 / years) - 1.0) * 100.0)
    return {
        "series": name,
        "months": len(returns),
        "cagr_pct": round(cagr, 2),
        "ann_vol_pct": round(float(returns.std(ddof=0) * math.sqrt(12.0) * 100.0), 2),
        "excess_sharpe": round(
            float(excess.mean() / excess.std(ddof=0) * math.sqrt(12.0)), 3
        ) if excess.std(ddof=0) > 0 else np.nan,
        "raw_sharpe": round(
            float(returns.mean() / returns.std(ddof=0) * math.sqrt(12.0)), 3
        ) if returns.std(ddof=0) > 0 else np.nan,
        "max_dd_pct": round(drawdown * 100.0, 2),
        "calmar": round(cagr / abs(drawdown * 100.0), 3) if drawdown < 0 else np.nan,
    }


def monitor(name: str, returns: pd.Series, cash: pd.Series,
            drawdown_limit: float) -> dict[str, object]:
    returns = returns.dropna()
    live = returns.loc[returns.index >= FIRST_LIVE_RETURN_DATE]
    if live.empty:
        return {
            "series": name,
            "freeze_date": FREEZE_DATE.date().isoformat(),
            "first_live_as_of": FIRST_LIVE_AS_OF.date().isoformat(),
            "first_live_return_date": FIRST_LIVE_RETURN_DATE.date().isoformat(),
            "completed_live_months": 0,
            "trailing_12m_live_excess_sharpe": np.nan,
            "live_drawdown_pct": np.nan,
            "drawdown_limit_pct": drawdown_limit * 100.0,
            "status": "PENDING: first post-freeze return due 2026-09-30",
        }

    live_cash = cash.reindex(live.index).fillna(0.0)
    live_equity = np.exp(live.cumsum())
    live_drawdown = float((live_equity / live_equity.cummax() - 1.0).iloc[-1])
    trailing = (live - live_cash).tail(12)
    sharpe = (
        float(trailing.mean() / trailing.std(ddof=0) * math.sqrt(12.0))
        if len(trailing) == 12 and trailing.std(ddof=0) > 0 else np.nan
    )
    flags: list[str] = []
    if np.isfinite(sharpe) and sharpe < MONITOR_MIN_SHARPE_12M:
        flags.append(f"12m live excess Sharpe {sharpe:.2f} < {MONITOR_MIN_SHARPE_12M}")
    if live_drawdown < drawdown_limit:
        flags.append(f"live drawdown {live_drawdown:.1%} beyond {drawdown_limit:.0%}")
    return {
        "series": name,
        "freeze_date": FREEZE_DATE.date().isoformat(),
        "first_live_as_of": FIRST_LIVE_AS_OF.date().isoformat(),
        "first_live_return_date": FIRST_LIVE_RETURN_DATE.date().isoformat(),
        "completed_live_months": int(len(live)),
        "trailing_12m_live_excess_sharpe": round(sharpe, 3) if np.isfinite(sharpe) else np.nan,
        "live_drawdown_pct": round(live_drawdown * 100.0, 2),
        "drawdown_limit_pct": drawdown_limit * 100.0,
        "status": "REVIEW: " + "; ".join(flags) if flags else "OK",
    }


def current_target_weights(prices: pd.DataFrame, probabilities: pd.DataFrame,
                           realrate_shift: pd.Series, credit_stress: pd.Series,
                           daily_index: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
    returns = prices.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    usable = probabilities.reindex(returns.index)
    season_weights = {
        season: m.template_weights_at(season, as_of, returns) for season in m.SEASONS
    }
    if any(weights.empty for weights in season_weights.values()):
        return pd.Series(dtype=float)
    p_row = usable.loc[as_of, m.SEASONS].astype(float)
    blended = pd.Series(dtype=float)
    for season in m.SEASONS:
        blended = blended.add(season_weights[season] * float(p_row[season]), fill_value=0.0)
    blended = m.apply_liquidity_overlay(
        blended / blended.sum(), float(usable.loc[as_of, "liquidity_z"])
    )
    blended = m.apply_realrate_rotation(blended, float(realrate_shift.get(as_of, np.nan)))
    if bool(credit_stress.get(as_of, False)):
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
    cash_pad = 1.0 - float(final.sum())
    if cash_pad > 1e-9:
        final["BIL"] = final.get("BIL", 0.0) + cash_pad
    return final.sort_values(ascending=False)


def current_ensemble_state(streams: pd.DataFrame) -> pd.DataFrame:
    """Next-month stream weights and final risk scale using returns through as-of."""
    history = streams.dropna()
    vol = history.tail(m.ERC_VOL_WINDOW).std(ddof=0)
    inverse = 1.0 / vol.replace(0.0, np.nan)
    weights = inverse / inverse.sum()

    historical_weights = m.inverse_vol_weights(history)
    simple = np.exp(history) - 1.0
    raw_combo = pd.Series(
        np.log1p((historical_weights * simple).sum(axis=1)), index=history.index
    )
    realized = float(raw_combo.tail(m.PORT_VOL_WINDOW).std(ddof=0) * math.sqrt(12.0))
    scale = float(np.clip(m.VOL_TARGET / realized, *m.LEVERED_SCALE_BOUNDS)) if realized > 0 else 1.0
    row = {f"weight_{name}": float(value) for name, value in weights.items()}
    row.update(
        {
            "as_of": history.index.max(),
            "effective_month": (history.index.max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
            "ensemble_vol_target_scale": scale,
            "trailing_raw_combo_vol": realized,
        }
    )
    return pd.DataFrame([row])


def main() -> None:
    allow_network = "--no-network" not in sys.argv
    source = m.DATA / "prices_macro_seasons_extended.csv"
    if not source.exists():
        source = m.DATA / "prices.csv"
    panel_end = pd.Timestamp(m.load_wide_csv(source).index.max())
    audit = m.refresh_and_validate_caches(panel_end, allow_network=allow_network)

    prices, _notes = m.build_price_panel(False, extend=True)
    factors_path = m.DATA / "factors_point_in_time.csv"
    factors_pit = m.load_wide_csv(factors_path) if factors_path.exists() else pd.DataFrame()
    print(
        f"[data] panel {prices.shape[0]} months x {prices.shape[1]} assets "
        f"({prices.index.min():%Y-%m} -> {prices.index.max():%Y-%m}); "
        f"freshness checks={len(audit)} PASS"
    )

    pillars = m.build_pillars(prices, factors_pit, allow_network=False)
    probabilities = m.season_probabilities(pillars.composites)
    timeline = probabilities.join(
        pillars.composites.rename(columns={"G": "growth_z", "I": "inflation_z", "L": "liquidity_pillar_z"})
    )
    month_end = pd.DatetimeIndex(prices.index)
    m.causality_self_check(prices, factors_pit)

    m.realrate_causality_check(month_end, False)
    shift, _ = m.build_realrate_shift(month_end, False)
    raw_spreads = m.load_credit_spreads(False)
    m.credit_causality_check(raw_spreads, month_end)
    stress = m.build_credit_stress(raw_spreads, month_end)["credit_stress"]

    tickers = sorted(set(m.daily_risk_tickers()) | set(m.DAILY_PROXIES.values()))
    daily_returns = m.stitched_daily_returns(m.download_yahoo_daily(tickers, m.DAILY_START, False))
    if daily_returns.empty:
        raise SystemExit("[daily] refreshed daily cache is empty")
    m.trend_causality_check(daily_returns, month_end)
    daily_index = m.daily_price_index(daily_returns)

    long_only = m.run_walk_forward(
        prices,
        probabilities,
        cost_bps=m.COST_BPS,
        use_xsmom=True,
        use_trend=True,
        realrate_shift=shift,
        credit_stress=stress,
        daily_index=daily_index,
    )
    long_ledger = long_only.ledger.set_index("return_date")
    cash = long_ledger["cash_return"]

    core = m.run_walk_forward(
        prices, probabilities, cost_bps=m.COST_BPS, use_xsmom=False, use_trend=False
    )
    core_ledger = core.ledger.set_index("return_date")
    tsmom = m.run_tsmom_sleeve(prices)
    core_levered = m.levered_stream(core_ledger["strategy_return"], cash, "seasons_core_lev")
    long_levered = m.levered_stream(long_ledger["strategy_return"], cash, "long_only_lev")
    streams = pd.concat(
        [core_levered, long_levered, tsmom["tsmom_return"].rename("tsmom")], axis=1
    )
    m.erc_causality_check(streams, cash)
    ensemble, diagnostics = m.combine_erc(streams, cash)
    ensemble_ledger = pd.DataFrame(
        {"strategy_return": ensemble, "cash_return": cash.reindex(ensemble.index)}
    )
    ensemble_ledger.index.name = "return_date"

    summary = pd.DataFrame(
        [
            stats_row("Macro Seasons v4 multi-strategy ensemble", ensemble, cash),
            stats_row("Macro Seasons v4 long-only season portfolio", long_ledger["strategy_return"], cash),
            stats_row("Macro Seasons v4 core season allocation", core_ledger["strategy_return"], cash),
            stats_row("SPY", long_ledger["spy_return"], cash),
            stats_row("60/40 SPY/AGG", long_ledger["sixty_forty_return"], cash),
            stats_row("GSMIF 70/30 VT/BIV", long_ledger["gsmif_policy_return"], cash),
        ]
    )
    monitor_table = pd.DataFrame(
        [
            monitor("multi-strategy ensemble", ensemble, cash, MONITOR_MAX_DD_ENSEMBLE),
            monitor("long-only season portfolio", long_ledger["strategy_return"], cash, MONITOR_MAX_DD_BOOK),
        ]
    )
    bl_views = m.build_bl_views(prices, probabilities)

    last_as_of = pd.Timestamp(prices.index.max())
    target = current_target_weights(
        prices, probabilities, shift, stress, daily_index, last_as_of
    )
    if target.empty or not np.isclose(float(target.sum()), 1.0, atol=1e-10):
        raise RuntimeError("V4 current long-only allocation is missing or does not sum to one")
    allocation = pd.DataFrame(
        {
            "as_of": last_as_of,
            "effective_month": (last_as_of + pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
            "ticker": target.index,
            "weight": target.values,
            "weight_pct": (target.values * 100.0).round(4),
        }
    )
    ensemble_state = current_ensemble_state(streams)

    EXPORTS.mkdir(exist_ok=True)
    long_only.ledger.to_csv(EXPORTS / "macro_seasons_v4_long_only_ledger.csv", index=False)
    long_only.weights.to_csv(EXPORTS / "macro_seasons_v4_long_only_weights.csv", index=False)
    core.ledger.to_csv(EXPORTS / "macro_seasons_v4_core_ledger.csv", index=False)
    allocation.to_csv(EXPORTS / "macro_seasons_v4_current_allocation.csv", index=False)
    ensemble_ledger.join(diagnostics).to_csv(EXPORTS / "macro_seasons_v4_ensemble_ledger.csv")
    streams.to_csv(EXPORTS / "macro_seasons_v4_streams.csv", index_label="return_date")
    ensemble_state.to_csv(EXPORTS / "macro_seasons_v4_current_ensemble_state.csv", index=False)
    summary.to_csv(EXPORTS / "macro_seasons_v4_summary.csv", index=False)
    timeline.to_csv(EXPORTS / "macro_seasons_v4_season_timeline.csv", index_label="date")
    bl_views.to_csv(EXPORTS / "macro_seasons_v4_bl_views.csv", index=False)
    monitor_table.to_csv(EXPORTS / "macro_seasons_v4_monitor.csv", index=False)

    latest = timeline.dropna(subset=m.SEASONS).iloc[-1]
    print(f"\n=== Macro Seasons v4 PIT - data through {prices.index.max():%b %Y} ===")
    print(summary.to_string(index=False))
    print(
        "\nCurrent season probabilities: "
        + ", ".join(f"{season} {float(latest[season]):.2f}" for season in m.SEASONS)
        + f" | modal {latest['modal_season']} | confidence {float(latest['confidence']):.2f}"
    )
    print("\nCurrent long-only allocation:")
    print(allocation[["ticker", "weight_pct"]].to_string(index=False))
    print("\n=== V4 live monitor ===")
    print(monitor_table.to_string(index=False))
    print("\nWrote exports/macro_seasons_v4_*.csv")


if __name__ == "__main__":
    main()
