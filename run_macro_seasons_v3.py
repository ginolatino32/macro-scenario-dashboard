"""Macro Seasons v3 — FROZEN production runner (Config A).

Model frozen 2026-07-16. See MODEL_FREEZE_MACRO_SEASONS_V3.md for the model
card, validation record, and change policy. This runner executes the frozen
configuration only — it exposes no tuning knobs by design:

    in-book stack : seasons + xsmom tilt + 200d MA trend gate
                    + real-rate gold/duration rotation + credit dimmer
                    + 10% vol target (no leverage)
    ensemble      : inverse-vol (24m, shift 1) combination of
                    [core seasons levered, enhanced stack, TSMOM sleeve],
                    vol-targeted to 10%

Monthly procedure (after month-end data refresh):
    python3 update_macro_seasons_research.py --append-only
    python3 run_macro_seasons_v3.py

Outputs to exports/: macro_seasons_v3_ledger.csv, macro_seasons_v3_ensemble_ledger.csv,
macro_seasons_v3_streams.csv, macro_seasons_v3_summary.csv,
macro_seasons_v3_season_timeline.csv, macro_seasons_v3_bl_views.csv,
macro_seasons_v3_monitor.csv.

The monitor compares post-freeze live months against the frozen walk-forward
record. REVIEW triggers (outside historical experience — investigate before
trading on the model): trailing 12m excess Sharpe < -0.5, in-book drawdown
beyond -8%, or ensemble drawdown beyond -15%.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research"))

import macro_seasons_v3 as m  # frozen model module (research/macro_seasons_v3.py)

FREEZE_DATE = "2026-07-16"
FREEZE_DATA_END = pd.Timestamp("2026-05-31")   # last month in the frozen validation record
MONITOR_MIN_SHARPE_12M = -0.5
MONITOR_MAX_DD_BOOK = -0.08
MONITOR_MAX_DD_ENSEMBLE = -0.15
EXPORTS = ROOT / "exports"


def stats_row(name: str, r: pd.Series, rf: pd.Series) -> dict[str, object]:
    r = r.dropna()
    rf = rf.reindex(r.index).fillna(0.0)
    excess = r - rf
    equity = np.exp(r.cumsum())
    years = len(r) / 12.0
    dd = float((equity / equity.cummax() - 1.0).min())
    cagr = float((equity.iloc[-1] ** (1.0 / years) - 1.0) * 100.0)
    return {
        "series": name,
        "months": len(r),
        "cagr_pct": round(cagr, 2),
        "ann_vol_pct": round(float(r.std(ddof=0) * math.sqrt(12.0) * 100.0), 2),
        "excess_sharpe": round(float(excess.mean() / excess.std(ddof=0) * math.sqrt(12.0)), 3) if excess.std(ddof=0) > 0 else np.nan,
        "raw_sharpe": round(float(r.mean() / r.std(ddof=0) * math.sqrt(12.0)), 3) if r.std(ddof=0) > 0 else np.nan,
        "max_dd_pct": round(dd * 100.0, 2),
        "calmar": round(cagr / abs(dd * 100.0), 3) if dd < 0 else np.nan,
    }


def monitor(name: str, r: pd.Series, rf: pd.Series, dd_limit: float) -> dict[str, object]:
    r = r.dropna()
    live = r.loc[r.index > FREEZE_DATA_END]
    equity = np.exp(r.cumsum())
    current_dd = float(equity.iloc[-1] / equity.cummax().iloc[-1] - 1.0)
    last12 = (r - rf.reindex(r.index).fillna(0.0)).tail(12)
    sharpe12 = float(last12.mean() / last12.std(ddof=0) * math.sqrt(12.0)) if len(last12) == 12 and last12.std(ddof=0) > 0 else np.nan
    flags = []
    if np.isfinite(sharpe12) and sharpe12 < MONITOR_MIN_SHARPE_12M:
        flags.append(f"12m excess Sharpe {sharpe12:.2f} < {MONITOR_MIN_SHARPE_12M}")
    if current_dd < dd_limit:
        flags.append(f"drawdown {current_dd:.1%} beyond {dd_limit:.0%}")
    return {
        "series": name,
        "live_months_since_freeze": int(len(live)),
        "trailing_12m_excess_sharpe": round(sharpe12, 3) if np.isfinite(sharpe12) else np.nan,
        "current_drawdown_pct": round(current_dd * 100.0, 2),
        "drawdown_limit_pct": dd_limit * 100.0,
        "status": "REVIEW: " + "; ".join(flags) if flags else "OK",
    }


def current_target_weights(prices: pd.DataFrame, probs: pd.DataFrame,
                           shift: pd.Series, stress: pd.Series,
                           daily_idx: pd.DataFrame,
                           as_of: pd.Timestamp) -> pd.Series:
    """Frozen Config A target weights at as_of (identical pipeline to the WF loop).

    The walk-forward ledger only records months whose next-month return exists;
    this reproduces the same weight construction at the latest month-end so the
    upcoming month's allocation can be published.
    """
    returns = prices.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    usable = probs.reindex(returns.index)
    season_w = {s: m.template_weights_at(s, as_of, returns) for s in m.SEASONS}
    if any(w.empty for w in season_w.values()):
        return pd.Series(dtype=float)
    p_row = usable.loc[as_of, m.SEASONS].astype(float)
    blended = pd.Series(dtype=float)
    for season in m.SEASONS:
        blended = blended.add(season_w[season] * float(p_row[season]), fill_value=0.0)
    blended = m.apply_liquidity_overlay(blended / blended.sum(),
                                        float(usable.loc[as_of, "liquidity_z"]))
    blended = m.apply_realrate_rotation(blended, float(shift.get(as_of, np.nan)))
    if bool(stress.get(as_of, False)):
        blended = m.apply_credit_dimmer(blended, True)
    blended = m.apply_xsmom_tilt(blended, prices, as_of)
    blended = m.apply_trend_gate(blended, prices, as_of, daily_idx)
    hist = returns.loc[returns.index <= as_of, blended.index].tail(m.PORT_VOL_WINDOW)
    port_hist = hist.mul(blended, axis=1).sum(axis=1, min_count=max(1, len(blended) // 2))
    realized = float(port_hist.dropna().std(ddof=0) * math.sqrt(12.0))
    scale = float(np.clip(m.VOL_TARGET / realized, *m.VOL_SCALE_BOUNDS)) if realized > 0 else 1.0
    final = blended * scale
    cash_pad = 1.0 - float(final.sum())
    if cash_pad > 1e-9:
        final["BIL"] = final.get("BIL", 0.0) + cash_pad
    return final.sort_values(ascending=False)


def main() -> None:
    allow_network = "--no-network" not in sys.argv

    prices, _notes = m.build_price_panel(allow_network, extend=True)
    pit_path = m.DATA / "factors_point_in_time.csv"
    factors_pit = m.load_wide_csv(pit_path) if pit_path.exists() else pd.DataFrame()
    print(f"[data] panel {prices.shape[0]} months x {prices.shape[1]} assets "
          f"({prices.index.min():%Y-%m} -> {prices.index.max():%Y-%m})")

    pillars = m.build_pillars(prices, factors_pit, allow_network)
    probs = m.season_probabilities(pillars.composites)
    month_end = pd.DatetimeIndex(prices.index)

    # Frozen signal builders, each with its causality self-check (fail-closed).
    m.realrate_causality_check(month_end, allow_network)
    shift, _ = m.build_realrate_shift(month_end, allow_network)
    raw_spreads = m.load_credit_spreads(allow_network)
    m.credit_causality_check(raw_spreads, month_end)
    stress = m.build_credit_stress(raw_spreads, month_end)["credit_stress"]
    tickers = sorted(set(m.daily_risk_tickers()) | set(m.DAILY_PROXIES.values()))
    daily_ret = m.stitched_daily_returns(m.download_yahoo_daily(tickers, m.DAILY_START, allow_network))
    if daily_ret.empty:
        raise SystemExit("[daily] no daily data — frozen config requires the 200d MA gate.")
    m.trend_causality_check(daily_ret, month_end)
    daily_idx = m.daily_price_index(daily_ret)

    # Frozen Config A book.
    wf = m.run_walk_forward(prices, probs, cost_bps=m.COST_BPS,
                            use_xsmom=True, use_trend=True,
                            realrate_shift=shift, credit_stress=stress,
                            daily_index=daily_idx)
    ledger = wf.ledger.set_index("return_date")
    cash = ledger["cash_return"]

    # Frozen ensemble.
    wf_core = m.run_walk_forward(prices, probs, cost_bps=m.COST_BPS,
                                 use_xsmom=False, use_trend=False)
    core_ledger = wf_core.ledger.set_index("return_date")
    tsmom = m.run_tsmom_sleeve(prices)
    core_lev = m.levered_stream(core_ledger["strategy_return"], cash, "seasons_core_lev")
    enh_lev = m.levered_stream(ledger["strategy_return"], cash, "fullstack_lev")
    streams = pd.concat([core_lev, enh_lev, tsmom["tsmom_return"].rename("tsmom")], axis=1)
    m.erc_causality_check(streams, cash)
    combo, diag = m.combine_erc(streams, cash)
    ens_ledger = pd.DataFrame({"strategy_return": combo, "cash_return": cash.reindex(combo.index)})
    ens_ledger.index.name = "return_date"

    summary = pd.DataFrame([
        stats_row("Macro Seasons v3 ensemble (frozen Config A)", combo, cash),
        stats_row("Macro Seasons v3 in-book stack (frozen Config A)", ledger["strategy_return"], cash),
        stats_row("SPY", ledger["spy_return"], cash),
        stats_row("60/40 SPY/AGG", ledger["sixty_forty_return"], cash),
        stats_row("GSMIF 70/30 VT/BIV", ledger["gsmif_policy_return"], cash),
    ])
    monitor_table = pd.DataFrame([
        monitor("ensemble", combo, cash, MONITOR_MAX_DD_ENSEMBLE),
        monitor("in-book stack", ledger["strategy_return"], cash, MONITOR_MAX_DD_BOOK),
    ])
    bl_views = m.build_bl_views(prices, probs)

    EXPORTS.mkdir(exist_ok=True)
    wf.ledger.to_csv(EXPORTS / "macro_seasons_v3_ledger.csv", index=False)
    wf.weights.to_csv(EXPORTS / "macro_seasons_v3_weights.csv", index=False)

    last_as_of = pd.Timestamp(prices.index.max())
    target = current_target_weights(prices, probs, shift, stress, daily_idx, last_as_of)
    allocation = pd.DataFrame({
        "as_of": last_as_of,
        "effective_month": (last_as_of + pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
        "ticker": target.index,
        "weight": target.values,
        "weight_pct": (target.values * 100.0).round(2),
    })
    allocation.to_csv(EXPORTS / "macro_seasons_v3_current_allocation.csv", index=False)
    ens_ledger.join(diag).to_csv(EXPORTS / "macro_seasons_v3_ensemble_ledger.csv")
    streams.to_csv(EXPORTS / "macro_seasons_v3_streams.csv", index_label="return_date")
    summary.to_csv(EXPORTS / "macro_seasons_v3_summary.csv", index=False)
    probs.to_csv(EXPORTS / "macro_seasons_v3_season_timeline.csv", index_label="date")
    bl_views.to_csv(EXPORTS / "macro_seasons_v3_bl_views.csv", index=False)
    monitor_table.to_csv(EXPORTS / "macro_seasons_v3_monitor.csv", index=False)

    latest = probs.dropna(subset=m.SEASONS).iloc[-1]
    print(f"\n=== Macro Seasons v3 (frozen {FREEZE_DATE}) — data through {prices.index.max():%b %Y} ===")
    print(summary.to_string(index=False))
    print(f"\nCurrent season probabilities: "
          + ", ".join(f"{s} {float(latest[s]):.2f}" for s in m.SEASONS)
          + f" | modal {latest['modal_season']} | confidence {float(latest['confidence']):.2f}")
    print("\n=== Live monitor (vs frozen validation record) ===")
    print(monitor_table.to_string(index=False))
    print(f"\nWrote exports/macro_seasons_v3_*.csv")


if __name__ == "__main__":
    main()
