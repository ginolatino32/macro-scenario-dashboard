"""Generate the executable IBKR position and cost overlay for Macro Seasons V4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research"))

import macro_seasons_v4 as m  # noqa: E402
from macro_seasons_v4_execution import (  # noqa: E402
    build_execution_artifacts,
    write_execution_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Use the existing daily cache instead of refreshing Yahoo adjusted closes.",
    )
    parser.add_argument(
        "--price-cutoff",
        help="Override the live-price cutoff (YYYY-MM-DD); defaults to prior New York day.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    live_cutoff = (
        pd.Timestamp(args.price_cutoff).normalize()
        if args.price_cutoff
        else m.latest_completed_price_cutoff()
    )
    if not args.no_network:
        print(f"Refreshing Yahoo adjusted closes through {live_cutoff:%Y-%m-%d} ...")
        m.refresh_yahoo_live_cache(live_cutoff)

    source = m.DATA / "prices_macro_seasons_extended.csv"
    if not source.exists():
        source = m.DATA / "prices.csv"
    panel_end = pd.Timestamp(m.load_wide_csv(source).index.max())
    m.refresh_and_validate_caches(panel_end, allow_network=False)

    prices, _notes = m.build_price_panel(False, extend=True)
    factors_path = m.DATA / "factors_point_in_time.csv"
    factors_pit = m.load_wide_csv(factors_path) if factors_path.exists() else pd.DataFrame()
    pillars = m.build_pillars(prices, factors_pit, allow_network=False)
    probabilities = m.season_probabilities(pillars.composites)
    month_end = pd.DatetimeIndex(prices.index)

    realrate_shift, _ = m.build_realrate_shift(month_end, False)
    credit_stress = m.build_credit_stress(m.load_credit_spreads(False), month_end)["credit_stress"]
    tickers = sorted(
        set(m.daily_risk_tickers()) | set(m.DAILY_PROXIES.values()) | set(m.LIVE_PRICE_TICKERS)
    )
    daily_prices = m.download_yahoo_daily(tickers, m.DAILY_START, False)
    live_daily_prices = m.load_yahoo_live_prices()
    if live_daily_prices.empty:
        raise SystemExit("Live daily cache is empty; rerun without --no-network")
    daily_returns = m.stitched_daily_returns(
        daily_prices
    )
    if daily_returns.empty:
        raise SystemExit("Daily cache is empty; run the V4 monthly refresh first")
    daily_index = m.daily_price_index(daily_returns)

    artifacts = build_execution_artifacts(
        prices,
        probabilities,
        realrate_shift,
        credit_stress,
        daily_index,
        live_daily_prices=live_daily_prices,
        live_cutoff=live_cutoff,
    )
    write_execution_artifacts(artifacts)

    print("\n=== Macro Seasons V4 IBKR execution overlay ===")
    print(artifacts.summary.to_string(index=False))
    print("\nCurrent positions:")
    print(
        artifacts.current_positions[
            ["ticker", "side", "target_weight_pct", "target_notional_usd", "estimated_shares"]
        ].to_string(index=False)
    )
    print("\nAverage annual modeled costs (bps):")
    print(artifacts.cost_summary[["cost_component", "average_annual_bps"]].to_string(index=False))
    if not artifacts.live_mtd.empty:
        live = artifacts.live_mtd.iloc[0]
        if live["status"] == "UPDATED":
            print(
                f"\nMTD through {pd.Timestamp(live['price_as_of']):%Y-%m-%d} close: "
                f"long only {float(live['long_only_simple_return']):+.2%}, "
                f"L/S {float(live['ls_simple_return']):+.2%}, "
                f"SPY {float(live['spy_simple_return']):+.2%}"
            )
        else:
            print(f"\nMTD price status: {live['status']}")
    print("\nWrote exports/macro_seasons_v4_execution_*")


if __name__ == "__main__":
    main()
