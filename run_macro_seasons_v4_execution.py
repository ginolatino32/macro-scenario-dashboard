"""Generate the executable IBKR position and cost overlay for Macro Seasons V4."""

from __future__ import annotations

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


def main() -> None:
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
    tickers = sorted(set(m.daily_risk_tickers()) | set(m.DAILY_PROXIES.values()))
    daily_returns = m.stitched_daily_returns(
        m.download_yahoo_daily(tickers, m.DAILY_START, False)
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
    print("\nWrote exports/macro_seasons_v4_execution_*")


if __name__ == "__main__":
    main()
