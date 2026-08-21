from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import macro_seasons_v4 as v4  # noqa: E402
import macro_seasons_v4_execution as execution  # noqa: E402


def test_ibkr_fixed_commission_respects_rate_minimum_and_cap() -> None:
    settings = execution.load_execution_settings()

    assert execution.ibkr_fixed_commission(1_000, 25_000, settings) == pytest.approx(5.0)
    assert execution.ibkr_fixed_commission(10, 2_500, settings) == pytest.approx(1.0)
    assert execution.ibkr_fixed_commission(10, 2.0, settings) == pytest.approx(0.02)


def test_ibkr_regulatory_fees_apply_sale_fees_and_cat_to_all_orders() -> None:
    settings = execution.load_execution_settings()

    buy = execution.ibkr_regulatory_fees(1_000, 25_000, False, settings)
    sell = execution.ibkr_regulatory_fees(1_000, 25_000, True, settings)

    assert buy == pytest.approx(1_000 * settings.finra_cat_per_share_usd)
    assert sell == pytest.approx(
        buy
        + 25_000 * settings.sec_transaction_fee_rate
        + 1_000 * settings.finra_taf_per_share_usd
    )


def test_margin_spread_uses_published_balance_tiers() -> None:
    tiers = execution.load_margin_tiers()

    assert execution.blended_margin_spread_bps(50_000, tiers) == pytest.approx(150.0)
    assert execution.blended_margin_spread_bps(200_000, tiers) == pytest.approx(125.0)
    assert execution.blended_margin_spread_bps(1_100_000, tiers) == pytest.approx(
        (100_000 * 150 + 900_000 * 100 + 100_000 * 75) / 1_100_000
    )


def test_short_proceeds_rate_uses_separate_ibkr_tiers() -> None:
    tiers = execution.load_short_proceeds_tiers()
    benchmark = 0.0363

    assert execution.blended_short_proceeds_rate(100_000, benchmark, tiers) == 0.0
    assert execution.blended_short_proceeds_rate(230_000, benchmark, tiers) == pytest.approx(
        130_000 * (benchmark - 0.0125) / 230_000
    )


def test_position_limits_net_bil_against_borrowing_and_enforce_caps() -> None:
    settings = execution.load_execution_settings()
    raw = pd.Series({"BIL": 0.50, "SPY": 0.60, "QQQ": 0.50, "TLT": -0.25})

    constrained, diagnostics = execution.enforce_position_limits(raw, settings)

    assert diagnostics["netted_bil_weight"] > 0
    assert diagnostics["gross_exposure"] <= settings.gross_exposure_limit + 1e-12
    assert diagnostics["short_gross_exposure"] <= settings.short_gross_limit + 1e-12
    assert diagnostics["borrow_weight"] <= settings.max_borrow_fraction + 1e-12
    assert abs(float(constrained["SPY"])) <= settings.per_asset_abs_limit + 1e-12
    assert abs(float(constrained["QQQ"])) <= settings.per_asset_abs_limit + 1e-12


def test_duplicate_stream_exposures_are_aggregated_and_netted() -> None:
    components, total = execution._aggregate_components(
        pd.Series({"SPY": 0.40, "TLT": 0.10}),
        pd.Series({"SPY": 0.20}),
        pd.Series({"SPY": -0.30, "TLT": -0.05}),
        1.0,
        1.0,
        pd.Series({"seasons_core_lev": 1.0, "long_only_lev": 1.0, "tsmom": 1.0}),
        1.0,
    )

    assert total["SPY"] == pytest.approx(0.30)
    assert total["TLT"] == pytest.approx(0.05)
    assert sum(component.get("SPY", 0.0) for component in components.values()) == pytest.approx(
        total["SPY"]
    )


def test_tsmom_positions_reconstruct_frozen_sleeve_and_include_both_sides() -> None:
    prices, _ = v4.build_price_panel(False, extend=True)
    positions, ledger, current = execution.build_tsmom_position_path(prices)

    execution.assert_tsmom_parity(prices, ledger)
    assert not positions.empty
    assert (current > 0).any()
    assert (current < 0).any()
    assert float(current.abs().max()) <= v4.TSMOM_MAX_ABS_WEIGHT * v4.LEVERED_SCALE_BOUNDS[1] + 1e-12


def test_current_execution_export_reconciles_components_cash_and_limits() -> None:
    path = ROOT / "exports" / "macro_seasons_v4_execution_current_positions.csv"
    if not path.exists():
        pytest.skip("V4 execution exports are not available")
    positions = pd.read_csv(path)
    securities = positions.loc[positions["ticker"] != "USD_CASH"].copy()
    cash = float(positions.loc[positions["ticker"] == "USD_CASH", "target_weight"].iloc[0])
    settings = execution.load_execution_settings()

    components = securities[["core_component", "long_only_component", "tsmom_component"]].sum(axis=1)
    assert np.allclose(components, securities["raw_weight"], atol=1e-12)
    assert float(securities["target_weight"].sum()) + cash == pytest.approx(1.0)
    assert float(securities["target_weight"].abs().sum()) <= settings.gross_exposure_limit + 1e-12
    assert float(-securities.loc[securities["target_weight"] < 0, "target_weight"].sum()) <= settings.short_gross_limit + 1e-12
    assert securities.loc[securities["ticker"] != "BIL", "target_weight"].abs().max() <= settings.per_asset_abs_limit + 1e-12
    assert set(securities["side"]) == {"LONG", "SHORT"}

    manifest = pd.read_json(
        ROOT / "exports" / "macro_seasons_v4_execution_manifest.json", typ="series"
    )
    margin_policy = manifest["margin_policy"]
    assert margin_policy["account_type"] == "PORTFOLIO_MARGIN"
    assert margin_policy["exact_requirement_source"] == "LIVE_IBKR_CHECK_MARGIN"

    pm_check = pd.read_csv(
        ROOT / "exports" / "macro_seasons_v4_execution_pm_pretrade_check.csv"
    )
    assert pm_check.loc[0, "status"] == "LIVE_IBKR_CHECK_REQUIRED"
    assert pd.isna(pm_check.loc[0, "ibkr_projected_maintenance_margin_usd"])


def test_execution_ledger_charges_every_requested_cost_and_no_futures_roll() -> None:
    ledger_path = ROOT / "exports" / "macro_seasons_v4_execution_ledger.csv"
    summary_path = ROOT / "exports" / "macro_seasons_v4_execution_summary.csv"
    if not ledger_path.exists() or not summary_path.exists():
        pytest.skip("V4 execution exports are not available")
    ledger = pd.read_csv(ledger_path)
    summary = pd.read_csv(summary_path).set_index("series")

    for column in [
        "commission_cost",
        "regulatory_fee_cost",
        "slippage_cost",
        "margin_benchmark_cost",
        "margin_spread_cost",
        "short_borrow_cost",
        "futures_roll_cost",
    ]:
        assert ledger[column].ge(0).all()
    assert ledger["commission_cost"].sum() > 0
    assert ledger["regulatory_fee_cost"].sum() > 0
    assert ledger["slippage_cost"].sum() > 0
    assert ledger["margin_spread_cost"].sum() > 0
    assert ledger["short_borrow_cost"].sum() > 0
    assert ledger["short_proceeds_interest"].sum() > 0
    assert ledger["futures_roll_cost"].sum() == pytest.approx(0.0)
    assert ledger["turnover"].sum() > 0

    no_cost = summary.loc["Executable physical accounting before explicit IBKR costs", "cagr_pct"]
    costed = summary.loc["IBKR-costed executable ensemble", "cagr_pct"]
    assert costed < no_cost


def test_execution_ledger_includes_complete_2007_history() -> None:
    ledger_path = ROOT / "exports" / "macro_seasons_v4_execution_ledger.csv"
    if not ledger_path.exists():
        pytest.skip("V4 execution exports are not available")
    ledger = pd.read_csv(ledger_path, parse_dates=["return_date"])
    rows_2007 = ledger.loc[ledger["return_date"].dt.year == 2007]

    assert execution.EXECUTION_BACKTEST_START == pd.Timestamp("2007-01-31")
    assert rows_2007["return_date"].dt.month.tolist() == list(range(1, 13))
    assert rows_2007["net_return"].notna().all()


def test_monthly_outer_rebalance_charges_drifted_target_turnover() -> None:
    dates = pd.to_datetime(["2020-12-31", "2021-01-31", "2021-02-28"])
    prices = pd.DataFrame(
        {"SPY": [100.0, 110.0, 110.0], "IEF": [100.0, 100.0, 100.0]},
        index=dates,
    )
    targets = pd.DataFrame(
        [
            {"as_of": dates[0], "return_date": dates[1], "ticker": ticker, "target_weight": 0.5}
            for ticker in ["SPY", "IEF"]
        ]
        + [
            {"as_of": dates[1], "return_date": dates[2], "ticker": ticker, "target_weight": 0.5}
            for ticker in ["SPY", "IEF"]
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {
                "return_date": date,
                "gross_exposure": 1.0,
                "net_exposure": 1.0,
                "short_gross_exposure": 0.0,
                "borrow_weight": 0.0,
                "regt_reference_requirement": 0.5,
                "regt_reference_buffer": 0.5,
                "portfolio_margin_minimum_cushion": 0.2,
                "portfolio_margin_status": "LIVE_IBKR_CHECK_REQUIRED",
                "limit_scale": 1.0,
                "netted_bil_weight": 0.0,
                "clipped_assets": "",
            }
            for date in dates[1:]
        ]
    )
    cash = pd.Series(0.0, index=dates[1:])
    ledger, _, _ = execution.simulate_execution(
        targets,
        diagnostics,
        prices,
        cash,
        execution.load_execution_settings(),
        execution.load_margin_tiers(),
        pd.Series(dtype=float),
        execution.load_short_proceeds_tiers(),
    )

    assert ledger.loc[1, "turnover"] > 0
    assert ledger.loc[1, "commission_cost"] > 0
    assert ledger.loc[1, "slippage_cost"] > 0
