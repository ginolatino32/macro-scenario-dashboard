from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bl_views import (
    _active_p_vector,
    _relative_p_vector,
    build_P_q_Omega,
    build_forward_return_labels,
    build_view_diagnostics,
    load_asset_master,
    load_benchmark_weights,
    load_bl_settings,
    load_view_pairs,
    run_black_litterman,
    validate_bl_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_forward_labels_obey_training_cutoff() -> None:
    as_of = pd.Timestamp("2026-03-31")
    horizon = 6
    cutoff = as_of - pd.DateOffset(months=horizon)
    assert cutoff == pd.Timestamp("2025-09-30")

    idx = pd.date_range("2025-01-31", periods=18, freq="ME")
    prices = pd.DataFrame({"SPY": np.linspace(100.0, 140.0, len(idx))}, index=idx)
    labels = build_forward_return_labels(prices, horizon_months=horizon)
    original = labels.loc[labels.index <= cutoff, "SPY"].copy()

    changed = prices.copy()
    changed.loc[changed.index > as_of, "SPY"] *= 10.0
    changed_labels = build_forward_return_labels(changed, horizon_months=horizon)
    pd.testing.assert_series_equal(original, changed_labels.loc[changed_labels.index <= cutoff, "SPY"])


def test_p_vector_construction_for_active_and_relative_views() -> None:
    asset_order = ["SPY", "XLE", "XLY"]
    benchmark_weights = pd.DataFrame(
        {
            "benchmark_id": ["SPY_BENCH"],
            "as_of": [pd.Timestamp("2026-03-31")],
            "ticker": ["SPY"],
            "weight": [1.0],
        }
    )

    active, error = _active_p_vector("XLE", "SPY_BENCH", asset_order, benchmark_weights, pd.Timestamp("2026-03-31"))
    assert error == ""
    assert active.tolist() == [-1.0, 1.0, 0.0]

    relative, error = _relative_p_vector("XLE", "XLY", asset_order)
    assert error == ""
    assert relative.tolist() == [0.0, 1.0, -1.0]


def test_build_matrices_excludes_blocked_views_and_preserves_shapes() -> None:
    views = pd.DataFrame(
        [
            {
                "view_id": "candidate",
                "status": "Candidate",
                "p_vector_json": '{"SPY":-1.0,"XLE":1.0}',
                "q_expected_return": 0.04,
                "omega": 0.02,
            },
            {
                "view_id": "blocked",
                "status": "Blocked",
                "p_vector_json": '{"SPY":1.0}',
                "q_expected_return": 0.01,
                "omega": 0.03,
            },
        ]
    )
    P, q, Omega = build_P_q_Omega(views, ["SPY", "XLE", "XLY"])
    assert P.shape == (1, 3)
    assert q.shape == (1, 2)
    assert Omega.shape == (1, 1)
    assert P.index.tolist() == ["candidate"]
    assert float(Omega.iloc[0, 0]) > 0


def test_black_litterman_zero_view_and_omega_sanity() -> None:
    idx = pd.date_range("2020-01-31", periods=84, freq="ME")
    trend = np.linspace(0.0, 1.0, len(idx))
    prices = pd.DataFrame(
        {
            "SPY": 100.0 * (1.004 + 0.001 * np.sin(trend)).cumprod(),
            "XLE": 90.0 * (1.005 + 0.002 * np.cos(trend)).cumprod(),
            "XLY": 95.0 * (1.003 + 0.0015 * np.sin(trend * 2)).cumprod(),
        },
        index=idx,
    )
    asset_master = pd.DataFrame(
        {
            "ticker": ["SPY", "XLE", "XLY"],
            "asset_name": ["SPY", "XLE", "XLY"],
            "sleeve": ["Equity", "Equity", "Equity"],
        }
    )
    benchmark_weights = pd.DataFrame(
        {
            "benchmark_id": ["SPY_BENCH"],
            "as_of": [pd.Timestamp("2026-03-31")],
            "ticker": ["SPY"],
            "weight": [1.0],
        }
    )
    settings = {
        "horizon_months": 6,
        "policy_benchmark_id": "SPY_BENCH",
        "covariance_lookback_months": 60,
        "covariance_shrinkage": 0.2,
        "tau": 0.05,
        "risk_aversion_delta": 2.5,
        "covariance_method": "shrunk_sample",
        "return_basis": "simple_excess_return",
    }
    asset_order = ["SPY", "XLE", "XLY"]

    empty_P = pd.DataFrame(columns=asset_order)
    empty_q = pd.DataFrame(columns=["view_id", "q_expected_return"])
    empty_Omega = pd.DataFrame()
    _, zero_post, _, _ = run_black_litterman(prices, asset_master, benchmark_weights, asset_order, empty_P, empty_q, empty_Omega, settings, idx[-1])
    assert np.allclose(zero_post["prior_return"], zero_post["posterior_return"])

    P = pd.DataFrame([[0.0, 1.0, -1.0]], index=["rel"], columns=asset_order)
    q = pd.DataFrame([{"view_id": "rel", "q_expected_return": 0.05}])
    high_Omega = pd.DataFrame([[1.0]], index=["rel"], columns=["rel"])
    low_Omega = pd.DataFrame([[0.0001]], index=["rel"], columns=["rel"])
    _, high_post, high_sigma, high_pi = run_black_litterman(prices, asset_master, benchmark_weights, asset_order, P, q, high_Omega, settings, idx[-1])
    _, low_post, _, _ = run_black_litterman(prices, asset_master, benchmark_weights, asset_order, P, q, low_Omega, settings, idx[-1])

    high_spread = float(high_post.set_index("ticker").loc["XLE", "posterior_minus_prior"] - high_post.set_index("ticker").loc["XLY", "posterior_minus_prior"])
    low_spread = float(low_post.set_index("ticker").loc["XLE", "posterior_minus_prior"] - low_post.set_index("ticker").loc["XLY", "posterior_minus_prior"])
    assert low_spread > high_spread
    assert low_spread > 0
    assert high_sigma.shape == (3, 3)
    assert high_pi.shape == (3, 2)


def test_view_diagnostics_report_bl_transmission() -> None:
    views = pd.DataFrame(
        [
            {
                "view_id": "rel",
                "status": "Candidate",
                "assets": "XLE - XLY",
                "view_type": "relative",
                "q_expected_return": 0.05,
                "confidence_score": 0.60,
                "omega": 0.01,
            }
        ]
    )
    P = pd.DataFrame([[0.0, 1.0, -1.0]], index=["rel"], columns=["SPY", "XLE", "XLY"])
    q = pd.DataFrame([{"view_id": "rel", "q_expected_return": 0.05}])
    posterior_returns = pd.DataFrame(
        [
            {"ticker": "SPY", "prior_return": 0.02, "posterior_return": 0.02},
            {"ticker": "XLE", "prior_return": 0.03, "posterior_return": 0.055},
            {"ticker": "XLY", "prior_return": 0.01, "posterior_return": 0.005},
        ]
    )
    diagnostics = build_view_diagnostics(views, P, q, posterior_returns)
    assert diagnostics.shape[0] == 1
    row = diagnostics.iloc[0]
    assert np.isclose(row["prior_view_return"], 0.02)
    assert np.isclose(row["posterior_view_return"], 0.05)
    assert np.isclose(row["remaining_gap_to_q"], 0.0)
    assert row["pull_to_q"] > 0
    assert row["transmission_flag"] == "Aligned"


def test_generated_exports_round_trip_contract() -> None:
    asset_order = pd.read_csv(ROOT / "exports" / "bl_asset_order.csv")["ticker"].astype(str).tolist()
    views = pd.read_csv(ROOT / "data" / "bl_macro_views.csv")
    P = pd.read_csv(ROOT / "exports" / "P_matrix.csv", index_col="view_id")
    q = pd.read_csv(ROOT / "exports" / "q_vector.csv")
    Omega = pd.read_csv(ROOT / "exports" / "Omega_matrix.csv", index_col="view_id")
    Sigma = pd.read_csv(ROOT / "exports" / "Sigma_matrix.csv", index_col="ticker")
    pi = pd.read_csv(ROOT / "exports" / "pi_vector.csv")

    candidates = views[views["status"].eq("Candidate")]
    assert len(asset_order) == len(P.columns)
    assert P.shape == (len(candidates), len(asset_order))
    assert q.shape[0] == len(candidates)
    assert Omega.shape == (len(candidates), len(candidates))
    assert P.columns.tolist() == asset_order
    assert Sigma.shape == (len(asset_order), len(asset_order))
    assert Sigma.columns.tolist() == asset_order
    assert pi.shape[0] == len(asset_order)
    assert np.all(np.diag(Omega.to_numpy(dtype=float)) > 0)
    assert np.allclose(Omega.to_numpy(dtype=float), np.diag(np.diag(Omega.to_numpy(dtype=float))))
    payload = json.loads((ROOT / "exports" / "bl_views.json").read_text())
    assert payload["asset_order"] == asset_order
    assert len(payload["views"]) == len(views)
    assert "view_diagnostics" in payload
    assert "Sigma" in payload
    assert "pi" in payload


def test_config_contract_loads_and_weights_sum_to_one() -> None:
    settings = load_bl_settings(ROOT / "config")
    asset_master = load_asset_master(ROOT / "config")
    benchmark_weights = load_benchmark_weights(ROOT / "config")
    assert settings["horizon_months"] == 6
    assert asset_master["is_bl_eligible"].sum() >= 30
    sums = benchmark_weights.groupby("benchmark_id")["weight"].sum().round(8)
    assert (sums == 1.0).all()
    audit = validate_bl_config(
        asset_master,
        benchmark_weights,
        load_view_pairs(ROOT / "config"),
        pd.read_csv(ROOT / "data" / "prices.csv", index_col=0),
        settings,
        pd.Timestamp("2026-04-30"),
    )
    assert "Review" in set(audit["status"])
