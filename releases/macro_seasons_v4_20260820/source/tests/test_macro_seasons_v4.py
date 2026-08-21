from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import macro_seasons_v4 as v4  # noqa: E402
import run_macro_seasons_v4 as runner  # noqa: E402


def test_alfred_asof_level_uses_only_vintages_known_at_decision_time(monkeypatch) -> None:
    vintages = pd.DataFrame(
        {
            "series_id": ["TEST"] * 5,
            "date": pd.to_datetime([
                "2025-12-01", "2025-12-01", "2026-01-01", "2026-01-01", "2026-02-01",
            ]),
            "realtime_start": pd.to_datetime([
                "2026-01-15", "2026-02-15", "2026-02-20", "2026-03-15", "2026-03-20",
            ]),
            "realtime_end": pd.to_datetime(["2026-12-31"] * 5),
            "value": [100.0, 101.0, 110.0, 111.0, 120.0],
        }
    )
    monkeypatch.setattr(v4, "load_alfred_table", lambda _series_id: vintages.copy())
    monkeypatch.setitem(v4.FRED_MAX_AGE_DAYS, "TEST", 75)

    result = v4.alfred_asof_level(
        "TEST", pd.DatetimeIndex(["2026-01-31", "2026-02-28", "2026-03-31"])
    )

    assert result.loc["2026-01-31"] == 100.0
    assert result.loc["2026-02-28"] == 110.0
    assert result.loc["2026-03-31"] == 120.0


def test_fred_freshness_marks_stale_cache_as_failure(tmp_path, monkeypatch) -> None:
    cache = tmp_path / "data" / "fred_cache"
    cache.mkdir(parents=True)
    pd.DataFrame(
        {"observation_date": ["2026-07-01"], "TEST": [1.0]}
    ).to_csv(cache / "TEST.csv", index=False)
    monkeypatch.setattr(v4, "ROOT", tmp_path)
    monkeypatch.setattr(v4, "CACHE", cache)
    monkeypatch.setattr(v4, "FRED_CACHE_SERIES", ("TEST",))
    monkeypatch.setattr(v4, "FRED_MAX_AGE_DAYS", {"TEST": 10})
    v4._FRED_MEMORY.clear()

    row = v4._fred_audit_rows(pd.Timestamp("2026-07-31"))[0]

    assert row["age_days"] == 30
    assert row["status"] == "FAIL"


def test_data_gate_fails_closed_when_any_required_input_is_stale(tmp_path, monkeypatch) -> None:
    failed = [{"component": "yahoo_daily", "identifier": "SPY", "status": "FAIL"}]
    monkeypatch.setattr(v4, "DATA_AUDIT_FILE", tmp_path / "audit.csv")
    monkeypatch.setattr(v4, "_fred_audit_rows", lambda _as_of: [])
    monkeypatch.setattr(v4, "_alfred_audit_rows", lambda _as_of: [])
    monkeypatch.setattr(v4, "_yahoo_audit_rows", lambda _as_of: failed)

    with pytest.raises(RuntimeError, match="yahoo_daily:SPY"):
        v4.refresh_and_validate_caches(pd.Timestamp("2026-07-31"), allow_network=False)


def test_live_monitor_begins_with_september_return() -> None:
    pending_returns = pd.Series(
        [0.01, -0.005], index=pd.to_datetime(["2026-07-31", "2026-08-31"])
    )
    pending = runner.monitor(
        "long-only", pending_returns, pd.Series(0.0, index=pending_returns.index), -0.08
    )
    assert pending["completed_live_months"] == 0
    assert pending["status"].startswith("PENDING")
    assert pending["first_live_return_date"] == "2026-09-30"

    live_returns = pd.concat(
        [pending_returns, pd.Series([0.006], index=pd.to_datetime(["2026-09-30"]))]
    )
    live = runner.monitor(
        "long-only", live_returns, pd.Series(0.0, index=live_returns.index), -0.08
    )
    assert live["completed_live_months"] == 1
    assert live["status"] == "OK"


def test_frozen_august_allocation_is_exact_and_uses_fresh_trend_gate() -> None:
    path = ROOT / "exports" / "macro_seasons_v4_current_allocation.csv"
    if not path.exists():
        pytest.skip("V4 production exports are not available")
    allocation = pd.read_csv(path).set_index("ticker")

    assert allocation["effective_month"].eq("2026-08").all()
    assert np.isclose(float(allocation["weight"].sum()), 1.0, atol=1e-10)
    assert allocation.loc["BIL", "weight"] == pytest.approx(0.40807267303060435)
    assert {"TIP", "XHB", "XLU"}.isdisjoint(allocation.index)
    assert len(allocation) == 18


def test_production_data_audit_passes_all_required_sources() -> None:
    path = ROOT / "data" / "macro_seasons_v4_data_audit.csv"
    if not path.exists():
        pytest.skip("V4 production data audit is not available")
    audit = pd.read_csv(path)

    assert len(audit) == 49
    assert audit["status"].eq("PASS").all()
    assert set(audit.loc[audit["component"] == "alfred_vintage", "identifier"]) == set(
        v4.ALFRED_SERIES
    )
