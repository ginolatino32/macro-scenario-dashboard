"""Macro Seasons v4 data layer over the frozen v3 portfolio machinery.

V4 keeps the v3 portfolio rules unchanged and replaces the historical data
contract. Revision-sensitive monthly macro series are reconstructed from
ALFRED vintages as they were known at each decision month. Every production
run refreshes the shared FRED/Yahoo caches and fails closed when required
inputs are stale or missing.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import macro_seasons_v3 as _v3


ROOT = _v3.ROOT
DATA = _v3.DATA
CACHE = _v3.CACHE
EXPORTS = _v3.EXPORTS
VINTAGE_DIR = DATA / "vintages"
DATA_AUDIT_FILE = DATA / "macro_seasons_v4_data_audit.csv"

MODEL_VERSION = "macro_seasons_v4_pit"
ALFRED_SERIES = ("CPIAUCSL", "INDPRO", "PAYEMS", "M2SL")
EXTRA_FRED_SERIES = ("DBAA",)
FRED_CACHE_SERIES = tuple(sorted(set(_v3.FRED_SERIES) | set(EXTRA_FRED_SERIES)))

FRED_MAX_AGE_DAYS = {
    "INDPRO": 75,
    "PAYEMS": 75,
    "CPIAUCSL": 75,
    "M2SL": 75,
    "TB3MS": 62,
    "ICSA": 21,
    "NFCI": 21,
}
DEFAULT_FRED_MAX_AGE_DAYS = 10
YAHOO_DAILY_MAX_AGE_DAYS = 7
YAHOO_MONTHLY_MAX_AGE_DAYS = 40
MIN_ALFRED_HISTORY_YEARS = 25

_FRED_MEMORY: dict[str, pd.Series] = {}
_ALFRED_MEMORY: dict[str, pd.DataFrame] = {}


def __getattr__(name: str):
    """Expose the frozen v3 portfolio API without copying its implementation."""
    return getattr(_v3, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def _atomic_write_frame(path: Path, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False).encode("utf-8")
    _atomic_write(path, payload)


def _fred_api_key() -> str:
    key = os.environ.get("FRED_API_KEY") or os.environ.get("ALFRED_API_KEY") or ""
    key_file = Path.home() / ".fred_api_key"
    if not key and key_file.exists():
        key = key_file.read_text().strip()
    return key


def _fred_frame_from_bytes(raw: bytes, series_id: str) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(raw))
    if frame.shape[1] < 2:
        raise ValueError(f"FRED response for {series_id} has no value column")
    date_col = frame.columns[0]
    value_col = series_id if series_id in frame.columns else frame.columns[1]
    out = pd.DataFrame(
        {
            "observation_date": pd.to_datetime(frame[date_col], errors="coerce"),
            series_id: pd.to_numeric(frame[value_col], errors="coerce"),
        }
    ).dropna(subset=["observation_date"])
    if out[series_id].notna().sum() == 0:
        raise ValueError(f"FRED response for {series_id} contains no numeric observations")
    return out


def refresh_fred_caches() -> None:
    """Refresh every FRED cache consumed by the v3/v4 model."""
    for series_id in FRED_CACHE_SERIES:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                raw = urllib.request.urlopen(url, timeout=60).read()
                _fred_frame_from_bytes(raw, series_id)
                _atomic_write(CACHE / f"{series_id}.csv", raw)
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Unable to refresh FRED {series_id}: {last_error}")
    _FRED_MEMORY.clear()


def fetch_fred(series_id: str, allow_network: bool = False) -> pd.Series:
    """Load a refreshed FRED cache; network refresh is centralized above."""
    if series_id in _FRED_MEMORY:
        return _FRED_MEMORY[series_id].copy()
    path = CACHE / f"{series_id}.csv"
    if not path.exists():
        if allow_network:
            refresh_fred_caches()
        if not path.exists():
            return pd.Series(dtype=float, name=series_id)
    frame = pd.read_csv(path)
    date_col = frame.columns[0]
    value_col = series_id if series_id in frame.columns else frame.columns[1]
    values = pd.to_numeric(frame[value_col], errors="coerce")
    series = pd.Series(
        values.to_numpy(),
        index=pd.to_datetime(frame[date_col], errors="coerce"),
        name=series_id,
    ).dropna().sort_index()
    _FRED_MEMORY[series_id] = series
    return series.copy()


def _download_alfred_table(series_id: str, as_of: pd.Timestamp) -> pd.DataFrame:
    key = _fred_api_key()
    if not key:
        raise RuntimeError("FRED_API_KEY is required for ALFRED point-in-time histories")
    params = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": "1900-01-01",
        "observation_end": as_of.strftime("%Y-%m-%d"),
        "realtime_start": "1776-07-04",
        "realtime_end": as_of.strftime("%Y-%m-%d"),
        "output_type": 1,
        "limit": 100000,
        "offset": 0,
    }
    endpoint = "https://api.stlouisfed.org/fred/series/observations"
    rows: list[dict[str, object]] = []
    while True:
        url = endpoint + "?" + urllib.parse.urlencode(params)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                payload = json.loads(urllib.request.urlopen(url, timeout=90).read().decode("utf-8"))
                break
            except Exception as exc:
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        else:
            raise RuntimeError(f"Unable to refresh ALFRED {series_id}: {last_error}")
        page = payload.get("observations", [])
        rows.extend(page)
        count = int(payload.get("count", len(rows)))
        if not page or len(rows) >= count:
            break
        params["offset"] = len(rows)

    table = pd.DataFrame(rows)
    required = {"date", "realtime_start", "realtime_end", "value"}
    if table.empty or not required.issubset(table.columns):
        raise ValueError(f"ALFRED response for {series_id} is incomplete")
    table = table[list(required)].copy()
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    table["realtime_start"] = pd.to_datetime(table["realtime_start"], errors="coerce")
    table["realtime_end"] = pd.to_datetime(table["realtime_end"], errors="coerce")
    table["value"] = pd.to_numeric(table["value"].replace(".", np.nan), errors="coerce")
    table = table.dropna(subset=["date", "realtime_start", "value"])
    table = table.sort_values(["date", "realtime_start"]).drop_duplicates(
        ["date", "realtime_start"], keep="last"
    )
    table.insert(0, "series_id", series_id)
    return table.reset_index(drop=True)


def refresh_alfred_vintages(as_of: pd.Timestamp) -> None:
    VINTAGE_DIR.mkdir(parents=True, exist_ok=True)
    for series_id in ALFRED_SERIES:
        table = _download_alfred_table(series_id, as_of)
        _atomic_write_frame(VINTAGE_DIR / f"{series_id}_vintages.csv", table)
    _ALFRED_MEMORY.clear()


def load_alfred_table(series_id: str) -> pd.DataFrame:
    if series_id in _ALFRED_MEMORY:
        return _ALFRED_MEMORY[series_id].copy()
    path = VINTAGE_DIR / f"{series_id}_vintages.csv"
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path, parse_dates=["date", "realtime_start", "realtime_end"])
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    table = table.dropna(subset=["date", "realtime_start", "value"])
    _ALFRED_MEMORY[series_id] = table
    return table.copy()


def alfred_asof_level(series_id: str, month_end: pd.DatetimeIndex) -> pd.Series:
    """Latest published level known at each month-end, including then-known revisions."""
    table = load_alfred_table(series_id)
    out = pd.Series(np.nan, index=month_end, name=series_id, dtype=float)
    if table.empty:
        return out
    max_age = FRED_MAX_AGE_DAYS.get(series_id, 75)
    for as_of in month_end:
        known = table.loc[
            (table["date"] <= as_of) & (table["realtime_start"] <= as_of),
            ["date", "realtime_start", "value"],
        ]
        if known.empty:
            continue
        latest_vintage = known.sort_values(["date", "realtime_start"]).drop_duplicates(
            "date", keep="last"
        )
        latest = latest_vintage.sort_values("date").iloc[-1]
        if (as_of - pd.Timestamp(latest["date"])).days <= max_age:
            out.loc[as_of] = float(latest["value"])
    return out


def _extract_yahoo_closes(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    elif "Close" in raw.columns:
        closes = raw[["Close"]]
    else:
        raise ValueError("Yahoo response has no Close field")
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)
    return closes.dropna(how="all").sort_index()


def refresh_yahoo_caches(as_of: pd.Timestamp) -> None:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for Macro Seasons cache refresh") from exc

    daily_tickers = sorted(set(_v3.daily_risk_tickers()) | set(_v3.DAILY_PROXIES.values()))
    end = (as_of + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw_daily = yf.download(
        daily_tickers,
        start=_v3.DAILY_START,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    daily = _extract_yahoo_closes(raw_daily)
    daily.index = daily.index.normalize()
    daily = daily.groupby(daily.index).last()
    daily_out = daily.copy()
    daily_out.index.name = "date"
    _atomic_write_frame(CACHE / "yahoo_daily.csv", daily_out.reset_index())

    monthly_tickers = sorted(set(_v3.EXTENSION_ETFS) | set(_v3.EXTENSION_PROXIES.values()))
    raw_monthly = yf.download(
        monthly_tickers,
        start=_v3.EXTENDED_START,
        end=end,
        interval="1mo",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    monthly = _extract_yahoo_closes(raw_monthly)
    monthly.index = pd.DatetimeIndex(monthly.index) + pd.offsets.MonthEnd(0)
    monthly = monthly.loc[monthly.index <= as_of].groupby(monthly.index).last()
    monthly_out = monthly.copy()
    monthly_out.index.name = "date"
    _atomic_write_frame(CACHE / "yahoo_extension.csv", monthly_out.reset_index())


def download_yahoo_daily(tickers: list[str], start: str, allow_network: bool) -> pd.DataFrame:
    path = CACHE / "yahoo_daily.csv"
    return _v3.load_wide_csv(path) if path.exists() else pd.DataFrame()


def download_yahoo_monthly(tickers: list[str], start: str, allow_network: bool) -> pd.DataFrame:
    path = CACHE / "yahoo_extension.csv"
    return _v3.load_wide_csv(path) if path.exists() else pd.DataFrame()


def _fred_audit_rows(as_of: pd.Timestamp) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for series_id in FRED_CACHE_SERIES:
        path = CACHE / f"{series_id}.csv"
        series = fetch_fred(series_id)
        through = series.loc[series.index <= as_of]
        last = through.index.max() if not through.empty else pd.NaT
        max_age = FRED_MAX_AGE_DAYS.get(series_id, DEFAULT_FRED_MAX_AGE_DAYS)
        age = int((as_of - last).days) if pd.notna(last) else None
        status = "PASS" if age is not None and age <= max_age else "FAIL"
        rows.append(
            {
                "component": "fred_cache",
                "identifier": series_id,
                "as_of": as_of,
                "last_observation": last,
                "age_days": age,
                "max_age_days": max_age,
                "rows": int(len(series)),
                "status": status,
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path) if path.exists() else "",
            }
        )
    return rows


def _alfred_audit_rows(as_of: pd.Timestamp) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for series_id in ALFRED_SERIES:
        path = VINTAGE_DIR / f"{series_id}_vintages.csv"
        table = load_alfred_table(series_id)
        known = table.loc[table["realtime_start"] <= as_of] if not table.empty else table
        last = known["date"].max() if not known.empty else pd.NaT
        first = known["date"].min() if not known.empty else pd.NaT
        age = int((as_of - last).days) if pd.notna(last) else None
        years = float((as_of - first).days / 365.25) if pd.notna(first) else 0.0
        max_age = FRED_MAX_AGE_DAYS.get(series_id, 75)
        status = "PASS" if age is not None and age <= max_age and years >= MIN_ALFRED_HISTORY_YEARS else "FAIL"
        rows.append(
            {
                "component": "alfred_vintage",
                "identifier": series_id,
                "as_of": as_of,
                "last_observation": last,
                "age_days": age,
                "max_age_days": max_age,
                "rows": int(len(table)),
                "status": status,
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path) if path.exists() else "",
            }
        )
    return rows


def _yahoo_audit_rows(as_of: pd.Timestamp) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    daily_path = CACHE / "yahoo_daily.csv"
    daily = _v3.load_wide_csv(daily_path) if daily_path.exists() else pd.DataFrame()
    for ticker in _v3.daily_risk_tickers():
        series = daily[ticker].dropna() if ticker in daily else pd.Series(dtype=float)
        through = series.loc[series.index <= as_of]
        last = through.index.max() if not through.empty else pd.NaT
        age = int((as_of - last).days) if pd.notna(last) else None
        status = "PASS" if age is not None and age <= YAHOO_DAILY_MAX_AGE_DAYS else "FAIL"
        rows.append(
            {
                "component": "yahoo_daily",
                "identifier": ticker,
                "as_of": as_of,
                "last_observation": last,
                "age_days": age,
                "max_age_days": YAHOO_DAILY_MAX_AGE_DAYS,
                "rows": int(len(series)),
                "status": status,
                "path": str(daily_path.relative_to(ROOT)),
                "sha256": _sha256(daily_path) if daily_path.exists() else "",
            }
        )

    monthly_path = CACHE / "yahoo_extension.csv"
    monthly = _v3.load_wide_csv(monthly_path) if monthly_path.exists() else pd.DataFrame()
    through = monthly.loc[monthly.index <= as_of].dropna(how="all") if not monthly.empty else monthly
    last = through.index.max() if not through.empty else pd.NaT
    age = int((as_of - last).days) if pd.notna(last) else None
    rows.append(
        {
            "component": "yahoo_monthly",
            "identifier": "extension_panel",
            "as_of": as_of,
            "last_observation": last,
            "age_days": age,
            "max_age_days": YAHOO_MONTHLY_MAX_AGE_DAYS,
            "rows": int(len(monthly)),
            "status": "PASS" if age is not None and age <= YAHOO_MONTHLY_MAX_AGE_DAYS else "FAIL",
            "path": str(monthly_path.relative_to(ROOT)),
            "sha256": _sha256(monthly_path) if monthly_path.exists() else "",
        }
    )
    return rows


def refresh_and_validate_caches(as_of: pd.Timestamp, allow_network: bool) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of).normalize() + pd.offsets.MonthEnd(0)
    if allow_network:
        refresh_fred_caches()
        refresh_alfred_vintages(as_of)
        refresh_yahoo_caches(as_of)

    # Ensure every v3 helper reads the validated V4 caches.
    _v3.fetch_fred = fetch_fred
    _v3.download_yahoo_daily = download_yahoo_daily
    _v3.download_yahoo_monthly = download_yahoo_monthly

    audit = pd.DataFrame(
        _fred_audit_rows(as_of) + _alfred_audit_rows(as_of) + _yahoo_audit_rows(as_of)
    )
    audit.insert(0, "checked_at_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    audit.to_csv(DATA_AUDIT_FILE, index=False)
    failed = audit.loc[audit["status"] != "PASS"]
    if not failed.empty:
        detail = ", ".join(f"{r.component}:{r.identifier}" for r in failed.itertuples())
        raise RuntimeError(f"Macro Seasons V4 data freshness failed: {detail}")
    return audit


def build_pillars(prices: pd.DataFrame, factors_pit: pd.DataFrame,
                  allow_network: bool) -> _v3.PillarBundle:
    """Build G/I/L pillars with ALFRED decision-time monthly histories."""
    month_end = pd.DatetimeIndex(prices.index)

    def fred_monthly(series_id: str) -> pd.Series:
        if series_id in ALFRED_SERIES:
            return alfred_asof_level(series_id, month_end)
        return _v3.monthly_from_lagged(fetch_fred(series_id), _v3.FRED_SERIES[series_id], month_end)

    subs: dict[str, pd.Series] = {}
    indpro = fred_monthly("INDPRO")
    payems = fred_monthly("PAYEMS")
    claims = fred_monthly("ICSA")
    subs["g_indpro_6m"] = _v3.rolling_z(_v3.log_momentum(indpro, 6))
    subs["g_payrolls_3m"] = _v3.rolling_z(_v3.log_momentum(payems, 3))
    subs["g_claims_13w"] = _v3.rolling_z(-_v3.log_momentum(claims.rolling(2, min_periods=1).mean(), 3))
    if {"XLI", "XLP"}.issubset(prices.columns):
        subs["g_cyc_def_6m"] = _v3.rolling_z(_v3.log_momentum(prices["XLI"] / prices["XLP"], 6))
    if {"SPY", "IEF"}.issubset(prices.columns):
        subs["g_risk_appetite_6m"] = _v3.rolling_z(_v3.log_momentum(prices["SPY"] / prices["IEF"], 6))

    cpi = fred_monthly("CPIAUCSL")
    cpi_yoy = _v3.log_momentum(cpi, 12)
    subs["i_cpi_delta3"] = _v3.rolling_z(cpi_yoy.diff(3))
    subs["i_breakeven_3m"] = _v3.rolling_z(fred_monthly("T5YIE").diff(3))
    subs["i_oil_6m"] = _v3.rolling_z(_v3.log_momentum(fred_monthly("DCOILWTICO"), 6))
    if {"TIP", "IEF"}.issubset(prices.columns):
        subs["i_tip_ief_6m"] = _v3.rolling_z(_v3.log_momentum(prices["TIP"] / prices["IEF"], 6))

    nfci = fred_monthly("NFCI")
    subs["l_nfci"] = 0.5 * _v3.rolling_z(-nfci) + 0.5 * _v3.rolling_z(-nfci.diff(3))
    subs["l_hy_oas_3m"] = _v3.rolling_z(-fred_monthly("BAMLH0A0HYM2").diff(3))
    m2 = fred_monthly("M2SL")
    if m2.dropna().empty and "liquidity" in factors_pit.columns:
        m2 = factors_pit["liquidity"].reindex(month_end)
    subs["l_m2_6m"] = _v3.rolling_z(_v3.log_momentum(m2, 6))
    subs["l_policy_12m"] = _v3.rolling_z(-fred_monthly("DGS2").diff(12))
    walcl = fred_monthly("WALCL")
    rrp = fred_monthly("RRPONTSYD")
    tga = fred_monthly("WTREGEN")
    if not walcl.dropna().empty:
        netliq = walcl - rrp.fillna(0.0) - tga.fillna(0.0)
        subs["l_netliq_3m"] = _v3.rolling_z(_v3.log_momentum(netliq, 3))

    subframe = pd.DataFrame(subs, index=month_end)

    def composite(prefix: str) -> pd.Series:
        cols = [c for c in subframe.columns if c.startswith(prefix)]
        block = subframe[cols]
        raw = block.mean(axis=1).where(block.notna().sum(axis=1) >= _v3.MIN_SUBSIGNALS)
        return raw.ewm(span=_v3.COMPOSITE_EMA_SPAN, adjust=False, min_periods=1).mean()

    composites = pd.DataFrame(
        {"G": composite("g_"), "I": composite("i_"), "L": composite("l_")},
        index=month_end,
    )
    coverage = pd.DataFrame(
        [
            {
                "subsignal": col,
                "first_valid": subframe[col].first_valid_index(),
                "last_valid": subframe[col].last_valid_index(),
            }
            for col in subframe.columns
        ]
    )
    return _v3.PillarBundle(composites=composites, subsignals=subframe, coverage=coverage)


def causality_self_check(prices: pd.DataFrame, factors_pit: pd.DataFrame) -> None:
    full = build_pillars(prices, factors_pit, allow_network=False).composites
    for fraction in (0.55, 0.75, 0.90):
        position = int(len(prices) * fraction)
        cut = prices.index[position]
        truncated = build_pillars(
            prices.iloc[: position + 1], factors_pit.loc[factors_pit.index <= cut], allow_network=False
        ).composites
        if not np.allclose(
            full.loc[cut].to_numpy(dtype=float),
            truncated.loc[cut].to_numpy(dtype=float),
            atol=1e-9,
            equal_nan=True,
        ):
            raise RuntimeError(f"V4 ALFRED pillar causality failed at {cut:%Y-%m-%d}")
    print("[v4 self-check] ALFRED point-in-time pillars passed truncation checks")


# Route frozen helper functions that resolve globals in the v3 module through
# the refreshed V4 cache readers.
_v3.fetch_fred = fetch_fred
_v3.download_yahoo_daily = download_yahoo_daily
_v3.download_yahoo_monthly = download_yahoo_monthly
