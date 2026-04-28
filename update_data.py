from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CONFIG = ROOT / "config"
STATE_FILE = DATA / "update_state.json"
SOURCE_AUDIT_FILE = DATA / "source_audit.csv"
SOURCE_AUDIT_SCHEMA_VERSION = 2

SOURCE = "yfinance_fred_public_macro_proxies_v1"
DEFAULT_START = "2008-01-01"
FACTOR_COLUMNS = ["risk", "growth", "inflation", "rates", "liquidity", "dollar", "oil"]

PRICE_TICKER_MAP = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}

# Public proxies. These are not proprietary macro series, but they make the
# dashboard self-updating and directionally interpretable.
YAHOO_FACTOR_SYMBOLS = {
    "ACWI": "ACWI",
    "IEF": "IEF",
    "XLI": "XLI",
    "XLP": "XLP",
    "TIP": "TIP",
    "UUP": "UUP",
    "USO": "USO",
}

FRED_SERIES = {
    "rates": "DGS2",      # 2-year Treasury yield, percent.
    "liquidity": "M2SL",  # M2 money stock, seasonally adjusted.
}

FACTOR_SOURCE_COMPONENTS = {
    "risk": ["ACWI", "IEF"],
    "growth": ["XLI", "XLP"],
    "inflation": ["TIP", "IEF"],
    "rates": ["DGS2"],
    "liquidity": ["M2SL"],
    "dollar": ["UUP"],
    "oil": ["USO"],
}


def _last_complete_month_end(now: pd.Timestamp | None = None) -> pd.Timestamp:
    if now is None:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    month_end = now.to_period("M").to_timestamp("M")
    if now.date() >= month_end.date():
        return month_end
    return (now - pd.offsets.MonthEnd(1)).normalize()


def _read_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def _write_state(payload: dict) -> None:
    DATA.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError(f"{path} must contain a date column")
    return df.set_index("date").sort_index()


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    DATA.mkdir(exist_ok=True)
    out = df.copy()
    out.index.name = "date"
    out.reset_index().to_csv(path, index=False)


def _merge_monthly(existing: pd.DataFrame, fresh: pd.DataFrame, force_full_refresh: bool, rewrite_tail_months: int) -> pd.DataFrame:
    if force_full_refresh or existing.empty:
        return fresh.copy()

    cutoff = fresh.index.min()
    if rewrite_tail_months > 0 and not existing.empty:
        cutoff = min(cutoff, existing.index.max() - pd.offsets.MonthEnd(rewrite_tail_months - 1))

    kept = existing.loc[existing.index < cutoff]
    merged = pd.concat([kept, fresh], axis=0)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def _close_field(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        for field in ("Close", "Adj Close"):
            if field in raw.columns.get_level_values(0):
                return raw[field]
            if field in raw.columns.get_level_values(1):
                return raw.xs(field, axis=1, level=1)
        raise ValueError("Yahoo response did not contain Close or Adj Close data")
    if "Close" in raw.columns:
        return raw[["Close"]]
    if "Adj Close" in raw.columns:
        return raw[["Adj Close"]]
    raise ValueError("Yahoo response did not contain Close or Adj Close data")


def download_yahoo_daily(symbols: list[str], start: str, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        symbols,
        start=start,
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    levels = _close_field(raw)
    if isinstance(levels, pd.Series):
        levels = levels.to_frame(symbols[0])
    if len(symbols) == 1 and list(levels.columns) == ["Close"]:
        levels.columns = symbols

    levels = levels.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return levels.sort_index()


def monthly_from_daily(levels: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    monthly = levels.sort_index().ffill().resample("ME").last().loc[:end]
    return monthly.dropna(how="all")


def download_yahoo_monthly(symbols: list[str], start: str, end: pd.Timestamp) -> pd.DataFrame:
    return monthly_from_daily(download_yahoo_daily(symbols, start=start, end=end), end=end)


def _business_days_after(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if pd.isna(start) or pd.isna(end) or start >= end:
        return 0
    days = pd.bdate_range(start + pd.offsets.BDay(1), end)
    return int(len(days))


def _flatline(series: pd.Series, flatline_months: int) -> bool:
    clean = series.dropna()
    if len(clean) < flatline_months:
        return False
    recent = clean.tail(flatline_months)
    return bool(recent.diff().abs().fillna(0.0).sum() == 0.0)


def _audit_row(
    dashboard_series: str,
    source_symbol: str,
    series_type: str,
    last_true_date: pd.Timestamp | pd.NaT,
    last_resampled_date: pd.Timestamp | pd.NaT,
    monthly: pd.Series,
    end: pd.Timestamp,
    source_audit_basis: str,
    allowed_lag_business_days: int = 10,
    flatline_months: int = 6,
) -> dict[str, object]:
    stale_days = _business_days_after(pd.to_datetime(last_true_date), end) if pd.notna(last_true_date) else np.nan
    flatline = _flatline(monthly, flatline_months)
    exclude_flag = bool((pd.notna(stale_days) and stale_days > allowed_lag_business_days) or flatline)
    return {
        "dashboard_series": dashboard_series,
        "series_type": series_type,
        "source_symbol": source_symbol,
        "last_true_trade_date": last_true_date,
        "last_resampled_date": last_resampled_date,
        "stale_business_days": stale_days,
        "allowed_lag_business_days": allowed_lag_business_days,
        "flatline_6m": flatline,
        "exclude_flag": exclude_flag,
        "source_audit_status": "Exclude" if exclude_flag else "Pass",
        "source_audit_basis": source_audit_basis,
    }


def build_source_audit(
    daily_levels: pd.DataFrame,
    monthly_levels: pd.DataFrame,
    factor_levels: pd.DataFrame,
    fred_raw: dict[str, pd.Series],
    fred_monthly: dict[str, pd.Series],
    end: pd.Timestamp,
    dashboard_symbol_map: dict[str, str],
    flatline_months: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in sorted(monthly_levels.columns):
        daily = daily_levels[symbol].dropna() if symbol in daily_levels else pd.Series(dtype=float)
        monthly = monthly_levels[symbol].dropna()
        last_true_date = daily.index.max() if not daily.empty else pd.NaT
        last_resampled_date = monthly.index.max() if not monthly.empty else pd.NaT
        rows.append(
            _audit_row(
                dashboard_series=dashboard_symbol_map.get(symbol, symbol),
                source_symbol=symbol,
                series_type="asset price / market proxy",
                last_true_date=last_true_date,
                last_resampled_date=last_resampled_date,
                monthly=monthly,
                end=end,
                source_audit_basis="raw Yahoo daily close resampled to month-end",
                allowed_lag_business_days=10,
                flatline_months=flatline_months,
            )
        )

    for factor, components in FACTOR_SOURCE_COMPONENTS.items():
        if factor not in factor_levels:
            continue
        monthly_factor = factor_levels[factor].dropna()
        if factor in FRED_SERIES:
            source_id = FRED_SERIES[factor]
            raw = fred_raw.get(factor, pd.Series(dtype=float)).dropna()
            monthly_source = fred_monthly.get(factor, pd.Series(dtype=float)).dropna()
            last_true_date = raw.index.max() if not raw.empty else pd.NaT
            last_resampled_date = monthly_source.index.max() if not monthly_source.empty else pd.NaT
            allowed_lag = 45 if factor == "liquidity" else 10
            basis = "raw FRED observations resampled to month-end"
        else:
            component_dates = []
            for component in components:
                component_daily = daily_levels[component].dropna() if component in daily_levels else pd.Series(dtype=float)
                if not component_daily.empty:
                    component_dates.append(component_daily.index.max())
            last_true_date = min(component_dates) if component_dates else pd.NaT
            last_resampled_date = monthly_factor.index.max() if not monthly_factor.empty else pd.NaT
            source_id = "/".join(components)
            allowed_lag = 10
            basis = "minimum raw Yahoo component date for derived factor"

        rows.append(
            _audit_row(
                dashboard_series=factor,
                source_symbol=source_id,
                series_type="macro factor",
                last_true_date=last_true_date,
                last_resampled_date=last_resampled_date,
                monthly=monthly_factor,
                end=end,
                source_audit_basis=basis,
                allowed_lag_business_days=allowed_lag,
                flatline_months=flatline_months,
            )
        )
    return pd.DataFrame(rows)


def download_fred_series(series_id: str, start: str, end: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = pd.read_csv(url)
    date_col = raw.columns[0]
    value_col = series_id if series_id in raw.columns else raw.columns[-1]
    raw[date_col] = pd.to_datetime(raw[date_col])
    values = pd.to_numeric(raw[value_col].replace(".", np.nan), errors="coerce")
    series = pd.Series(values.to_numpy(), index=raw[date_col], name=series_id).sort_index()
    series = series.loc[series.index >= pd.Timestamp(start)]
    monthly = series.ffill().resample("ME").last().loc[:end]
    return series.loc[:end], monthly


def download_fred_monthly(series_id: str, start: str, end: pd.Timestamp) -> pd.Series:
    return download_fred_series(series_id, start=start, end=end)[1]


def build_prices(universe: pd.DataFrame, yahoo_monthly: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=yahoo_monthly.index)
    for ticker in universe["ticker"]:
        yahoo_symbol = PRICE_TICKER_MAP.get(ticker, ticker)
        if yahoo_symbol in yahoo_monthly.columns:
            out[ticker] = yahoo_monthly[yahoo_symbol]
    return out.dropna(how="all")


def build_factors(yahoo_monthly: pd.DataFrame, fred_monthly: dict[str, pd.Series]) -> pd.DataFrame:
    required = set(YAHOO_FACTOR_SYMBOLS.values())
    missing = sorted(required - set(yahoo_monthly.columns))
    if missing:
        raise ValueError(f"Missing required Yahoo factor proxies: {missing}")

    idx = yahoo_monthly.index
    factors = pd.DataFrame(index=idx)
    factors["risk"] = 100.0 * yahoo_monthly["ACWI"] / yahoo_monthly["IEF"]
    factors["growth"] = 100.0 * yahoo_monthly["XLI"] / yahoo_monthly["XLP"]
    factors["inflation"] = 100.0 * yahoo_monthly["TIP"] / yahoo_monthly["IEF"]
    factors["rates"] = fred_monthly["rates"].reindex(idx).ffill()
    factors["liquidity"] = fred_monthly["liquidity"].reindex(idx).ffill()
    factors["dollar"] = yahoo_monthly["UUP"]
    factors["oil"] = yahoo_monthly["USO"]
    return factors[FACTOR_COLUMNS].dropna(how="all")


def update_data(
    start: str = DEFAULT_START,
    force_full_refresh: bool | None = None,
    rewrite_tail_months: int = 2,
) -> dict:
    DATA.mkdir(exist_ok=True)
    universe = pd.read_csv(CONFIG / "universe.csv")
    state = _read_state()
    if force_full_refresh is None:
        force_full_refresh = state.get("source") != SOURCE

    end = _last_complete_month_end()
    price_symbols = {PRICE_TICKER_MAP.get(t, t) for t in universe["ticker"]}
    yahoo_symbols = sorted(price_symbols | set(YAHOO_FACTOR_SYMBOLS.values()))
    yahoo_daily = download_yahoo_daily(yahoo_symbols, start=start, end=end)
    yahoo_monthly = monthly_from_daily(yahoo_daily, end=end)

    fred_raw: dict[str, pd.Series] = {}
    fred_monthly: dict[str, pd.Series] = {}
    for name, series_id in FRED_SERIES.items():
        raw_series, monthly_series = download_fred_series(series_id, start=start, end=end)
        fred_raw[name] = raw_series
        fred_monthly[name] = monthly_series

    fresh_prices = build_prices(universe, yahoo_monthly)
    fresh_factors = build_factors(yahoo_monthly, fred_monthly)

    existing_prices = _load_existing(DATA / "prices.csv")
    existing_factors = _load_existing(DATA / "factors.csv")

    prices = _merge_monthly(existing_prices, fresh_prices, bool(force_full_refresh), rewrite_tail_months)
    factors = _merge_monthly(existing_factors, fresh_factors, bool(force_full_refresh), rewrite_tail_months)

    _write_csv(DATA / "prices.csv", prices)
    _write_csv(DATA / "factors.csv", factors)
    reverse_price_map = {PRICE_TICKER_MAP.get(ticker, ticker): ticker for ticker in universe["ticker"]}
    source_audit = build_source_audit(
        yahoo_daily,
        yahoo_monthly,
        fresh_factors,
        fred_raw,
        fred_monthly,
        end=end,
        dashboard_symbol_map=reverse_price_map,
    )
    source_audit.to_csv(SOURCE_AUDIT_FILE, index=False)

    payload = {
        "source": SOURCE,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": start,
        "last_complete_month": end.strftime("%Y-%m-%d"),
        "force_full_refresh": bool(force_full_refresh),
        "rewrite_tail_months": rewrite_tail_months,
        "prices_rows": int(len(prices)),
        "factors_rows": int(len(factors)),
        "prices_last_date": prices.index.max().strftime("%Y-%m-%d") if not prices.empty else None,
        "factors_last_date": factors.index.max().strftime("%Y-%m-%d") if not factors.empty else None,
        "price_symbols": yahoo_symbols,
        "fred_series": FRED_SERIES,
        "source_audit_rows": int(len(source_audit)),
        "source_audit_schema_version": SOURCE_AUDIT_SCHEMA_VERSION,
    }
    _write_state(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and append monthly market/macro data for the dashboard.")
    parser.add_argument("--start", default=DEFAULT_START, help="Download start date for full refreshes.")
    parser.add_argument("--force-full-refresh", action="store_true", help="Overwrite data CSVs with a fresh full history.")
    parser.add_argument("--append-only", action="store_true", help="Keep existing history and append/update recent months only.")
    parser.add_argument("--rewrite-tail-months", type=int, default=2, help="Recent existing months to refresh during append mode.")
    args = parser.parse_args()

    if args.force_full_refresh and args.append_only:
        raise SystemExit("Choose either --force-full-refresh or --append-only, not both.")

    force = True if args.force_full_refresh else False if args.append_only else None
    result = update_data(start=args.start, force_full_refresh=force, rewrite_tail_months=args.rewrite_tail_months)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
