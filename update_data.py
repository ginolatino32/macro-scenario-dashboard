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


def download_yahoo_monthly(symbols: list[str], start: str, end: pd.Timestamp) -> pd.DataFrame:
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
    monthly = levels.sort_index().ffill().resample("ME").last().loc[:end]
    return monthly.dropna(how="all")


def download_fred_monthly(series_id: str, start: str, end: pd.Timestamp) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    raw = pd.read_csv(url)
    date_col = raw.columns[0]
    value_col = series_id if series_id in raw.columns else raw.columns[-1]
    raw[date_col] = pd.to_datetime(raw[date_col])
    values = pd.to_numeric(raw[value_col].replace(".", np.nan), errors="coerce")
    series = pd.Series(values.to_numpy(), index=raw[date_col], name=series_id).sort_index()
    series = series.loc[series.index >= pd.Timestamp(start)]
    return series.ffill().resample("ME").last().loc[:end]


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
    yahoo_monthly = download_yahoo_monthly(yahoo_symbols, start=start, end=end)

    fred_monthly = {
        name: download_fred_monthly(series_id, start=start, end=end)
        for name, series_id in FRED_SERIES.items()
    }

    fresh_prices = build_prices(universe, yahoo_monthly)
    fresh_factors = build_factors(yahoo_monthly, fred_monthly)

    existing_prices = _load_existing(DATA / "prices.csv")
    existing_factors = _load_existing(DATA / "factors.csv")

    prices = _merge_monthly(existing_prices, fresh_prices, bool(force_full_refresh), rewrite_tail_months)
    factors = _merge_monthly(existing_factors, fresh_factors, bool(force_full_refresh), rewrite_tail_months)

    _write_csv(DATA / "prices.csv", prices)
    _write_csv(DATA / "factors.csv", factors)

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
