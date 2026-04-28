from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import (
    FACTOR_COLUMNS,
    build_model,
    compare_scenarios,
    estimate_scenario_probabilities,
    load_wide_csv,
    optimize_probability_weighted_portfolio,
    probability_weighted_asset_ranking,
    scenario_overlay_breakdown,
    top_bottom_by_bucket,
    walk_forward_scenario_calibration,
    walk_forward_predicted_scenario_portfolio,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CONFIG = ROOT / "config"
STATE_FILE = DATA / "update_state.json"
SOURCE_AUDIT_FILE = DATA / "source_audit.csv"
SOURCE_AUDIT_SCHEMA_VERSION = 2
BACKTEST_CACHE_VERSION = "predicted_scenario_backtest_equity_v6"

COLORS = {
    "positive": "#14b8a6",
    "negative": "#f97316",
    "neutral": "#64748b",
    "accent": "#3b82f6",
    "unknown": "#94a3b8",
    "grid": "rgba(148, 163, 184, 0.20)",
    "paper": "#0f172a",
    "plot": "#111827",
}

st.set_page_config(page_title="Macro Scenario Dashboard", layout="wide")
px.defaults.template = "plotly_dark"
st.title("Macro Scenario Dashboard")
st.caption(
    "Scenario-conditioned macro tilts, factor attribution, and model-quality diagnostics using public monthly market data and macro proxies. "
    "Validate the proxy choices and model behavior before using capital."
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1320px; padding-top: 2.3rem;}
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.40);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        padding: 0.85rem 0.95rem;
    }
    div[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #cbd5e1;}
    div[data-testid="stMetricValue"] {letter-spacing: 0;}
    .stDataFrame {border: 1px solid rgba(148, 163, 184, 0.16); border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def last_completed_month_end(now: pd.Timestamp | None = None) -> pd.Timestamp:
    if now is None:
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    month_end = now.to_period("M").to_timestamp("M")
    if now.date() >= month_end.date():
        return month_end
    return (now - pd.offsets.MonthEnd(1)).normalize()


def read_update_state_file() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def source_audit_file_is_current(state: dict) -> bool:
    if int(state.get("source_audit_schema_version", 0) or 0) < SOURCE_AUDIT_SCHEMA_VERSION:
        return False
    if not SOURCE_AUDIT_FILE.exists():
        return False
    try:
        audit = pd.read_csv(SOURCE_AUDIT_FILE, usecols=["dashboard_series", "series_type", "exclude_flag"])
    except (ValueError, FileNotFoundError):
        return False
    required_factors = set(FACTOR_COLUMNS)
    available = set(audit["dashboard_series"].astype(str))
    required_columns = {"dashboard_series", "series_type", "exclude_flag"}
    return required_columns.issubset(audit.columns) and required_factors.issubset(available)


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def refresh_data_on_startup() -> dict:
    state = read_update_state_file()
    expected_month = last_completed_month_end().strftime("%Y-%m-%d")
    source_audit_current = source_audit_file_is_current(state)
    snapshot_current = (
        state.get("last_complete_month") == expected_month
        and state.get("prices_last_date") == expected_month
        and source_audit_current
    )
    payload = dict(state)
    payload.update(
        {
            "ok": snapshot_current,
            "skipped": True,
            "skip_reason": "read-only-snapshot-current" if snapshot_current else "read-only-snapshot-stale",
            "expected_month": expected_month,
            "read_only_runtime": True,
            "source_audit_current": source_audit_current,
            "updated_at_utc": state.get("updated_at_utc", "snapshot-not-available"),
        }
    )
    return payload


@st.cache_data
def load_inputs(refresh_token: str):
    prices = load_wide_csv(DATA / "prices.csv")
    factors = load_wide_csv(DATA / "factors.csv")
    universe = pd.read_csv(CONFIG / "universe.csv")
    scenarios = pd.read_csv(CONFIG / "scenarios.csv")
    return prices, factors, universe, scenarios


@st.cache_data
def build_scenario_matrices(prices: pd.DataFrame, factors: pd.DataFrame, universe: pd.DataFrame, scenarios: pd.DataFrame, half_life: float):
    names = [s for s in scenarios["scenario"].tolist() if s != "Custom"]
    expected_cols = []
    conviction_cols = []
    for name in names:
        scenario_row = scenarios.set_index("scenario").loc[name, FACTOR_COLUMNS].astype(float)
        scenario_result = build_model(prices, factors, universe, scenario_row, half_life=half_life)
        expected_cols.append(scenario_result.expected["expected_return_pct"].rename(name))
        conviction_cols.append(scenario_result.expected["conviction_score"].rename(name))
    meta = universe.set_index("ticker")[["name", "bucket"]]
    expected = pd.concat(expected_cols, axis=1).join(meta, how="left")
    conviction = pd.concat(conviction_cols, axis=1).join(meta, how="left")
    return expected, conviction


@st.cache_data
def build_calibration_report(factors: pd.DataFrame, scenarios: pd.DataFrame):
    return walk_forward_scenario_calibration(factors, scenarios, lookback=84, horizon=1, max_periods=96)


@st.cache_data(show_spinner=False)
def build_predicted_portfolio_backtest(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    universe: pd.DataFrame,
    scenarios: pd.DataFrame,
    half_life: float,
    n_each: int,
    transaction_cost_bps: float,
    cache_version: str,
):
    return walk_forward_predicted_scenario_portfolio(
        prices,
        factors,
        universe,
        scenarios,
        lookback=84,
        half_life=half_life,
        n_each=n_each,
        max_per_bucket=2,
        transaction_cost_bps=transaction_cost_bps,
    )


def format_pct_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out


def polish_figure(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e5e7eb", "size": 13},
        margin={"l": 12, "r": 12, "t": 24, "b": 24},
        legend={"orientation": "v", "bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor="rgba(229,231,235,0.55)")
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor="rgba(229,231,235,0.55)")
    if height is not None:
        fig.update_layout(height=height)
    return fig


def add_starting_value(frame: pd.DataFrame, start_date: pd.Timestamp, value: float = 1.0) -> pd.DataFrame:
    start = pd.DataFrame({col: [value] for col in frame.columns}, index=[start_date])
    start.index.name = frame.index.name
    return pd.concat([start, frame]).sort_index()


def load_update_state() -> dict:
    return read_update_state_file()


def load_source_audit() -> pd.DataFrame:
    if not SOURCE_AUDIT_FILE.exists():
        return pd.DataFrame()
    try:
        audit = pd.read_csv(SOURCE_AUDIT_FILE, parse_dates=["last_true_trade_date", "last_resampled_date"])
    except (ValueError, FileNotFoundError):
        return pd.DataFrame()
    required = {"dashboard_series", "last_true_trade_date", "last_resampled_date", "exclude_flag"}
    if not required.issubset(audit.columns):
        return pd.DataFrame()
    if not set(FACTOR_COLUMNS).issubset(set(audit["dashboard_series"].astype(str))):
        return pd.DataFrame()
    return audit


def dashboard_run_id() -> str:
    digest = hashlib.sha256()
    for path in [
        ROOT / "app.py",
        ROOT / "model.py",
        DATA / "prices.csv",
        DATA / "factors.csv",
        SOURCE_AUDIT_FILE,
        CONFIG / "universe.csv",
        CONFIG / "scenarios.csv",
    ]:
        if path.exists():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def status_label(current: int, total: int, amber_reason: bool = False) -> str:
    if total <= 0:
        return "Red"
    if current == total and not amber_reason:
        return "Green"
    if current > 0:
        return "Amber"
    return "Red"


def confidence_label(value: float) -> str:
    if value >= 0.70:
        return "High"
    if value >= 0.45:
        return "Medium"
    return "Low"


def data_freshness_table(refresh_info: dict, prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    state = load_update_state()
    rows = [
        {
            "item": "Asset prices",
            "last_date": prices.index.max().strftime("%Y-%m-%d") if not prices.empty else None,
            "rows": len(prices),
            "source": state.get("source", refresh_info.get("source")),
        },
        {
            "item": "Macro factors",
            "last_date": factors.index.max().strftime("%Y-%m-%d") if not factors.empty else None,
            "rows": len(factors),
            "source": state.get("source", refresh_info.get("source")),
        },
        {
            "item": "Last completed month",
            "last_date": state.get("last_complete_month", refresh_info.get("last_complete_month")),
            "rows": None,
            "source": "calendar cutoff",
        },
        {
            "item": "Last refresh run",
            "last_date": state.get("updated_at_utc", refresh_info.get("updated_at_utc")),
            "rows": None,
            "source": "local updater",
        },
    ]
    return pd.DataFrame(rows)


def _month_gap(latest: pd.Timestamp | None, expected: pd.Timestamp | None) -> int | None:
    if latest is None or expected is None or pd.isna(latest) or pd.isna(expected):
        return None
    return max(0, (expected.year - latest.year) * 12 + expected.month - latest.month)


def data_quality_table(frame: pd.DataFrame, expected_last_month: pd.Timestamp | None, names: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    names = names or {}
    total_rows = len(frame)
    for col in frame.columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        valid = series.dropna()
        last_valid = valid.index.max() if not valid.empty else None
        first_valid = valid.index.min() if not valid.empty else None
        stale_months = _month_gap(last_valid, expected_last_month)
        latest_value = valid.iloc[-1] if not valid.empty else np.nan
        coverage = float(valid.shape[0] / total_rows * 100.0) if total_rows else np.nan
        if last_valid is None:
            status = "No data"
        elif stale_months == 0:
            status = "Current"
        else:
            status = "Stale"
        rows.append(
            {
                "series": col,
                "name": names.get(col, col),
                "status": status,
                "first_valid": first_valid,
                "last_valid": last_valid,
                "stale_months": stale_months,
                "observations": int(valid.shape[0]),
                "coverage_pct": coverage,
                "missing_values": int(series.isna().sum()),
                "latest_value": latest_value,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    status_rank = {"No data": 2, "Stale": 1, "Current": 0}
    out["_status_rank"] = out["status"].map(status_rank).fillna(3)
    out = out.sort_values(["_status_rank", "stale_months", "series"], ascending=[False, False, True])
    return out.drop(columns=["_status_rank"])


def stale_flatline_fallback(frame: pd.DataFrame, names: dict[str, str] | None = None, label: str = "asset") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col in frame.columns:
        series = frame[col].dropna()
        if series.empty:
            continue
        recent = series.tail(6)
        stale_months = (frame.index.max().year - series.index.max().year) * 12 + (frame.index.max().month - series.index.max().month)
        flatline = bool(len(recent) >= 6 and recent.diff().abs().fillna(0.0).sum() == 0.0)
        exclude_flag = bool(stale_months > 0 or flatline)
        rows.append(
            {
                "dashboard_series": col,
                "name": names.get(col, col) if names else col,
                "series_type": label,
                "source_symbol": "stored monthly CSV",
                "last_true_trade_date": pd.NaT,
                "last_resampled_date": series.index.max(),
                "stale_business_days": np.nan,
                "allowed_lag_business_days": np.nan,
                "stale_months": int(max(stale_months, 0)),
                "flatline_6m": flatline,
                "exclude_flag": exclude_flag,
                "source_audit_status": "Fallback exclude" if exclude_flag else "Fallback only",
                "source_audit_basis": "monthly fallback; run data workflow to populate last true trade dates",
            }
        )
    return pd.DataFrame(rows)


def bucket_liquidity_score(bucket: str) -> float:
    scores = {
        "Asset Class": 92.0,
        "Equity Sector": 90.0,
        "Equity Region": 86.0,
        "Fixed Income": 84.0,
        "Style Factor": 82.0,
        "Commodity": 72.0,
        "FX": 70.0,
        "Crypto": 55.0,
    }
    return scores.get(str(bucket), 65.0)


def bucket_cost_score(bucket: str) -> float:
    scores = {
        "Asset Class": 90.0,
        "Equity Sector": 88.0,
        "Equity Region": 82.0,
        "Fixed Income": 82.0,
        "Style Factor": 78.0,
        "Commodity": 70.0,
        "FX": 68.0,
        "Crypto": 45.0,
    }
    return scores.get(str(bucket), 65.0)


def universe_quality_table(
    universe: pd.DataFrame,
    prices: pd.DataFrame,
    price_quality: pd.DataFrame,
    source_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    returns = np.exp(np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)) - 1.0
    rows = []
    quality = price_quality.set_index("series") if not price_quality.empty else pd.DataFrame()
    source_excluded: set[str] = set()
    if source_audit is not None and not source_audit.empty:
        audit = source_audit.copy()
        if {"dashboard_series", "series_type", "exclude_flag"}.issubset(audit.columns):
            asset_audit = audit[
                audit["series_type"].astype(str).str.contains("asset price", case=False, na=False)
                & audit["exclude_flag"].fillna(False).astype(bool)
            ]
            source_excluded = set(asset_audit["dashboard_series"].astype(str))
    for _, row in universe.iterrows():
        ticker = str(row["ticker"])
        bucket = str(row["bucket"])
        q = quality.loc[ticker] if ticker in quality.index else pd.Series(dtype=object)
        coverage = float(q.get("coverage_pct", 0.0) or 0.0)
        stale_months = q.get("stale_months", np.nan)
        stale_penalty = min(float(stale_months) * 10.0, 40.0) if pd.notna(stale_months) else 40.0
        proxy_quality = float(np.clip(coverage - stale_penalty, 0.0, 100.0))
        liquidity = bucket_liquidity_score(bucket)
        cost = bucket_cost_score(bucket)
        data_quality = proxy_quality
        investability = 0.4 * liquidity + 0.3 * proxy_quality + 0.2 * cost + 0.1 * data_quality
        asset_returns = returns[ticker].dropna() if ticker in returns else pd.Series(dtype=float)
        annual_vol = float(asset_returns.std(ddof=0) * np.sqrt(12.0) * 100.0) if not asset_returns.empty else np.nan
        max_abs_month = float(asset_returns.abs().max() * 100.0) if not asset_returns.empty else np.nan
        source_exclude = ticker in source_excluded
        if source_exclude:
            investability = min(investability, 49.0)
        long_only = investability >= 55.0 and q.get("status") in {"Current", "Stale"} and not source_exclude
        long_short = investability >= 70.0 and q.get("status") == "Current" and bucket != "Crypto" and not source_exclude
        if source_exclude:
            reason = "excluded by raw-source stale/flatline audit"
        elif q.get("status") != "Current":
            reason = "stale or missing price history"
        elif investability < 55.0:
            reason = "research-only: low investability score"
        elif not long_short:
            reason = "long-only or research-only liquidity/cost profile"
        else:
            reason = "eligible"
        rows.append(
            {
                "ticker": ticker,
                "name": row.get("name", ticker),
                "bucket": bucket,
                "status": q.get("status", "No data"),
                "first_valid": q.get("first_valid", pd.NaT),
                "last_valid": q.get("last_valid", pd.NaT),
                "coverage_pct": coverage,
                "proxy_quality_score": proxy_quality,
                "liquidity_score": liquidity,
                "cost_score": cost,
                "data_quality_score": data_quality,
                "investability_score": investability,
                "raw_source_excluded": bool(source_exclude),
                "annual_vol_pct": annual_vol,
                "max_abs_month_pct": max_abs_month,
                "allowed_long_only": bool(long_only),
                "allowed_long_short": bool(long_short),
                "reason": reason,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["allowed_long_short", "investability_score", "ticker"], ascending=[True, True, True])


def bucket_tilt_label(score: float) -> str:
    if score >= 0.35:
        return "Overweight"
    if score <= -0.35:
        return "Underweight / hedge"
    return "Neutral"


def scenario_summary_line(scenario: pd.Series) -> str:
    if scenario.abs().sum() == 0:
        return "Manual scenario assumptions: no active macro shock; the manual playbook should stay close to neutral."
    top = scenario.abs().sort_values(ascending=False).head(3).index
    drivers = ", ".join(f"{factor} {scenario[factor]:+.1f}z" for factor in top)
    return f"Manual scenario assumptions: {drivers}."


def best_worst_bucket(bucket_summary: pd.DataFrame) -> tuple[str, str]:
    if bucket_summary.empty:
        return "n/a", "n/a"
    best = bucket_summary.sort_values("avg_conviction_score", ascending=False).iloc[0]["bucket"]
    worst = bucket_summary.sort_values("avg_conviction_score", ascending=True).iloc[0]["bucket"]
    return str(best), str(worst)

refresh_info = refresh_data_on_startup()

try:
    prices, factors, universe, scenarios = load_inputs(refresh_info.get("updated_at_utc", "no-refresh-token"))
except FileNotFoundError:
    st.error("Missing data files. Run `python sample_data.py` first, then restart the app.")
    st.stop()

if refresh_info["ok"]:
    st.caption(
        f"Read-only data snapshot current through {refresh_info.get('prices_last_date') or refresh_info.get('last_complete_month')} "
        f"from {refresh_info.get('source')}."
    )
else:
    st.error(
        "Read-only data snapshot is stale or missing its source audit. "
        "Run the scheduled/manual GitHub data workflow before using model output."
    )

scenario_names = list(scenarios["scenario"])
state = load_update_state()
expected_month = pd.to_datetime(state.get("last_complete_month") or refresh_info.get("last_complete_month"), errors="coerce")
asset_names = universe.set_index("ticker")["name"].to_dict()
price_quality = data_quality_table(prices, expected_month, asset_names)
factor_quality = data_quality_table(factors, expected_month)
source_audit_exact = load_source_audit()
source_audit_is_current = source_audit_file_is_current(state)
universe_quality = universe_quality_table(universe, prices, price_quality, source_audit_exact)

with st.sidebar:
    st.header("Data")
    st.caption("Streamlit runtime is read-only. Refresh data with the GitHub Action or run update_data.py before deployment.")
    if os.getenv("MACRO_DASHBOARD_ALLOW_RUNTIME_REFRESH", "0") == "1":
        if st.button("Local research refresh", use_container_width=True):
            try:
                from update_data import update_data

                payload = update_data(force_full_refresh=False)
                st.session_state["manual_refresh_status"] = (
                    f"Local refresh complete through {payload.get('prices_last_date') or payload.get('factors_last_date')}."
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.session_state["manual_refresh_status"] = f"Local refresh failed: {type(exc).__name__}: {exc}"
        if st.session_state.get("manual_refresh_status"):
            st.caption(st.session_state["manual_refresh_status"])

    st.header("Scenario")
    preset = st.selectbox("Preset", scenario_names, index=scenario_names.index("Summer") if "Summer" in scenario_names else 0)
    preset_row = scenarios.set_index("scenario").loc[preset, FACTOR_COLUMNS].astype(float)

    st.write("Adjust macro factor shocks. Values are standardized units by default.")
    scenario_values = {}
    for c in FACTOR_COLUMNS:
        scenario_values[c] = st.slider(c, min_value=-3.0, max_value=3.0, value=float(preset_row[c]), step=0.1)
    scenario = pd.Series(scenario_values)

    st.header("Model")
    half_life = st.slider("Recency half-life, months", min_value=6, max_value=120, value=36, step=6)
    top_n = st.slider("Top/bottom names", min_value=1, max_value=10, value=3, step=1)
    basket_n = st.slider("Basket names per side", min_value=3, max_value=10, value=6, step=1)
    backtest_cost_bps = st.slider("Backtest cost, bps", min_value=0, max_value=50, value=5, step=1)
    apply_investability_gate = st.checkbox("Apply investability gate", value=True)
    min_investability_score = st.slider("Minimum investability score", min_value=0, max_value=100, value=60, step=5)

if not source_audit_is_current:
    st.stop()

if not apply_investability_gate:
    st.warning("Research override active: the investability gate is disabled. Do not treat rankings or portfolio output as investable.")

eligible_tickers = universe_quality[
    (universe_quality["status"] == "Current")
    & (universe_quality["investability_score"] >= float(min_investability_score))
    & (universe_quality["allowed_long_only"])
]["ticker"].tolist()
active_universe = universe[universe["ticker"].isin(eligible_tickers)].copy() if apply_investability_gate else universe.copy()
if active_universe.empty:
    st.error("Investability gate excluded the full universe. Model output is stopped instead of falling back to unapproved assets.")
    st.stop()

result = build_model(prices, factors, active_universe, scenario, half_life=float(half_life))
scenario_expected, scenario_conviction = build_scenario_matrices(prices, factors, active_universe, scenarios, half_life=float(half_life))
auto_regime = estimate_scenario_probabilities(factors, scenarios)
probability_rank = probability_weighted_asset_ranking(scenario_expected, auto_regime.probabilities, result.expected)
overlay_breakdown = scenario_overlay_breakdown(auto_regime.probabilities, scenarios)
calibration = build_calibration_report(factors, scenarios)
optimized_portfolio = optimize_probability_weighted_portfolio(
    scenario_expected,
    auto_regime.probabilities,
    result.expected,
    result.rel_returns,
    active_universe,
)
predicted_portfolio_backtest = build_predicted_portfolio_backtest(
    prices,
    factors,
    active_universe,
    scenarios,
    half_life=float(half_life),
    n_each=basket_n,
    transaction_cost_bps=float(backtest_cost_bps),
    cache_version=BACKTEST_CACHE_VERSION,
)
point_in_time_audit = getattr(predicted_portfolio_backtest, "point_in_time_audit", pd.DataFrame()).copy()
result.trade_basket = result.trade_basket.copy()
if not result.trade_basket.empty:
    # Rebuild display basket from the full expected table if the user changes basket size.
    from model import build_trade_basket

    result.trade_basket = build_trade_basket(result.expected, n_each=basket_n, max_per_bucket=2)

best_bucket, worst_bucket = best_worst_bucket(result.bucket_summary)
basket_edge = result.trade_basket["expected_contribution_pct"].sum() if not result.trade_basket.empty else 0.0
auto_probs = auto_regime.probabilities.copy()
modal_row = auto_probs[~auto_probs["is_unknown"]].sort_values("probability", ascending=False).iloc[0]
unknown_probability = float(auto_probs.loc[auto_probs["is_unknown"], "probability"].sum())
pit_flags = (
    point_in_time_audit["lookahead_flag"].sum()
    if not point_in_time_audit.empty and "lookahead_flag" in point_in_time_audit
    else np.nan
)
date_boundary_status = "Unavailable" if not np.isfinite(pit_flags) else ("Pass" if pit_flags == 0 else "Fail")
refresh_date = pd.to_datetime(state.get("updated_at_utc", refresh_info.get("updated_at_utc")), errors="coerce")
cache_age_hours = (
    (pd.Timestamp.now(tz="UTC").tz_localize(None) - refresh_date.tz_localize(None)).total_seconds() / 3600.0
    if pd.notna(refresh_date)
    else np.nan
)
data_status = status_label(
    int((price_quality["status"] == "Current").sum()) + int((factor_quality["status"] == "Current").sum()),
    len(price_quality) + len(factor_quality),
    amber_reason=True,
)
data_status_display = f"{data_status} (proxy/vintage limits)" if data_status == "Amber" else data_status
st.warning(
    "Mode: Research only | "
    f"Data through: {expected_month.strftime('%Y-%m-%d') if pd.notna(expected_month) else 'n/a'} | "
    f"Run ID: {dashboard_run_id()} | "
    f"Date-boundary check: {date_boundary_status} | "
    f"Vintage status: Not available; latest-revised proxies, not ALFRED vintages | "
    f"Transaction costs: {backtest_cost_bps} bps | "
    f"Cache age: {cache_age_hours:.1f}h | "
    f"Data status: {data_status_display} | "
    f"Model confidence: {confidence_label(auto_regime.confidence)}"
)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Assets modeled", f"{len(result.expected):,}/{len(universe):,}" if apply_investability_gate else f"{len(result.expected):,}")
c2.metric("Auto modal scenario", str(modal_row["scenario"]))
c3.metric("Unknown / mixed", f"{unknown_probability * 100:.1f}%")
c4.metric("Auto confidence", f"{auto_regime.confidence * 100:.1f}%")
c5.metric("Median model R²", f"{result.expected['model_r2'].median():.2f}")

st.info(scenario_summary_line(scenario))
st.caption(
    f"Selected manual scenario: {preset}. Auto-regime metrics and probability rankings are computed from the latest data and do not use the manual sidebar preset."
)

tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Auto Regime",
        "Investment Brief",
        "Probability Rankings",
        "Portfolio",
        "Names",
        "Factor Attribution",
        "Scenario Playbook",
        "Diagnostics",
        "Data",
    ]
)

with tab0:
    st.subheader("Automatic scenario probabilities")
    st.caption("Computed from the latest macro state only, then transition-smoothed. Asset rankings do not influence these probabilities.")
    p_view = auto_probs.copy()
    p_view["probability_pct"] = p_view["probability"] * 100.0
    p_view["distance"] = p_view["normalized_distance"]
    p_colors = np.where(p_view["is_unknown"], COLORS["unknown"], COLORS["accent"])
    fig = go.Figure(
        go.Bar(
            x=p_view["probability_pct"],
            y=p_view["scenario"],
            orientation="h",
            marker_color=p_colors,
            text=p_view["probability_pct"].map(lambda x: f"{x:.1f}%"),
            textposition="auto",
        )
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Probability",
        yaxis_title="",
        height=420,
    )
    polish_figure(fig, height=420)
    st.plotly_chart(fig, width="stretch")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Modal scenario", str(modal_row["scenario"]))
    r2.metric("Modal probability", f"{float(modal_row['probability']) * 100:.1f}%")
    r3.metric("Entropy", f"{auto_regime.entropy:.2f}")
    r4.metric("Confidence", f"{auto_regime.confidence * 100:.1f}%")
    raw_view = auto_regime.raw_probabilities[["scenario", "probability"]].rename(columns={"probability": "raw_probability"})
    prior_view = auto_regime.transition_prior.rename(columns={"transition_prior": "transition_prior_probability"})
    probability_detail = p_view[["scenario", "probability_pct", "distance", "score", "is_unknown"]].merge(raw_view, on="scenario", how="left")
    probability_detail = probability_detail.merge(prior_view, on="scenario", how="left")
    probability_detail["raw_probability_pct"] = probability_detail["raw_probability"] * 100.0
    probability_detail["transition_prior_pct"] = probability_detail["transition_prior_probability"] * 100.0
    st.dataframe(
        format_pct_columns(
            probability_detail[
                [
                    "scenario",
                    "probability_pct",
                    "raw_probability_pct",
                    "transition_prior_pct",
                    "distance",
                    "score",
                    "is_unknown",
                ]
            ],
            ["probability_pct", "raw_probability_pct", "transition_prior_pct", "distance", "score"],
        ),
        width="stretch",
    )

    st.subheader("Core regimes and overlays")
    st.caption("Preset probabilities are regrouped into core macro regime, policy/liquidity overlay, and stress overlay.")
    overlay_view = overlay_breakdown.copy()
    fig = px.bar(
        overlay_view,
        x="probability_pct",
        y="state",
        color="layer",
        facet_col="layer",
        orientation="h",
        text=overlay_view["probability_pct"].map(lambda x: f"{x:.1f}%"),
        labels={"probability_pct": "Probability", "state": ""},
        color_discrete_sequence=[COLORS["accent"], COLORS["positive"], COLORS["negative"]],
    )
    fig.update_yaxes(matches=None, showticklabels=True)
    polish_figure(fig, height=360)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(format_pct_columns(overlay_view, ["probability", "probability_pct"]), width="stretch")

    st.subheader("Current macro state used by the probability engine")
    feature_view = auto_regime.latest_features.rename("value").to_frame()
    feature_view["abs_value"] = feature_view["value"].abs()
    feature_view = feature_view.sort_values("abs_value", ascending=False).head(12).drop(columns=["abs_value"])
    st.dataframe(format_pct_columns(feature_view, ["value"]), width="stretch")

with tab1:
    st.subheader("Scenario allocation view")
    st.caption(f"Manual playbook for the selected sidebar scenario: {preset}.")
    bucket_view = result.bucket_summary.copy()
    if not bucket_view.empty:
        bucket_view["tilt"] = bucket_view["avg_conviction_score"].apply(bucket_tilt_label)
        fig = px.bar(
            bucket_view.sort_values("avg_conviction_score"),
            x="avg_conviction_score",
            y="bucket",
            color="tilt",
            orientation="h",
            hover_data=["avg_expected_return_pct", "top_name", "bottom_name", "avg_model_r2"],
            labels={"avg_conviction_score": "Conviction score", "bucket": ""},
            color_discrete_map={"Overweight": COLORS["positive"], "Neutral": COLORS["neutral"], "Underweight / hedge": COLORS["negative"]},
        )
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color=COLORS["unknown"])
        polish_figure(fig, height=420)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            format_pct_columns(
                bucket_view[
                    [
                        "bucket",
                        "tilt",
                        "avg_expected_return_pct",
                        "avg_conviction_score",
                        "avg_model_r2",
                        "avg_residual_vol_pct",
                        "top_name",
                        "top_expected_return_pct",
                        "bottom_name",
                        "bottom_expected_return_pct",
                    ]
                ],
                ["avg_expected_return_pct", "avg_conviction_score", "avg_model_r2", "avg_residual_vol_pct", "top_expected_return_pct", "bottom_expected_return_pct"],
            ),
            width="stretch",
        )

    st.subheader("Diversified scenario playbook")
    if result.trade_basket.empty:
        st.warning("No long/short basket could be formed from this scenario.")
    else:
        b1, b2, b3 = st.columns(3)
        b1.metric("Expected basket contribution", f"{basket_edge:.2f}%")
        b2.metric("Average model R²", f"{result.trade_basket['model_r2'].mean():.2f}")
        b3.metric("Gross exposure", f"{result.trade_basket['portfolio_weight_pct'].abs().sum():.0f}%")
        basket_cols = [
            "side",
            "name",
            "bucket",
            "portfolio_weight_pct",
            "expected_return_pct",
            "expected_contribution_pct",
            "conviction_score",
            "model_r2",
            "policy_signal",
        ]
        st.dataframe(
            format_pct_columns(
                result.trade_basket[basket_cols],
                ["portfolio_weight_pct", "expected_return_pct", "expected_contribution_pct", "conviction_score", "model_r2"],
            ),
            width="stretch",
        )

with tab2:
    st.subheader("Probability-weighted robust ranking")
    st.caption("Uses the automated scenario probability distribution above, not the manual sidebar scenario.")
    rank_view = probability_rank.copy()
    fig = px.scatter(
        rank_view,
        x="downside_loss_pct",
        y="weighted_expected_return_pct",
        size="rank_stability",
        color="robust_score",
        hover_name="name",
        hover_data=["bucket", "best_scenario", "worst_scenario", "scenario_dispersion_pct", "weighted_regret_pct"],
        labels={
            "downside_loss_pct": "Downside loss in plausible scenarios",
            "weighted_expected_return_pct": "Probability-weighted expected return",
            "robust_score": "Robust score",
            "rank_stability": "Rank stability",
        },
        color_continuous_scale=[COLORS["negative"], "#facc15", COLORS["positive"]],
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["unknown"])
    polish_figure(fig, height=520)
    st.plotly_chart(fig, width="stretch")

    robust_cols = [
        "name",
        "bucket",
        "robust_score",
        "weighted_expected_return_pct",
        "scenario_dispersion_pct",
        "downside_loss_pct",
        "rank_stability",
        "weighted_regret_pct",
        "best_scenario",
        "worst_scenario",
        "model_r2",
    ]
    left, right = st.columns(2)
    with left:
        st.markdown("**Top robust assets**")
        st.dataframe(
            format_pct_columns(
                rank_view.head(12)[robust_cols],
                [
                    "robust_score",
                    "weighted_expected_return_pct",
                    "scenario_dispersion_pct",
                    "downside_loss_pct",
                    "rank_stability",
                    "weighted_regret_pct",
                    "model_r2",
                ],
            ),
            width="stretch",
        )
    with right:
        st.markdown("**Most fragile high-upside assets**")
        fragile = rank_view.sort_values(["fragility_score", "best_scenario_return_pct"], ascending=False).head(12)
        fragile_cols = [
            "name",
            "bucket",
            "fragility_score",
            "best_scenario_return_pct",
            "weighted_expected_return_pct",
            "scenario_dispersion_pct",
            "downside_loss_pct",
            "best_scenario",
            "worst_scenario",
        ]
        st.dataframe(
            format_pct_columns(
                fragile[fragile_cols],
                [
                    "fragility_score",
                    "best_scenario_return_pct",
                    "weighted_expected_return_pct",
                    "scenario_dispersion_pct",
                    "downside_loss_pct",
                ],
            ),
            width="stretch",
        )

    st.subheader("Assets to avoid under current probability distribution")
    avoid = rank_view.sort_values(["robust_score", "downside_loss_pct"], ascending=[True, False]).head(15)
    st.dataframe(
        format_pct_columns(
            avoid[robust_cols],
            [
                "robust_score",
                "weighted_expected_return_pct",
                "scenario_dispersion_pct",
                "downside_loss_pct",
                "rank_stability",
                "weighted_regret_pct",
                "model_r2",
            ],
        ),
        width="stretch",
    )

with tab3:
    st.subheader("Probability-weighted optimized portfolio")
    st.caption("Uses scenario probabilities, return dispersion, and the recent covariance matrix. This is a tilt optimizer, not an execution order.")
    if optimized_portfolio.weights.empty:
        st.warning("The optimizer could not build a portfolio from the available data.")
    else:
        opt_stats = optimized_portfolio.stats.iloc[0]
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("Expected return", f"{opt_stats['expected_return_pct']:.2f}%")
        o2.metric("Vol estimate", f"{opt_stats['volatility_estimate_pct']:.2f}%")
        o3.metric("Return / risk", f"{opt_stats['return_to_risk']:.2f}")
        o4.metric("Gross exposure", f"{opt_stats['gross_exposure_pct']:.0f}%")
        o5.metric("Net exposure", f"{opt_stats['net_exposure_pct']:.0f}%")

        weights_view = optimized_portfolio.weights.copy()
        fig = px.bar(
            weights_view.sort_values("weight_pct"),
            x="weight_pct",
            y="name",
            color="side",
            orientation="h",
            hover_data=["bucket", "expected_return_pct", "expected_contribution_pct", "risk_contribution_pct", "model_r2"],
            labels={"weight_pct": "Portfolio weight %", "name": ""},
            color_discrete_map={
                "Long / overweight": COLORS["positive"],
                "Short / underweight": COLORS["negative"],
            },
        )
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color=COLORS["unknown"])
        polish_figure(fig, height=520)
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns([2, 1])
        with left:
            st.markdown("**Optimized asset weights**")
            st.dataframe(
                format_pct_columns(
                    weights_view[
                        [
                            "side",
                            "name",
                            "bucket",
                            "weight_pct",
                            "expected_return_pct",
                            "expected_contribution_pct",
                            "risk_contribution_pct",
                            "model_r2",
                            "residual_vol_pct",
                        ]
                    ],
                    [
                        "weight_pct",
                        "expected_return_pct",
                        "expected_contribution_pct",
                        "risk_contribution_pct",
                        "model_r2",
                        "residual_vol_pct",
                    ],
                ),
                width="stretch",
            )
        with right:
            st.markdown("**Bucket exposures**")
            st.dataframe(
                format_pct_columns(
                    optimized_portfolio.bucket_weights,
                    ["net_weight_pct", "gross_weight_pct", "expected_contribution_pct", "risk_contribution_pct"],
                ),
                width="stretch",
            )
            st.markdown("**Optimization controls**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"constraint": "Max asset weight", "value": "12%"},
                        {"constraint": "Max bucket gross", "value": "35%"},
                        {"constraint": "Max net exposure", "value": "25%"},
                        {"constraint": "Gross target", "value": "100%"},
                        {"constraint": "Covariance lookback", "value": "60 months"},
                        {"constraint": "Optimizer", "value": str(opt_stats.get("optimizer_status", "constrained SLSQP"))},
                    ]
                ),
                width="stretch",
            )
            if not optimized_portfolio.constraint_audit.empty:
                st.markdown("**Constraint audit**")
                st.dataframe(
                    format_pct_columns(optimized_portfolio.constraint_audit, ["actual_pct", "limit_pct"]),
                    width="stretch",
                )

    st.subheader("Walk-forward predicted-scenario portfolio vs SPY")
    st.caption(
        "Each month uses data available at that month-end, selects the modal non-unknown scenario, builds the diversified long/short scenario basket, "
        "then applies it to the following month. Performance uses actual asset returns, with SPY shown on the chart and a broader benchmark tear sheet below."
    )
    bt = predicted_portfolio_backtest
    if bt.returns.empty:
        st.warning("Not enough history to run the predicted-scenario portfolio backtest.")
    else:
        summary = bt.summary.copy()
        summary_idx = summary.set_index("series")
        strategy_row = summary_idx.loc["Predicted scenario portfolio"]
        spy_row = summary_idx.loc["SPY"]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Strategy equity", f"{strategy_row['final_equity']:.2f}x")
        m2.metric("SPY equity", f"{spy_row['final_equity']:.2f}x")
        m3.metric("Strategy Sharpe", f"{strategy_row['sharpe']:.2f}")
        m4.metric("SPY Sharpe", f"{spy_row['sharpe']:.2f}")
        m5.metric("Strategy max DD", f"{strategy_row['max_drawdown_pct']:.1f}%")
        m6.metric("Months tested", f"{int(strategy_row['months'])}")

        first_as_of = pd.to_datetime(bt.returns["as_of"]).min()
        equity = add_starting_value(bt.equity, first_as_of, value=1.0)
        fig = go.Figure()
        line_styles = {
            "Predicted scenario portfolio": {"color": "#2dd4bf", "width": 4},
            "SPY": {"color": "#60a5fa", "width": 4},
        }
        for series, style in line_styles.items():
            if series not in equity:
                continue
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity[series],
                    name=series,
                    mode="lines",
                    line=style,
                    connectgaps=True,
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}x<extra>" + series + "</extra>",
                )
            )
        plotted_equity = equity[[series for series in line_styles if series in equity]]
        max_equity = float(plotted_equity.max(numeric_only=True).max())
        fig.update_layout(xaxis_title="", yaxis_title="Growth of $1")
        fig.update_yaxes(range=[0, max(1.1, max_equity * 1.08)])
        polish_figure(fig, height=460)
        st.plotly_chart(fig, width="stretch")

        drawdowns = add_starting_value(bt.drawdowns, first_as_of, value=0.0)
        fig = go.Figure()
        for series, style in line_styles.items():
            if series not in drawdowns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns[series],
                    name=series,
                    mode="lines",
                    line=style,
                    connectgaps=True,
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra>" + series + "</extra>",
                )
            )
        fig.update_layout(xaxis_title="", yaxis_title="Drawdown %")
        fig.add_hline(y=0, line_width=1, line_color=COLORS["unknown"])
        polish_figure(fig, height=360)
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns(2)
        with left:
            st.markdown("**Performance summary**")
            st.dataframe(
                format_pct_columns(
                    summary,
                    [
                        "final_equity",
                        "total_return_pct",
                        "cagr_pct",
                        "annual_vol_pct",
                        "sharpe",
                        "max_drawdown_pct",
                        "hit_rate_pct",
                        "avg_monthly_return_pct",
                        "best_month_pct",
                        "worst_month_pct",
                        "correlation_to_spy",
                        "avg_turnover_pct",
                    ],
                ),
                width="stretch",
            )
        with right:
            st.markdown("**Scenario usage**")
            st.dataframe(
                format_pct_columns(
                    bt.scenario_counts,
                    [
                        "avg_modal_probability_pct",
                        "avg_strategy_return_pct",
                        "avg_spy_return_pct",
                        "hit_rate_pct",
                    ],
                ),
                width="stretch",
            )

        if not bt.benchmark_tear_sheet.empty:
            st.markdown("**Benchmark tear sheet**")
            st.dataframe(
                format_pct_columns(
                    bt.benchmark_tear_sheet,
                    [
                        "strategy_cagr_pct",
                        "benchmark_cagr_pct",
                        "active_cagr_spread_pct",
                        "strategy_sharpe",
                        "benchmark_sharpe",
                        "sharpe_spread",
                        "tracking_error_pct",
                        "information_ratio",
                        "beta",
                        "correlation",
                        "strategy_max_dd_pct",
                        "benchmark_max_dd_pct",
                    ],
                ),
                width="stretch",
            )

        if not bt.benchmark_diagnostics.empty:
            st.markdown("**Institutional risk audit**")
            diag = bt.benchmark_diagnostics.set_index("metric")["value"]
            r1, r2, r3, r4, r5, r6 = st.columns(6)
            r1.metric("Active return", f"{diag.get('Annualized active return', np.nan):.1f}%")
            r2.metric("Tracking error", f"{diag.get('Tracking error', np.nan):.1f}%")
            r3.metric("Information ratio", f"{diag.get('Information ratio', np.nan):.2f}")
            r4.metric("Beta to SPY", f"{diag.get('Beta to SPY', np.nan):.2f}")
            r5.metric("Upside capture", f"{diag.get('Upside capture', np.nan):.0f}%")
            r6.metric("Downside capture", f"{diag.get('Downside capture', np.nan):.0f}%")

            audit_left, audit_right = st.columns([1.25, 1.0])
            with audit_left:
                if not bt.rolling_metrics.empty:
                    rolling = bt.rolling_metrics.copy()
                    rolling["date"] = pd.to_datetime(rolling["date"])
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=rolling["date"],
                            y=rolling["strategy_sharpe"],
                            name="Strategy 36m Sharpe",
                            mode="lines",
                            line={"color": "#2dd4bf", "width": 3},
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=rolling["date"],
                            y=rolling["spy_sharpe"],
                            name="SPY 36m Sharpe",
                            mode="lines",
                            line={"color": "#60a5fa", "width": 3},
                        )
                    )
                    fig.add_hline(y=0, line_width=1, line_color=COLORS["unknown"])
                    fig.update_layout(xaxis_title="", yaxis_title="36-month rolling Sharpe")
                    polish_figure(fig, height=340)
                    st.plotly_chart(fig, width="stretch")
            with audit_right:
                st.markdown("**Benchmark diagnostics**")
                st.dataframe(format_pct_columns(bt.benchmark_diagnostics, ["value"]), width="stretch")

            cost_left, stress_right = st.columns([1.0, 1.25])
            with cost_left:
                st.markdown("**Transaction-cost sensitivity**")
                st.dataframe(
                    format_pct_columns(
                        bt.cost_sensitivity,
                        [
                            "final_equity",
                            "total_return_pct",
                            "cagr_pct",
                            "annual_vol_pct",
                            "sharpe",
                            "max_drawdown_pct",
                            "hit_rate_pct",
                        ],
                    ),
                    width="stretch",
                )
            with stress_right:
                st.markdown("**Worst SPY months stress test**")
                stress_cols = [
                    "return_date",
                    "modal_scenario",
                    "modal_probability_pct",
                    "unknown_probability_pct",
                    "strategy_return_pct",
                    "spy_return_pct",
                    "excess_return_pct",
                    "turnover_pct",
                    "n_assets",
                ]
                st.dataframe(
                    format_pct_columns(
                        bt.stress_months[stress_cols],
                        [
                            "modal_probability_pct",
                            "unknown_probability_pct",
                            "strategy_return_pct",
                            "spy_return_pct",
                            "excess_return_pct",
                            "turnover_pct",
                        ],
                    ),
                    width="stretch",
                )

        st.markdown("**Recent backtest months**")
        recent_bt = bt.returns.sort_values("return_date", ascending=False).head(24).copy()
        recent_bt["modal_probability_pct"] = recent_bt["modal_probability"] * 100.0
        recent_bt["unknown_probability_pct"] = recent_bt["unknown_probability"] * 100.0
        recent_bt["confidence_pct"] = recent_bt["confidence"] * 100.0
        recent_bt["strategy_return_pct"] = (np.exp(recent_bt["strategy_return"]) - 1.0) * 100.0
        recent_bt["spy_return_pct"] = (np.exp(recent_bt["spy_return"]) - 1.0) * 100.0
        recent_bt["excess_return_pct"] = (np.exp(recent_bt["excess_return"]) - 1.0) * 100.0
        recent_bt["turnover_pct"] = recent_bt["turnover"] * 100.0
        st.dataframe(
            format_pct_columns(
                recent_bt[
                    [
                        "as_of",
                        "return_date",
                        "modal_scenario",
                        "modal_probability_pct",
                        "unknown_probability_pct",
                        "strategy_return_pct",
                        "spy_return_pct",
                        "excess_return_pct",
                        "turnover_pct",
                        "n_longs",
                        "n_shorts",
                    ]
                ],
                [
                    "modal_probability_pct",
                    "unknown_probability_pct",
                    "strategy_return_pct",
                    "spy_return_pct",
                    "excess_return_pct",
                    "turnover_pct",
                ],
            ),
            width="stretch",
        )

with tab4:
    st.subheader("Universe quality and investability gate")
    st.caption("Scores are operational guardrails for dashboard use. They combine data freshness, proxy coverage, liquidity, and implementation-cost heuristics.")
    uq = universe_quality.copy()
    uq["optimizer_included"] = uq["ticker"].isin(active_universe["ticker"])
    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Eligible now", f"{int(uq['optimizer_included'].sum())}/{len(uq)}")
    g2.metric("Research-only", f"{int((~uq['allowed_long_only']).sum())}")
    g3.metric("Long/short eligible", f"{int(uq['allowed_long_short'].sum())}")
    g4.metric("Raw-source excluded", f"{int(uq['raw_source_excluded'].sum())}")
    g5.metric("Median investability", f"{uq['investability_score'].median():.0f}")
    st.dataframe(
        format_pct_columns(
            uq[
                [
                    "ticker",
                    "name",
                    "bucket",
                    "optimizer_included",
                    "status",
                    "raw_source_excluded",
                    "investability_score",
                    "proxy_quality_score",
                    "liquidity_score",
                    "cost_score",
                    "coverage_pct",
                    "annual_vol_pct",
                    "max_abs_month_pct",
                    "allowed_long_only",
                    "allowed_long_short",
                    "reason",
                ]
            ],
            [
                "investability_score",
                "proxy_quality_score",
                "liquidity_score",
                "cost_score",
                "coverage_pct",
                "annual_vol_pct",
                "max_abs_month_pct",
            ],
        ),
        width="stretch",
    )

    st.subheader("Top and bottom expected outperformers by bucket")
    grouped = top_bottom_by_bucket(result.expected, n=top_n)
    for bucket, tb in grouped.items():
        st.markdown(f"### {bucket}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Top**")
            st.dataframe(format_pct_columns(tb["top"], ["expected_return_pct", "confidence_t_proxy"]), width="stretch")
        with right:
            st.markdown("**Bottom**")
            st.dataframe(format_pct_columns(tb["bottom"], ["expected_return_pct", "confidence_t_proxy"]), width="stretch")

    st.subheader("Full ranking")
    full = result.expected.sort_values("conviction_score", ascending=False).copy()
    full_cols = [
        "name",
        "bucket",
        "policy_signal",
        "expected_return_pct",
        "conviction_score",
        "model_r2",
        "residual_vol_pct",
        "trailing_12m_return_pct",
        "confidence_t_proxy",
    ]
    bucket_pick = st.multiselect(
        "Filter buckets",
        sorted(full["bucket"].dropna().unique()),
        default=sorted(full["bucket"].dropna().unique()),
        key="manual_bucket_filter",
    )
    full = full[full["bucket"].isin(bucket_pick)]
    st.dataframe(
        format_pct_columns(
            full[full_cols],
            ["expected_return_pct", "conviction_score", "model_r2", "residual_vol_pct", "trailing_12m_return_pct", "confidence_t_proxy"],
        ),
        width="stretch",
    )

with tab5:
    st.subheader("Factor attribution")
    labels = result.expected.sort_values("abs_conviction_score", ascending=False).copy()
    fallback_names = pd.Series(labels.index, index=labels.index)
    labels["label"] = labels.index.to_series() + " - " + labels["name"].fillna(fallback_names)
    selected_label = st.selectbox("Asset", labels["label"].tolist(), index=0)
    ticker = selected_label.split(" - ", 1)[0]
    contribution = result.contributions.loc[ticker].rename(lambda c: c.replace("_contribution", "")) * 100.0
    attribution = pd.DataFrame(
        {
            "factor": FACTOR_COLUMNS,
            "scenario_shock_z": scenario.reindex(FACTOR_COLUMNS).values,
            "beta": result.exposures.loc[ticker, FACTOR_COLUMNS].values,
            "contribution_pct": contribution.reindex(FACTOR_COLUMNS).values,
            "t_stat": result.tstats.loc[ticker, FACTOR_COLUMNS].values,
        }
    )
    a1, a2, a3, a4 = st.columns(4)
    asset_row = result.expected.loc[ticker]
    a1.metric("Expected relative return", f"{asset_row['expected_return_pct']:.2f}%")
    a2.metric("Conviction score", f"{asset_row['conviction_score']:.2f}")
    a3.metric("Model R²", f"{asset_row['model_r2']:.2f}")
    a4.metric("Residual vol", f"{asset_row['residual_vol_pct']:.2f}%")
    colors = np.where(attribution["contribution_pct"] >= 0, COLORS["positive"], COLORS["negative"])
    fig = go.Figure(
        go.Bar(
            x=attribution["contribution_pct"],
            y=attribution["factor"],
            orientation="h",
            marker_color=colors,
            text=attribution["contribution_pct"].map(lambda x: f"{x:.2f}%"),
            textposition="auto",
        )
    )
    fig.update_layout(xaxis_title="Contribution to expected relative return", yaxis_title="", height=380)
    polish_figure(fig, height=380)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(format_pct_columns(attribution, ["scenario_shock_z", "beta", "contribution_pct", "t_stat"]), width="stretch")

with tab6:
    st.subheader("Scenario comparison")
    st.caption("Compares the selected manual sidebar scenario against another preset.")
    other = st.selectbox("Compare against", scenario_names, index=scenario_names.index("Custom") if "Custom" in scenario_names else 0)
    other_scenario = scenarios.set_index("scenario").loc[other, FACTOR_COLUMNS].astype(float)
    comp = compare_scenarios(result.exposures, universe, scenario, other_scenario)
    st.write("Positive difference means the current sidebar scenario favors the asset more than the comparison scenario.")
    comp_view = comp[["name", "bucket", "scenario_a", "scenario_b", "difference_pct"]].copy()
    comp_view["scenario_a_pct"] = comp_view["scenario_a"] * 100.0
    comp_view["scenario_b_pct"] = comp_view["scenario_b"] * 100.0
    st.dataframe(
        format_pct_columns(
            comp_view[["name", "bucket", "scenario_a_pct", "scenario_b_pct", "difference_pct"]],
            ["scenario_a_pct", "scenario_b_pct", "difference_pct"],
        ),
        width="stretch",
    )

    st.subheader("Preset stress map")
    scenario_cols = [c for c in scenario_expected.columns if c not in {"name", "bucket"}]
    stress = scenario_expected.copy()
    stress["best_scenario"] = stress[scenario_cols].idxmax(axis=1)
    stress["worst_scenario"] = stress[scenario_cols].idxmin(axis=1)
    stress["scenario_range_pct"] = stress[scenario_cols].max(axis=1) - stress[scenario_cols].min(axis=1)
    stress["average_pct"] = stress[scenario_cols].mean(axis=1)
    stress["consistency_score"] = stress["average_pct"] / stress[scenario_cols].std(axis=1).replace(0, np.nan)
    stress_view = stress.sort_values("scenario_range_pct", ascending=False)
    st.dataframe(
        format_pct_columns(
            stress_view[["name", "bucket", "best_scenario", "worst_scenario", "scenario_range_pct", "average_pct", "consistency_score"]],
            ["scenario_range_pct", "average_pct", "consistency_score"],
        ),
        width="stretch",
    )

    heatmap_bucket = st.selectbox("Stress-map bucket", ["All"] + sorted(stress["bucket"].dropna().unique()))
    heatmap_source = stress if heatmap_bucket == "All" else stress[stress["bucket"] == heatmap_bucket]
    heatmap_names = heatmap_source.sort_values("scenario_range_pct", ascending=False).head(20)
    fig = px.imshow(
        heatmap_names[scenario_cols],
        aspect="auto",
        color_continuous_midpoint=0,
        labels={"x": "Scenario", "y": "Ticker", "color": "Expected %"},
        color_continuous_scale=[COLORS["negative"], "#111827", COLORS["positive"]],
    )
    polish_figure(fig, height=520)
    st.plotly_chart(fig, width="stretch")

with tab7:
    st.subheader("Model diagnostics")
    diag_cols = [
        "name",
        "bucket",
        "expected_return_pct",
        "conviction_score",
        "model_r2",
        "residual_vol_pct",
        "realized_vol_annual_pct",
        "trailing_3m_return_pct",
        "trailing_12m_return_pct",
        "nobs",
    ]
    diag = result.expected.sort_values("model_r2", ascending=False)
    fig = px.scatter(
        diag,
        x="residual_vol_pct",
        y="expected_return_pct",
        color="bucket",
        size="model_r2",
        hover_name="name",
        hover_data=["conviction_score", "trailing_12m_return_pct", "policy_signal"],
        labels={"residual_vol_pct": "Residual vol, monthly %", "expected_return_pct": "Expected relative return %"},
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=COLORS["unknown"])
    polish_figure(fig, height=520)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        format_pct_columns(
            diag[diag_cols],
            [
                "expected_return_pct",
                "conviction_score",
                "model_r2",
                "residual_vol_pct",
                "realized_vol_annual_pct",
                "trailing_3m_return_pct",
                "trailing_12m_return_pct",
            ],
        ),
        width="stretch",
    )

    st.subheader("Walk-forward scenario calibration")
    st.caption("Checks whether the probability engine assigned meaningful probability to the next-month nearest realized macro regime.")
    if calibration.summary.empty:
        st.warning("Not enough history to run walk-forward scenario calibration.")
    else:
        cal_row = calibration.summary.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Calibration periods", f"{int(cal_row['periods'])}")
        k2.metric("Top hit rate", f"{cal_row['top_hit_rate'] * 100:.1f}%")
        k3.metric("Avg assigned prob", f"{cal_row['avg_realized_probability'] * 100:.1f}%")
        k4.metric("Avg Brier score", f"{cal_row['avg_brier_score']:.2f}")
        left, right = st.columns(2)
        with left:
            st.markdown("**Probability buckets**")
            bucket_view = calibration.probability_buckets.copy()
            bucket_view["avg_assigned_probability_pct"] = bucket_view["avg_assigned_probability"] * 100.0
            bucket_view["hit_rate_pct"] = bucket_view["hit_rate"] * 100.0
            st.dataframe(
                format_pct_columns(
                    bucket_view[
                        [
                            "probability_bucket",
                            "observations",
                            "avg_assigned_probability_pct",
                            "hit_rate_pct",
                            "avg_brier_score",
                        ]
                    ],
                    ["avg_assigned_probability_pct", "hit_rate_pct", "avg_brier_score"],
                ),
                width="stretch",
            )
        with right:
            st.markdown("**Recent predictions**")
            recent = calibration.recent_predictions.copy()
            recent["realized_probability_pct"] = recent["realized_probability"] * 100.0
            recent["confidence_pct"] = recent["confidence"] * 100.0
            recent["unknown_probability_pct"] = recent["unknown_probability"] * 100.0
            st.dataframe(
                format_pct_columns(
                    recent[
                        [
                            "as_of",
                            "modal_prediction",
                            "realized_scenario",
                            "realized_probability_pct",
                            "top_hit",
                            "confidence_pct",
                            "unknown_probability_pct",
                        ]
                    ],
                    ["realized_probability_pct", "top_hit", "confidence_pct", "unknown_probability_pct"],
                ),
                width="stretch",
            )

    st.subheader("Reproducibility and mechanical date-boundary audit")
    st.caption(
        "Checks the mechanical backtest boundary: every portfolio decision must use only rows dated on or before the rebalance month-end. "
        "This is not a true macro release-vintage audit because the macro layer still uses latest-revised public proxy data."
    )
    audit = point_in_time_audit.copy()
    if audit.empty:
        st.warning("No predicted-portfolio audit rows are available.")
    else:
        lookahead_count = int(audit["lookahead_flag"].sum())
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Audit rows", f"{len(audit):,}")
        a2.metric("Lookahead flags", f"{lookahead_count}")
        a3.metric("Date-boundary", "Pass" if lookahead_count == 0 else "Fail")
        a4.metric("Run ID", dashboard_run_id())
        a5.metric("Cache age", f"{cache_age_hours:.1f}h" if np.isfinite(cache_age_hours) else "n/a")
        st.markdown("**Acceptance rule:** no backtest result should render as production-grade unless `lookahead_flag_count == 0`; vintage-data limitations remain separate and disclosed.")
        st.dataframe(
            audit.sort_values("return_date", ascending=False).head(36),
            width="stretch",
        )

    st.subheader("Correlation heatmap: relative returns vs macro factors")
    corr = result.corr.join(universe.set_index("ticker")[["name", "bucket"]], how="left")
    bucket_filter = st.multiselect("Buckets", sorted(corr["bucket"].dropna().unique()), default=sorted(corr["bucket"].dropna().unique()))
    corr_filtered = corr[corr["bucket"].isin(bucket_filter)][FACTOR_COLUMNS]
    fig = px.imshow(
        corr_filtered,
        aspect="auto",
        color_continuous_midpoint=0,
        labels={"x": "Macro factor", "y": "Ticker", "color": "Corr"},
        color_continuous_scale=[COLORS["negative"], "#111827", COLORS["positive"]],
    )
    polish_figure(fig, height=540)
    st.plotly_chart(fig, width="stretch")

with tab8:
    st.subheader("Inputs")
    st.markdown("**Data freshness and source status**")
    st.caption("This dashboard uses completed monthly public market and macro proxy data. It is not an intraday live-data terminal.")
    freshness = data_freshness_table(refresh_info, prices, factors)
    state = load_update_state()
    expected_month = pd.to_datetime(state.get("last_complete_month") or refresh_info.get("last_complete_month"), errors="coerce")
    asset_names = universe.set_index("ticker")["name"].to_dict()
    price_quality = data_quality_table(prices, expected_month, asset_names)
    factor_quality = data_quality_table(factors, expected_month)
    refresh_date = pd.to_datetime(state.get("updated_at_utc", refresh_info.get("updated_at_utc")), errors="coerce")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Data month", expected_month.strftime("%b %Y") if pd.notna(expected_month) else "n/a")
    q2.metric("Current assets", f"{int((price_quality['status'] == 'Current').sum())}/{len(price_quality)}")
    q3.metric("Stale assets", f"{int((price_quality['status'] == 'Stale').sum())}")
    q4.metric("Current factors", f"{int((factor_quality['status'] == 'Current').sum())}/{len(factor_quality)}")
    q5.metric("Refreshed", refresh_date.strftime("%b %d") if pd.notna(refresh_date) else "n/a")
    st.markdown("**Data provenance**")
    source_name = state.get("source", refresh_info.get("source", "unknown"))
    yahoo_count = len(state.get("price_symbols", refresh_info.get("price_symbols", [])) or [])
    fred_series = state.get("fred_series", refresh_info.get("fred_series", {})) or {}
    provenance = pd.DataFrame(
        [
            {
                "area": "Asset universe",
                "source": "Yahoo Finance via yfinance",
                "frequency": "Completed month-end closes",
                "live_status": "Public market data, refreshed by scheduled/manual GitHub workflow",
                "limitation": "ETF/proxy universe, not full security master or execution venue feed",
            },
            {
                "area": "Macro factors",
                "source": f"{source_name}; Yahoo factor proxies plus FRED {', '.join(fred_series.values()) if fred_series else 'series'}",
                "frequency": "Monthly proxy levels",
                "live_status": "Latest-revised public data",
                "limitation": "Not ALFRED point-in-time vintages; revisions are disclosed in diagnostics",
            },
            {
                "area": "Scenario definitions",
                "source": "Local config/scenarios.csv",
                "frequency": "Static until edited",
                "live_status": "Model inputs, not downloaded observations",
                "limitation": "Hand-defined macro states require validation and calibration",
            },
            {
                "area": "Backtest decisions",
                "source": "Walk-forward model using rows dated on or before rebalance month-end",
                "frequency": "Monthly",
                "live_status": "Mechanical date-boundary audited",
                "limitation": "Still uses latest-revised macro proxies and simplified transaction costs",
            },
        ]
    )
    st.caption(f"Tracked Yahoo/FRED symbols: {yahoo_count}. Amber status means the latest rows are present, but proxy and vintage limitations remain.")
    st.dataframe(provenance, width="stretch")
    st.markdown("**Stale and flatline source audit**")
    source_audit = source_audit_exact.copy()
    using_source_audit_fallback = source_audit.empty
    if source_audit.empty:
        source_audit = pd.concat(
            [
                stale_flatline_fallback(prices, asset_names, "asset price"),
                stale_flatline_fallback(factors, label="macro factor"),
            ],
            ignore_index=True,
        )
    if using_source_audit_fallback:
        st.warning("Exact raw-source audit is not available in this app session; showing a conservative fallback from stored monthly rows.")
    else:
        excluded_count = int(pd.Series(source_audit.get("exclude_flag", False)).fillna(False).astype(bool).sum())
        source_audit_kind = str(state.get("source_audit_kind", "raw_source_audit"))
        if source_audit_kind == "committed_monthly_snapshot_fallback":
            st.caption(
                f"Committed source-audit snapshot loaded for {len(source_audit)} rows. "
                f"Exclude flags: {excluded_count}. The snapshot is fail-closed for flatlines/stale monthly rows; run update_data.py with Yahoo/FRED access for raw source-date audit rows."
            )
        else:
            st.caption(
                f"Exact raw-source audit loaded for {len(source_audit)} rows. "
                f"Exclude flags: {excluded_count}. Yahoo daily sources use a 10-business-day lag rule; slower FRED liquidity data uses a wider documented lag rule."
            )
    st.dataframe(source_audit, width="stretch")
    st.dataframe(freshness, width="stretch")

    st.markdown("**Asset price quality**")
    st.dataframe(
        format_pct_columns(
            price_quality[
                [
                    "series",
                    "name",
                    "status",
                    "first_valid",
                    "last_valid",
                    "stale_months",
                    "observations",
                    "coverage_pct",
                    "missing_values",
                    "latest_value",
                ]
            ],
            ["coverage_pct", "latest_value"],
        ),
        width="stretch",
    )
    st.markdown("**Macro factor quality**")
    st.dataframe(
        format_pct_columns(
            factor_quality[
                [
                    "series",
                    "status",
                    "first_valid",
                    "last_valid",
                    "stale_months",
                    "observations",
                    "coverage_pct",
                    "missing_values",
                    "latest_value",
                ]
            ],
            ["coverage_pct", "latest_value"],
        ),
        width="stretch",
    )
    st.markdown("**Macro scenario vector**")
    st.dataframe(format_pct_columns(pd.DataFrame(scenario, columns=["value"]), ["value"]))
    st.markdown("**Latest standardized macro impulse**")
    latest_macro = result.factors_z.tail(1).T.rename(columns={result.factors_z.index[-1]: "latest_z"})
    st.dataframe(format_pct_columns(latest_macro, ["latest_z"]))
    st.markdown("**Universe**")
    st.dataframe(universe, width="stretch")
    st.markdown("**Latest factor rows**")
    st.dataframe(factors.tail(), width="stretch")
