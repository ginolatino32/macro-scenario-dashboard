from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import FACTOR_COLUMNS, estimate_scenario_probabilities, factor_changes

MODEL_VERSION = "bl_macro_views_v1"
FEATURE_VERSION = "pit_macro_features_v1"

DEFAULT_SETTINGS: dict[str, object] = {
    "horizon_months": 6,
    "secondary_horizon_months": 12,
    "policy_benchmark_id": "GSMIF_SAMPLE_POLICY",
    "return_basis": "active_return_vs_benchmark",
    "relative_return_basis": "relative_spread_return",
    "covariance_frequency": "monthly",
    "covariance_lookback_months": 60,
    "covariance_method": "shrunk_sample",
    "covariance_shrinkage": 0.20,
    "tau": 0.05,
    "risk_aversion_delta": 2.50,
    "confidence_floor": 0.25,
    "confidence_cap": 0.75,
    "min_candidate_confidence": 0.45,
    "omega_floor": 0.0001,
    "omega_cap": 0.2500,
    "q_absolute_cap": 0.1500,
    "q_relative_cap": 0.1200,
    "min_active_q_threshold": 0.0200,
    "min_train_observations": 60,
    "walkforward_embargo_months": 6,
    "unknown_mixed_reduce_threshold": 0.40,
    "unknown_mixed_block_threshold": 0.60,
    "macro_feature_lag_months": 1,
    "max_model_features": 5,
    "validation_stride_months": 3,
    "max_validation_points": 72,
    "max_views_per_run": 10,
    "view_expiry_months": 1,
}

BOOL_COLUMNS = {
    "is_investable",
    "is_bl_eligible",
    "is_macro_indicator_only",
    "view_eligible_absolute",
    "view_eligible_relative",
    "allow_short",
    "is_placeholder",
    "cross_sleeve_allowed",
    "is_approved",
}


@dataclass
class BLResult:
    as_of: pd.Timestamp
    horizon_months: int
    asset_order: list[str]
    settings: dict[str, object]
    config_audit: pd.DataFrame
    predictions: pd.DataFrame
    views: pd.DataFrame
    P: pd.DataFrame
    q: pd.DataFrame
    Omega: pd.DataFrame
    Sigma: pd.DataFrame
    pi: pd.DataFrame
    bl_runs: pd.DataFrame
    posterior_returns: pd.DataFrame
    view_diagnostics: pd.DataFrame
    scenario_context: dict[str, object]


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _coerce_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in BOOL_COLUMNS.intersection(out.columns):
        out[col] = out[col].map(_parse_bool)
    return out


def _coerce_setting(value: object) -> object:
    if pd.isna(value):
        return value
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and "." not in text:
        return int(number)
    return number


def load_bl_settings(config_dir: str | Path) -> dict[str, object]:
    path = Path(config_dir) / "bl_settings.csv"
    settings = DEFAULT_SETTINGS.copy()
    if path.exists():
        raw = pd.read_csv(path)
        for _, row in raw.iterrows():
            settings[str(row["setting"])] = _coerce_setting(row["value"])
    return settings


def load_asset_master(config_dir: str | Path) -> pd.DataFrame:
    path = Path(config_dir) / "asset_master.csv"
    out = pd.read_csv(path)
    out["ticker"] = out["ticker"].astype(str)
    return _coerce_bool_columns(out)


def load_benchmark_weights(config_dir: str | Path) -> pd.DataFrame:
    path = Path(config_dir) / "benchmark_weights.csv"
    out = pd.read_csv(path, parse_dates=["as_of"])
    out["ticker"] = out["ticker"].astype(str)
    out["benchmark_id"] = out["benchmark_id"].astype(str)
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")
    return _coerce_bool_columns(out)


def load_view_pairs(config_dir: str | Path) -> pd.DataFrame:
    path = Path(config_dir) / "view_pairs.csv"
    out = pd.read_csv(path)
    for col in ["long_ticker", "short_ticker", "view_pair_id"]:
        out[col] = out[col].astype(str)
    for col in ["min_q_threshold", "min_confidence", "max_q_abs"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return _coerce_bool_columns(out)


def get_bl_asset_order(asset_master: pd.DataFrame, prices: pd.DataFrame) -> list[str]:
    eligible = asset_master[
        asset_master["is_bl_eligible"].map(_parse_bool)
        & ~asset_master["is_macro_indicator_only"].map(_parse_bool)
        & asset_master["ticker"].isin(prices.columns)
    ].copy()
    return eligible["ticker"].astype(str).tolist()


def _latest_benchmark_weights(
    benchmark_weights: pd.DataFrame,
    benchmark_id: str,
    as_of: pd.Timestamp,
) -> pd.Series:
    if not benchmark_id or pd.isna(benchmark_id):
        return pd.Series(dtype=float)
    rows = benchmark_weights[benchmark_weights["benchmark_id"].eq(str(benchmark_id))].copy()
    if rows.empty:
        return pd.Series(dtype=float)
    rows = rows[rows["as_of"] <= pd.Timestamp(as_of)]
    if rows.empty:
        return pd.Series(dtype=float)
    latest = rows["as_of"].max()
    rows = rows[rows["as_of"].eq(latest)]
    weights = rows.set_index("ticker")["weight"].astype(float)
    total = float(weights.sum())
    if total > 0:
        weights = weights / total
    return weights


def validate_bl_config(
    asset_master: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    view_pairs: pd.DataFrame,
    prices: pd.DataFrame,
    settings: dict[str, object],
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of) if as_of is not None else prices.index.max()
    rows: list[dict[str, str]] = []

    def add(check: str, status: str, detail: str) -> None:
        rows.append({"check": check, "status": status, "detail": detail})

    asset_order = get_bl_asset_order(asset_master, prices)
    add("asset_order", "Pass" if asset_order else "Fail", f"{len(asset_order)} BL-eligible assets in deterministic order.")

    missing_prices = sorted(set(asset_master["ticker"]) - set(prices.columns))
    add(
        "asset_master_prices",
        "Pass" if not missing_prices else "Fail",
        "Every asset has a price column." if not missing_prices else f"Missing price columns: {missing_prices}",
    )

    for benchmark_id, rows_for_id in benchmark_weights.groupby("benchmark_id"):
        latest_rows = rows_for_id[rows_for_id["as_of"] <= as_of]
        if latest_rows.empty:
            add(f"benchmark_{benchmark_id}", "Fail", "No benchmark weights available as of run date.")
            continue
        latest = latest_rows["as_of"].max()
        weights = latest_rows[latest_rows["as_of"].eq(latest)]
        total = float(weights["weight"].sum())
        missing_components = sorted(set(weights["ticker"]) - set(prices.columns))
        status = "Pass" if abs(total - 1.0) < 1e-6 and not missing_components else "Fail"
        detail = f"sum={total:.6f}, as_of={latest.date()}"
        if missing_components:
            detail += f", missing components={missing_components}"
        add(f"benchmark_{benchmark_id}", status, detail)
        if "is_placeholder" in weights and weights["is_placeholder"].map(_parse_bool).any():
            add(
                f"benchmark_{benchmark_id}_source",
                "Review",
                "Benchmark uses placeholder/sample weights; replace with GSMIF policy or current holdings before investment use.",
            )

    pair_assets = set(view_pairs["long_ticker"]).union(set(view_pairs["short_ticker"]))
    missing_pair_assets = sorted(pair_assets - set(asset_master["ticker"]))
    add(
        "view_pairs_assets",
        "Pass" if not missing_pair_assets else "Fail",
        "All approved-pair assets exist in asset master." if not missing_pair_assets else f"Missing pair assets: {missing_pair_assets}",
    )

    horizon = int(settings.get("horizon_months", 0) or 0)
    return_basis = str(settings.get("return_basis", ""))
    unit_ok = horizon > 0 and return_basis in {
        "simple_total_return",
        "log_total_return",
        "simple_excess_return",
        "log_excess_return",
        "active_return_vs_benchmark",
        "relative_spread_return",
    }
    add("unit_contract", "Pass" if unit_ok else "Fail", f"horizon_months={horizon}, return_basis={return_basis}")
    add("tau", "Pass" if float(settings.get("tau", 0.0) or 0.0) > 0 else "Fail", f"tau={settings.get('tau')}")
    return pd.DataFrame(rows)


def _as_of_date(prices: pd.DataFrame, factors: pd.DataFrame, as_of: str | pd.Timestamp | None) -> pd.Timestamp:
    if as_of is not None:
        requested = pd.Timestamp(as_of)
        candidates = prices.index[prices.index <= requested].intersection(factors.index[factors.index <= requested])
        if candidates.empty:
            raise ValueError(f"No aligned price/factor rows are available on or before {requested.date()}")
        return pd.Timestamp(candidates.max())
    candidates = prices.dropna(how="all").index.intersection(factors.dropna(how="all").index)
    if candidates.empty:
        raise ValueError("No aligned price/factor rows are available")
    return pd.Timestamp(candidates.max())


def build_forward_return_labels(
    prices: pd.DataFrame,
    horizon_months: int,
    return_basis: str = "simple_total_return",
) -> pd.DataFrame:
    future_ratio = prices.shift(-int(horizon_months)) / prices
    if return_basis.startswith("log_"):
        labels = np.log(future_ratio)
    else:
        labels = future_ratio - 1.0
    return labels.replace([np.inf, -np.inf], np.nan)


def _benchmark_forward_return(
    prices: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    benchmark_id: str,
    as_of: pd.Timestamp,
    horizon_months: int,
    return_basis: str,
) -> pd.Series:
    weights = _latest_benchmark_weights(benchmark_weights, benchmark_id, as_of)
    if weights.empty:
        return pd.Series(np.nan, index=prices.index, name=benchmark_id)
    components = [ticker for ticker in weights.index if ticker in prices.columns]
    if not components:
        return pd.Series(np.nan, index=prices.index, name=benchmark_id)
    returns = build_forward_return_labels(prices[components], horizon_months, return_basis="simple_total_return")
    weighted = returns.mul(weights.reindex(components), axis=1).sum(axis=1, min_count=1)
    if return_basis.startswith("log_"):
        weighted = np.log1p(weighted)
    return weighted.rename(str(benchmark_id))


def build_active_return_labels(
    prices: pd.DataFrame,
    asset_master: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    as_of: str | pd.Timestamp,
    horizon_months: int,
    return_basis: str = "active_return_vs_benchmark",
) -> pd.DataFrame:
    base_return_basis = "log_total_return" if return_basis.startswith("log_") else "simple_total_return"
    total_returns = build_forward_return_labels(prices, horizon_months, base_return_basis)
    out: dict[str, pd.Series] = {}
    as_of_ts = pd.Timestamp(as_of)
    for _, row in asset_master.iterrows():
        ticker = str(row["ticker"])
        if ticker not in total_returns.columns:
            continue
        benchmark_id = str(row.get("benchmark_id", "") or "")
        if return_basis == "active_return_vs_benchmark" and benchmark_id:
            benchmark_return = _benchmark_forward_return(prices, benchmark_weights, benchmark_id, as_of_ts, horizon_months, base_return_basis)
            out[ticker] = total_returns[ticker] - benchmark_return
        else:
            out[ticker] = total_returns[ticker]
    return pd.DataFrame(out, index=prices.index)


def point_in_time_macro_features(
    factors: pd.DataFrame,
    macro_feature_lag_months: int = 1,
    min_periods: int = 36,
) -> pd.DataFrame:
    changes = factor_changes(factors).replace([np.inf, -np.inf], np.nan)
    means = changes.expanding(min_periods=min_periods).mean()
    stds = changes.expanding(min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    shocks = ((changes - means) / stds).clip(-5.0, 5.0)
    blocks = {
        "shock": shocks,
        "trend_3m": shocks.rolling(3, min_periods=2).mean(),
        "trend_6m": shocks.rolling(6, min_periods=3).mean(),
    }
    blocks["accel"] = blocks["trend_3m"] - blocks["trend_6m"]
    parts = []
    for block_name, block in blocks.items():
        renamed = block.reindex(columns=FACTOR_COLUMNS).copy()
        renamed.columns = [f"{block_name}_{factor}" for factor in FACTOR_COLUMNS]
        parts.append(renamed)
    features = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    if int(macro_feature_lag_months) > 0:
        features = features.shift(int(macro_feature_lag_months))
    return features


def _exp_weights(n: int, half_life: float = 60.0) -> np.ndarray:
    if n <= 0:
        return np.array([])
    age = np.arange(n - 1, -1, -1)
    return 0.5 ** (age / max(float(half_life), 1.0))


def _select_features(y: pd.Series, x: pd.DataFrame, max_features: int) -> list[str]:
    data = pd.concat([y.rename("target"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return []
    scores: list[tuple[str, float]] = []
    for col in x.columns:
        sample = data[["target", col]].dropna()
        if len(sample) < 24 or float(sample[col].std(ddof=0)) <= 0:
            continue
        corr = sample["target"].corr(sample[col])
        if np.isfinite(corr):
            scores.append((col, abs(float(corr))))
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    return [col for col, _ in scores[: int(max_features)]]


def _fit_linear_prediction(
    y: pd.Series,
    x: pd.DataFrame,
    x_latest: pd.Series,
    selected_features: list[str],
    min_obs: int,
    half_life: float = 60.0,
) -> tuple[float, float, dict[str, float], int, float]:
    if not selected_features:
        valid_y = y.dropna()
        return float(valid_y.mean()) if not valid_y.empty else np.nan, np.nan, {}, int(len(valid_y)), np.nan
    data = pd.concat([y.rename("target"), x[selected_features]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < max(int(min_obs), len(selected_features) + 12):
        valid_y = y.dropna()
        return float(valid_y.mean()) if not valid_y.empty else np.nan, np.nan, {}, int(len(data)), np.nan
    yv = data["target"].to_numpy(dtype=float)
    xv = data[selected_features].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(data)), xv])
    weights = _exp_weights(len(data), half_life=half_life)
    sw = np.sqrt(weights)
    coef = np.linalg.lstsq(X * sw[:, None], yv * sw, rcond=None)[0]
    latest = x_latest.reindex(selected_features).astype(float).fillna(0.0).to_numpy()
    prediction = float(coef[0] + latest @ coef[1:])
    fitted = X @ coef
    resid = yv - fitted
    resid_std = float(np.sqrt(np.average(resid**2, weights=weights)))
    ybar = float(np.average(yv, weights=weights))
    sst = float(np.sum(weights * (yv - ybar) ** 2))
    sse = float(np.sum(weights * resid**2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    contrib = {feature: float(value) for feature, value in zip(selected_features, latest * coef[1:])}
    return prediction, resid_std, contrib, int(len(data)), float(r2)


def _walk_forward_validation(
    y: pd.Series,
    x: pd.DataFrame,
    min_obs: int,
    max_features: int,
    stride_months: int = 3,
    max_points: int = 72,
) -> pd.DataFrame:
    data = pd.concat([y.rename("realized"), x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) <= int(min_obs) + 12:
        return pd.DataFrame(columns=["date", "prediction", "realized"])
    rows = []
    indices = list(range(int(min_obs), len(data), max(int(stride_months), 1)))
    if max_points > 0:
        indices = indices[-int(max_points) :]
    for i in indices:
        train = data.iloc[:i]
        test = data.iloc[i]
        selected = _select_features(train["realized"], train.drop(columns=["realized"]), max_features=max_features)
        pred, _, _, nobs, _ = _fit_linear_prediction(
            train["realized"],
            train.drop(columns=["realized"]),
            test.drop(labels=["realized"]),
            selected,
            min_obs=max(24, min(int(min_obs), len(train))),
        )
        if np.isfinite(pred):
            rows.append(
                {
                    "date": data.index[i],
                    "prediction": pred,
                    "realized": float(test["realized"]),
                    "n_train": nobs,
                }
            )
    return pd.DataFrame(rows)


def _json_dumps(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _data_freshness_flag(ticker: str, source_audit: pd.DataFrame | None) -> str:
    if source_audit is None or source_audit.empty or "dashboard_series" not in source_audit.columns:
        return "Unknown"
    rows = source_audit[source_audit["dashboard_series"].astype(str).eq(str(ticker))]
    if rows.empty:
        return "Unknown"
    row = rows.iloc[0]
    if _parse_bool(row.get("exclude_flag", False)):
        return "Blocked"
    status = str(row.get("source_audit_status", "") or "")
    return "Current" if status.lower() == "pass" else "Review"


def _benchmark_label(benchmark_weights: pd.DataFrame, benchmark_id: str, as_of: pd.Timestamp) -> str:
    weights = _latest_benchmark_weights(benchmark_weights, benchmark_id, as_of)
    if weights.empty:
        return ""
    if len(weights) == 1:
        return str(weights.index[0])
    return str(benchmark_id)


def build_model_predictions(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    asset_master: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    settings: dict[str, object],
    as_of: str | pd.Timestamp,
    source_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of)
    horizon = int(settings["horizon_months"])
    return_basis = str(settings["return_basis"])
    min_obs = int(settings["min_train_observations"])
    max_features = int(settings["max_model_features"])
    labels = build_active_return_labels(prices, asset_master, benchmark_weights, as_of_ts, horizon, return_basis=return_basis)
    features = point_in_time_macro_features(
        factors,
        macro_feature_lag_months=int(settings["macro_feature_lag_months"]),
    )
    features = features.loc[features.index <= as_of_ts].ffill()
    if features.empty:
        raise ValueError("No point-in-time macro features are available")
    latest_features = features.iloc[-1]
    training_cutoff = as_of_ts - pd.DateOffset(months=horizon)
    rows: list[dict[str, object]] = []
    master = asset_master.set_index("ticker")
    eligible_tickers = get_bl_asset_order(asset_master, prices)
    for ticker in eligible_tickers:
        row = master.loc[ticker]
        y = labels[ticker].loc[labels.index <= training_cutoff].replace([np.inf, -np.inf], np.nan)
        x = features.loc[features.index <= training_cutoff]
        aligned = pd.concat([y.rename("target"), x], axis=1).dropna(subset=["target"])
        n_train = int(len(aligned.dropna()))
        training_start = aligned.index.min() if not aligned.empty else pd.NaT
        training_end = aligned.index.max() if not aligned.empty else pd.NaT
        selected = _select_features(aligned["target"], aligned.drop(columns=["target"], errors="ignore"), max_features=max_features)
        forecast_raw, residual_std, contrib, n_model, model_r2 = _fit_linear_prediction(
            y,
            x,
            latest_features,
            selected,
            min_obs=min_obs,
        )
        hist_mean = float(y.dropna().tail(max(min_obs, 24)).mean()) if not y.dropna().empty else np.nan
        if not np.isfinite(forecast_raw):
            forecast_raw = hist_mean
        validation = _walk_forward_validation(
            y,
            x,
            min_obs=min_obs,
            max_features=max_features,
            stride_months=int(settings["validation_stride_months"]),
            max_points=int(settings["max_validation_points"]),
        )
        if not validation.empty:
            residuals = validation["realized"] - validation["prediction"]
            wf_std = float(np.sqrt(np.mean(residuals**2)))
            validation_ic = float(validation["prediction"].corr(validation["realized"])) if len(validation) >= 6 else np.nan
            validation_hit_rate = float((np.sign(validation["prediction"]) == np.sign(validation["realized"])).mean())
            pos_realized = validation.loc[validation["prediction"] > 0, "realized"]
            neg_realized = validation.loc[validation["prediction"] <= 0, "realized"]
            validation_spread = float(pos_realized.mean() - neg_realized.mean()) if not pos_realized.empty and not neg_realized.empty else np.nan
            validation_rmse = float(np.sqrt(np.mean(residuals**2)))
            validation_mae = float(np.mean(np.abs(residuals)))
            if np.isfinite(wf_std) and wf_std > 0:
                residual_std = wf_std
        else:
            validation_ic = np.nan
            validation_hit_rate = np.nan
            validation_spread = np.nan
            validation_rmse = np.nan
            validation_mae = np.nan
        train_std = float(y.dropna().std(ddof=0)) if len(y.dropna()) > 1 else np.nan
        if not np.isfinite(residual_std) or residual_std <= 0:
            residual_std = train_std if np.isfinite(train_std) and train_std > 0 else 0.05
        validation_quality = 0.50
        if np.isfinite(validation_ic):
            validation_quality = float(np.clip(0.50 + 0.50 * validation_ic, 0.15, 0.85))
        coverage_quality = float(np.clip(n_model / max(min_obs * 2.0, 1.0), 0.25, 1.0))
        shrink_weight = float(np.clip(0.25 + 0.45 * validation_quality + 0.30 * coverage_quality, 0.25, 0.80))
        forecast_shrunk = shrink_weight * float(forecast_raw) + (1.0 - shrink_weight) * float(hist_mean if np.isfinite(hist_mean) else 0.0)
        cap = float(settings["q_absolute_cap"])
        forecast_final = float(np.clip(forecast_shrunk, -cap, cap))
        forecast_error_var = float(max(residual_std**2, 1e-8))
        top_contrib = dict(sorted(contrib.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
        data_flag = _data_freshness_flag(ticker, source_audit)
        benchmark_id = str(row.get("benchmark_id", "") or "")
        rows.append(
            {
                "prediction_id": f"{as_of_ts:%Y%m%d}_{horizon}m_{return_basis}_{ticker}",
                "as_of": as_of_ts.date().isoformat(),
                "ticker": ticker,
                "horizon_months": horizon,
                "target_type": return_basis,
                "benchmark_id": benchmark_id,
                "benchmark_ticker": _benchmark_label(benchmark_weights, benchmark_id, as_of_ts),
                "forecast_return_raw": float(forecast_raw) if np.isfinite(forecast_raw) else np.nan,
                "forecast_return_shrunk": forecast_shrunk,
                "forecast_return_final": forecast_final,
                "return_basis": return_basis,
                "model_name": "macro_ols_selected_plus_historical_mean",
                "model_version": MODEL_VERSION,
                "feature_version": FEATURE_VERSION,
                "training_start": training_start.date().isoformat() if pd.notna(training_start) else "",
                "training_end": training_end.date().isoformat() if pd.notna(training_end) else "",
                "n_train": n_model,
                "selected_features_json": _json_dumps(selected),
                "top_driver_contrib_json": _json_dumps(top_contrib),
                "forecast_error_std": float(residual_std),
                "forecast_error_var": forecast_error_var,
                "validation_ic": validation_ic,
                "validation_hit_rate": validation_hit_rate,
                "validation_spread": validation_spread,
                "validation_rmse": validation_rmse,
                "validation_mae": validation_mae,
                "feature_stability_score": float(np.clip(abs(validation_ic), 0.0, 1.0)) if np.isfinite(validation_ic) else np.nan,
                "data_freshness_flag": data_flag,
                "created_at": pd.Timestamp.now(tz="UTC").replace(microsecond=0).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _prediction_confidence(
    prediction: pd.Series,
    settings: dict[str, object],
    unknown_probability: float,
) -> float:
    floor = float(settings["confidence_floor"])
    cap = float(settings["confidence_cap"])
    error = float(prediction.get("forecast_error_std", np.nan))
    forecast = float(prediction.get("forecast_return_final", np.nan))
    if not np.isfinite(error) or error <= 0 or not np.isfinite(forecast):
        return floor
    signal_to_noise = abs(forecast) / error
    base = floor + (cap - floor) * (1.0 - np.exp(-signal_to_noise))
    validation_ic = prediction.get("validation_ic", np.nan)
    hit_rate = prediction.get("validation_hit_rate", np.nan)
    validation_multiplier = 1.0
    if np.isfinite(validation_ic):
        validation_multiplier *= float(np.clip(1.0 + 0.35 * validation_ic, 0.70, 1.20))
    if np.isfinite(hit_rate):
        validation_multiplier *= float(np.clip(1.0 + 0.50 * (hit_rate - 0.50), 0.80, 1.15))
    if str(prediction.get("data_freshness_flag", "")).lower() != "current":
        validation_multiplier *= 0.85
    if unknown_probability >= float(settings["unknown_mixed_reduce_threshold"]):
        validation_multiplier *= 0.82
    return float(np.clip(base * validation_multiplier, floor, cap))


def _view_status(
    q_value: float,
    confidence: float,
    min_q: float,
    min_confidence: float,
    data_flag: str,
    unknown_probability: float,
    settings: dict[str, object],
) -> tuple[str, str]:
    if str(data_flag).lower() == "blocked":
        return "Blocked", "Data source is stale or excluded"
    if unknown_probability >= float(settings["unknown_mixed_block_threshold"]):
        return "Blocked", "Unknown/Mixed regime probability is above the automatic-view block threshold"
    if not np.isfinite(q_value):
        return "Blocked", "Forecast q is unavailable"
    if abs(q_value) < float(min_q):
        return "Blocked", "Forecast q is below the minimum materiality threshold"
    if confidence < float(min_confidence):
        return "Needs Review", "Confidence is below the candidate threshold"
    return "Candidate", ""


def _sparse_json_from_vector(vector: pd.Series) -> str:
    sparse = {str(k): float(v) for k, v in vector.items() if np.isfinite(v) and abs(float(v)) > 1e-12}
    return _json_dumps(sparse)


def _active_p_vector(
    ticker: str,
    benchmark_id: str,
    asset_order: list[str],
    benchmark_weights: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.Series, str]:
    vector = pd.Series(0.0, index=asset_order)
    if ticker not in vector.index:
        return vector, f"{ticker} is not in BL asset order"
    vector.loc[ticker] += 1.0
    weights = _latest_benchmark_weights(benchmark_weights, benchmark_id, as_of)
    if weights.empty:
        return vector * np.nan, f"Benchmark {benchmark_id} has no weights"
    missing = [component for component in weights.index if component not in vector.index]
    if missing:
        return vector * np.nan, f"Benchmark components missing from BL asset order: {missing}"
    for component, weight in weights.items():
        vector.loc[component] -= float(weight)
    return vector, ""


def _absolute_p_vector(ticker: str, asset_order: list[str]) -> tuple[pd.Series, str]:
    vector = pd.Series(0.0, index=asset_order)
    if ticker not in vector.index:
        return vector, f"{ticker} is not in BL asset order"
    vector.loc[ticker] = 1.0
    return vector, ""


def _relative_p_vector(long_ticker: str, short_ticker: str, asset_order: list[str]) -> tuple[pd.Series, str]:
    vector = pd.Series(0.0, index=asset_order)
    missing = [ticker for ticker in [long_ticker, short_ticker] if ticker not in vector.index]
    if missing:
        return vector * np.nan, f"Relative view assets missing from BL asset order: {missing}"
    vector.loc[long_ticker] = 1.0
    vector.loc[short_ticker] = -1.0
    return vector, ""


def _scenario_context_dict(scenarios: pd.DataFrame, factors: pd.DataFrame) -> dict[str, object]:
    try:
        result = estimate_scenario_probabilities(factors, scenarios)
        probs = result.probabilities.copy()
        modal = probs[~probs["is_unknown"]].sort_values("probability", ascending=False).iloc[0]
        return {
            "modal_scenario": str(modal["scenario"]),
            "modal_probability": float(modal["probability"]),
            "unknown_probability": float(result.unknown_probability),
            "scenario_confidence": float(result.confidence),
        }
    except Exception as exc:
        return {
            "modal_scenario": "Unavailable",
            "modal_probability": np.nan,
            "unknown_probability": 0.0,
            "scenario_confidence": np.nan,
            "scenario_error": f"{type(exc).__name__}: {exc}",
        }


def generate_bl_views(
    predictions: pd.DataFrame,
    asset_master: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    view_pairs: pd.DataFrame,
    asset_order: list[str],
    settings: dict[str, object],
    as_of: str | pd.Timestamp,
    scenario_context: dict[str, object],
) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of)
    horizon = int(settings["horizon_months"])
    unknown_probability = float(scenario_context.get("unknown_probability", 0.0) or 0.0)
    expiry_date = (as_of_ts + pd.offsets.MonthEnd(int(settings["view_expiry_months"]))).date().isoformat()
    pred = predictions.set_index("ticker")
    master = asset_master.set_index("ticker")
    rows: list[dict[str, object]] = []

    def add_row(
        view_id: str,
        view_type: str,
        sleeve: str,
        assets: str,
        p_vector: pd.Series,
        q_value: float,
        q_basis: str,
        confidence: float,
        omega: float,
        omega_method: str,
        forecast_ids: list[str],
        forecast_error_std: float,
        validation_ic: float,
        validation_hit_rate: float,
        validation_spread: float,
        top_drivers_json: str,
        rationale: str,
        risks: str,
        status: str,
        block_reason: str,
        forecast_model: str,
    ) -> None:
        rows.append(
            {
                "view_id": view_id,
                "as_of": as_of_ts.date().isoformat(),
                "horizon_months": horizon,
                "view_type": view_type,
                "sleeve": sleeve,
                "assets": assets,
                "p_vector_json": _sparse_json_from_vector(p_vector),
                "p_l1_norm": float(p_vector.abs().sum()) if p_vector.notna().all() else np.nan,
                "q_expected_return": q_value,
                "q_return_basis": q_basis,
                "q_units": f"decimal_{horizon}m_return",
                "confidence_score": confidence,
                "omega": omega,
                "omega_method": omega_method,
                "omega_units": f"variance_of_decimal_{horizon}m_return",
                "forecast_model": forecast_model,
                "forecast_ids": _json_dumps(forecast_ids),
                "forecast_error_std": forecast_error_std,
                "validation_ic": validation_ic,
                "validation_hit_rate": validation_hit_rate,
                "validation_spread": validation_spread,
                "scenario_context_json": _json_dumps(scenario_context),
                "top_drivers_json": top_drivers_json,
                "rationale_short": rationale,
                "risks": risks,
                "status": status,
                "block_reason": block_reason,
                "expiry_date": expiry_date,
                "model_version": MODEL_VERSION,
                "data_version": as_of_ts.date().isoformat(),
                "generated_at": pd.Timestamp.now(tz="UTC").replace(microsecond=0).isoformat(),
                "_candidate_score": abs(q_value) * max(confidence, 0.0) / max(forecast_error_std, 1e-6) if np.isfinite(q_value) else 0.0,
            }
        )

    for ticker, p_row in pred.iterrows():
        if ticker not in master.index:
            continue
        m_row = master.loc[ticker]
        if not _parse_bool(m_row.get("view_eligible_absolute", False)):
            continue
        q_value = float(p_row.get("forecast_return_final", np.nan))
        confidence = _prediction_confidence(p_row, settings, unknown_probability)
        error_var = float(p_row.get("forecast_error_var", np.nan))
        omega = float(np.clip(error_var / max(confidence, 1e-6), float(settings["omega_floor"]), float(settings["omega_cap"])))
        p_vector, p_error = _active_p_vector(ticker, str(m_row.get("benchmark_id", "") or ""), asset_order, benchmark_weights, as_of_ts)
        status, reason = _view_status(
            q_value,
            confidence,
            min_q=float(settings["min_active_q_threshold"]),
            min_confidence=float(settings["min_candidate_confidence"]),
            data_flag=str(p_row.get("data_freshness_flag", "")),
            unknown_probability=unknown_probability,
            settings=settings,
        )
        if p_error:
            status, reason = "Blocked", p_error
        drivers = json.loads(str(p_row.get("top_driver_contrib_json", "{}") or "{}"))
        driver_text = ", ".join(list(drivers.keys())[:3]) or "macro OLS baseline"
        direction = "outperform" if q_value >= 0 else "underperform"
        benchmark_label = str(p_row.get("benchmark_ticker") or m_row.get("benchmark_id") or "benchmark")
        rationale = (
            f"{ticker} is forecast to {direction} {benchmark_label} by {q_value:+.1%} over {horizon}m; "
            f"main drivers: {driver_text}."
        )
        add_row(
            view_id=f"{as_of_ts:%Y%m%d}_{horizon}m_ACTIVE_{ticker}",
            view_type="active",
            sleeve=str(m_row.get("sleeve", "")),
            assets=f"{ticker} vs {benchmark_label}",
            p_vector=p_vector,
            q_value=q_value,
            q_basis=str(p_row.get("return_basis", settings["return_basis"])),
            confidence=confidence,
            omega=omega,
            omega_method="forecast_error_var_divided_by_confidence",
            forecast_ids=[str(p_row.get("prediction_id", ""))],
            forecast_error_std=float(p_row.get("forecast_error_std", np.nan)),
            validation_ic=float(p_row.get("validation_ic", np.nan)),
            validation_hit_rate=float(p_row.get("validation_hit_rate", np.nan)),
            validation_spread=float(p_row.get("validation_spread", np.nan)),
            top_drivers_json=str(p_row.get("top_driver_contrib_json", "{}") or "{}"),
            rationale=rationale,
            risks="Public proxy data uses latest revised vintage; macro features are lagged one month.",
            status=status,
            block_reason=reason,
            forecast_model=str(p_row.get("model_name", "")),
        )

    approved_pairs = view_pairs[view_pairs["is_approved"].map(_parse_bool)].copy()
    for _, pair in approved_pairs.iterrows():
        long_ticker = str(pair["long_ticker"])
        short_ticker = str(pair["short_ticker"])
        if long_ticker not in pred.index or short_ticker not in pred.index:
            continue
        long_pred = pred.loc[long_ticker]
        short_pred = pred.loc[short_ticker]
        q_value = float(long_pred["forecast_return_final"] - short_pred["forecast_return_final"])
        q_value = float(np.clip(q_value, -float(pair["max_q_abs"]), float(pair["max_q_abs"])))
        q_value = float(np.clip(q_value, -float(settings["q_relative_cap"]), float(settings["q_relative_cap"])))
        confidence = min(
            _prediction_confidence(long_pred, settings, unknown_probability),
            _prediction_confidence(short_pred, settings, unknown_probability),
        )
        error_var = float(long_pred.get("forecast_error_var", np.nan)) + float(short_pred.get("forecast_error_var", np.nan))
        omega = float(np.clip(error_var / max(confidence, 1e-6), float(settings["omega_floor"]), float(settings["omega_cap"])))
        p_vector, p_error = _relative_p_vector(long_ticker, short_ticker, asset_order)
        status, reason = _view_status(
            q_value,
            confidence,
            min_q=float(pair["min_q_threshold"]),
            min_confidence=float(pair["min_confidence"]),
            data_flag="Current",
            unknown_probability=unknown_probability,
            settings=settings,
        )
        if p_error:
            status, reason = "Blocked", p_error
        direction = "outperform" if q_value >= 0 else "underperform"
        rationale = f"{long_ticker} is forecast to {direction} {short_ticker} by {q_value:+.1%} over {horizon}m."
        long_drivers = json.loads(str(long_pred.get("top_driver_contrib_json", "{}") or "{}"))
        short_drivers = json.loads(str(short_pred.get("top_driver_contrib_json", "{}") or "{}"))
        pair_drivers = {"long": long_drivers, "short": short_drivers}
        add_row(
            view_id=f"{as_of_ts:%Y%m%d}_{horizon}m_REL_{long_ticker}_{short_ticker}",
            view_type="relative",
            sleeve=str(pair.get("sleeve", "")),
            assets=f"{long_ticker} - {short_ticker}",
            p_vector=p_vector,
            q_value=q_value,
            q_basis=str(settings["relative_return_basis"]),
            confidence=confidence,
            omega=omega,
            omega_method="relative_error_var_sum_divided_by_confidence",
            forecast_ids=[str(long_pred["prediction_id"]), str(short_pred["prediction_id"])],
            forecast_error_std=float(np.sqrt(max(error_var, 0.0))),
            validation_ic=float(np.nanmean([long_pred.get("validation_ic", np.nan), short_pred.get("validation_ic", np.nan)])),
            validation_hit_rate=float(np.nanmean([long_pred.get("validation_hit_rate", np.nan), short_pred.get("validation_hit_rate", np.nan)])),
            validation_spread=float(np.nanmean([long_pred.get("validation_spread", np.nan), short_pred.get("validation_spread", np.nan)])),
            top_drivers_json=_json_dumps(pair_drivers),
            rationale=rationale,
            risks="Relative view uses diagonal residual-error approximation in v1.",
            status=status,
            block_reason=reason,
            forecast_model=str(long_pred.get("model_name", "")),
        )

    views = pd.DataFrame(rows)
    if views.empty:
        return views
    candidate_idx = views[views["status"].eq("Candidate")].sort_values("_candidate_score", ascending=False).index
    max_views = int(settings["max_views_per_run"])
    overflow = candidate_idx[max_views:]
    if len(overflow) > 0:
        views.loc[overflow, "status"] = "Needs Review"
        views.loc[overflow, "block_reason"] = "Candidate cap reached; not exported to P/q/Omega"
    return views.drop(columns=["_candidate_score"]).sort_values(["status", "confidence_score", "q_expected_return"], ascending=[True, False, False])


def build_P_q_Omega(views: pd.DataFrame, asset_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = views[views["status"].eq("Candidate")].copy() if not views.empty else pd.DataFrame()
    if candidates.empty:
        return (
            pd.DataFrame(columns=asset_order, dtype=float),
            pd.DataFrame(columns=["view_id", "q_expected_return"], dtype=float),
            pd.DataFrame(dtype=float),
        )
    p_rows = []
    for _, row in candidates.iterrows():
        sparse = json.loads(row["p_vector_json"])
        vector = pd.Series(0.0, index=asset_order)
        for ticker, value in sparse.items():
            if ticker in vector.index:
                vector.loc[ticker] = float(value)
        vector.name = row["view_id"]
        p_rows.append(vector)
    P = pd.DataFrame(p_rows, columns=asset_order)
    q = candidates[["view_id", "q_expected_return"]].copy()
    omega_values = candidates.set_index("view_id")["omega"].astype(float)
    Omega = pd.DataFrame(np.diag(omega_values.to_numpy()), index=omega_values.index, columns=omega_values.index)
    return P, q, Omega


def _covariance_matrix(
    prices: pd.DataFrame,
    asset_order: list[str],
    settings: dict[str, object],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    returns = prices.reindex(columns=asset_order).pct_change().replace([np.inf, -np.inf], np.nan)
    sample = returns.loc[returns.index <= as_of].tail(int(settings["covariance_lookback_months"])).dropna(axis=1, how="all")
    sample = sample.reindex(columns=asset_order)
    cov = sample.cov(min_periods=max(12, min(36, len(sample) // 2))).fillna(0.0)
    diagonal = pd.DataFrame(np.diag(np.diag(cov.to_numpy())), index=cov.index, columns=cov.columns)
    shrink = float(settings.get("covariance_shrinkage", 0.20) or 0.0)
    cov = (1.0 - shrink) * cov + shrink * diagonal
    cov = cov * int(settings["horizon_months"])
    values = (cov.to_numpy(dtype=float) + cov.to_numpy(dtype=float).T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(values)
    eigvals = np.clip(eigvals, 1e-8, None)
    repaired = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return pd.DataFrame(repaired, index=asset_order, columns=asset_order)


def _benchmark_vector(
    benchmark_weights: pd.DataFrame,
    benchmark_id: str,
    asset_order: list[str],
    as_of: pd.Timestamp,
) -> pd.Series:
    weights = _latest_benchmark_weights(benchmark_weights, benchmark_id, as_of)
    vector = pd.Series(0.0, index=asset_order)
    for ticker, weight in weights.items():
        if ticker in vector.index:
            vector.loc[ticker] = float(weight)
    total = float(vector.sum())
    if total > 0:
        vector = vector / total
    return vector


def run_black_litterman(
    prices: pd.DataFrame,
    asset_master: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    asset_order: list[str],
    P: pd.DataFrame,
    q: pd.DataFrame,
    Omega: pd.DataFrame,
    settings: dict[str, object],
    as_of: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    as_of_ts = pd.Timestamp(as_of)
    Sigma = _covariance_matrix(prices, asset_order, settings, as_of_ts)
    benchmark_id = str(settings.get("policy_benchmark_id", "GSMIF_SAMPLE_POLICY"))
    benchmark = _benchmark_vector(benchmark_weights, benchmark_id, asset_order, as_of_ts)
    delta = float(settings["risk_aversion_delta"])
    tau = float(settings["tau"])
    pi = pd.Series(delta * Sigma.to_numpy().dot(benchmark.to_numpy()), index=asset_order, name="prior_return")
    pi_frame = pi.rename_axis("ticker").reset_index()

    if P.empty or q.empty or Omega.empty:
        posterior = pi.copy()
        status = "No candidate views; posterior equals prior"
    else:
        sigma_tau = tau * Sigma.to_numpy(dtype=float)
        Pm = P.reindex(columns=asset_order).to_numpy(dtype=float)
        qv = q.set_index("view_id").reindex(P.index)["q_expected_return"].to_numpy(dtype=float)
        Om = Omega.reindex(index=P.index, columns=P.index).to_numpy(dtype=float)
        inv_tau_sigma = np.linalg.pinv(sigma_tau)
        inv_omega = np.linalg.pinv(Om)
        lhs = inv_tau_sigma + Pm.T @ inv_omega @ Pm
        rhs = inv_tau_sigma @ pi.to_numpy(dtype=float) + Pm.T @ inv_omega @ qv
        posterior_values = np.linalg.pinv(lhs) @ rhs
        posterior = pd.Series(posterior_values, index=asset_order, name="posterior_return")
        status = "Posterior computed"

    diff = posterior - pi
    abs_sum = float(diff.abs().sum())
    active_budget = 0.10
    active_weight = diff / abs_sum * active_budget if abs_sum > 0 else diff * 0.0
    suggested = benchmark + active_weight
    posterior_rows = []
    master = asset_master.set_index("ticker")
    for ticker in asset_order:
        posterior_rows.append(
            {
                "run_id": f"{as_of_ts:%Y%m%d}_{int(settings['horizon_months'])}m_{benchmark_id}",
                "ticker": ticker,
                "prior_return": float(pi.loc[ticker]),
                "posterior_return": float(posterior.loc[ticker]),
                "posterior_minus_prior": float(diff.loc[ticker]),
                "benchmark_weight": float(benchmark.loc[ticker]),
                "suggested_weight": float(suggested.loc[ticker]),
                "suggested_active_weight": float(active_weight.loc[ticker]),
                "asset_name": str(master.loc[ticker, "asset_name"]) if ticker in master.index else ticker,
                "sleeve": str(master.loc[ticker, "sleeve"]) if ticker in master.index else "",
            }
        )
    posterior_returns = pd.DataFrame(posterior_rows)
    run = pd.DataFrame(
        [
            {
                "run_id": f"{as_of_ts:%Y%m%d}_{int(settings['horizon_months'])}m_{benchmark_id}",
                "as_of": as_of_ts.date().isoformat(),
                "horizon_months": int(settings["horizon_months"]),
                "benchmark_id": benchmark_id,
                "num_assets": len(asset_order),
                "num_views": int(len(P)),
                "tau": tau,
                "risk_aversion_delta": delta,
                "covariance_method": str(settings["covariance_method"]),
                "return_basis": str(settings["return_basis"]),
                "prior_return_json": _json_dumps(pi.round(8).to_dict()),
                "posterior_return_json": _json_dumps(posterior.round(8).to_dict()),
                "active_weight_suggestion_json": _json_dumps(active_weight.round(8).to_dict()),
                "status": status,
                "created_at": pd.Timestamp.now(tz="UTC").replace(microsecond=0).isoformat(),
                "notes": "BL v1 uses horizon-scaled shrunk covariance and sample benchmark weights unless replaced.",
            }
        ]
    )
    return run, posterior_returns, Sigma, pi_frame


def build_view_diagnostics(
    views: pd.DataFrame,
    P: pd.DataFrame,
    q: pd.DataFrame,
    posterior_returns: pd.DataFrame,
) -> pd.DataFrame:
    if views.empty or P.empty or q.empty or posterior_returns.empty:
        return pd.DataFrame(
            columns=[
                "view_id",
                "assets",
                "view_type",
                "q_expected_return",
                "prior_view_return",
                "posterior_view_return",
                "posterior_minus_prior_view",
                "remaining_gap_to_q",
                "pull_to_q",
                "transmission_flag",
                "confidence_score",
                "omega",
                "status",
            ]
        )

    posterior = posterior_returns.set_index("ticker")
    prior_vector = posterior["prior_return"].astype(float)
    posterior_vector = posterior["posterior_return"].astype(float)
    q_vector = q.set_index("view_id")["q_expected_return"].astype(float)
    view_meta = views.set_index("view_id")
    rows: list[dict[str, object]] = []
    for view_id, p_vector in P.iterrows():
        aligned_p = p_vector.reindex(prior_vector.index).fillna(0.0).astype(float)
        prior_view = float(aligned_p.dot(prior_vector))
        posterior_view = float(aligned_p.dot(posterior_vector))
        q_value = float(q_vector.get(view_id, np.nan))
        original_gap = q_value - prior_view
        remaining_gap = q_value - posterior_view
        moved = posterior_view - prior_view
        pull_to_q = moved / original_gap if abs(original_gap) > 1e-10 else np.nan
        if not np.isfinite(pull_to_q):
            transmission_flag = "Neutral"
        elif pull_to_q < -0.01:
            transmission_flag = "Opposite"
        elif pull_to_q < 0.02:
            transmission_flag = "Weak"
        else:
            transmission_flag = "Aligned"
        meta = view_meta.loc[view_id] if view_id in view_meta.index else pd.Series(dtype=object)
        rows.append(
            {
                "view_id": view_id,
                "assets": str(meta.get("assets", "")),
                "view_type": str(meta.get("view_type", "")),
                "q_expected_return": q_value,
                "prior_view_return": prior_view,
                "posterior_view_return": posterior_view,
                "posterior_minus_prior_view": moved,
                "remaining_gap_to_q": remaining_gap,
                "pull_to_q": float(pull_to_q) if np.isfinite(pull_to_q) else np.nan,
                "transmission_flag": transmission_flag,
                "confidence_score": float(meta.get("confidence_score", np.nan)),
                "omega": float(meta.get("omega", np.nan)),
                "status": str(meta.get("status", "")),
            }
        )
    return pd.DataFrame(rows)


def generate_bl_inputs(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    universe: pd.DataFrame,
    scenarios: pd.DataFrame,
    config_dir: str | Path,
    source_audit: pd.DataFrame | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> BLResult:
    settings = load_bl_settings(config_dir)
    as_of_ts = _as_of_date(prices, factors, as_of)
    asset_master = load_asset_master(config_dir)
    benchmark_weights = load_benchmark_weights(config_dir)
    view_pairs = load_view_pairs(config_dir)
    asset_order = get_bl_asset_order(asset_master, prices)
    config_audit = validate_bl_config(asset_master, benchmark_weights, view_pairs, prices, settings, as_of=as_of_ts)
    scenario_context = _scenario_context_dict(scenarios, factors.loc[factors.index <= as_of_ts])
    predictions = build_model_predictions(
        prices,
        factors.loc[factors.index <= as_of_ts],
        asset_master,
        benchmark_weights,
        settings,
        as_of=as_of_ts,
        source_audit=source_audit,
    )
    views = generate_bl_views(
        predictions,
        asset_master,
        benchmark_weights,
        view_pairs,
        asset_order,
        settings,
        as_of=as_of_ts,
        scenario_context=scenario_context,
    )
    P, q, Omega = build_P_q_Omega(views, asset_order)
    bl_runs, posterior_returns, Sigma, pi = run_black_litterman(
        prices,
        asset_master,
        benchmark_weights,
        asset_order,
        P,
        q,
        Omega,
        settings,
        as_of=as_of_ts,
    )
    view_diagnostics = build_view_diagnostics(views, P, q, posterior_returns)
    return BLResult(
        as_of=as_of_ts,
        horizon_months=int(settings["horizon_months"]),
        asset_order=asset_order,
        settings=settings,
        config_audit=config_audit,
        predictions=predictions,
        views=views,
        P=P,
        q=q,
        Omega=Omega,
        Sigma=Sigma,
        pi=pi,
        bl_runs=bl_runs,
        posterior_returns=posterior_returns,
        view_diagnostics=view_diagnostics,
        scenario_context=scenario_context,
    )


def write_bl_outputs(result: BLResult, root: str | Path) -> None:
    root = Path(root)
    data_dir = root / "data"
    exports_dir = root / "exports"
    data_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)

    result.predictions.to_csv(data_dir / "model_predictions.csv", index=False)
    result.views.to_csv(data_dir / "bl_macro_views.csv", index=False)
    result.bl_runs.to_csv(data_dir / "bl_runs.csv", index=False)
    result.posterior_returns.to_csv(data_dir / "bl_posterior_returns.csv", index=False)
    result.view_diagnostics.to_csv(data_dir / "bl_view_diagnostics.csv", index=False)

    pd.DataFrame({"ticker": result.asset_order, "column_index": range(len(result.asset_order))}).to_csv(
        exports_dir / "bl_asset_order.csv",
        index=False,
    )
    result.P.to_csv(exports_dir / "P_matrix.csv", index=True, index_label="view_id")
    result.q.to_csv(exports_dir / "q_vector.csv", index=False)
    result.Omega.to_csv(exports_dir / "Omega_matrix.csv", index=True, index_label="view_id")
    result.Sigma.to_csv(exports_dir / "Sigma_matrix.csv", index=True, index_label="ticker")
    result.pi.to_csv(exports_dir / "pi_vector.csv", index=False)
    payload = {
        "as_of": result.as_of.date().isoformat(),
        "horizon_months": result.horizon_months,
        "asset_order": result.asset_order,
        "views": result.views.to_dict(orient="records"),
        "P": result.P.to_dict(orient="index"),
        "q": result.q.to_dict(orient="records"),
        "Omega": result.Omega.to_dict(orient="index"),
        "Sigma": result.Sigma.to_dict(orient="index"),
        "pi": result.pi.to_dict(orient="records"),
        "view_diagnostics": result.view_diagnostics.to_dict(orient="records"),
        "scenario_context": result.scenario_context,
    }
    (exports_dir / "bl_views.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
