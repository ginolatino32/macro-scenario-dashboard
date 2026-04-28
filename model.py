from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ["risk", "growth", "inflation", "rates", "liquidity", "dollar", "oil"]
FEATURE_BLOCK_WEIGHTS = {
    "shock": 0.45,
    "trend_3m": 0.25,
    "trend_6m": 0.20,
    "accel": 0.10,
}
FEATURE_BLOCK_SCENARIO_MULTIPLIERS = {
    "shock": 1.00,
    "trend_3m": 0.75,
    "trend_6m": 0.50,
    "accel": 0.25,
}


@dataclass
class ModelResult:
    exposures: pd.DataFrame
    tstats: pd.DataFrame
    alphas: pd.Series
    diagnostics: pd.DataFrame
    contributions: pd.DataFrame
    expected: pd.DataFrame
    bucket_summary: pd.DataFrame
    trade_basket: pd.DataFrame
    rel_returns: pd.DataFrame
    factors_z: pd.DataFrame
    corr: pd.DataFrame


@dataclass
class ScenarioProbabilityResult:
    probabilities: pd.DataFrame
    raw_probabilities: pd.DataFrame
    transition_prior: pd.DataFrame
    transition_matrix: pd.DataFrame
    latest_features: pd.Series
    scenario_features: pd.DataFrame
    feature_history: pd.DataFrame
    entropy: float
    confidence: float
    unknown_probability: float


@dataclass
class ScenarioCalibrationResult:
    summary: pd.DataFrame
    probability_buckets: pd.DataFrame
    recent_predictions: pd.DataFrame


@dataclass
class PortfolioOptimizationResult:
    weights: pd.DataFrame
    bucket_weights: pd.DataFrame
    stats: pd.DataFrame
    covariance: pd.DataFrame


@dataclass
class PortfolioBacktestResult:
    returns: pd.DataFrame
    equity: pd.DataFrame
    drawdowns: pd.DataFrame
    summary: pd.DataFrame
    weights: pd.DataFrame
    scenario_counts: pd.DataFrame
    benchmark_diagnostics: pd.DataFrame
    rolling_metrics: pd.DataFrame
    stress_months: pd.DataFrame
    cost_sensitivity: pd.DataFrame


def load_wide_csv(path: str | Path) -> pd.DataFrame:
    """Load a date-indexed wide CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns:
        raise ValueError(f"{path} must contain a date column")
    df = df.set_index("date").sort_index()
    return df.apply(pd.to_numeric, errors="coerce")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly log returns from price/index levels."""
    return np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)


def factor_changes(factors: pd.DataFrame) -> pd.DataFrame:
    """Convert factor levels to changes; rates/growth/inflation use level changes, dollar/oil/liquidity use log changes."""
    missing = [c for c in FACTOR_COLUMNS if c not in factors.columns]
    if missing:
        raise ValueError(f"factors.csv is missing required columns: {missing}")

    x = pd.DataFrame(index=factors.index)
    for c in FACTOR_COLUMNS:
        if c in {"risk", "growth", "inflation", "rates"}:
            x[c] = factors[c].diff()
        else:
            x[c] = np.log(factors[c] / factors[c].shift(1))
    return x.replace([np.inf, -np.inf], np.nan)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)


def softmax(scores: pd.Series) -> pd.Series:
    scores = scores.astype(float)
    finite = scores.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return pd.Series(1.0 / len(scores), index=scores.index)
    shifted = scores - finite.max()
    exp_scores = np.exp(shifted.replace([np.inf, -np.inf], np.nan).fillna(-1e9))
    total = exp_scores.sum()
    if total <= 0 or not np.isfinite(total):
        return pd.Series(1.0 / len(scores), index=scores.index)
    return exp_scores / total


def make_relative_returns(returns: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Apply benchmark-relative return mapping from universe.csv.

    Blank benchmark => absolute return.
    Nonblank benchmark => asset return minus benchmark return.
    """
    universe = universe.copy()
    universe["benchmark"] = universe["benchmark"].fillna("")
    out: dict[str, pd.Series] = {}
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        if ticker not in returns.columns:
            continue
        bm = str(row.get("benchmark", "") or "").strip()
        if bm and bm in returns.columns:
            out[ticker] = returns[ticker] - returns[bm]
        else:
            out[ticker] = returns[ticker]
    return pd.DataFrame(out, index=returns.index)


def exp_weights(n: int, half_life: float | None) -> np.ndarray:
    if half_life is None or half_life <= 0:
        return np.ones(n)
    # Oldest to newest. Newest weight = 1.0.
    age = np.arange(n - 1, -1, -1)
    return 0.5 ** (age / half_life)


def weighted_ols(
    y: pd.Series,
    x: pd.DataFrame,
    half_life: float | None = 36.0,
) -> tuple[pd.Series, pd.Series, float, int, float, float]:
    """Fit y = alpha + X beta using diagonal recency weights.

    Returns beta, beta_tstats, alpha, nobs, weighted_r2, residual_vol.
    """
    data = pd.concat([y.rename("y"), x], axis=1).dropna()
    if len(data) < len(x.columns) + 12:
        empty = pd.Series(np.nan, index=x.columns)
        return empty, empty.copy(), np.nan, len(data), np.nan, np.nan

    yv = data["y"].to_numpy(dtype=float)
    xv = data[x.columns].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(data)), xv])
    w = exp_weights(len(data), half_life)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = yv * sw

    try:
        coef = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        residual = yv - X @ coef
        dof = max(len(data) - X.shape[1], 1)
        sse = float(np.sum(w * residual**2))
        sigma2 = sse / dof
        ybar = float(np.average(yv, weights=w))
        sst = float(np.sum(w * (yv - ybar) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        residual_vol = float(np.sqrt(max(sigma2, 0.0)))
        xtwx_inv = np.linalg.pinv(X.T @ (w[:, None] * X))
        se = np.sqrt(np.diag(sigma2 * xtwx_inv))
        t = coef / se
    except np.linalg.LinAlgError:
        beta = pd.Series(np.nan, index=x.columns)
        tstats = pd.Series(np.nan, index=x.columns)
        return beta, tstats, np.nan, len(data), np.nan, np.nan

    beta = pd.Series(coef[1:], index=x.columns)
    tstats = pd.Series(t[1:], index=x.columns)
    alpha = float(coef[0])
    return beta, tstats, alpha, len(data), float(r2), residual_vol


def fit_exposures(
    rel_returns: pd.DataFrame,
    factors_z: pd.DataFrame,
    half_life: float | None = 36.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    betas = {}
    tstats = {}
    alphas = {}
    nobs = {}
    r2 = {}
    residual_vol = {}
    for col in rel_returns.columns:
        b, t, alpha, n, fit_r2, resid = weighted_ols(rel_returns[col], factors_z, half_life=half_life)
        betas[col] = b
        tstats[col] = t
        alphas[col] = alpha
        nobs[col] = n
        r2[col] = fit_r2
        residual_vol[col] = resid
    return (
        pd.DataFrame(betas).T,
        pd.DataFrame(tstats).T,
        pd.Series(alphas, name="alpha"),
        pd.Series(nobs, name="nobs"),
        pd.Series(r2, name="model_r2"),
        pd.Series(residual_vol, name="residual_vol"),
    )


def trailing_log_return(rel_returns: pd.DataFrame, periods: int) -> pd.Series:
    if len(rel_returns) < periods:
        return pd.Series(np.nan, index=rel_returns.columns)
    return rel_returns.tail(periods).sum(min_count=max(1, periods // 2))


def macro_feature_history(factors: pd.DataFrame) -> pd.DataFrame:
    """Create macro state features from standardized factor shocks.

    Features intentionally use only factor data, not asset returns. That keeps
    the scenario probability engine separate from portfolio desirability.
    """
    shocks = zscore(factor_changes(factors))
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
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def scenario_feature_centers(scenarios: pd.DataFrame) -> pd.DataFrame:
    """Expand seven-factor scenario presets into level/trend/acceleration features."""
    rows = {}
    scenario_rows = scenarios[scenarios["scenario"] != "Custom"].copy()
    for _, row in scenario_rows.iterrows():
        base = row[FACTOR_COLUMNS].astype(float)
        expanded = {}
        for block, multiplier in FEATURE_BLOCK_SCENARIO_MULTIPLIERS.items():
            for factor in FACTOR_COLUMNS:
                expanded[f"{block}_{factor}"] = float(base[factor]) * multiplier
        rows[row["scenario"]] = expanded
    return pd.DataFrame(rows).T


def _feature_weight_vector(columns: pd.Index) -> pd.Series:
    weights = {}
    for col in columns:
        col_name = str(col)
        block = next((name for name in FEATURE_BLOCK_WEIGHTS if col_name.startswith(f"{name}_")), "shock")
        weights[col] = np.sqrt(FEATURE_BLOCK_WEIGHTS.get(block, 1.0))
    return pd.Series(weights)


def _normalize_probabilities(values: pd.Series) -> pd.Series:
    values = values.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(values.sum())
    if total <= 0:
        return pd.Series(1.0 / len(values), index=values.index)
    return values / total


def _scenario_taxonomy_from_row(row: pd.Series) -> dict[str, str]:
    growth = float(row.get("growth", 0.0))
    inflation = float(row.get("inflation", 0.0))
    rates = float(row.get("rates", 0.0))
    liquidity = float(row.get("liquidity", 0.0))
    dollar = float(row.get("dollar", 0.0))
    oil = float(row.get("oil", 0.0))
    risk = float(row.get("risk", 0.0))

    if growth >= 0 and inflation < 0:
        core = "Reflation / Spring"
    elif growth >= 0 and inflation >= 0:
        core = "Expansion / Summer"
    elif growth < 0 and inflation >= 0:
        core = "Stagflation / Fall"
    else:
        core = "Deflation / Winter"

    if rates >= 0.6 or liquidity <= -0.6:
        policy = "Tightening / liquidity drain"
    elif rates <= -0.6 or liquidity >= 0.6:
        policy = "Easing / liquidity support"
    else:
        policy = "Neutral / mixed policy"

    if oil >= 0.9:
        stress = "Oil shock"
    elif dollar >= 0.9:
        stress = "Dollar squeeze"
    elif risk <= -0.6:
        stress = "Risk-off stress"
    elif risk >= 1.0 and liquidity >= 0.5:
        stress = "Melt-up / speculative risk"
    else:
        stress = "No dominant stress"

    return {"core_regime": core, "policy_overlay": policy, "stress_overlay": stress}


def scenario_taxonomy(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in scenarios[scenarios["scenario"] != "Custom"].iterrows():
        labels = _scenario_taxonomy_from_row(row)
        labels["scenario"] = row["scenario"]
        rows.append(labels)
    return pd.DataFrame(rows)


def scenario_overlay_breakdown(probabilities: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    taxonomy = scenario_taxonomy(scenarios)
    probs = probabilities[["scenario", "probability"]].copy()
    merged = taxonomy.merge(probs, on="scenario", how="left")
    merged["probability"] = merged["probability"].fillna(0.0)
    rows = []
    for layer, label_col in [
        ("Core regime", "core_regime"),
        ("Policy / liquidity overlay", "policy_overlay"),
        ("Stress overlay", "stress_overlay"),
    ]:
        grouped = merged.groupby(label_col, as_index=False)["probability"].sum()
        for _, row in grouped.iterrows():
            rows.append({"layer": layer, "state": row[label_col], "probability": row["probability"]})
    unknown = float(probs.loc[probs["scenario"].eq("Unknown / Mixed"), "probability"].sum())
    if unknown > 0:
        rows.append({"layer": "Core regime", "state": "Unknown / Mixed", "probability": unknown})
        rows.append({"layer": "Policy / liquidity overlay", "state": "Unknown / Mixed", "probability": unknown})
        rows.append({"layer": "Stress overlay", "state": "Unknown / Mixed", "probability": unknown})
    out = pd.DataFrame(rows)
    out["probability_pct"] = out["probability"] * 100.0
    return out.sort_values(["layer", "probability"], ascending=[True, False]).reset_index(drop=True)


def default_transition_matrix(scenarios: pd.DataFrame) -> pd.DataFrame:
    names = [s for s in scenarios["scenario"].tolist() if s != "Custom"]
    taxonomy = scenario_taxonomy(scenarios).set_index("scenario")
    matrix = pd.DataFrame(0.02, index=names, columns=names, dtype=float)

    for from_s in names:
        matrix.loc[from_s, from_s] += 0.55
        for to_s in names:
            if from_s == to_s:
                continue
            if taxonomy.loc[from_s, "core_regime"] == taxonomy.loc[to_s, "core_regime"]:
                matrix.loc[from_s, to_s] += 0.12
            if taxonomy.loc[from_s, "policy_overlay"] == taxonomy.loc[to_s, "policy_overlay"]:
                matrix.loc[from_s, to_s] += 0.07
            if taxonomy.loc[from_s, "stress_overlay"] == taxonomy.loc[to_s, "stress_overlay"]:
                matrix.loc[from_s, to_s] += 0.05

    cycle_edges = {
        "Spring": ["Summer", "Easing + Rate Cuts"],
        "Summer": ["Fall", "Market Melt Up"],
        "Fall": ["Winter", "Tightening + Rate Hikes", "Oil Shock"],
        "Winter": ["Easing + Rate Cuts", "Spring"],
        "Dollar Wrecking Ball": ["Winter", "Easing + Rate Cuts"],
        "Tightening + Rate Hikes": ["Fall", "Winter"],
        "Easing + Rate Cuts": ["Spring", "Market Melt Up"],
        "Oil Shock": ["Fall", "Tightening + Rate Hikes"],
        "Market Melt Up": ["Tightening + Rate Hikes", "Summer"],
    }
    for from_s, targets in cycle_edges.items():
        if from_s not in matrix.index:
            continue
        for to_s in targets:
            if to_s in matrix.columns:
                matrix.loc[from_s, to_s] += 0.12

    return matrix.div(matrix.sum(axis=1), axis=0)


def _distance_table(
    history: pd.DataFrame,
    latest: pd.Series,
    centers: pd.DataFrame,
    temperature: float,
) -> pd.DataFrame:
    valid_cols = latest.dropna().index
    history = history[valid_cols].dropna()
    centers = centers[valid_cols]
    latest = latest[valid_cols]
    if len(valid_cols) < len(FACTOR_COLUMNS):
        raise ValueError("Not enough macro features are available for scenario probability estimation")

    weights = _feature_weight_vector(pd.Index(valid_cols))
    history_w = history.mul(weights, axis=1)
    latest_w = latest.mul(weights)
    centers_w = centers.mul(weights, axis=1)

    cov = np.cov(history_w.to_numpy(dtype=float), rowvar=False)
    ridge = np.eye(cov.shape[0]) * 1e-4
    inv_cov = np.linalg.pinv(cov + ridge)

    rows = []
    for scenario_name, center in centers_w.iterrows():
        diff = latest_w.to_numpy(dtype=float) - center.to_numpy(dtype=float)
        distance_sq = float(diff.T @ inv_cov @ diff)
        normalized_distance = float(np.sqrt(max(distance_sq, 0.0) / len(valid_cols)))
        rows.append(
            {
                "scenario": scenario_name,
                "distance_sq": distance_sq,
                "normalized_distance": normalized_distance,
                "score": -0.5 * distance_sq / max(temperature, 1e-6),
                "is_unknown": False,
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def _probability_table_from_scores(
    distance_table: pd.DataFrame,
    unknown_threshold: float,
    unknown_scale: float,
    unknown_max: float,
    transition_prior: pd.Series | None = None,
    transition_smoothing: float = 0.0,
) -> pd.DataFrame:
    raw_modeled = softmax(distance_table["score"])
    min_distance = float(distance_table["normalized_distance"].min())
    unknown_probability = unknown_max / (1.0 + np.exp(-(min_distance - unknown_threshold) / max(unknown_scale, 1e-6)))
    unknown_probability = float(np.clip(unknown_probability, 0.0, unknown_max))

    modeled = raw_modeled.copy()
    if transition_prior is not None and transition_smoothing > 0:
        prior = _normalize_probabilities(transition_prior.reindex(modeled.index).fillna(0.0) + 1e-6)
        modeled = (modeled.clip(lower=1e-12) ** (1.0 - transition_smoothing)) * (
            prior.clip(lower=1e-12) ** transition_smoothing
        )
        modeled = _normalize_probabilities(modeled)

    probabilities = distance_table.copy()
    probabilities["probability"] = modeled * (1.0 - unknown_probability)
    unknown_row = pd.DataFrame(
        {
            "distance_sq": [np.nan],
            "normalized_distance": [min_distance],
            "score": [np.nan],
            "is_unknown": [True],
            "probability": [unknown_probability],
        },
        index=["Unknown / Mixed"],
    )
    return (
        pd.concat([probabilities, unknown_row], axis=0)
        .reset_index()
        .rename(columns={"index": "scenario"})
        .sort_values("probability", ascending=False)
    )


def estimate_scenario_probabilities(
    factors: pd.DataFrame,
    scenarios: pd.DataFrame,
    temperature: float = 1.0,
    unknown_threshold: float = 1.25,
    unknown_scale: float = 0.35,
    unknown_max: float = 0.35,
    transition_smoothing: float = 0.35,
) -> ScenarioProbabilityResult:
    """Estimate a soft probability distribution over scenario presets.

    Uses Mahalanobis distance between today's macro feature vector and each
    hand-defined scenario center. Unknown/Mixed receives weight when the latest
    macro state is far from all presets.
    """
    history = macro_feature_history(factors)
    centers = scenario_feature_centers(scenarios)
    common = [c for c in centers.columns if c in history.columns]
    if not common:
        raise ValueError("No common macro features are available for scenario probability estimation")

    history = history[common].dropna(how="all")
    latest = history.dropna(how="all").iloc[-1].reindex(common)

    distance_table = _distance_table(history, latest, centers, temperature)
    raw_probabilities = _probability_table_from_scores(
        distance_table,
        unknown_threshold=unknown_threshold,
        unknown_scale=unknown_scale,
        unknown_max=unknown_max,
    )

    transition_matrix = default_transition_matrix(scenarios)
    transition_prior = pd.Series(1.0 / len(transition_matrix), index=transition_matrix.index, name="transition_prior")
    if transition_smoothing > 0 and history.dropna().shape[0] >= 24:
        previous_history = history.iloc[:-1].dropna()
        if len(previous_history) >= 24:
            previous_latest = previous_history.iloc[-1].reindex(common)
            previous_distance = _distance_table(previous_history, previous_latest, centers, temperature)
            previous_probabilities = _probability_table_from_scores(
                previous_distance,
                unknown_threshold=unknown_threshold,
                unknown_scale=unknown_scale,
                unknown_max=unknown_max,
            )
            previous_modeled = previous_probabilities[~previous_probabilities["is_unknown"]].set_index("scenario")["probability"]
            previous_modeled = _normalize_probabilities(previous_modeled)
            transition_prior = previous_modeled.reindex(transition_matrix.index).fillna(0.0).dot(transition_matrix)
            transition_prior = _normalize_probabilities(pd.Series(transition_prior, index=transition_matrix.columns, name="transition_prior"))

    probabilities = _probability_table_from_scores(
        distance_table,
        unknown_threshold=unknown_threshold,
        unknown_scale=unknown_scale,
        unknown_max=unknown_max,
        transition_prior=transition_prior,
        transition_smoothing=transition_smoothing,
    )

    p = probabilities["probability"].clip(lower=0.0)
    entropy = float(-(p * np.log(p.replace(0, np.nan))).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    confidence = float(np.clip(1.0 - entropy, 0.0, 1.0))

    return ScenarioProbabilityResult(
        probabilities=probabilities,
        raw_probabilities=raw_probabilities,
        transition_prior=transition_prior.rename("transition_prior").reset_index().rename(columns={"index": "scenario"}),
        transition_matrix=transition_matrix,
        latest_features=latest,
        scenario_features=centers,
        feature_history=history,
        entropy=entropy,
        confidence=confidence,
        unknown_probability=float(probabilities.loc[probabilities["is_unknown"], "probability"].sum()),
    )


def walk_forward_scenario_calibration(
    factors: pd.DataFrame,
    scenarios: pd.DataFrame,
    lookback: int = 84,
    horizon: int = 1,
    temperature: float = 1.0,
    max_periods: int = 96,
    unknown_threshold: float = 1.25,
    unknown_scale: float = 0.35,
    unknown_max: float = 0.35,
) -> ScenarioCalibrationResult:
    """Walk-forward calibration against next-period nearest realized macro regime."""
    history = macro_feature_history(factors)
    centers = scenario_feature_centers(scenarios)
    common = [c for c in centers.columns if c in history.columns]
    history = history[common].dropna()
    centers = centers[common]
    rows = []
    start = max(lookback, len(FACTOR_COLUMNS) * 3)
    end = len(history) - horizon
    if end <= start:
        empty = pd.DataFrame()
        return ScenarioCalibrationResult(summary=empty, probability_buckets=empty, recent_predictions=empty)

    indices = range(start, end)
    if max_periods > 0:
        indices = list(indices)[-max_periods:]

    for i in indices:
        train = history.iloc[: i + 1]
        latest = train.iloc[-1]
        realized = history.iloc[i + horizon]
        try:
            score_table = _distance_table(train, latest, centers, temperature)
            probs = _probability_table_from_scores(
                score_table,
                unknown_threshold=unknown_threshold,
                unknown_scale=unknown_scale,
                unknown_max=unknown_max,
            )
            realized_scores = _distance_table(train, realized, centers, temperature)
        except ValueError:
            continue

        realized_min = float(realized_scores["normalized_distance"].min())
        realized_scenario = str(realized_scores["normalized_distance"].idxmin())
        if realized_min > unknown_threshold:
            realized_scenario = "Unknown / Mixed"

        p = probs.set_index("scenario")["probability"]
        all_labels = list(p.index)
        y = pd.Series(0.0, index=all_labels)
        if realized_scenario in y.index:
            y.loc[realized_scenario] = 1.0
        brier = float(((p - y) ** 2).sum())
        assigned = float(p.get(realized_scenario, 0.0))
        modal = str(p.idxmax())
        p_clip = p.clip(lower=1e-12)
        entropy = float(-(p_clip * np.log(p_clip)).sum() / np.log(len(p_clip))) if len(p_clip) > 1 else 0.0
        rows.append(
            {
                "as_of": history.index[i],
                "realized_date": history.index[i + horizon],
                "modal_prediction": modal,
                "realized_scenario": realized_scenario,
                "realized_probability": assigned,
                "brier_score": brier,
                "log_score": float(-np.log(max(assigned, 1e-12))),
                "top_hit": float(modal == realized_scenario),
                "confidence": float(np.clip(1.0 - entropy, 0.0, 1.0)),
                "unknown_probability": float(p.get("Unknown / Mixed", 0.0)),
            }
        )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return ScenarioCalibrationResult(summary=pd.DataFrame(), probability_buckets=pd.DataFrame(), recent_predictions=predictions)

    summary = pd.DataFrame(
        [
            {
                "periods": len(predictions),
                "avg_brier_score": predictions["brier_score"].mean(),
                "avg_log_score": predictions["log_score"].mean(),
                "top_hit_rate": predictions["top_hit"].mean(),
                "avg_realized_probability": predictions["realized_probability"].mean(),
                "avg_confidence": predictions["confidence"].mean(),
                "avg_unknown_probability": predictions["unknown_probability"].mean(),
            }
        ]
    )
    buckets = predictions.copy()
    buckets["probability_bucket"] = pd.cut(
        buckets["realized_probability"],
        bins=[-0.001, 0.05, 0.20, 0.40, 0.60, 0.80, 1.001],
        labels=["0-5%", "5-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )
    bucket_summary = (
        buckets.groupby("probability_bucket", observed=False)
        .agg(
            observations=("realized_probability", "size"),
            avg_assigned_probability=("realized_probability", "mean"),
            hit_rate=("top_hit", "mean"),
            avg_brier_score=("brier_score", "mean"),
        )
        .reset_index()
    )
    return ScenarioCalibrationResult(
        summary=summary,
        probability_buckets=bucket_summary,
        recent_predictions=predictions.sort_values("as_of", ascending=False).head(24),
    )


def probability_weighted_asset_ranking(
    scenario_expected: pd.DataFrame,
    scenario_probabilities: pd.DataFrame,
    diagnostics: pd.DataFrame,
    plausible_probability_floor: float = 0.05,
) -> pd.DataFrame:
    """Blend scenario returns into robust asset rankings.

    `scenario_expected` is expected-return percent by scenario. Unknown/Mixed is
    treated as a zero-return scenario that increases dispersion and lowers
    confidence instead of pretending a precise forecast exists.
    """
    probs = scenario_probabilities.set_index("scenario")["probability"].astype(float)
    scenario_cols = [c for c in scenario_expected.columns if c in probs.index and c != "Unknown / Mixed"]
    if not scenario_cols:
        raise ValueError("No scenario columns match the probability table")

    returns = scenario_expected[scenario_cols].apply(pd.to_numeric, errors="coerce")
    modeled_probs = probs.reindex(scenario_cols).fillna(0.0)
    unknown_prob = float(probs.get("Unknown / Mixed", 0.0))

    weighted_return = returns.mul(modeled_probs, axis=1).sum(axis=1)
    centered_sq = returns.sub(weighted_return, axis=0).pow(2).mul(modeled_probs, axis=1).sum(axis=1)
    if unknown_prob > 0:
        centered_sq = centered_sq + unknown_prob * weighted_return.pow(2)
    scenario_dispersion = np.sqrt(centered_sq)

    plausible_cols = [c for c in scenario_cols if modeled_probs[c] >= plausible_probability_floor]
    if not plausible_cols:
        plausible_cols = list(modeled_probs.sort_values(ascending=False).head(3).index)
    worst_plausible = returns[plausible_cols].min(axis=1)
    downside_loss = (-worst_plausible).clip(lower=0.0)

    top_quartile = pd.DataFrame(index=returns.index)
    regret = pd.Series(0.0, index=returns.index)
    for col in scenario_cols:
        top_quartile[col] = returns[col].rank(ascending=False, pct=True) <= 0.25
        regret = regret + modeled_probs[col] * (returns[col].max() - returns[col])
    rank_stability = top_quartile.mul(modeled_probs, axis=1).sum(axis=1)

    out = scenario_expected[["name", "bucket"]].copy()
    out["weighted_expected_return_pct"] = weighted_return
    out["scenario_dispersion_pct"] = scenario_dispersion
    out["worst_plausible_return_pct"] = worst_plausible
    out["downside_loss_pct"] = downside_loss
    out["rank_stability"] = rank_stability
    out["weighted_regret_pct"] = regret
    out["best_scenario"] = returns.idxmax(axis=1)
    out["worst_scenario"] = returns.idxmin(axis=1)
    out["best_scenario_return_pct"] = returns.max(axis=1)
    out["worst_scenario_return_pct"] = returns.min(axis=1)

    diag_cols = [
        "model_r2",
        "residual_vol_pct",
        "realized_vol_annual_pct",
        "trailing_3m_return_pct",
        "trailing_12m_return_pct",
    ]
    out = out.join(diagnostics.reindex(out.index)[[c for c in diag_cols if c in diagnostics.columns]], how="left")
    residual_penalty = out.get("residual_vol_pct", pd.Series(0.0, index=out.index)).fillna(0.0)
    out["robust_score"] = (
        out["weighted_expected_return_pct"]
        - 0.45 * out["scenario_dispersion_pct"]
        - 0.35 * out["downside_loss_pct"]
        - 0.10 * residual_penalty
    )
    out["fragility_score"] = (
        out["best_scenario_return_pct"]
        - out["weighted_expected_return_pct"]
        + out["scenario_dispersion_pct"]
        + out["downside_loss_pct"]
    )
    return out.sort_values("robust_score", ascending=False)


def optimize_probability_weighted_portfolio(
    scenario_expected: pd.DataFrame,
    scenario_probabilities: pd.DataFrame,
    diagnostics: pd.DataFrame,
    rel_returns: pd.DataFrame,
    universe: pd.DataFrame,
    gross_target: float = 1.0,
    max_abs_weight: float = 0.12,
    max_bucket_abs_weight: float = 0.35,
    risk_aversion: float = 6.0,
    covariance_lookback: int = 60,
) -> PortfolioOptimizationResult:
    """Build a covariance-aware long/short macro tilt from scenario probabilities."""
    probs = scenario_probabilities.set_index("scenario")["probability"].astype(float)
    scenario_cols = [c for c in scenario_expected.columns if c in probs.index and c != "Unknown / Mixed"]
    if not scenario_cols:
        empty = pd.DataFrame()
        return PortfolioOptimizationResult(weights=empty, bucket_weights=empty, stats=empty, covariance=empty)

    expected_pct = scenario_expected[scenario_cols].apply(pd.to_numeric, errors="coerce")
    expected_decimal = expected_pct / 100.0
    modeled_probs = probs.reindex(scenario_cols).fillna(0.0)
    unknown_prob = float(probs.get("Unknown / Mixed", 0.0))
    common_assets = [a for a in expected_decimal.index if a in rel_returns.columns]
    expected_decimal = expected_decimal.loc[common_assets]
    diagnostics = diagnostics.reindex(common_assets)
    if not common_assets:
        empty = pd.DataFrame()
        return PortfolioOptimizationResult(weights=empty, bucket_weights=empty, stats=empty, covariance=empty)

    mu = expected_decimal.mul(modeled_probs, axis=1).sum(axis=1)
    hist = rel_returns[common_assets].tail(covariance_lookback).dropna(how="all")
    hist_cov = hist.cov().reindex(index=common_assets, columns=common_assets).fillna(0.0)

    scenario_diff = expected_decimal.sub(mu, axis=0).fillna(0.0)
    between = scenario_diff.mul(modeled_probs, axis=1).dot(scenario_diff.T)
    if unknown_prob > 0:
        zero_diff = (-mu).to_frame()
        between = between + unknown_prob * zero_diff.dot(zero_diff.T)

    diag_floor = np.nanmedian(np.diag(hist_cov.to_numpy(dtype=float)))
    if not np.isfinite(diag_floor) or diag_floor <= 0:
        diag_floor = 0.0025**2
    cov = hist_cov + between
    cov = cov + pd.DataFrame(np.eye(len(common_assets)) * max(diag_floor * 0.15, 1e-6), index=common_assets, columns=common_assets)

    cov_values = cov.to_numpy(dtype=float)
    mu_values = mu.to_numpy(dtype=float)
    ridge = np.eye(len(common_assets)) * max(float(np.trace(cov_values)) / max(len(common_assets), 1) * 0.25, 1e-6)
    try:
        raw_weights = np.linalg.pinv(cov_values + ridge).dot(mu_values) / max(risk_aversion, 1e-6)
    except np.linalg.LinAlgError:
        raw_weights = mu_values / np.maximum(np.diag(cov_values), 1e-6) / max(risk_aversion, 1e-6)

    raw = pd.Series(raw_weights, index=common_assets).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw[mu.abs() < 0.001] = 0.0
    raw[(raw > 0) & (mu <= 0)] = 0.0
    raw[(raw < 0) & (mu >= 0)] = 0.0
    raw = raw.clip(lower=-max_abs_weight, upper=max_abs_weight)
    gross = float(raw.abs().sum())
    if gross > 0:
        raw = raw * (gross_target / gross)
    raw = raw.clip(lower=-max_abs_weight, upper=max_abs_weight)

    meta = universe.set_index("ticker")[["name", "bucket"]]
    weights = pd.DataFrame({"weight": raw}).join(meta, how="left")
    for _ in range(4):
        for bucket, idx in weights.groupby("bucket").groups.items():
            bucket_gross = float(weights.loc[idx, "weight"].abs().sum())
            if bucket_gross > max_bucket_abs_weight:
                weights.loc[idx, "weight"] *= max_bucket_abs_weight / bucket_gross
        gross = float(weights["weight"].abs().sum())
        if gross > 0:
            weights["weight"] *= gross_target / gross
        weights["weight"] = weights["weight"].clip(lower=-max_abs_weight, upper=max_abs_weight)

    weights = weights[weights["weight"].abs() > 1e-4].copy()
    if weights.empty:
        empty = pd.DataFrame()
        return PortfolioOptimizationResult(weights=empty, bucket_weights=empty, stats=empty, covariance=cov)

    w = weights["weight"].reindex(common_assets).fillna(0.0)
    port_return = float(w.dot(mu.reindex(common_assets).fillna(0.0)))
    port_var = float(w.to_numpy().T @ cov.reindex(index=common_assets, columns=common_assets).to_numpy(dtype=float) @ w.to_numpy())
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    marginal_risk = cov.reindex(index=common_assets, columns=common_assets).dot(w)
    risk_contribution = (w * marginal_risk / port_var) if port_var > 0 else pd.Series(0.0, index=common_assets)

    weights["expected_return_pct"] = mu.reindex(weights.index) * 100.0
    weights["expected_contribution_pct"] = weights["weight"] * mu.reindex(weights.index) * 100.0
    weights["risk_contribution_pct"] = risk_contribution.reindex(weights.index).fillna(0.0) * 100.0
    weights = weights.join(diagnostics[["model_r2", "residual_vol_pct"]], how="left")
    weights["weight_pct"] = weights["weight"] * 100.0
    weights["side"] = np.where(weights["weight"] >= 0, "Long / overweight", "Short / underweight")
    weights["abs_weight"] = weights["weight"].abs()
    weights = weights.sort_values("abs_weight", ascending=False)

    bucket_weights = (
        weights.groupby("bucket", as_index=False)
        .agg(
            net_weight_pct=("weight_pct", "sum"),
            gross_weight_pct=("weight_pct", lambda x: float(np.abs(x).sum())),
            expected_contribution_pct=("expected_contribution_pct", "sum"),
            risk_contribution_pct=("risk_contribution_pct", "sum"),
        )
        .sort_values("gross_weight_pct", ascending=False)
    )
    stats = pd.DataFrame(
        [
            {
                "expected_return_pct": port_return * 100.0,
                "volatility_estimate_pct": port_vol * 100.0,
                "return_to_risk": port_return / port_vol if port_vol > 0 else np.nan,
                "gross_exposure_pct": float(weights["weight"].abs().sum() * 100.0),
                "net_exposure_pct": float(weights["weight"].sum() * 100.0),
                "long_exposure_pct": float(weights.loc[weights["weight"] > 0, "weight"].sum() * 100.0),
                "short_exposure_pct": float(weights.loc[weights["weight"] < 0, "weight"].sum() * 100.0),
                "asset_count": int(len(weights)),
            }
        ]
    )
    return PortfolioOptimizationResult(weights=weights, bucket_weights=bucket_weights, stats=stats, covariance=cov)


def policy_signal(expected_return: pd.Series, conviction: pd.Series) -> pd.Series:
    """Convert model output into a simple portfolio-action label."""
    out = pd.Series("Neutral / watch", index=expected_return.index, dtype=object)
    out[(expected_return > 0) & (conviction >= 0.35)] = "Overweight"
    out[(expected_return > 0) & (conviction >= 0.75)] = "High-conviction overweight"
    out[(expected_return < 0) & (conviction <= -0.35)] = "Underweight / hedge"
    out[(expected_return < 0) & (conviction <= -0.75)] = "High-conviction underweight"
    return out


def summarize_buckets(expected: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scenario attractiveness by asset bucket."""
    rows = []
    for bucket, g in expected.dropna(subset=["bucket"]).groupby("bucket"):
        g_sorted = g.sort_values("expected_return", ascending=False)
        rows.append(
            {
                "bucket": bucket,
                "asset_count": len(g),
                "avg_expected_return_pct": g["expected_return_pct"].mean(),
                "median_expected_return_pct": g["expected_return_pct"].median(),
                "avg_conviction_score": g["conviction_score"].mean(),
                "avg_model_r2": g["model_r2"].mean(),
                "avg_residual_vol_pct": g["residual_vol_pct"].mean(),
                "top_name": g_sorted.iloc[0]["name"],
                "top_expected_return_pct": g_sorted.iloc[0]["expected_return_pct"],
                "bottom_name": g_sorted.iloc[-1]["name"],
                "bottom_expected_return_pct": g_sorted.iloc[-1]["expected_return_pct"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("avg_conviction_score", ascending=False).reset_index(drop=True)


def _diversified_pick(candidates: pd.DataFrame, n: int, max_per_bucket: int) -> pd.DataFrame:
    picked = []
    bucket_counts: dict[str, int] = {}
    for idx, row in candidates.iterrows():
        bucket = str(row.get("bucket", ""))
        if bucket_counts.get(bucket, 0) >= max_per_bucket:
            continue
        picked.append(idx)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if len(picked) >= n:
            break
    return candidates.loc[picked]


def build_trade_basket(expected: pd.DataFrame, n_each: int = 6, max_per_bucket: int = 2) -> pd.DataFrame:
    """Create a diversified scenario long/short playbook from conviction scores."""
    candidates = expected.dropna(subset=["bucket", "expected_return", "conviction_score"]).copy()
    longs = candidates[candidates["expected_return"] > 0].sort_values("conviction_score", ascending=False)
    shorts = candidates[candidates["expected_return"] < 0].sort_values("conviction_score", ascending=True)
    longs = _diversified_pick(longs, n_each, max_per_bucket)
    shorts = _diversified_pick(shorts, n_each, max_per_bucket)

    pieces = []
    if not longs.empty:
        long_weight = 0.5 / len(longs)
        long_table = longs.copy()
        long_table["side"] = "Long / overweight"
        long_table["portfolio_weight"] = long_weight
        pieces.append(long_table)
    if not shorts.empty:
        short_weight = -0.5 / len(shorts)
        short_table = shorts.copy()
        short_table["side"] = "Short / underweight"
        short_table["portfolio_weight"] = short_weight
        pieces.append(short_table)

    if not pieces:
        return pd.DataFrame()

    basket = pd.concat(pieces).copy()
    basket["expected_contribution"] = basket["portfolio_weight"] * basket["expected_return"]
    basket["portfolio_weight_pct"] = basket["portfolio_weight"] * 100.0
    basket["expected_contribution_pct"] = basket["expected_contribution"] * 100.0
    basket["side_order"] = np.where(basket["portfolio_weight"] > 0, 0, 1)
    return basket.sort_values(["side_order", "abs_conviction_score"], ascending=[True, False]).drop(columns=["side_order"])


def fisher_corr_z(r: pd.Series | pd.DataFrame, n: int) -> pd.Series | pd.DataFrame:
    """Approximate z-stat for correlation coefficients using Fisher transform."""
    clipped = r.clip(-0.999999, 0.999999)
    return np.arctanh(clipped) * np.sqrt(max(n - 3, 1))


def build_model(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    universe: pd.DataFrame,
    scenario: pd.Series,
    half_life: float | None = 36.0,
) -> ModelResult:
    returns = log_returns(prices)
    rel = make_relative_returns(returns, universe)
    x_raw = factor_changes(factors)
    xz = zscore(x_raw)

    idx = rel.index.intersection(xz.index)
    rel = rel.loc[idx]
    xz = xz.loc[idx]

    exposures, tstats, alphas, nobs, r2, residual_vol = fit_exposures(rel, xz, half_life=half_life)
    scenario = scenario.reindex(FACTOR_COLUMNS).astype(float)
    contributions = exposures.mul(scenario, axis=1)
    contributions.columns = [f"{c}_contribution" for c in contributions.columns]
    exp_ret = contributions.sum(axis=1).rename("expected_return")

    # Scenario-level confidence proxy: weighted average absolute factor t-stat, weighted by scenario magnitude.
    weights = scenario.abs()
    if weights.sum() == 0:
        confidence = pd.Series(0.0, index=exposures.index, name="confidence_t_proxy")
    else:
        confidence = (tstats.abs() * weights).sum(axis=1) / weights.sum()
        confidence.name = "confidence_t_proxy"

    meta = universe.set_index("ticker")[["name", "bucket"]]
    diagnostics = pd.concat(
        [
            alphas,
            nobs,
            r2,
            residual_vol,
            rel.std(ddof=0).rename("realized_vol"),
            trailing_log_return(rel, 3).rename("trailing_3m_return"),
            trailing_log_return(rel, 12).rename("trailing_12m_return"),
        ],
        axis=1,
    )
    diagnostics["model_r2"] = diagnostics["model_r2"].clip(lower=0.0)
    diagnostics["residual_vol_pct"] = diagnostics["residual_vol"] * 100.0
    diagnostics["realized_vol_annual_pct"] = diagnostics["realized_vol"] * np.sqrt(12.0) * 100.0
    diagnostics["trailing_3m_return_pct"] = diagnostics["trailing_3m_return"] * 100.0
    diagnostics["trailing_12m_return_pct"] = diagnostics["trailing_12m_return"] * 100.0

    expected = pd.concat([exp_ret, confidence], axis=1).join(meta, how="left").join(diagnostics, how="left")
    expected = expected.dropna(subset=["expected_return"])
    expected["expected_return_pct"] = expected["expected_return"] * 100.0
    fit_quality = np.sqrt(expected["model_r2"].clip(lower=0.0, upper=1.0)) * (expected["nobs"] / 84.0).clip(upper=1.0)
    risk_scale = pd.concat(
        [
            expected["residual_vol"],
            expected["realized_vol"] * 0.50,
            pd.Series(0.0025, index=expected.index),
        ],
        axis=1,
    ).max(axis=1)
    expected["risk_scale"] = risk_scale
    expected["risk_adjusted_edge"] = (expected["expected_return"] / risk_scale.replace(0, np.nan)).clip(-10.0, 10.0)
    expected["conviction_score"] = expected["risk_adjusted_edge"] * (0.35 + 0.65 * fit_quality)
    expected["abs_conviction_score"] = expected["conviction_score"].abs()
    expected["policy_signal"] = policy_signal(expected["expected_return"], expected["conviction_score"])
    expected["expected_return_pct"] = expected["expected_return_pct"].replace([np.inf, -np.inf], np.nan)

    corr = pd.concat([rel, xz], axis=1).corr().loc[rel.columns, FACTOR_COLUMNS]
    bucket_summary = summarize_buckets(expected)
    trade_basket = build_trade_basket(expected)
    return ModelResult(
        exposures=exposures,
        tstats=tstats,
        alphas=alphas,
        diagnostics=diagnostics,
        contributions=contributions,
        expected=expected,
        bucket_summary=bucket_summary,
        trade_basket=trade_basket,
        rel_returns=rel,
        factors_z=xz,
        corr=corr,
    )


def _performance_stats(name: str, log_return: pd.Series, periods_per_year: int = 12) -> dict[str, float | str | int]:
    r = log_return.replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {"series": name, "months": 0}

    equity = np.exp(r.cumsum())
    years = len(r) / periods_per_year
    final_equity = float(equity.iloc[-1])
    total_return = final_equity - 1.0
    cagr = final_equity ** (1.0 / years) - 1.0 if years > 0 and final_equity > 0 else np.nan
    vol = float(r.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(r.mean() / r.std(ddof=0) * np.sqrt(periods_per_year)) if r.std(ddof=0) > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "series": name,
        "months": int(len(r)),
        "final_equity": final_equity,
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0 if np.isfinite(cagr) else np.nan,
        "annual_vol_pct": vol * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "hit_rate_pct": float((r > 0).mean() * 100.0),
        "avg_monthly_return_pct": float((np.exp(r.mean()) - 1.0) * 100.0),
        "best_month_pct": float((np.exp(r.max()) - 1.0) * 100.0),
        "worst_month_pct": float((np.exp(r.min()) - 1.0) * 100.0),
    }


def _rebalance_basket_weights(basket: pd.DataFrame, next_returns: pd.Series) -> pd.Series:
    if basket.empty or "portfolio_weight" not in basket.columns:
        return pd.Series(dtype=float)
    weights = basket["portfolio_weight"].astype(float)
    tradable = next_returns.reindex(weights.index).notna()
    weights = weights.loc[tradable]
    if weights.empty:
        return pd.Series(dtype=float)

    longs = weights[weights > 0]
    shorts = weights[weights < 0]
    if longs.empty or shorts.empty:
        return pd.Series(dtype=float)

    balanced = pd.Series(0.0, index=weights.index)
    balanced.loc[longs.index] = longs / longs.sum() * 0.5
    balanced.loc[shorts.index] = shorts / shorts.abs().sum() * 0.5
    return balanced[balanced.abs() > 1e-12]


def _benchmark_diagnostics(returns_table: pd.DataFrame) -> pd.DataFrame:
    data = returns_table[["strategy_return", "spy_return"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return pd.DataFrame()

    strategy = data["strategy_return"]
    benchmark = data["spy_return"]
    active = strategy - benchmark
    benchmark_var = float(benchmark.var(ddof=0))
    beta = float(strategy.cov(benchmark, ddof=0) / benchmark_var) if benchmark_var > 0 else np.nan
    alpha_monthly = float(strategy.mean() - beta * benchmark.mean()) if np.isfinite(beta) else np.nan
    tracking_error = float(active.std(ddof=0) * np.sqrt(12.0))
    information_ratio = float(active.mean() / active.std(ddof=0) * np.sqrt(12.0)) if active.std(ddof=0) > 0 else np.nan
    correlation = float(strategy.corr(benchmark))

    strategy_arith = np.exp(strategy) - 1.0
    benchmark_arith = np.exp(benchmark) - 1.0
    up_mask = benchmark_arith > 0
    down_mask = benchmark_arith < 0
    upside_capture = np.nan
    downside_capture = np.nan
    if up_mask.any() and abs(float(benchmark_arith.loc[up_mask].mean())) > 1e-12:
        upside_capture = float(strategy_arith.loc[up_mask].mean() / benchmark_arith.loc[up_mask].mean() * 100.0)
    if down_mask.any() and abs(float(benchmark_arith.loc[down_mask].mean())) > 1e-12:
        downside_capture = float(strategy_arith.loc[down_mask].mean() / benchmark_arith.loc[down_mask].mean() * 100.0)

    return pd.DataFrame(
        [
            {"metric": "Annualized active return", "value": float((np.exp(active.mean() * 12.0) - 1.0) * 100.0), "unit": "%"},
            {"metric": "Tracking error", "value": tracking_error * 100.0, "unit": "%"},
            {"metric": "Information ratio", "value": information_ratio, "unit": "ratio"},
            {"metric": "Beta to SPY", "value": beta, "unit": "ratio"},
            {"metric": "Correlation to SPY", "value": correlation, "unit": "ratio"},
            {"metric": "Annualized alpha", "value": float((np.exp(alpha_monthly * 12.0) - 1.0) * 100.0) if np.isfinite(alpha_monthly) else np.nan, "unit": "%"},
            {"metric": "Upside capture", "value": upside_capture, "unit": "%"},
            {"metric": "Downside capture", "value": downside_capture, "unit": "%"},
        ]
    )


def _rolling_backtest_metrics(returns_table: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    data = returns_table[["return_date", "strategy_return", "spy_return"]].replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, float | pd.Timestamp]] = []
    if len(data) < window:
        return pd.DataFrame()

    for i in range(window - 1, len(data)):
        sample = data.iloc[i - window + 1 : i + 1]
        strategy = sample["strategy_return"]
        benchmark = sample["spy_return"]
        active = strategy - benchmark
        benchmark_var = float(benchmark.var(ddof=0))
        beta = float(strategy.cov(benchmark, ddof=0) / benchmark_var) if benchmark_var > 0 else np.nan
        rows.append(
            {
                "date": sample["return_date"].iloc[-1],
                "strategy_sharpe": float(strategy.mean() / strategy.std(ddof=0) * np.sqrt(12.0)) if strategy.std(ddof=0) > 0 else np.nan,
                "spy_sharpe": float(benchmark.mean() / benchmark.std(ddof=0) * np.sqrt(12.0)) if benchmark.std(ddof=0) > 0 else np.nan,
                "information_ratio": float(active.mean() / active.std(ddof=0) * np.sqrt(12.0)) if active.std(ddof=0) > 0 else np.nan,
                "beta_to_spy": beta,
                "correlation_to_spy": float(strategy.corr(benchmark)),
                "strategy_cagr_pct": float((np.exp(strategy.mean() * 12.0) - 1.0) * 100.0),
                "spy_cagr_pct": float((np.exp(benchmark.mean() * 12.0) - 1.0) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def _stress_months(returns_table: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    if returns_table.empty:
        return pd.DataFrame()
    cols = [
        "as_of",
        "return_date",
        "modal_scenario",
        "modal_probability",
        "unknown_probability",
        "strategy_return",
        "spy_return",
        "excess_return",
        "turnover",
        "n_assets",
    ]
    stress = returns_table.sort_values("spy_return").head(n)[cols].copy()
    for col in ["modal_probability", "unknown_probability", "turnover"]:
        stress[f"{col}_pct"] = stress[col] * 100.0
    for col in ["strategy_return", "spy_return", "excess_return"]:
        stress[f"{col}_pct"] = (np.exp(stress[col]) - 1.0) * 100.0
    return stress.sort_values("return_date", ascending=False)


def _cost_sensitivity(returns_table: pd.DataFrame, costs_bps: tuple[int, ...] = (0, 5, 10, 25, 50)) -> pd.DataFrame:
    if returns_table.empty or "gross_strategy_return" not in returns_table:
        return pd.DataFrame()
    rows = []
    for bps in costs_bps:
        adjusted = returns_table["gross_strategy_return"] - returns_table["turnover"] * bps / 10000.0
        stats = _performance_stats(f"{bps} bps", adjusted)
        stats["cost_bps"] = bps
        rows.append(stats)
    out = pd.DataFrame(rows)
    return out[
        [
            "cost_bps",
            "months",
            "final_equity",
            "total_return_pct",
            "cagr_pct",
            "annual_vol_pct",
            "sharpe",
            "max_drawdown_pct",
            "hit_rate_pct",
        ]
    ]


def walk_forward_predicted_scenario_portfolio(
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    universe: pd.DataFrame,
    scenarios: pd.DataFrame,
    lookback: int = 84,
    half_life: float | None = 36.0,
    n_each: int = 6,
    max_per_bucket: int = 2,
    transaction_cost_bps: float = 0.0,
) -> PortfolioBacktestResult:
    """Backtest the scenario basket chosen by the walk-forward modal scenario.

    At each month-end, the model uses data available through that date, selects
    the modal non-unknown scenario, builds the diversified long/short basket,
    and applies those weights to the next month's asset returns.
    """
    returns = log_returns(prices)
    common_index = returns.index.intersection(factors.index).sort_values()
    scenario_lookup = scenarios.set_index("scenario")
    meta = universe.set_index("ticker")[["name", "bucket"]]
    rows: list[dict[str, object]] = []
    weight_rows: list[pd.DataFrame] = []
    previous_weights = pd.Series(dtype=float)

    start = max(int(lookback), len(FACTOR_COLUMNS) * 3)
    for i in range(start, len(common_index) - 1):
        as_of = common_index[i]
        next_date = common_index[i + 1]
        price_history = prices.loc[prices.index <= as_of]
        factor_history = factors.loc[factors.index <= as_of]
        next_returns = returns.loc[next_date]

        try:
            probabilities = estimate_scenario_probabilities(factor_history, scenarios)
            modeled = probabilities.probabilities[~probabilities.probabilities["is_unknown"]]
            if modeled.empty:
                continue
            modal_row = modeled.sort_values("probability", ascending=False).iloc[0]
            modal_scenario = str(modal_row["scenario"])
            scenario = scenario_lookup.loc[modal_scenario, FACTOR_COLUMNS].astype(float)
            model = build_model(price_history, factor_history, universe, scenario, half_life=half_life)
            basket = build_trade_basket(model.expected, n_each=n_each, max_per_bucket=max_per_bucket)
        except (ValueError, KeyError, IndexError, np.linalg.LinAlgError):
            continue

        weights = _rebalance_basket_weights(basket, next_returns)
        if weights.empty:
            continue

        aligned_returns = next_returns.reindex(weights.index).dropna()
        weights = weights.reindex(aligned_returns.index).dropna()
        if weights.empty:
            continue

        turnover_index = weights.index.union(previous_weights.index)
        turnover = float((weights.reindex(turnover_index).fillna(0.0) - previous_weights.reindex(turnover_index).fillna(0.0)).abs().sum() * 0.5)
        cost_return = turnover * transaction_cost_bps / 10000.0
        gross_strategy_return = float(weights.dot(aligned_returns))
        strategy_return = float(gross_strategy_return - cost_return)
        spy_return = float(next_returns.get("SPY", np.nan))

        rows.append(
            {
                "as_of": as_of,
                "return_date": next_date,
                "modal_scenario": modal_scenario,
                "modal_probability": float(modal_row["probability"]),
                "unknown_probability": float(probabilities.unknown_probability),
                "confidence": float(probabilities.confidence),
                "gross_strategy_return": gross_strategy_return,
                "strategy_return": strategy_return,
                "spy_return": spy_return,
                "excess_return": strategy_return - spy_return if np.isfinite(spy_return) else np.nan,
                "gross_exposure": float(weights.abs().sum()),
                "net_exposure": float(weights.sum()),
                "turnover": turnover,
                "cost_return": cost_return,
                "n_assets": int(len(weights)),
                "n_longs": int((weights > 0).sum()),
                "n_shorts": int((weights < 0).sum()),
            }
        )

        weight_detail = weights.rename("weight").to_frame()
        weight_detail["as_of"] = as_of
        weight_detail["return_date"] = next_date
        weight_detail["modal_scenario"] = modal_scenario
        weight_detail["side"] = np.where(weight_detail["weight"] > 0, "Long", "Short")
        weight_detail = weight_detail.join(meta, how="left")
        weight_rows.append(weight_detail.reset_index().rename(columns={"index": "ticker"}))
        previous_weights = weights

    returns_table = pd.DataFrame(rows)
    if returns_table.empty:
        empty = pd.DataFrame()
        return PortfolioBacktestResult(
            returns=empty,
            equity=empty,
            drawdowns=empty,
            summary=empty,
            weights=empty,
            scenario_counts=empty,
            benchmark_diagnostics=empty,
            rolling_metrics=empty,
            stress_months=empty,
            cost_sensitivity=empty,
        )

    returns_table = returns_table.sort_values("return_date").reset_index(drop=True)
    equity = pd.DataFrame(
        {
            "Predicted scenario portfolio": np.exp(returns_table["strategy_return"].fillna(0.0).cumsum()).to_numpy(),
            "SPY": np.exp(returns_table["spy_return"].fillna(0.0).cumsum()).to_numpy(),
            "Excess vs SPY": np.exp(returns_table["excess_return"].fillna(0.0).cumsum()).to_numpy(),
        },
        index=returns_table["return_date"],
    )
    equity.index.name = "date"
    drawdowns = equity / equity.cummax() - 1.0
    drawdowns = drawdowns * 100.0

    summary = pd.DataFrame(
        [
            _performance_stats("Predicted scenario portfolio", returns_table["strategy_return"]),
            _performance_stats("SPY", returns_table["spy_return"]),
            _performance_stats("Excess vs SPY", returns_table["excess_return"]),
        ]
    )
    if returns_table[["strategy_return", "spy_return"]].dropna().shape[0] > 2:
        corr = float(returns_table[["strategy_return", "spy_return"]].corr().iloc[0, 1])
        summary.loc[summary["series"] == "Predicted scenario portfolio", "correlation_to_spy"] = corr
    summary["avg_turnover_pct"] = np.nan
    summary.loc[summary["series"] == "Predicted scenario portfolio", "avg_turnover_pct"] = float(returns_table["turnover"].mean() * 100.0)

    scenario_counts = (
        returns_table.groupby("modal_scenario", as_index=False)
        .agg(
            months=("strategy_return", "size"),
            avg_modal_probability_pct=("modal_probability", lambda x: float(x.mean() * 100.0)),
            avg_strategy_return_pct=("strategy_return", lambda x: float((np.exp(x.mean()) - 1.0) * 100.0)),
            avg_spy_return_pct=("spy_return", lambda x: float((np.exp(x.mean()) - 1.0) * 100.0)),
            hit_rate_pct=("strategy_return", lambda x: float((x > 0).mean() * 100.0)),
        )
        .sort_values("months", ascending=False)
    )
    weights_table = pd.concat(weight_rows, ignore_index=True) if weight_rows else pd.DataFrame()
    benchmark_diagnostics = _benchmark_diagnostics(returns_table)
    rolling_metrics = _rolling_backtest_metrics(returns_table)
    stress_months = _stress_months(returns_table)
    cost_sensitivity = _cost_sensitivity(returns_table)
    return PortfolioBacktestResult(
        returns=returns_table,
        equity=equity,
        drawdowns=drawdowns,
        summary=summary,
        weights=weights_table,
        scenario_counts=scenario_counts,
        benchmark_diagnostics=benchmark_diagnostics,
        rolling_metrics=rolling_metrics,
        stress_months=stress_months,
        cost_sensitivity=cost_sensitivity,
    )


def top_bottom_by_bucket(expected: pd.DataFrame, n: int = 3) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for bucket, g in expected.dropna(subset=["bucket"]).groupby("bucket"):
        g = g.sort_values("expected_return", ascending=False)
        cols = ["name", "expected_return_pct", "confidence_t_proxy"]
        out[bucket] = {"top": g.head(n)[cols], "bottom": g.tail(n).sort_values("expected_return")[cols]}
    return out


def compare_scenarios(exposures: pd.DataFrame, universe: pd.DataFrame, scenario_a: pd.Series, scenario_b: pd.Series) -> pd.DataFrame:
    a = exposures.dot(scenario_a.reindex(FACTOR_COLUMNS).astype(float))
    b = exposures.dot(scenario_b.reindex(FACTOR_COLUMNS).astype(float))
    diff = (a - b).rename("difference")
    meta = universe.set_index("ticker")[["name", "bucket"]]
    out = pd.concat([a.rename("scenario_a"), b.rename("scenario_b"), diff], axis=1).join(meta, how="left")
    out["difference_pct"] = out["difference"] * 100.0
    return out.sort_values("difference", ascending=False)
