# Macro Scenario Dashboard Starter

This dashboard provides a macro scenario analysis workflow with:

- macro scenario sliders / presets
- exposure matrix across asset buckets
- top/bottom expected outperformers
- correlation heatmap
- relative returns by benchmark bucket
- recency-weighted regressions
- investor-facing bucket tilts, long/short playbook, factor attribution, scenario stress map, and model diagnostics
- automatic scenario probabilities, unknown/mixed regime detection, probability-weighted asset rankings, robustness, downside, regret, and fragility diagnostics

It will not reproduce the exact proprietary GMI/MIT output unless you have the same historical data, factor definitions, scenario presets, and weighting rules.
The dashboard now refreshes public monthly Yahoo/FRED data by default. Treat the macro factors as transparent proxies and validate the proxy choices before using the outputs for capital allocation.

## Files

```text
app.py                 Streamlit dashboard
model.py               Core calculation engine
sample_data.py         Generates synthetic data so the app can run immediately
requirements.txt       Python dependencies
config/universe.csv    Asset universe, bucket, benchmark mapping
config/scenarios.csv   Editable scenario presets
data/prices.csv        Public monthly market/ETF/crypto price history after refresh
data/factors.csv       Public monthly macro proxy history after refresh
data/update_state.json Public-data updater metadata
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

On startup the app runs `update_data.py`, downloads public monthly Yahoo/FRED proxy data, and appends or refreshes `data/prices.csv` and `data/factors.csv`. The first updater run replaces the synthetic starter CSVs with a full public-history refresh; later runs keep existing history and rewrite only the recent overlapping months plus new completed months.

Manual update commands:

```bash
python update_data.py
python update_data.py --force-full-refresh
python update_data.py --append-only
```

Disable startup refresh when needed:

```bash
MACRO_DASHBOARD_AUTO_UPDATE=0 streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, choose that repository and set the main file path to `app.py`.
3. Keep `requirements.txt`, `config/`, and `data/` in the repository. The app can refresh public Yahoo/FRED proxy data on startup, but the committed CSVs make the first load deterministic.
4. Do not add private data, API keys, broker credentials, or paid data exports to this repository. If a future version needs secrets, put them in Streamlit secrets, not in the files.

## Bring-your-own data format

You can replace `data/prices.csv` and `data/factors.csv` with your own verified monthly data if you want to move beyond the default public proxies.

`prices.csv` should be wide format:

```text
date,ACWI,SPY,EEM,EFA,XLK,XLE,...
2008-01-31,100.0,100.0,100.0,100.0,100.0,100.0,...
```

`factors.csv` should be wide format with these columns:

```text
date,risk,growth,inflation,rates,liquidity,dollar,oil
2008-01-31,100.0,48.2,3.1,3.2,100.0,92.0,78.0
```

Recommended monthly factor definitions:

- `risk`: broad risk proxy, for example global equities versus government bonds, or another risk appetite index.
- `growth`: ISM/PMI composite or global leading-indicator composite.
- `inflation`: CPI/breakeven/commodity inflation composite.
- `rates`: 2-year yield.
- `liquidity`: global net liquidity proxy.
- `dollar`: DXY/BBDXY or broad USD index.
- `oil`: WTI or Brent.

## Model

For each asset series:

1. Compute monthly log return.
2. Convert to relative return when appropriate:
   - asset classes: absolute return by default
   - equity regions/sectors/style: relative to global equities or configured benchmark
   - fixed income: relative to configured bond benchmark
   - commodities: relative to configured commodity benchmark
   - FX: return versus USD or configured proxy
3. Standardize factor changes.
4. Estimate recency-weighted OLS:

```text
relative_return(asset) = alpha + beta_1*risk + ... + beta_7*oil + error
```

5. Scenario expected return:

```text
expected_return = beta dot scenario_vector
```

6. Rank by expected return and display top/bottom names with a simple t-stat confidence proxy.
7. Convert expected return into conviction using residual volatility, model fit, and observation count.
8. Build bucket tilts and a diversified long/short playbook from the highest-conviction positive and negative names.

## Auto Regime Engine

The automatic regime view does not choose the nicest portfolio. It first looks only at current macro factors, then assigns probabilities to the preset scenarios.

1. Build macro state features from the factor data:
   - latest standardized shock
   - 3-month trend
   - 6-month trend
   - acceleration
2. Expand each scenario preset into the same feature space.
3. Measure Mahalanobis distance from today's macro state to each scenario.
4. Convert distances into soft probabilities.
5. Add `Unknown / Mixed` probability when today's macro state is far from every preset.

The probability-weighted ranking then blends asset expected returns across the scenario distribution:

```text
weighted_expected_return(asset) =
    sum probability(scenario) * expected_return(asset | scenario)
```

The robust ranking penalizes assets that only work in one narrow scenario:

```text
robust_score =
    weighted_expected_return
  - scenario_dispersion_penalty
  - downside_penalty
  - residual_risk_penalty
```

The dashboard also reports:

- `rank_stability`: probability that an asset ranks in the top quartile across plausible scenarios.
- `weighted_regret`: average distance from the best asset in each scenario.
- `fragility_score`: high upside in one scenario combined with high dispersion or downside.

## Implemented improvement layers

1. Data freshness and source transparency: add a visible per-source update table, expose the latest completed data month, and add a manual refresh control for the dashboard.
2. Probability calibration: walk-forward test the scenario probabilities, tune temperature, and report Brier/log score plus calibration by probability bucket.
3. Scenario structure: split the flat preset list into core growth/inflation regimes plus policy/liquidity and stress overlays.
4. Transition smoothing: add monthly transition priors so the automatic regime probabilities do not flip too aggressively on one noisy macro print.
5. Portfolio construction: move beyond long/short rankings by adding mixture covariance, asset caps, bucket caps, and downside-aware robust rankings.
6. Visualization and design quality: standardize colors, improve contrast, make positive/negative/unknown states visually consistent, tighten chart labels, round table values everywhere, and run desktop/mobile screenshot checks before relying on the dashboard.
