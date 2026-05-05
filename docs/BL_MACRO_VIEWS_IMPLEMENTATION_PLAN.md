# BL-Ready Automatic Macro Views Implementation Plan

Date: 2026-05-05

Purpose: turn the current Streamlit macro research dashboard into a CIO-to-PM automatic macro views pipeline that produces auditable Black-Litterman inputs.

This plan incorporates the GPT Pro validation pass. The main correction from that review is sequencing: the formal BL contract must come first. Forecast-model upgrades are useful, but they should not block the minimum viable CIO deliverable.

## Executive Verdict

The existing dashboard is already useful as a macro regime, scenario, ranking, and diagnostics tool. It is not yet a true CIO "automatic macro views" system because it does not produce formal, portfolio-construction-ready view objects.

The critical deliverable is:

```text
generate_bl_inputs(as_of, horizon)
  -> asset_order
  -> P
  -> q
  -> Omega
  -> view_rationale
  -> status
```

Once this function exists, passes tests, and appears in the Streamlit app as a CIO review table, the dashboard becomes a genuine automatic macro views pipeline rather than another macro research dashboard.

## GPT Pro Validation Summary

GPT Pro approved the direction but made these changes mandatory:

1. Move the BL schema, unit conventions, matrix contract, and tests to the front.
2. Make the MVP convert the existing model into valid `P / q / Omega` views before adding Elastic Net or Bayesian Ridge.
3. Treat 6-month views as the primary horizon; 12-month views should be lower confidence until validation is stronger.
4. Enforce hard no-lookahead rules for 6m/12m labels.
5. Never use full-sample z-scores in walk-forward forecasts.
6. Lag or disclose public macro data because FRED-style series have release lags and revisions.
7. Distinguish absolute views from active views. An active view must subtract the benchmark in `P`.
8. Do not over-expand the asset universe, pair universe, feature set, or UI before the BL contract is working.

## Final Priority Order

1. BL unit contract and configuration.
2. Asset master, benchmark weights, and approved view pairs.
3. 6m forward-return labels with no-lookahead enforcement.
4. Forecast table using existing model plus simple baseline.
5. Confidence and `Omega`.
6. Formal BL view object table.
7. `P/q/Omega` matrix export.
8. Minimal BL posterior engine.
9. Streamlit CIO Views page.
10. View-level validation.
11. Regularized model upgrade.
12. Documentation and PM handoff.

Do not start with Elastic Net, extra scenarios, broader asset coverage, or UI polish.

## Core Design Principles

The implementation should preserve the existing app and add a new BL layer beside it. The current pages, recency-weighted OLS model, probability engine, optimizer diagnostics, and data audit should remain intact.

The BL layer should be narrow and institutional:

```text
macro data
  -> point-in-time features
  -> 6m/12m forecasts
  -> confidence and forecast error
  -> BL view object
  -> P, q, Omega
  -> CIO review
  -> BL posterior returns
  -> PM export
```

Everything should be explicit:

- `q` units.
- `Omega` units.
- forecast horizon.
- covariance horizon.
- return basis.
- benchmark basis.
- `tau` assumption.
- data freshness gate.
- research-only caveats.

## BL Unit Contract for MVP

The MVP must use one internally consistent return convention per BL run.

Primary convention:

```text
horizon_months = 6
return_basis = simple_excess_return or active_return_vs_benchmark
q = 6-month expected excess or active return
Sigma = 6-month covariance matrix
pi = 6-month implied equilibrium return
Omega = 6-month view error variance
```

If covariance is estimated from monthly returns, scale it into the BL horizon:

```text
Sigma_6m ~= 6 * Sigma_monthly
Sigma_12m ~= 12 * Sigma_monthly
```

This approximation is acceptable for the student MVP if documented. It is not acceptable to mix:

```text
q = 6-month return
Sigma = monthly covariance
pi = annualized return
Omega = 6-month error variance
```

Allowed `return_basis` values:

```text
simple_total_return
log_total_return
simple_excess_return
log_excess_return
active_return_vs_benchmark
relative_spread_return
```

For v1, prefer:

- Relative views: `relative_spread_return`.
- Active views: `active_return_vs_benchmark`.
- Absolute views: `simple_excess_return` only when the benchmark/excess-return basis is clear.

## Phase A: BL Contract and Configuration

Goal: define exactly what the system must output before changing the model.

Create these files:

```text
config/asset_master.csv
config/benchmark_weights.csv
config/view_pairs.csv
config/bl_settings.csv
```

### `config/asset_master.csv`

Columns:

```text
ticker
asset_name
asset_class
sleeve
sub_sleeve
currency
benchmark_id
is_investable
is_bl_eligible
is_macro_indicator_only
view_eligible_absolute
view_eligible_relative
min_weight
max_weight
allow_short
view_group
bl_group
covariance_group
return_source
return_type
min_history_months
max_missing_pct
stale_after_days
inception_date
notes
```

Initial policy:

- Include only practical GSMIF / ETF sleeves by default.
- Mark crypto as research-only unless manually approved.
- Mark stale/flatline assets as ineligible.
- Mark obscure commodity proxies as research-only unless they are explicitly part of the fund universe.
- Keep macro indicator proxies out of BL views.
- Use `is_bl_eligible` as the hard gate for automatic BL views.
- Use `view_eligible_absolute` and `view_eligible_relative` to avoid treating all assets as valid in all view types.
- Use `return_type`, `min_history_months`, and `stale_after_days` as mechanical quality gates.

### `config/benchmark_weights.csv`

Columns:

```text
benchmark_id
as_of
ticker
weight
source
is_placeholder
notes
```

Rules:

- Weights must sum to `1.0` within each `benchmark_id` / `as_of`.
- If actual GSMIF holdings are unavailable, use placeholder benchmark weights and label them clearly.
- The benchmark can be a single ticker or a basket.
- Placeholder rows must set `is_placeholder = TRUE` and use a source such as `sample benchmark, replace with GSMIF current holdings`.

### `config/view_pairs.csv`

Columns:

```text
view_pair_id
long_ticker
short_ticker
pair_type
sleeve
cross_sleeve_allowed
is_approved
min_q_threshold
min_confidence
max_q_abs
notes
```

Initial approved examples:

```text
XLE > XLY
IEF > HYG
EFA > SPY
GLD > SPY
XLU > XLY
```

No broad pairwise generation in v1. Use only approved pairs.

### `config/bl_settings.csv`

Columns:

```text
setting
value
notes
```

Required settings:

```text
horizon_months,6
return_basis,active_return_vs_benchmark
covariance_frequency,monthly
covariance_lookback_months,60
covariance_method,shrunk_sample
tau,0.05
risk_aversion_delta,2.5
confidence_floor,0.25
confidence_cap,0.75
omega_floor,0.0001
omega_cap,0.2500
q_absolute_cap,0.1500
q_relative_cap,0.1000
min_train_observations,60
walkforward_embargo_months,6
unknown_mixed_reduce_threshold,0.40
unknown_mixed_block_threshold,0.60
max_views_per_run,10
macro_feature_lag_months,1
```

Acceptance criteria:

- BL asset universe is deterministic.
- Benchmark weights sum to `1.0`.
- Every view asset exists in `data/prices.csv`.
- Research-only assets are excluded from active BL views.
- Return basis and horizon are explicit.
- `tau` is explicit and documented.

Implementation functions:

```python
get_bl_asset_order()
load_benchmark_weights()
load_view_pairs()
load_bl_settings()
validate_bl_config()
```

## Phase B: Horizon Labels and No-Lookahead Forecast Dataset

Goal: create valid 6m and 12m training labels.

Implement:

```python
build_forward_return_labels(prices, horizon_months)
build_active_return_labels(prices, benchmark_weights, horizon_months)
build_forecast_training_panel(prices, factors, asset_master, as_of, horizon_months)
```

Primary target:

```text
6m active return versus benchmark
```

Secondary target:

```text
12m active return versus benchmark
```

Also support absolute total return, but do not confuse it with active return.

### No-Lookahead Rule

When forecasting as of date `T`, a training row with feature date `s` is eligible only if:

```text
s + horizon_months <= T
```

For example, if `as_of = 2026-03-31`:

```text
6m model latest eligible training feature date: 2025-09-30
12m model latest eligible training feature date: 2025-03-31
```

Rows after those dates may be scored, but they cannot be used as completed training labels.

### Feature Standardization Rule

Do not use full-sample z-scores in walk-forward forecasting.

Bad:

```text
z-score using full-sample mean and std through 2026
```

Good:

```text
z-score using expanding or rolling history available as of each date
```

This applies to:

- macro factor shocks.
- scenario probabilities.
- asset momentum.
- asset volatility.
- residual volatility.

### Macro Release-Lag Rule

For v1, lag macro features by one month and disclose that public proxy data may contain revision bias.

Acceptance criteria:

- No training row uses a label whose realized return window ends after `as_of`.
- Forecast rows and training rows are clearly separated.
- 6m labels are available and usable.
- 12m labels are available but marked lower confidence until validated.
- All feature standardization is point-in-time safe.

## Phase C: Minimal Forecast Engine

Goal: produce auditable 6m forecasts without overbuilding.

Implement initially:

```text
historical mean baseline
existing OLS macro model adapted to 6m active-return target
optional Ridge model
```

Defer Elastic Net and Bayesian Ridge until the end-to-end BL pipeline works.

Output:

```text
data/model_predictions.csv
```

Suggested columns:

```text
prediction_id
as_of
ticker
horizon_months
target_type
benchmark_id
benchmark_ticker
forecast_return_raw
forecast_return_shrunk
forecast_return_final
return_basis
model_name
model_version
feature_version
training_start
training_end
n_train
selected_features_json
top_driver_contrib_json
forecast_error_std
forecast_error_var
validation_ic
validation_hit_rate
validation_spread
validation_rmse
validation_mae
feature_stability_score
data_freshness_flag
created_at
```

Acceptance criteria:

- One row per eligible asset / horizon / target type.
- Forecast return exists.
- Forecast error exists.
- Validation metrics exist.
- Model version exists.
- Selected features or drivers are recorded.

## Phase D: Confidence and Omega

Goal: convert forecast quality into BL view uncertainty.

Confidence is not a probability that the view is correct. It is a model confidence score used to scale view uncertainty.

Use this language:

```text
confidence = 0.62 means this view receives lower Omega than a weak view, not that it has a 62% probability of being correct.
```

Initial confidence components:

```text
base_signal = abs(forecast_return_final) / forecast_error_std
base_confidence = scaled function of base_signal
validation_adjustment
data_quality_penalty
unknown_mixed_regime_penalty
fragility_penalty
```

Keep other diagnostics visible, but do not bury too many arbitrary ingredients inside the v1 score.

Cap confidence:

```text
confidence_floor <= confidence <= confidence_cap
```

Recommended v1 range:

```text
0.25 to 0.75
```

Omega rule:

```text
Omega_i = forecast_error_variance_i / confidence_i
```

Requirements:

- `forecast_error_variance_i` must be in the same horizon units as `q`.
- `Omega` must be positive.
- `Omega` must be floored and capped.
- Higher confidence lowers `Omega`.
- Higher forecast error raises `Omega`.

For relative views, use the variance of the relative forecast error:

```text
Omega_A_B = Var(error_A - error_B)
```

Approximation for v1:

```text
Omega_A_B = Var(error_A) + Var(error_B)
```

Better later:

```text
Omega_A_B = Var(error_A) + Var(error_B) - 2 * Cov(error_A, error_B)
```

Acceptance criteria:

- `Omega` positive.
- `Omega` same horizon units as `q`.
- Blocked views do not enter `Omega`.
- Higher confidence lowers `Omega`.
- Higher forecast error raises `Omega`.

## Phase E: BL View Object Generation

Goal: generate formal view objects.

Implement:

```python
generate_absolute_views()
generate_active_views()
generate_relative_views()
build_P_q_Omega()
export_bl_inputs()
```

Use approved pairs only.

Output:

```text
data/bl_macro_views.csv
exports/bl_asset_order.csv
exports/P_matrix.csv
exports/q_vector.csv
exports/Omega_matrix.csv
exports/bl_views.json
```

### View Object Schema

Columns:

```text
view_id
as_of
horizon_months
view_type
sleeve
assets
p_vector_json
p_l1_norm
q_expected_return
q_return_basis
q_units
confidence_score
omega
omega_method
omega_units
forecast_model
forecast_ids
forecast_error_std
validation_ic
validation_hit_rate
validation_spread
scenario_context_json
top_drivers_json
rationale_short
risks
status
block_reason
expiry_date
model_version
data_version
generated_at
```

Statuses:

```text
Candidate
Needs Review
Blocked
```

Do not show a non-persistent `Approved manually` state in v1. If manual approval is added later, persist it in:

```text
data/view_overrides.csv
```

### Absolute Views

For absolute views:

```text
P = {asset: +1}
q = expected total return over horizon
```

Only allow if `view_eligible_absolute = true`.

### Active Views

For active views:

```text
asset expected to outperform benchmark by q
```

Do not set:

```text
P = {asset: +1}
```

Instead:

```text
P = {asset: +1, benchmark: -1}
```

If benchmark is a basket:

```text
P = +asset - benchmark_weighted_basket
```

Example:

```text
P = +XLE - GSMIF_US_EQUITY_BENCHMARK
```

This is critical. Otherwise absolute and active views are mixed incorrectly.

### Relative Views

For relative views:

```text
P = {asset_a: +1, asset_b: -1}
q = forecast_return_asset_a - forecast_return_asset_b
```

Only approved pairs in `config/view_pairs.csv` are eligible.

Acceptance criteria:

- Each candidate view has `P`, `q`, `confidence`, `Omega`, rationale, and status.
- `P/q/Omega` round-trip into BL engine.
- No stale, research-only, macro-indicator, crypto, or flatline asset appears in active views unless explicitly eligible.
- No-view case produces an explicit explanation.

## Phase F: Minimal Black-Litterman Engine

Goal: prove the view objects are mathematically usable.

Implement:

```python
compute_covariance_matrix()
compute_implied_equilibrium_returns()
run_black_litterman()
compute_posterior_diagnostics()
```

Outputs:

```text
data/bl_runs.csv
data/bl_posterior_returns.csv
```

### `data/bl_runs.csv`

Columns:

```text
run_id
as_of
horizon_months
benchmark_id
num_assets
num_views
tau
risk_aversion_delta
covariance_method
return_basis
prior_return_json
posterior_return_json
active_weight_suggestion_json
status
created_at
notes
```

### `data/bl_posterior_returns.csv`

Columns:

```text
run_id
ticker
prior_return
posterior_return
posterior_minus_prior
benchmark_weight
suggested_weight
suggested_active_weight
```

Required inputs:

```text
asset_order
Sigma
benchmark_weights
delta
tau
pi
P
q
Omega
```

BL formula:

```text
posterior_mean =
  inv(inv(tau * Sigma) + P.T @ inv(Omega) @ P)
  @ (inv(tau * Sigma) @ pi + P.T @ inv(Omega) @ q)
```

Use a numerically stable implementation. Add small diagonal regularization if needed.

Acceptance criteria:

- Posterior returns are produced.
- Zero-view case returns prior.
- High-Omega views have low posterior impact.
- Low-Omega views have higher posterior impact.
- Relative views move assets in the correct direction.
- Dimensions match: `P` is `k x n`, `q` is `k`, `Omega` is `k x k`, `Sigma` is `n x n`.

Do not spend time on complex portfolio optimization yet.

## Phase G: Streamlit CIO Views Page

Goal: make the output usable by CIO and PM.

Add a new page:

```text
CIO Views
```

Put it near the top of `VIEW_NAMES`, after `Auto Regime` or before `Investment Brief`.

Default sections:

```text
Readiness strip
Top candidate views
Blocked / Needs Review views
Formal P/q/Omega table
BL posterior impact
Export buttons
Methodology expander
```

The page should show decision objects, not raw research tables.

### Readiness Strip

Metrics:

```text
Data month
BL eligible assets
Candidate views
Blocked views
Unknown/Mixed probability
BL status
Primary horizon
```

### Candidate View Cards

Each card should show:

```text
view sentence
q
confidence
Omega
top drivers
risk flags
status
```

### Formal Table

Columns:

```text
View Type
Assets
Horizon
P Vector
q
Confidence
Omega
Rationale
Risks
Status
```

### Export Buttons

Offer:

```text
CSV: bl_macro_views.csv
JSON: bl_views.json
CSV: P_matrix.csv
CSV: q_vector.csv
CSV: Omega_matrix.csv
```

Acceptance criteria:

- Page renders without errors.
- CSV/JSON export works.
- User can see why each view exists.
- User can see why each blocked view was blocked.
- Posterior impact is visible.

## Phase H: View-Level Validation

Goal: validate the actual view object, not only generic rankings.

Implement:

```python
historical_view_generation_backtest()
evaluate_realized_view_returns()
confidence_bucket_calibration()
```

Report:

```text
number of historical views
realized return by view type
hit rate by view type
average q versus realized return
performance by confidence bucket
blocked-view statistics
```

This matters more than another optimizer backtest.

Acceptance criteria:

- Historical generated views table exists.
- Realized view returns are measured.
- Hit rate by horizon is shown.
- Average realized return by confidence bucket is shown.
- View validation uses the same `P/q/Omega` view generation logic used live.

## Phase I: Regularized Model Upgrade

Goal: improve forecasts after the BL pipeline works.

Add later:

```text
Elastic Net / LASSO feature selection
Ridge or Bayesian Ridge forecast model
model comparison table
feature stability table
```

Do not make this a prerequisite for the MVP. The MVP can use the existing OLS macro engine and simple baselines as long as it produces valid BL views with honest confidence and validation.

Acceptance criteria:

- New model beats or matches baseline out-of-sample by horizon.
- Feature selection is stable enough to explain.
- Forecast error is recorded.
- Model version is recorded.
- If model is weak, it lowers confidence rather than forcing fake precision.

## Phase J: Documentation and PM Handoff

Documentation should answer:

```text
What assets are eligible?
What data is used?
What horizon is forecast?
How are q and Omega created?
How is P constructed?
How does the BL posterior use the views?
Where does human CIO judgment enter?
What are the limitations?
```

Add to `README.md`:

```text
How automatic macro views become BL inputs
```

Add or update:

```text
docs/BL_MACRO_VIEWS_IMPLEMENTATION_PLAN.md
docs/BL_METHOD_NOTES.md
```

Short PM-facing output example:

```text
View:
Over 6 months, XLE is expected to outperform XLY by +4.0%.

BL inputs:
P = {XLE: +1, XLY: -1}
q = 0.040
confidence = 0.62
Omega = 0.0011

Rationale:
Oil and inflation shocks remain supportive for Energy relative to Consumer Discretionary; historical validation is positive; regime uncertainty is moderate.

Status:
Candidate, CIO review required.
```

## 3-5 Day MVP Build Plan

If time is limited, build only the path that proves the CIO requirement.

### Day 1: BL Contract

- Add `asset_master.csv`, `benchmark_weights.csv`, `view_pairs.csv`, `bl_settings.csv`.
- Implement config loaders and validators.
- Add unit tests for asset order, benchmark weights, and eligible assets.

### Day 2: Labels and Forecasts

- Implement 6m forward active-return labels.
- Enforce no-lookahead training eligibility.
- Produce `data/model_predictions.csv` using existing model plus baseline.
- Add point-in-time feature standardization or explicitly restrict the first MVP to current forecast plus historical validation where leakage is avoided.

### Day 3: Confidence, Omega, and Views

- Implement confidence scoring.
- Implement `Omega`.
- Implement absolute, active, and approved-pair relative view generation.
- Export `P`, `q`, `Omega`, and view JSON.
- Add tests for active benchmark `P`, relative `P`, and blocked assets.

### Day 4: Minimal BL Engine and CIO Views Page

- Implement BL posterior calculation.
- Add zero-view and high/low-Omega tests.
- Add Streamlit `CIO Views` page.
- Add export buttons.

### Day 5: Validation and Handoff

- Add view-level historical validation table.
- Add blocked-view statistics.
- Update README and method notes.
- Run AppTest across all pages.
- Verify Streamlit render locally.

## Tests Required Before Merge

### BL Matrix Contract Tests

```text
P shape = K x N
q shape = K
Omega shape = K x K
Sigma shape = N x N
pi shape = N
P columns match asset_order
Omega is positive diagonal or positive definite
Sigma is positive semidefinite after shrinkage
```

### Unit Consistency Tests

```text
q horizon == Sigma horizon == pi horizon == Omega horizon
q return_basis == pi return_basis
Omega units match q units
```

These tests must fail if, for example, `q` is a 6-month return while `Sigma` is monthly covariance or `pi` is annualized.

### No-Lookahead Tests

```text
For horizon H and as_of T:
training feature date s is allowed only if s + H <= T
```

Specific fixtures:

```text
6m as_of 2026-03-31 latest training feature date <= 2025-09-30
12m as_of 2026-03-31 latest training feature date <= 2025-03-31
changing prices after as_of does not change generated forecasts or views
```

### P-Vector Tests

```text
absolute view, asset order [SPY, XLE, XLY], XLE absolute -> [0, 1, 0]
relative view, XLE > XLY -> [0, 1, -1]
active view, XLE vs SPY -> [-1, 1, 0]
active view vs benchmark basket -> +asset - benchmark_weights
```

### Omega Behavior Tests

```text
Omega > 0
Omega is floored
Omega is capped
higher confidence lowers Omega
higher forecast error raises Omega
Unknown/Mixed penalty lowers confidence
Blocked views do not enter P/q/Omega
```

### Export Round-Trip Test

```text
generated bl_views.csv/json
  -> loaded back into Python
  -> reconstructs P, q, Omega
  -> runs BL posterior without Streamlit
```

### Data Eligibility Tests

```text
stale assets cannot generate automatic views
research-only assets cannot generate BL views
crypto cannot generate BL views unless explicitly eligible
flatline assets are blocked
assets with insufficient history are blocked
```

### Unit Tests

```text
test_bl_config_weights_sum_to_one
test_bl_asset_order_is_stable
test_forward_labels_no_lookahead
test_no_future_price_change_affects_asof_views
test_bl_unit_contract_rejects_mixed_horizons
test_active_view_p_vector_subtracts_benchmark
test_relative_view_p_vector
test_blocked_assets_do_not_generate_views
test_omega_positive_and_scaled
test_zero_view_bl_returns_prior
test_high_omega_has_lower_posterior_impact
test_low_omega_has_higher_posterior_impact
test_relative_view_moves_assets_in_correct_direction
```

### App Tests

```text
all existing Streamlit pages render
CIO Views page renders
CIO Views page shows candidate views or a no-view explanation
exports are present
blocked views show reason
```

### Manual QA

Open the dashboard and confirm:

- Header still shows research-only status.
- CIO Views page is easy to understand without raw data tables.
- A PM can identify `P`, `q`, `Omega`, confidence, rationale, and status.
- Exported files match the on-screen table.
- No ineligible asset appears in candidate views.

## Out-of-Scope Until MVP Works

Do not spend time on:

- More hand-defined macro scenarios.
- Broader asset universe.
- More pair combinations.
- Complex portfolio optimizer.
- UI polish beyond making CIO Views readable.
- Elastic Net / Bayesian Ridge as a blocker.
- XGBoost, random forests, transformers, or AutoML.
- Persistent manual approval workflow.
- Proprietary Bloomberg/current holdings integration unless the data is supplied.

## Definition of Done

The dashboard is BL-ready when this can run deterministically:

```python
views = generate_bl_inputs(as_of="2026-03-31", horizon_months=6)
asset_order, P, q, Omega = views.asset_order, views.P, views.q, views.Omega
posterior = run_black_litterman(asset_order, P, q, Omega)
```

And the CIO can see:

- the formal view sentence.
- `P`.
- `q`.
- confidence.
- `Omega`.
- rationale.
- risks.
- status.
- posterior impact.
- export files.

Until that exists, the dashboard remains a strong macro research dashboard, not a complete automatic macro views pipeline.
