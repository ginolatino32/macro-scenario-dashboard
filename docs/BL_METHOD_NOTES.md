# BL Method Notes

Date: 2026-05-05

This note documents the implemented automatic macro views layer. It is intentionally narrower than the broader research dashboard.

## Implemented Contract

The production handoff is:

```text
generate_bl_inputs(as_of, horizon)
  -> asset_order
  -> P
  -> q
  -> Omega
  -> view_rationale
  -> status
  -> BL posterior return impact
```

The generated files are:

```text
data/model_predictions.csv
data/bl_macro_views.csv
data/bl_runs.csv
data/bl_posterior_returns.csv
exports/bl_asset_order.csv
exports/P_matrix.csv
exports/q_vector.csv
exports/Omega_matrix.csv
exports/bl_views.json
```

## Unit Convention

The MVP uses:

```text
horizon_months = 6
q = 6-month active or relative-spread return
Sigma = 6-month covariance
pi = 6-month implied equilibrium return
Omega = 6-month view error variance
```

Monthly covariance is scaled approximately:

```text
Sigma_6m = 6 * Sigma_monthly
```

This approximation is acceptable for the current student version and is disclosed in the dashboard.

## Forecasting

The forecast engine uses:

```text
historical mean baseline
selected-feature macro OLS
shrinkage back toward historical mean
walk-forward validation diagnostics
```

The regularized-model upgrade is deferred until the BL contract is stable.

## No-Lookahead Rule

For horizon `H` and as-of `T`, a training label is eligible only when:

```text
feature_date + H <= T
```

For `T = 2026-03-31`:

```text
6m latest training feature date = 2025-09-30
12m latest training feature date = 2025-03-31
```

Macro features are standardized point-in-time and lagged one month to reduce release-lag and revision-bias risk in public proxy data.

## View Construction

Active view:

```text
P = +asset - benchmark basket
q = expected active return
```

Relative view:

```text
P = +long asset - short asset
q = expected spread return
```

Only approved pairs from `config/view_pairs.csv` can create automatic relative views.

## Confidence and Omega

Confidence is a model uncertainty scaling score, not a probability of correctness.

```text
Omega = forecast_error_variance / confidence
```

The implementation applies:

```text
confidence_floor
confidence_cap
omega_floor
omega_cap
Unknown/Mixed regime penalty
data freshness penalty
```

Blocked and Needs Review views do not enter `P/q/Omega`.

## Black-Litterman Posterior

The posterior mean uses:

```text
mu_bl =
  inv(inv(tau * Sigma) + P.T @ inv(Omega) @ P)
  @ (inv(tau * Sigma) @ pi + P.T @ inv(Omega) @ q)
```

The dashboard displays:

```text
prior return
posterior return
posterior minus prior
benchmark weight
suggested active weight
```

The suggested active weight is a diagnostic proportional tilt, not a final optimizer output.

## Current Limitations

- Benchmark weights are sample placeholders until actual GSMIF policy/current holdings are supplied.
- Public macro data uses latest revised proxy series and is not a full real-time vintage dataset.
- 12-month views are configured for later validation but are not the primary MVP output.
- Relative-view Omega uses the conservative diagonal residual approximation in v1.
- Manual approval is not persisted. The app uses only `Candidate`, `Needs Review`, and `Blocked`.
