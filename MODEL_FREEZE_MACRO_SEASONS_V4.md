# Model Freeze: Macro Seasons v4 PIT

**Frozen:** 2026-08-20

**Frozen data end:** 2026-07-31

**Effective allocation:** August 2026

**First post-freeze signal date:** 2026-08-31

**First post-freeze realized return:** 2026-09-30

**Completed live months at freeze:** 0

V4 preserves the frozen V3 portfolio rules and changes the data contract. It
rebuilds revision-sensitive macro history from ALFRED, refreshes every required
FRED and Yahoo cache before production, fails closed on stale or missing inputs,
and reports the implementable long-only portfolio separately from the leveraged
multi-strategy ensemble.

## Immutable identity

The V3 baseline is Git commit `95e51fe1a7ac2ed60b5d91b38b1541d7486f4896`.

| File | SHA-256 |
|---|---|
| `research/macro_seasons_v3.py` | `545b4dd566f8f8c68f5d343325fca7bd9862277ffae38bac6ae7e960e70d6d85` |
| `run_macro_seasons_v3.py` | `d04154b64d60174030e245ab6ee23875f594e3d180a16cfb0c71d01b5f41acb5` |
| `research/macro_seasons_v4.py` | `f92615e5727d8d971354affe08c7ea292b876dd0fcacde62b1f064c359d5228d` |
| `refresh_macro_seasons_v4_data.py` | `307950ecd6a6aaba720d648c61c971789e1b21f1e5b45bc80640ae9e5b263617` |
| `run_macro_seasons_v4.py` | `d500641ae5d3c5fd80954b2bd2deb9624dd2e8f37e27cda90806b19fffb938fb` |
| `make_website.py` | `5d21d338eaad0abd4c4d4aab0fe222fb369f331421243cc6742bf16897698df1` |
| `research/make_onepager.py` | `cbd5b51d3185046bd1cd26b5e06c5eddfa98aadc88e496cd17e7762b78e4c3dd` |
| `scripts/macro_monthly_update.command` | `6193c69b11d3ccf0d7cf87cb288ea9ca53bf00cfa2275391223c0d8c7207d017` |
| `tests/test_macro_seasons_v4.py` | `2834453a2a7d0439f885e6251bb9e65c571dd19584aa31e682ace16d4e05c8a1` |
| `freeze_macro_seasons_v4.py` | `6d7d22f15cc26f2c73016e5b8ad4b11862c5cb4aaf5bc0fa7da85d781e0d16d2` |

The immutable release is `releases/macro_seasons_v4_20260820/`.
Its `manifest.csv` covers 52 files: 27 inputs, 10 source snapshots, 14 outputs,
and one metadata record. The manifest SHA-256 is:

`e5e2a5b94af1b540ac6a5141abab346594ca09c292d4f572a59964eef32e0ae9`

Every frozen file is byte-checked against its manifest. `manifest.sha256`
protects the manifest itself.

## Point-in-time data contract

Every monthly production run performs these steps before model execution:

1. Refresh all 16 FRED caches consumed by V3/V4.
2. Refresh the 28 required Yahoo daily ETF series and the monthly extension panel.
3. Download and persist ALFRED vintage histories for CPIAUCSL, INDPRO, PAYEMS,
   and M2SL.
4. At each historical month-end, select the newest observation and revision
   whose `realtime_start` was no later than that decision date.
5. Enforce series-specific freshness thresholds and stop the run if any required
   input fails.
6. Run prefix-truncation causality tests for the ALFRED pillars, real-rate layer,
   credit layer, daily trend gate, and ensemble weights.

The frozen July run passed 49/49 checks:

| Component | Passed |
|---|---:|
| FRED caches | 16/16 |
| ALFRED vintage histories | 4/4 |
| Yahoo daily series | 28/28 |
| Yahoo monthly extension panel | 1/1 |

Raw FRED caches can contain observations published after July 31 because the
refresh captures the current source file. Every model reader slices those files
to the decision date, and the frozen prefix-causality tests verify that later
rows do not change an earlier result.

## Model definition

The four labels remain the canonical growth/inflation quadrants:

- Spring: growth rising, inflation cooling.
- Summer: growth rising, inflation heating.
- Fall: growth falling, inflation heating.
- Winter: growth falling, inflation cooling.

Liquidity changes the probability mix and overall risk intensity; it does not
silently change the quadrant labels. The Growth, Inflation, and Liquidity
pillars combine trailing transformations that can be computed with information
available at the decision month.

The long-only portfolio keeps the V3 order of operations:

`season probability blend -> liquidity overlay -> real-rate rotation -> credit dimmer -> cross-sectional momentum -> 200-day trend gate -> 10% volatility target without leverage`

The multi-strategy ensemble is separate:

`inverse-volatility blend of levered core + levered long-only + time-series momentum -> final 10% volatility target`

For August 2026 its stream mix is 29.97% levered core, 31.13% levered
long-only, and 38.90% time-series momentum, followed by a 1.4586x final risk
scale. Those are strategy sleeves, not the ETF weights below.

## Frozen July signal and August allocation

The point-in-time July 31 season probabilities are:

| Season | Probability |
|---|---:|
| Spring | 25.46% |
| Summer | 30.73% |
| Fall | 23.96% |
| Winter | 19.85% |

Summer is modal, but the confidence gap is only 0.87 percentage points. V4
therefore describes the call as low conviction and blends all four season
portfolios.

The exact long-only target for August 2026 is:

| Ticker | Weight | Ticker | Weight |
|---|---:|---|---:|
| BIL | 40.8073% | XLI | 3.3070% |
| HYG | 6.0600% | VLUE | 3.2961% |
| XLE | 5.8093% | XLB | 2.9483% |
| DBC | 5.6296% | XLP | 2.3541% |
| IWM | 5.5877% | XLV | 2.2391% |
| SPY | 5.2389% | SMH | 2.1752% |
| QQQ | 3.7035% | UUP | 1.4269% |
| EEM | 3.5528% | CPER | 1.3836% |
| SHY | 3.3717% | USMV | 1.1091% |

Weights sum to 100% without hiding or redistributing positions under 2%. The
fresh 200-day trend gate moved TIP, XHB, and XLU to BIL for this rebalance.

## Historical results reported separately

All figures below are point-in-time historical simulations after modeled costs.
They are not live results.

| Track | Months | CAGR | Vol | Excess Sharpe | Max drawdown | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| Long-only season portfolio | 319 | 5.57% | 4.53% | 0.786 | -5.48% | 1.016 |
| Multi-strategy leveraged ensemble | 319 | 9.21% | 8.47% | 0.823 | -12.36% | 0.746 |
| Core season allocation | 319 | 6.21% | 6.41% | 0.653 | -11.12% | 0.558 |
| SPY | 319 | 8.20% | 15.23% | 0.397 | -50.78% | 0.161 |
| 60/40 SPY/AGG | 319 | 6.70% | 9.52% | 0.488 | -32.32% | 0.207 |

The public site uses the long-only series for the allocation and monthly return
table. It displays the ensemble only as a separately identified research track.

## Live-monitor boundary

The July 31 allocation was generated before the V4 freeze and is a frozen
historical/research result. The August 31 decision is the first signal produced
fully after the August 20 freeze. Its September 30 return is the first live
observation. V4 therefore reports `PENDING` and zero completed live months until
that return exists.

The former 2019-2026 lockbox has been opened and used in research review. It is
spent and is not represented as a new untouched holdout for V4.

## Known limitations

- ALFRED reconstruction is mandatory for CPI, industrial production, payrolls,
  and M2. Other FRED inputs use explicit publication lags but are not all rebuilt
  from full vintage histories.
- The current HY OAS cache begins in August 2023. Earlier credit states use the
  frozen Moody's Baa minus 10-year Treasury fallback rather than splicing an
  unavailable OAS history.
- Yahoo Finance is an acceptable public adjusted-price source for this research
  build, not an institutional security-master or execution feed.
- Trading costs are modeled. Taxes, borrow constraints, market impact, futures
  margin, and implementation slippage beyond the frozen cost rule are not live
  observations.
- Historical performance cannot validate the new V4 data contract as live. Only
  post-freeze monthly observations can do that.

## Monthly operation

Run `scripts/macro_monthly_update.command` after month-end. It refreshes the
broad panel, refreshes and gates every V4 external cache, regenerates BL outputs,
runs V4 from the validated local snapshot, rebuilds the PDF and website, and
deploys only if all prior steps pass. Missing ALFRED credentials, stale inputs,
failed causality tests, or build errors prevent deployment.

Any change to the frozen model logic, data-selection rules, thresholds, universe,
or allocation machinery requires a new version and a new immutable release.
