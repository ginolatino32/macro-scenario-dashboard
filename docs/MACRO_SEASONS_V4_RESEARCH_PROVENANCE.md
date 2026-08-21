# Macro Seasons V4 research provenance

This note records what the repository can establish about the construction of
Macro Seasons V3/V4. It separates executable facts from statements that appear
only in retrospective research notes.

## Version lineage

### June 2026 research prototype

`season_sharpe_research.py` built season portfolios with a constrained Sharpe
optimizer. It ranked a broad asset universe by season, selected candidates and
optimized the final weights using historical season returns. The June 7 review
package records those optimized portfolios and their historical results.

Those optimized weights are not used by V2, V3 or V4.

### V2 redesign

`macro_seasons_v2.py` replaced the earlier optimizer with:

- symmetric Growth x Inflation seasons;
- a separate Liquidity risk overlay;
- four literal `TEMPLATES` dictionaries containing ETF names and starting
  weights;
- a 50/50 blend of each template's fixed weights and a trailing 36-month
  inverse-volatility adjustment;
- probability-weighted blending of all four templates;
- trailing momentum, trend and volatility controls.

No optimizer estimates the V2 template constituents or starting weights. The
source comments describe them as "fixed macro-logical weights." The repository
does not contain a separate contemporaneous memo that documents the selection
of every ETF and weight, or proves that the template decisions were made before
their historical returns were inspected. Public methodology should therefore
call them manually specified macro templates, not empirically selected,
optimized or independently validated portfolios.

### V3 research

V3 copied the V2 templates without changing a ticker or starting weight. The
same template dictionaries also appear in every H1-H8 research variant.

The available V3 research files evaluated eight changes using returns through
December 2018:

1. H1: curve and duration timing.
2. H2: daily 200-day trend and an EWMA volatility variant.
3. H3: credit-spread risk reduction.
4. H4: dual-momentum asset selection.
5. H5: inverse-volatility combination of strategy streams.
6. H6: season-conditional mean tilt.
7. H7: downside-volatility targeting and a drawdown governor.
8. H8: real-yield-based rotation between gold and nominal Treasuries.

The merged V3 source retained H8, H3, the 200-day trend component of H2 and the
three-stream H5 ensemble. It did not retain H1, H4, H6, H7 or H2's EWMA
volatility component. The merged script runs five acceptance checks to
reproduce the individual development-period results, then evaluates two
combined configurations. Config A combines H8, H3 and the 200-day trend gate;
Config B combines H8 with the original monthly trend gate. Both use the H5
ensemble.

`MODEL_FREEZE_MACRO_SEASONS_V3.md` records that Config A was chosen using data
through December 2018 and that January 2019 through May 2026 was opened once
after selection. It also records the selection score as excess Sharpe plus
half Calmar. The first Git commit containing the V3 source and freeze documents
is `95e51fe`, committed on August 20, 2026. Git therefore confirms the frozen
files from that commit onward; it does not independently timestamp the claimed
pre-registration or lockbox sequence before August 20.

### V4 data release

V4 imports the frozen V3 portfolio functions. It does not redefine the season
templates, overlays or stream construction. Its changes are in the data and
release process:

- ALFRED decision-date histories for CPI, industrial production, payrolls and
  M2;
- refreshed FRED and Yahoo caches;
- freshness checks that stop a run on missing or stale inputs;
- immutable input and output manifests;
- separate reporting for the funded long-only portfolio and the leveraged
  ensemble.

The V4 freeze is commit `d30e26e`, dated August 20, 2026. Its data end is July
31, 2026 and its effective allocation is August 2026. At the freeze, no return
generated after the freeze had completed.

### Physical L/S execution overlay

Commit `3b8e39b`, later on August 20, added the IBKR execution overlay. It
reconstructs ETF positions from the three V3 streams, nets duplicate and
opposing exposures, applies exposure limits and models trading, financing,
short-borrow and short-proceeds costs. It does not change the season engine or
the long-only rules. Its 2007-2026 history is a retrospective execution
simulation, not part of the original V3 selection lockbox.

## Known V4 liquidity-unit issue

The V3/V4 Liquidity pillar calculates Fed net liquidity as `WALCL -
RRPONTSYD - WTREGEN`. FRED reports `WALCL` and `WTREGEN` in millions of
dollars, while `RRPONTSYD` is in billions of dollars. The frozen V4 code does
not multiply `RRPONTSYD` by 1,000 before subtraction, so the reverse-repo term
is understated. Correcting this input would change the frozen signal and must
therefore be released as a new model version rather than silently modifying
V4.

## ETF templates used by V3/V4

The following are the literal starting weights in
`research/macro_seasons_v3.py`:

| Season | Starting weights |
|---|---|
| Spring | SPY 20%, QQQ 15%, SMH 10%, IWM 10%, XHB 7.5%, HYG 15%, LQD 7.5%, IEF 10%, GLD 5% |
| Summer | XLE 15%, XLB 10%, XLI 10%, VLUE 10%, IWM 10%, EEM 10%, DBC 12.5%, CPER 5%, GLD 7.5%, TIP 10% |
| Fall | GLD 17.5%, DBC 10%, XLE 7.5%, TIP 15%, UUP 10%, XLP 10%, XLU 7.5%, XLV 7.5%, SHY 7.5%, BIL 7.5% |
| Winter | TLT 17.5%, IEF 17.5%, SHY 7.5%, GLD 12.5%, USMV 10%, XLP 10%, XLV 10%, FXY 5%, FXF 5%, BIL 5% |

These are starting weights, not final monthly holdings. At each month-end the
model filters unavailable assets, applies the trailing inverse-volatility
adjustment, blends the four season portfolios and then applies the V3 overlays.

## Historical evidence labels

- V3 development period: returns through December 2018.
- V3 recorded lockbox evaluation: January 2019 through May 2026.
- V4 PIT rerun: retrospective history through July 2026 using the new data
  contract.
- V4 first post-freeze signal: August 31, 2026.
- V4 first post-freeze return: September 30, 2026.
- Physical L/S history: retrospective simulation from January 2007.

The current website and PDF should describe all pre-freeze and reconstructed
results as historical simulations. They should not describe the full V4 or
physical L/S history as untouched out-of-sample evidence.

## Source record

- `season_sharpe_research.py`
- `macro_seasons_v2.py`
- `research/h1_duration_carry.py` through `research/h8_gold_realrate.py`
- `research/macro_seasons_v3.py`
- `MODEL_FREEZE_MACRO_SEASONS_V3.md`
- `research/macro_seasons_v4.py`
- `MODEL_FREEZE_MACRO_SEASONS_V4.md`
- `research/macro_seasons_v4_execution.py`
- `config/ibkr_execution_settings.csv`
- Git commits `95e51fe`, `d30e26e` and `3b8e39b`
