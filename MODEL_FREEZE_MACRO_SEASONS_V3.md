# Model Freeze: Macro Seasons v3 (Config A)

**Frozen:** 2026-07-16
**Frozen data snapshot:** monthly panel through 2026-05-31 (61 assets, proxy-extended to 1990)
**Model file:** `research/macro_seasons_v3.py` — SHA256 `545b4dd566f8f8c68f5d343325fca7bd9862277ffae38bac6ae7e960e70d6d85`
**Production runner:** `run_macro_seasons_v3.py` — SHA256 `d04154b64d60174030e245ab6ee23875f594e3d180a16cfb0c71d01b5f41acb5`
(re-stamped 2026-07-22: added weight-history and current-allocation exports —
a reporting change under the change policy; model logic and constants untouched,
and `current_target_weights` reproduces the WF loop's weights to 1e-16)

If either hash changes, the model is no longer this model. Re-hash with
`shasum -a 256 research/macro_seasons_v3.py run_macro_seasons_v3.py`.

## Frozen configuration (Config A)

Season engine: point-in-time macro pillar composites (Growth: INDPRO, PAYEMS,
ICSA, XLI/XLP, SPY/IEF; Inflation: CPI, T5YIE, WTI, TIP/IEF; Liquidity: NFCI,
HY OAS, M2 vintage, DGS2, Fed net liquidity), publication-lag shifted;
canonical G×I quadrant seasons with logistic probabilities (τ=0.75);
liquidity as a bounded risk overlay.

In-book stack (per month, in order): probability-weighted template blend →
liquidity overlay → real-rate gold/duration rotation (H8) → credit-stress
dimmer (H3: HY OAS > 1.10× trailing 36m median AND 3m widening → risky sleeve
× 0.50) → 12-1 cross-sectional momentum tilt → 200-day MA trend gate (H2,
below-trend weight to cash) → 10% vol target, no leverage. Costs 10bps on
turnover.

Ensemble: inverse-vol (24m rolling, min 12, shift 1) combination of
[core seasons levered, enhanced in-book stack levered, TSMOM long/short
sleeve (MOP 2012)], vol-targeted to 10%, financed at BIL.

All constants are in the model file header sections and are part of the
freeze. Every signal passes fail-closed causality self-checks at runtime.

## Validation record (all walk-forward, net of costs; 317 months Dec 1999 – May 2026)

Protocol: development window ≤ 2018-12; lockbox 2019-01 → 2026-05 opened once
after config selection by a pre-registered rule (excess Sharpe + 0.5·Calmar).
Research program: 8 hypothesis agents + adversarial verification; ~60 total
configurations evaluated (used for the deflated Sharpe below).

| Stream | Window | CAGR | Vol | Excess Sharpe | Raw Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|---|---|
| Ensemble | Dev | 8.83% | 8.38% | 0.827 | — | −11.0% | 0.80 |
| Ensemble | **Lockbox** | 10.93% | 8.56% | **0.908** | — | −8.5% | **1.29** |
| Ensemble | Full | 9.42% | 8.43% | 0.850 | 1.07 | −12.3% | 0.77 |
| In-book | Dev | 5.13% | 4.54% | 0.762 | — | −5.2% | 0.98 |
| In-book | **Lockbox** | 7.14% | 4.53% | **0.946** | — | −3.0% | **2.38** |
| In-book | Full | 5.69% | 4.54% | 0.813 | 1.22 | −5.2% | 1.09 |
| SPY | Full | 8.29% | 15.27% | 0.402 | 0.52 | −50.8% | 0.16 |
| 60/40 | Full | 6.79% | 9.55% | 0.496 | 0.69 | −32.3% | 0.21 |

Deflated Sharpe (Bailey–López de Prado, 60 trials): ensemble 0.975,
in-book 0.963. Season-timing permutation test (1000 circular shifts):
p = 0.001. Sharpe-difference bootstrap vs SPY: p ≈ 0.04.

Forward expectation should be set below the backtest: excess Sharpe 0.6–0.8
is the planning number, not the lockbox's 0.9+.

## Monitoring (run monthly via `python3 run_macro_seasons_v3.py`)

REVIEW triggers (outside the frozen record's historical experience):
- trailing 12-month excess Sharpe < −0.5
- in-book stack drawdown beyond −8%
- ensemble drawdown beyond −15%

A REVIEW flag means investigate and re-validate — not silently retune.

## Change policy

The 2019–2026 lockbox is spent. Any change to constants, templates, layers,
or universe requires new out-of-sample evidence (live months or genuinely new
data) and a new freeze document with a version bump. Permitted without
re-freeze: data refreshes, bug fixes that reproduce the frozen validation
numbers bit-identically, and reporting changes.

Sanctioned information upgrades for a future v4 (in priority order):
1. ALFRED first-print vintages for CPIAUCSL / INDPRO / PAYEMS (reuse the
   M2SL machinery in `update_data.py`).
2. Genuine carry sleeve (rates/FX) or futures universe for the TSMOM sleeve.
3. Crypto satellite, capped, as a separate stream.

## BL handoff

`exports/macro_seasons_v3_bl_views.csv` provides season-probability-weighted
relative views (q shrunk 50%, Ω scaled by regime confidence). Wire the
baskets through `config/view_pairs.csv` so views enter the production
P/q/Ω exports behind the existing gates. Season probabilities and the credit/
real-rate state flags are the sanctioned confidence inputs for Ω scaling.
