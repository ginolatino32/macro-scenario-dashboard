# Macro Seasons V4 IBKR execution overlay

## Purpose and scope

`macro_seasons_v4_ibkr_execution_v1` is an execution overlay on the frozen
Macro Seasons V4 signal model. It does not change macro seasons, ETF templates,
trend signals, or stream forecasts. It reconstructs positions that the frozen
return-level ensemble implies and then applies physical ETF cash accounting,
portfolio limits, and configurable Interactive Brokers cost assumptions.

The executable historical track begins on January 31, 2008. Earlier V4 research
returns rely on pre-inception mutual-fund and futures proxies. Those observations
remain useful for signal research but cannot support share-level IBKR commission
or physical ETF short-sale calculations.

## Broker sources checked on August 20, 2026

- [IBKR US stock and ETF commissions](https://www.interactivebrokers.com/en/pricing/commissions-stocks.php):
  IBKR Pro Fixed charges USD 0.005 per share, USD 1 minimum per order, and a
  maximum of 1% of trade value. IBKR lists US regulatory fees separately. The
  model therefore also charges the currently published SEC sale-value fee,
  FINRA Trading Activity Fee, and FINRA CAT fee.
- [IBKR USD margin rates](https://www.interactivebrokers.com/en/trading/pricing-margin-rates.php):
  IBKR Pro begins at benchmark plus 1.50% for the first USD 100,000 and declines
  by published balance tiers. The model applies those tiers to each monthly
  debit balance.
- [IBKR positive cash interest](https://www.interactivebrokers.com/en/accounts/fees/pricing-interest-rates.php):
  eligible USD cash earns benchmark minus 0.50%; the first USD 10,000 earns no
  interest. The reference account exceeds USD 100,000 NAV.
- [IBKR short-sale costs](https://www.interactivebrokers.com/en/pricing/short-sale-cost.php):
  borrow fees and short-sale proceeds interest are separate and security-specific.
  USD proceeds earn zero on the first USD 100,000, then benchmark minus the
  published tier spread. Rates and availability change during the day.
- [IBKR short collateral convention](https://www.interactivebrokers.com/campus/glossary-terms/collateral-short-sale/):
  USD stock-loan collateral generally uses 102% of the prior settlement-price
  mark, rounded up at the share-price level. The monthly simulation uses a 1.02x
  approximation because it does not retain historical whole-share loan marks.
- [IBKR short availability](https://www.interactivebrokers.com/en/trading/short-securities-availability.php):
  IBKR exposes current and historical indicative rates through its SLB tools,
  but the project does not possess a complete historical SLB dataset.
- [IBKR US stock margin requirements](https://www.interactivebrokers.com/en/trading/margin-stocks.php):
  Reg T end-of-day initial margin is generally 50% of eligible stock value;
  broker house requirements may be higher.
- [IBKR real-time Portfolio Margin monitoring](https://www.ibkrguides.com/traderworkstation/margin-monitoring.htm):
  Portfolio Margin is risk-based and recognizes portfolio offsets; the broker's
  current initial, maintenance, look-ahead, and overnight requirements must be
  checked before transmitting an order.
- [IBKR Portfolio Margin eligibility](https://www.ibkrguides.com/clientportal/accounttype.htm):
  an approved account currently requires at least USD 110,000 NLV and options
  permission.

Pricing can change. The monthly process must freshness-review these assumptions
before treating the result as an order recommendation.

## Configured assumptions

The source of truth is:

- `config/ibkr_execution_settings.csv`
- `config/ibkr_margin_tiers.csv`
- `config/ibkr_short_borrow_assumptions.csv`
- `config/ibkr_short_proceeds_tiers.csv`

The reference account is USD 1,000,000 under IBKR Pro Fixed pricing. A separate
1 basis-point allowance is charged to every traded dollar for bid/ask spread and
execution slippage.

Historical ticker-level borrow data is not available. Every liquid ETF short is
therefore charged a conservative 1.00% annual borrow rate. This is a model
assumption, not a quoted IBKR rate. Production orders require a same-day SLB
availability and borrow-rate check. An unavailable or hard-to-borrow ETF must be
blocked or replaced before execution.

Short-sale proceeds are not treated as freely available cash. The model finances
long ETF exposure above 1.00x NAV at the margin-loan rate, marks short collateral
at 102%, and credits that collateral using IBKR's separate short-proceeds tiers.
This prevents the short book from artificially reducing the modeled debit loan.

The current strategy holds no direct futures. DBC and GLD are ETFs whose adjusted
returns already reflect their product structure and any fund-level roll effects.
The explicit futures-roll charge is therefore zero. The engine supports a
ticker-level roll charge if direct futures are added later.

## Position reconstruction

The executable ensemble contains three streams:

1. Levered core macro-season allocation.
2. Levered enhanced long-only season allocation.
3. Long/short 12-month time-series-momentum ETF overlay.

The two season streams are independently scaled from 0.50x to 1.50x using only
trailing returns. TSMOM positions are reconstructed ticker by ticker from the
frozen 12-month excess-return sign, 36-month inverse-volatility sizing, and 20%
per-position cap before the whole TSMOM sleeve is scaled to target volatility.
A parity test requires the reconstructed sleeve return
to equal the frozen TSMOM return to numerical precision.

The three streams are weighted by trailing inverse volatility and the combined
stream receives the frozen 0.50x to 1.50x outer volatility scale. Duplicate ETF
positions are summed before any limits or trading costs are calculated.

If the aggregate book simultaneously holds BIL and has a negative USD cash
balance, BIL is sold first to repay the debit. Borrowing to hold Treasury bills
would otherwise create an artificial negative-carry position.

## Hard limits

- Gross ETF exposure: 1.75x NAV.
- Net ETF exposure: 0.25x to 1.50x NAV.
- Gross short exposure: 0.35x NAV.
- Absolute non-cash ETF exposure: 0.25x NAV per ticker.
- BIL exposure: 1.00x NAV.
- Explicit USD debit balance: 0.50x NAV.
- Live Portfolio Margin cushion: at least 20% above IBKR's projected maintenance
  requirement after the complete target is submitted to Check Margin / what-if.

Limits are applied after duplicate positions are netted. A constrained target is
the only target admitted to the execution simulation and current position file.

The configured account type is Portfolio Margin. The engine does not claim to
reproduce IBKR's TIMS and house-margin calculation from public data. It retains a
50%-of-gross Reg T equivalent as a conservative reference diagnostic only; that
number is neither the modeled Portfolio Margin requirement nor an execution
approval. `macro_seasons_v4_execution_pm_pretrade_check.csv` remains
`LIVE_IBKR_CHECK_REQUIRED` until the whole target passes the broker's live margin
preview with the configured cushion. Portfolio Margin changes available buying
power, not the cash debit, financing rate, short-collateral segregation, or stock
borrow charge.

## Monthly accounting

At each month-end the engine compares the new constrained outer-portfolio target
with the prior aggregate portfolio after market drift. It estimates shares from
the decision-date adjusted ETF price and charges IBKR Fixed commission,
regulatory fees, and the slippage allowance for every nonzero order. Historical
share-level commission is therefore an approximation: current IBKR fee rates and
split-adjusted Yahoo prices are applied across the executable backtest.

Unencumbered positive cash receives the configured IBKR credit rate. Long ETF
purchases above 1.00x NAV are charged the benchmark rate plus the balance-weighted
IBKR margin spread even when short-sale collateral makes the economic net cash
balance appear less negative. Every short ETF is charged its configured borrow
rate on 102% collateral; tiered proceeds interest is credited separately. All
financing uses the actual number of calendar days in the holding month and a
360-day convention.

The following files make the accounting auditable:

- `exports/macro_seasons_v4_execution_current_tsmom.csv`
- `exports/macro_seasons_v4_execution_current_positions.csv`
- `exports/macro_seasons_v4_execution_current_orders.csv`
- `exports/macro_seasons_v4_execution_pm_pretrade_check.csv`
- `exports/macro_seasons_v4_execution_position_history.csv`
- `exports/macro_seasons_v4_execution_ledger.csv`
- `exports/macro_seasons_v4_execution_summary.csv`
- `exports/macro_seasons_v4_execution_cost_summary.csv`
- `exports/macro_seasons_v4_execution_manifest.json`

`current_orders.csv` is the model's rebalance from its simulated prior holdings.
It is not an instruction to trade an actual account. Actual holdings, tax lots,
restricted securities, available margin, SLB availability, and order timing must
be reconciled first.
