from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CONFIG = ROOT / "config"
FACTOR_COLUMNS = ["risk", "growth", "inflation", "rates", "liquidity", "dollar", "oil"]


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2008-01-31", "2026-03-31", freq="ME")
    n = len(dates)

    # Create plausible but synthetic macro factor levels.
    factors_z = rng.normal(size=(n, len(FACTOR_COLUMNS)))
    factors_z = pd.DataFrame(factors_z, index=dates, columns=FACTOR_COLUMNS).rolling(3, min_periods=1).mean()

    factors = pd.DataFrame(index=dates)
    factors["risk"] = 100 + factors_z["risk"].cumsum()
    factors["growth"] = 50 + factors_z["growth"].cumsum() / 4
    factors["inflation"] = 2.5 + factors_z["inflation"].cumsum() / 20
    factors["rates"] = 3.0 + factors_z["rates"].cumsum() / 25
    factors["liquidity"] = 100 * np.exp((factors_z["liquidity"] / 50).cumsum())
    factors["dollar"] = 100 * np.exp((factors_z["dollar"] / 60).cumsum())
    factors["oil"] = 75 * np.exp((factors_z["oil"] / 40).cumsum())

    universe = pd.read_csv(CONFIG / "universe.csv")
    tickers = list(universe["ticker"])

    # Synthetic exposures by bucket. These are not financial recommendations.
    base = {
        "Asset Class":       [0.35, 0.20, 0.15, -0.10, 0.20, -0.10, 0.10],
        "Equity Region":     [0.25, 0.25, 0.10, 0.00, 0.10, -0.20, 0.10],
        "Equity Sector":     [0.25, 0.20, 0.10, 0.00, 0.10, -0.10, 0.05],
        "Fixed Income":      [-0.15, -0.10, -0.05, -0.30, 0.10, 0.05, -0.05],
        "FX":                [0.10, 0.15, 0.05, 0.00, 0.00, -0.35, 0.05],
        "Commodity":         [0.10, 0.15, 0.35, 0.00, -0.05, -0.10, 0.35],
        "Style Factor":      [0.25, 0.15, 0.00, 0.00, 0.15, -0.10, 0.00],
        "Crypto":            [0.45, 0.20, 0.00, -0.05, 0.35, -0.20, 0.00],
    }

    returns = pd.DataFrame(index=dates)
    x = factors_z.to_numpy()
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        bucket = row["bucket"]
        b = np.array(base.get(bucket, [0, 0, 0, 0, 0, 0, 0]), dtype=float)
        # Add ticker-specific tilt.
        b += rng.normal(scale=0.08, size=len(FACTOR_COLUMNS))
        drift = 0.004 if bucket not in {"Fixed Income", "FX"} else 0.001
        vol = 0.035 if bucket not in {"Fixed Income", "FX"} else 0.015
        if bucket == "Crypto":
            vol = 0.09
        returns[ticker] = drift + (x @ b) / 100 + rng.normal(scale=vol, size=n)

    prices = 100 * np.exp(returns.cumsum())
    DATA.mkdir(exist_ok=True)
    prices.reset_index(names="date").to_csv(DATA / "prices.csv", index=False)
    factors.reset_index(names="date").to_csv(DATA / "factors.csv", index=False)
    print(f"Wrote {DATA / 'prices.csv'}")
    print(f"Wrote {DATA / 'factors.csv'}")


if __name__ == "__main__":
    main()
