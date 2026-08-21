"""Refresh and validate every external cache used by Macro Seasons v4."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "research"))

import macro_seasons_v4 as m  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--as-of", help="Month-end to validate; defaults to the latest price month")
    args = parser.parse_args()

    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
    else:
        source = m.DATA / "prices_macro_seasons_extended.csv"
        if not source.exists():
            source = m.DATA / "prices.csv"
        panel = m.load_wide_csv(source)
        as_of = pd.Timestamp(panel.index.max())

    audit = m.refresh_and_validate_caches(as_of, allow_network=not args.no_network)
    print(f"Macro Seasons v4 data gate PASS through {as_of:%Y-%m-%d}")
    print(audit.groupby(["component", "status"]).size().rename("count").to_string())
    print(f"Wrote {m.DATA_AUDIT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
