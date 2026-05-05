from __future__ import annotations

from pathlib import Path

import pandas as pd

from bl_views import generate_bl_inputs, write_bl_outputs
from model import load_wide_csv


def main() -> None:
    root = Path(__file__).resolve().parent
    prices = load_wide_csv(root / "data" / "prices.csv")
    factors = load_wide_csv(root / "data" / "factors.csv")
    universe = pd.read_csv(root / "config" / "universe.csv")
    scenarios = pd.read_csv(root / "config" / "scenarios.csv")
    source_audit_path = root / "data" / "source_audit.csv"
    source_audit = pd.read_csv(source_audit_path) if source_audit_path.exists() else pd.DataFrame()

    result = generate_bl_inputs(
        prices,
        factors,
        universe,
        scenarios,
        root / "config",
        source_audit=source_audit,
    )
    failures = result.config_audit[result.config_audit["status"].eq("Fail")]
    if not failures.empty:
        raise SystemExit("BL config validation failed:\n" + failures.to_string(index=False))
    write_bl_outputs(result, root)
    candidate_count = int(result.views["status"].eq("Candidate").sum()) if not result.views.empty else 0
    print(
        {
            "as_of": result.as_of.date().isoformat(),
            "horizon_months": result.horizon_months,
            "assets": len(result.asset_order),
            "predictions": len(result.predictions),
            "candidate_views": candidate_count,
            "matrix_shape": result.P.shape,
        }
    )


if __name__ == "__main__":
    main()
