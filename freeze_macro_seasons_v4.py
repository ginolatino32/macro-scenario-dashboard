"""Create the immutable Macro Seasons v4 PIT release and SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RELEASE_ID = "macro_seasons_v4_20260820"
RELEASES = ROOT / "releases"
RELEASE = RELEASES / RELEASE_ID
STAGING = RELEASES / f".{RELEASE_ID}.staging"

CORE_INPUTS = [
    "data/prices_macro_seasons_extended.csv",
    "data/factors_point_in_time.csv",
    "data/macro_seasons_v4_data_audit.csv",
    "data/macro_seasons_extended_price_audit.csv",
    "data/macro_vintage_audit.csv",
]
SOURCE_FILES = [
    "MODEL_FREEZE_MACRO_SEASONS_V3.md",
    "research/macro_seasons_v3.py",
    "run_macro_seasons_v3.py",
    "research/macro_seasons_v4.py",
    "refresh_macro_seasons_v4_data.py",
    "run_macro_seasons_v4.py",
    "research/make_onepager.py",
    "make_website.py",
    "scripts/macro_monthly_update.command",
    "tests/test_macro_seasons_v4.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_release_file(category: str, relative: str) -> dict[str, object]:
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(f"Required V4 release file is missing: {relative}")
    destination = STAGING / category / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    frozen_hash = sha256(destination)
    if source_hash != frozen_hash:
        raise RuntimeError(f"Frozen copy hash mismatch: {relative}")
    return {
        "category": category,
        "source_path": relative,
        "frozen_path": str(destination.relative_to(STAGING)),
        "bytes": destination.stat().st_size,
        "sha256": frozen_hash,
    }


def release_metadata() -> dict[str, object]:
    summary = pd.read_csv(ROOT / "exports" / "macro_seasons_v4_summary.csv")
    timeline = pd.read_csv(
        ROOT / "exports" / "macro_seasons_v4_season_timeline.csv", parse_dates=["date"]
    ).dropna(subset=["modal_season"])
    allocation = pd.read_csv(ROOT / "exports" / "macro_seasons_v4_current_allocation.csv")
    latest = timeline.iloc[-1]
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return {
        "release_id": RELEASE_ID,
        "model_version": "macro_seasons_v4_pit",
        "freeze_date": "2026-08-20",
        "frozen_data_end": "2026-07-31",
        "effective_allocation_month": "2026-08",
        "first_live_as_of": "2026-08-31",
        "first_live_return_date": "2026-09-30",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "v3_baseline_commit": git_head,
        "current_season": {
            "modal": str(latest["modal_season"]),
            "SPRING": float(latest["SPRING"]),
            "SUMMER": float(latest["SUMMER"]),
            "FALL": float(latest["FALL"]),
            "WINTER": float(latest["WINTER"]),
            "confidence": float(latest["confidence"]),
        },
        "summary": summary.to_dict(orient="records"),
        "allocation_weight_sum": float(allocation["weight"].sum()),
        "allocation_rows": int(len(allocation)),
        "completed_live_months": 0,
    }


def main() -> None:
    if RELEASE.exists():
        raise SystemExit(f"Release already exists and is immutable: {RELEASE}")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    audit = pd.read_csv(ROOT / "data" / "macro_seasons_v4_data_audit.csv")
    if audit.empty or not audit["status"].eq("PASS").all():
        raise RuntimeError("V4 data audit must be complete and all-PASS before freezing")
    audit_inputs = sorted(set(audit["path"].dropna().astype(str)))
    input_files = sorted(set(CORE_INPUTS + audit_inputs))
    output_files = sorted(
        str(path.relative_to(ROOT))
        for pattern in ("macro_seasons_v4_*.csv", "macro_seasons_v4_onepager.pdf")
        for path in (ROOT / "exports").glob(pattern)
    )
    output_files.extend([
        "exports/website/index.html",
        "exports/website/macro_seasons_v4_onepager.pdf",
    ])

    records: list[dict[str, object]] = []
    try:
        for relative in input_files:
            records.append(copy_release_file("inputs", relative))
        for relative in SOURCE_FILES:
            records.append(copy_release_file("source", relative))
        for relative in sorted(set(output_files)):
            records.append(copy_release_file("outputs", relative))

        metadata_path = STAGING / "release_metadata.json"
        metadata_path.write_text(
            json.dumps(release_metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records.append({
            "category": "metadata",
            "source_path": "generated",
            "frozen_path": "release_metadata.json",
            "bytes": metadata_path.stat().st_size,
            "sha256": sha256(metadata_path),
        })

        manifest = pd.DataFrame(records).sort_values(
            ["category", "source_path"], kind="stable"
        )
        manifest_path = STAGING / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        manifest_hash = sha256(manifest_path)
        (STAGING / "manifest.sha256").write_text(
            f"{manifest_hash}  manifest.csv\n", encoding="ascii"
        )

        RELEASES.mkdir(parents=True, exist_ok=True)
        STAGING.rename(RELEASE)
    except Exception:
        if STAGING.exists():
            shutil.rmtree(STAGING)
        raise

    print(f"Frozen {len(records)} files in {RELEASE.relative_to(ROOT)}")
    print(f"Manifest SHA-256: {manifest_hash}")


if __name__ == "__main__":
    main()
