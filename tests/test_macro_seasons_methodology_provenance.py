from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value = node.value
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(value)
    raise AssertionError(f"{name} not found in {path}")


def test_v3_and_all_hypothesis_files_retain_the_v2_templates() -> None:
    v2_templates = _literal_assignment(ROOT / "macro_seasons_v2.py", "TEMPLATES")
    comparison_files = [
        ROOT / "research" / "macro_seasons_v3.py",
        *sorted((ROOT / "research").glob("h[1-8]_*.py")),
    ]

    assert len(comparison_files) == 9
    for path in comparison_files:
        assert _literal_assignment(path, "TEMPLATES") == v2_templates, path


def test_pdf_copy_does_not_repeat_unsupported_selection_claims() -> None:
    source = (ROOT / "research" / "make_onepager.py").read_text().lower()
    unsupported_claims = [
        "base weights were set from each etf's economic exposure before the backtest",
        "the whole curve is out-of-sample by construction",
        "the 2019-2026 results influenced later research decisions",
    ]

    for claim in unsupported_claims:
        assert claim not in source
