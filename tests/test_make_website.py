from __future__ import annotations

import re

from make_website import build_page, load_data


def _page() -> str:
    return build_page(load_data())


def test_season_definitions_precede_portfolio_summary() -> None:
    page = _page()

    assert page.index("How the seasons are defined") < page.index(
        "Long Only and L/S Portfolio"
    )
    assert "Broad stocks, tech, corporate credit, housing" in page
    assert "e.g. 2006-07, 2021" in page
    assert "e.g. early 2008, 2022" in page
    assert "e.g. 2001-02, 2008, 2020" in page


def test_execution_section_explains_bil_netting() -> None:
    page = _page()

    assert "Why BIL is absent" in page
    assert "negative-carry round trip" in page
    assert "financing netting" in page
    assert "Two strategies, reported separately" not in page


def test_monthly_returns_offer_long_only_and_costed_ls_views_from_2007() -> None:
    page = _page()

    assert re.findall(r"data-return-view='([^']+)'", page) == ["long-only", "ls"]
    assert re.findall(r"data-return-panel='([^']+)'", page) == ["long-only", "ls"]
    assert page.count("<tbody><tr><td>2007</td>") == 2
    assert "physical history begins January 2008" in page
