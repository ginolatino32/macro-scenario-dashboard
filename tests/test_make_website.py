from __future__ import annotations

import re

import pandas as pd

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


def test_public_page_omits_internal_monitoring_and_bil_note() -> None:
    page = _page()

    assert "Live monitor" not in page
    assert "First live return" not in page
    assert "confidence gap" not in page
    assert "live-monitor definition" not in page
    assert "Current monitor status" not in page
    assert "Why BIL is absent" not in page
    assert "negative-carry round trip" not in page
    assert "financing netting" not in page
    assert "Input freshness" in page
    assert "ALFRED vintages" in page
    assert "Two strategies, reported separately" not in page


def test_allocation_sections_use_reader_facing_labels() -> None:
    data = load_data()
    page = build_page(data)

    assert "Long-only allocation for August 2026" in page
    assert "L/S allocation for August 2026" in page
    assert "Exact long-only allocation" not in page
    assert "Executable leveraged portfolio" not in page
    assert "Generated from July month-end signals" not in page
    assert "Aggregated and netted ETF positions" not in page
    assert "Model target, not a broker order" not in page
    assert "PM pre-trade check" not in page
    assert "strip = monthly modal season" not in page
    assert "Leveraged ensemble" not in page
    assert "historical PIT simulation" not in page
    assert "in one line each" not in page
    assert "L/S portfolio" in page

    long_only = page.split("<section data-allocation='long-only'>", 1)[1].split(
        "</section>", 1
    )[0]
    ls = page.split("<section id='execution' data-allocation='ls'>", 1)[1].split(
        "</section>", 1
    )[0]
    assert long_only.count("class='allocation-row'") == len(data["allocation"])
    assert ls.count("class='exec-row'") == len(data["execution_positions"])


def test_equity_chart_keeps_all_series_without_end_value_labels() -> None:
    page = _page()
    chart = page.split("<svg id='eqchart'", 1)[1].split("</svg>", 1)[0]

    assert re.findall(r"data-series='([^']+)'", chart) == [
        "6040",
        "spy",
        "long-only",
        "ls",
    ]
    text_labels = re.findall(r"<text[^>]*>([^<]+)</text>", chart)
    assert text_labels
    assert all(re.fullmatch(r"(?:1|2|4|8|16)x|20\d{2}", label) for label in text_labels)


def test_page_requests_fresh_html() -> None:
    page = _page()

    assert "content='no-cache, no-store, must-revalidate'" in page


def test_page_uses_bitcoin_favicon_set() -> None:
    page = _page()

    assert "href='favicon.svg?v=bitcoin-1'" in page
    assert "href='favicon.ico?v=bitcoin-1'" in page
    assert "href='favicon-32x32.png?v=bitcoin-1'" in page
    assert "href='apple-touch-icon.png?v=bitcoin-1'" in page
    assert "content='#f7931a'" in page


def test_monthly_returns_offer_long_only_and_costed_ls_views_from_2007() -> None:
    page = _page()

    assert re.findall(r"data-return-view='([^']+)'", page) == ["long-only", "ls"]
    assert re.findall(r"data-return-panel='([^']+)'", page) == ["long-only", "ls"]
    assert page.count("<tbody><tr><td>2007</td>") == 2
    ls_panel = page.split("data-return-panel='ls'", 1)[1].split("</section>", 1)[0]
    row_2007 = ls_panel.split("<tr><td>2007</td>", 1)[1].split("</tr>", 1)[0]
    assert row_2007.count("class='na'") == 0
    assert "history begins January 2008" not in page


def test_provisional_mtd_row_updates_both_tables_without_moving_macro_signal() -> None:
    data = load_data()
    data["live_mtd"] = pd.DataFrame(
        [
            {
                "status": "UPDATED",
                "signal_date": pd.Timestamp("2026-07-31"),
                "cutoff_date": pd.Timestamp("2026-08-20"),
                "base_price_date": pd.Timestamp("2026-07-31"),
                "price_as_of": pd.Timestamp("2026-08-20"),
                "long_only_log_return": 0.02,
                "ls_log_return": 0.03,
                "spy_log_return": 0.01,
                "sixty_forty_log_return": 0.008,
            }
        ]
    )

    page = build_page(data)

    assert "August MTD through Aug 20 close" in page
    assert "point-in-time data through Jul 2026" in page
    long_panel = page.split("data-return-panel='long-only'", 1)[1].split("</section>", 1)[0]
    ls_panel = page.split("data-return-panel='ls'", 1)[1].split("</section>", 1)[0]
    long_2026 = long_panel.split("<tr><td>2026</td>", 1)[1].split("</tr>", 1)[0]
    ls_2026 = ls_panel.split("<tr><td>2026</td>", 1)[1].split("</tr>", 1)[0]
    assert "title='Aug 2026: +2.0%'" in long_2026
    assert "title='Aug 2026: +3.0%'" in ls_2026
