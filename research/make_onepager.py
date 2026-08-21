"""Build the two-page Macro Seasons v4 methodology PDF."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / "exports"
PDF_OUT = EXPORTS / "macro_seasons_v4_onepager.pdf"

INK = colors.HexColor("#17222C")
NAVY = colors.HexColor("#183246")
TEAL = colors.HexColor("#147C70")
GOLD = colors.HexColor("#B9852C")
MUTED = colors.HexColor("#596672")
SOFT = colors.HexColor("#F3F6F7")
SOFT_BLUE = colors.HexColor("#EAF1F4")
LINE = colors.HexColor("#D6DEE3")
WHITE = colors.white
SPRING = colors.HexColor("#2F8F68")
SUMMER = colors.HexColor("#B77A10")
FALL = colors.HexColor("#B6533C")
WINTER = colors.HexColor("#3F73A8")

PAGE_WIDTH, PAGE_HEIGHT = letter
LEFT_MARGIN = 0.48 * inch
RIGHT_MARGIN = 0.48 * inch
TOP_MARGIN = 0.44 * inch
BOTTOM_MARGIN = 0.42 * inch
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN


def _register_fonts() -> tuple[str, str]:
    """Use a native sans serif when available and fall back to Helvetica."""
    regular_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    bold_candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]
    regular = next((path for path in regular_candidates if path.exists()), None)
    bold = next((path for path in bold_candidates if path.exists()), None)
    if regular and bold:
        pdfmetrics.registerFont(TTFont("MethodRegular", str(regular)))
        pdfmetrics.registerFont(TTFont("MethodBold", str(bold)))
        return "MethodRegular", "MethodBold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = _register_fonts()


def text(
    value: str,
    size: float = 7.7,
    color=MUTED,
    *,
    bold: bool = False,
    leading: float | None = None,
    align: int = TA_LEFT,
) -> Paragraph:
    style = ParagraphStyle(
        name=f"method-{size}-{bold}-{align}-{leading}",
        fontName=FONT_BOLD if bold else FONT,
        fontSize=size,
        leading=leading or size * 1.30,
        textColor=color,
        alignment=align,
        spaceAfter=0,
        allowWidows=0,
        allowOrphans=0,
    )
    return Paragraph(value, style)


def section_heading(number: str, title: str, note: str = "") -> Table:
    number_box = Table(
        [[text(number, 7.4, WHITE, bold=True, align=TA_CENTER)]],
        colWidths=[0.28 * inch],
        rowHeights=[0.25 * inch],
    )
    number_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), TEAL),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    title_text = f"<b>{title}</b>"
    if note:
        title_text += f" <font color='#596672' size='7'>{note}</font>"
    table = Table(
        [[number_box, text(title_text, 10.0, NAVY, leading=11.8)]],
        colWidths=[0.35 * inch, CONTENT_WIDTH - 0.35 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return table


def page_header(title: str, subtitle: str, eyebrow: str) -> list:
    return [
        text(eyebrow.upper(), 7.2, TEAL, bold=True, leading=8.0),
        Spacer(1, 3),
        text(title, 18.5, NAVY, bold=True, leading=20.5),
        Spacer(1, 2),
        text(subtitle, 9.0, MUTED, leading=11.2),
        Spacer(1, 8),
    ]


def season_table() -> Table:
    rows = [
        ["Season", "Growth and inflation", "Economic reading", "Fixed sleeve emphasis"],
        ["SPRING", "Growth rising\nInflation cooling", "Recovery with disinflation", "Broad equities, technology, credit, housing and modest duration"],
        ["SUMMER", "Growth rising\nInflation heating", "Reflationary expansion", "Cyclicals, value, energy, materials, emerging markets, commodities and TIPS"],
        ["FALL", "Growth falling\nInflation heating", "Stagflation", "Gold, broad commodities, inflation protection, dollar, defensive equities and short duration"],
        ["WINTER", "Growth falling\nInflation cooling", "Disinflationary slowdown", "Treasuries, defensive and low-volatility equities, gold, safe-haven currencies and cash"],
    ]
    season_colors = [None, SPRING, SUMMER, FALL, WINTER]
    formatted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        formatted_row = []
        for column_index, cell in enumerate(row):
            if row_index == 0:
                formatted_row.append(text(cell, 6.8, WHITE, bold=True, leading=8.0))
            elif column_index == 0:
                formatted_row.append(text(cell, 7.2, season_colors[row_index], bold=True, leading=8.4))
            else:
                formatted_row.append(text(cell.replace("\n", "<br/>"), 7.1, INK, leading=8.7))
        formatted.append(formatted_row)
    table = Table(
        formatted,
        colWidths=[0.70 * inch, 1.35 * inch, 1.58 * inch, CONTENT_WIDTH - 3.63 * inch],
        repeatRows=1,
    )
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SOFT]),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for index, color in enumerate(season_colors[1:], start=1):
        style.append(("LINEBEFORE", (0, index), (0, index), 2.2, color))
    table.setStyle(TableStyle(style))
    return table


def pillar_cards() -> Table:
    cards = [
        (
            "GROWTH",
            "Is activity gaining or losing momentum?",
            "Industrial production and payrolls use ALFRED vintages. Jobless claims use an explicit release lag. "
            "Cyclical versus defensive equities and equities versus intermediate Treasuries confirm the reading.",
            "Main windows: 3 to 6 months",
        ),
        (
            "INFLATION",
            "Is inflation pressure building or cooling?",
            "CPI momentum uses ALFRED vintages. Five-year breakevens and oil use release-lagged FRED data. "
            "TIPS relative to nominal Treasuries adds market confirmation.",
            "Main windows: 3 to 12 months",
        ),
        (
            "LIQUIDITY",
            "Are financial conditions adding or removing support?",
            "M2 uses ALFRED vintages. Fed net liquidity, financial conditions, high-yield spreads and the "
            "two-year yield use release-lagged FRED data to capture money, credit and policy conditions.",
            "Role: changes risk intensity",
        ),
    ]
    cells = []
    for label, question, body, footer in cards:
        cells.append(
            [
                text(label, 7.0, TEAL, bold=True, leading=8.0),
                Spacer(1, 2),
                text(question, 8.0, NAVY, bold=True, leading=9.5),
                Spacer(1, 3),
                text(body, 7.0, MUTED, leading=8.7),
                Spacer(1, 4),
                text(footer, 6.4, GOLD, bold=True, leading=7.3),
            ]
        )
    table = Table([cells], colWidths=[CONTENT_WIDTH / 3.0] * 3)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def monthly_process() -> Table:
    steps = [
        ("1", "Cut the data", "Freeze the information available at the completed month-end."),
        ("2", "Build the pillars", "Measure trailing change, standardize it against prior history and combine the sub-signals."),
        ("3", "Set probabilities", "Growth and inflation distribute weight across all four seasons. No hard regime switch is required."),
        ("4", "Adjust risk", "Liquidity, real rates, credit, momentum and trend alter the blended portfolio within fixed limits."),
        ("5", "Trade next month", "Month-end weights govern the following calendar month's return. Turnover costs are charged."),
        ("6", "Lock the record", "Later data and revisions cannot change an earlier decision in the prefix-causality tests."),
    ]
    rows: list[list[list]] = [[], []]
    for index, (number, title, body) in enumerate(steps):
        rows[index // 3].append(
            [
                text(f"{number}. {title}", 7.5, NAVY, bold=True, leading=8.8),
                Spacer(1, 2),
                text(body, 6.8, MUTED, leading=8.3),
            ]
        )
    table = Table(rows, colWidths=[CONTENT_WIDTH / 3.0] * 3)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WHITE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 5.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def causal_box(audit_pass: int, audit_total: int) -> Table:
    body = (
        "<b>Causal</b> has a narrow meaning in this research. A decision for the next month can depend only "
        "on information available by the current month-end. Appending future observations must leave every "
        "earlier signal, weight and return unchanged. The model does not claim that a macro pillar proves "
        "economic causation."
    )
    data_note = (
        "CPI, industrial production, payrolls and M2 use ALFRED vintages. Other FRED inputs use explicit "
        "publication lags with latest-revised history. Yahoo adjusted prices supply the traded series. "
        f"The latest production refresh passed {audit_pass}/{audit_total} freshness checks."
    )
    table = Table(
        [[text(body, 7.2, INK, leading=9.1), text(data_note, 7.0, MUTED, leading=8.8)]],
        colWidths=[CONTENT_WIDTH * 0.53, CONTENT_WIDTH * 0.47],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT_BLUE),
                ("LINEBEFORE", (0, 0), (0, 0), 2.3, TEAL),
                ("LINEBEFORE", (1, 0), (1, 0), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def probability_note() -> Table:
    reading = (
        "<b>How to read a season percentage</b><br/>"
        "The percentages are soft classification weights. A 31% Summer reading means the current growth and "
        "inflation scores assign 31% of the season blend to Summer. It is not a calibrated 31% chance that a "
        "future event will occur."
    )
    horizon = (
        "<b>Time horizon and stability</b><br/>"
        "The reading describes the macro state at the completed month-end and sets the next calendar month's "
        "portfolio. It does not predict how long the season will last. A close top-two split keeps both sleeves "
        "active instead of forcing a full switch."
    )
    calibration = (
        "<b>Symmetric scoring</b><br/>"
        "Each sub-signal is compared with its own trailing 10-year history after a minimum 36-month warm-up. "
        "Extreme readings are capped and the pillar average receives light smoothing. Growth and inflation are "
        "then mapped symmetrically, so no season receives a built-in advantage."
    )
    table = Table(
        [[
            text(reading, 6.9, INK, leading=8.6),
            text(horizon, 6.9, INK, leading=8.6),
            text(calibration, 6.9, INK, leading=8.6),
        ]],
        colWidths=[CONTENT_WIDTH / 3.0] * 3,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBF7EE")),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def portfolio_comparison() -> Table:
    long_only = [
        text("LONG-ONLY SEASON PORTFOLIO", 7.1, GOLD, bold=True, leading=8.2),
        Spacer(1, 3),
        text("A fully funded ETF allocation designed to reduce drawdowns while retaining exposure to the active macro mix.", 8.0, NAVY, bold=True, leading=9.7),
        Spacer(1, 5),
        text("1. Blend the four fixed season sleeves using the current probabilities.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("2. Combine each sleeve's economic weights with a trailing 36-month inverse-volatility tilt.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("3. Let liquidity shift a bounded share between risky and defensive assets.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("4. Rotate the defensive sleeve between gold and nominal duration as real yields change.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("5. Halve risky positions when credit spreads are both high and widening.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("6. Apply relative momentum measured from 12 months ago to one month ago, followed by the 200-day trend gate. Failed trend positions move to BIL.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("7. Target 10% volatility from trailing 24-month risk. Scaling can reduce exposure; this track does not borrow.", 7.1, INK, leading=8.8),
        Spacer(1, 5),
        text("Backtest cost: 10 basis points per unit of monthly turnover.", 6.5, GOLD, bold=True, leading=7.6),
    ]
    long_short = [
        text("L/S PORTFOLIO", 7.1, TEAL, bold=True, leading=8.2),
        Spacer(1, 3),
        text("A separate 10% volatility strategy combining macro allocation with an independent trend-following return stream.", 8.0, NAVY, bold=True, leading=9.7),
        Spacer(1, 5),
        text("1. Start with a risk-scaled core season stream and a risk-scaled enhanced long-only stream.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("2. Add time-series momentum across 13 liquid ETFs covering equities, rates, credit, real assets and currencies.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("3. Hold an ETF long when its trailing 12-month return beats BIL and short when it trails BIL.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("4. Size trend positions from trailing 36-month volatility and cap each position at 20% of the sleeve.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("5. Weight the three streams from their trailing 24-month volatility, shifted one month, then target 10% total volatility.", 7.1, INK, leading=8.8),
        Spacer(1, 2),
        text("6. Aggregate duplicate ETF exposures, net longs against shorts, use BIL before margin borrowing and enforce gross, net, short and ticker limits.", 7.1, INK, leading=8.8),
        Spacer(1, 5),
        text("Cost model: commissions, regulatory fees, 1 bp slippage, financing, stock borrow and interest on short proceeds.", 6.5, TEAL, bold=True, leading=7.6),
    ]
    table = Table([[long_only, long_short]], colWidths=[CONTENT_WIDTH / 2.0] * 2)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#FBF7EE")),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#EEF7F5")),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def selection_and_walkforward() -> Table:
    selection = (
        "<b>How the ETF sleeves were chosen</b><br/>"
        "The season templates were set from economic exposure, liquidity and trading history. They were not "
        "selected by maximizing returns in the same season samples. Equity beta and cyclicals express growth; "
        "Treasuries and defensive equities express slowdown; TIPS, commodities and gold express inflation; "
        "BIL and major currencies provide defense or funding. Pre-inception mutual funds and futures extend "
        "selected ETF histories and introduce basis risk."
    )
    walkforward = (
        "<b>What walk-forward means</b><br/>"
        "At each month-end the model rebuilds the decision from that date's data, sets weights, and applies them "
        "to the next month's returns. Rolling risk estimates and stream weights use prior months and are shifted "
        "before use. Turnover and financing costs are charged as positions change. Prefix tests rerun the model "
        "at earlier cutoffs and require the old path to match exactly."
    )
    table = Table(
        [[text(selection, 7.2, INK, leading=9.0), text(walkforward, 7.2, INK, leading=9.0)]],
        colWidths=[CONTENT_WIDTH / 2.0] * 2,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def evidence_table(summary: pd.DataFrame, execution_summary: pd.DataFrame) -> Table:
    long_row = summary.loc["Macro Seasons v4 long-only season portfolio"]
    ls_row = execution_summary.loc["IBKR-costed executable ensemble"]
    spy_row = summary.loc["SPY"]
    benchmark_row = summary.loc["60/40 SPY/AGG"]
    rows = [
        ["Historical simulation", "Window", "CAGR", "Volatility", "Excess Sharpe", "Max drawdown"],
        ["Long-only season portfolio", "Jan 2000 - Jul 2026", f"{long_row['cagr_pct']:.2f}%", f"{long_row['ann_vol_pct']:.2f}%", f"{long_row['excess_sharpe']:.2f}", f"{long_row['max_dd_pct']:.2f}%"],
        ["L/S portfolio after IBKR costs", "Jan 2007 - Jul 2026", f"{ls_row['cagr_pct']:.2f}%", f"{ls_row['ann_vol_pct']:.2f}%", f"{ls_row['excess_sharpe']:.2f}", f"{ls_row['max_dd_pct']:.2f}%"],
        ["S&amp;P 500", "Jan 2000 - Jul 2026", f"{spy_row['cagr_pct']:.2f}%", f"{spy_row['ann_vol_pct']:.2f}%", f"{spy_row['excess_sharpe']:.2f}", f"{spy_row['max_dd_pct']:.2f}%"],
        ["60/40 SPY/AGG", "Jan 2000 - Jul 2026", f"{benchmark_row['cagr_pct']:.2f}%", f"{benchmark_row['ann_vol_pct']:.2f}%", f"{benchmark_row['excess_sharpe']:.2f}", f"{benchmark_row['max_dd_pct']:.2f}%"],
    ]
    formatted = []
    for row_index, row in enumerate(rows):
        formatted.append(
            [
                text(str(cell), 6.6, WHITE if row_index == 0 else INK, bold=(row_index == 0 or column_index == 0), leading=7.8)
                for column_index, cell in enumerate(row)
            ]
        )
    table = Table(
        formatted,
        colWidths=[1.63 * inch, 1.34 * inch, 0.70 * inch, 0.78 * inch, 0.95 * inch, CONTENT_WIDTH - 5.40 * inch],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SOFT]),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3.4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def limitations_box() -> Table:
    left = (
        "<b>What the evidence supports</b><br/>"
        "The historical record tests a fixed decision process across repeated monthly vintages. It supports "
        "comparison of drawdown, volatility and return after the stated cost rules. The 2019-2026 period has "
        "already informed research review and is not presented as an untouched holdout."
    )
    right = (
        "<b>Material limits</b><br/>"
        "Most non-core FRED series use publication lags with revised history. Yahoo prices are a public research "
        "feed. Proxy splices can differ from later ETFs. Borrow availability, taxes and market impact are modeled "
        "imperfectly. Portfolio Margin must be checked on the live IBKR account before trading."
    )
    table = Table(
        [[text(left, 6.9, INK, leading=8.6), text(right, 6.9, INK, leading=8.6)]],
        colWidths=[CONTENT_WIDTH / 2.0] * 2,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def risk_controls() -> Table:
    long_only = (
        "<b>Long-only boundary</b><br/>"
        "Weights remain fully funded. The 10% volatility target acts as a ceiling because exposure can be "
        "reduced but cannot exceed 100%. Trend failures move to BIL. This track has no short positions or margin debit."
    )
    long_short = (
        "<b>L/S hard limits</b><br/>"
        "Gross exposure is capped at 175%, net exposure at 150%, gross shorts at 35%, each non-cash ETF at "
        "25%, and margin debit at 50%. The whole book still requires an IBKR Portfolio Margin preview."
    )
    operations = (
        "<b>Monthly operation</b><br/>"
        "The completed month-end signal is frozen before the next allocation. Current-month prices update "
        "performance only and cannot alter that signal. Stale data, missing vintages or failed causality tests stop deployment."
    )
    table = Table(
        [[
            text(long_only, 6.8, INK, leading=8.4),
            text(long_short, 6.8, INK, leading=8.4),
            text(operations, 6.8, INK, leading=8.4),
        ]],
        colWidths=[CONTENT_WIDTH / 3.0] * 3,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WHITE),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def load_outputs() -> dict[str, pd.DataFrame]:
    return {
        "timeline": pd.read_csv(
            EXPORTS / "macro_seasons_v4_season_timeline.csv", parse_dates=["date"]
        ).set_index("date"),
        "summary": pd.read_csv(EXPORTS / "macro_seasons_v4_summary.csv").set_index("series"),
        "execution_summary": pd.read_csv(
            EXPORTS / "macro_seasons_v4_execution_summary.csv"
        ).set_index("series"),
        "audit": pd.read_csv(ROOT / "data" / "macro_seasons_v4_data_audit.csv"),
    }


def draw_footer(canvas, doc) -> None:
    canvas.saveState()
    y = 0.26 * inch
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.45)
    canvas.line(LEFT_MARGIN, y + 0.13 * inch, PAGE_WIDTH - RIGHT_MARGIN, y + 0.13 * inch)
    canvas.setFont(FONT, 6.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT_MARGIN, y, "newsixtyforty.com | Macro Seasons v4 methodology | research only")
    canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, y, f"Page {doc.page} of 2")
    canvas.restoreState()


def build_pdf(data: dict[str, pd.DataFrame]) -> None:
    timeline = data["timeline"].dropna(subset=["modal_season"])
    as_of = timeline.index[-1]
    audit = data["audit"]
    audit_pass = int(audit["status"].eq("PASS").sum())
    audit_total = int(len(audit))

    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=letter,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="Macro Seasons v4 - Methodology",
        author="New Sixty Forty",
        subject="Point-in-time macro seasons and portfolio construction methodology",
    )

    story = []
    story.extend(
        page_header(
            "Macro Seasons v4",
            "How point-in-time macro evidence becomes monthly season probabilities and two investable ETF portfolios.",
            f"Methodology | latest completed signal {as_of:%B %d, %Y}",
        )
    )

    intro = Table(
        [[text(
            "The model reads growth, inflation and liquidity at each completed month-end. Growth and inflation "
            "define four economic seasons. Liquidity changes how much risk the portfolio takes. The output is a "
            "probability mix across all four seasons, so one noisy release changes the portfolio gradually.",
            8.1,
            INK,
            leading=10.2,
        )]],
        colWidths=[CONTENT_WIDTH],
    )
    intro.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT_BLUE),
                ("LINEBEFORE", (0, 0), (0, 0), 2.5, TEAL),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.extend([intro, Spacer(1, 9)])

    story.extend([section_heading("1", "The four seasons"), Spacer(1, 4), season_table(), Spacer(1, 9)])
    story.extend([section_heading("2", "What enters the signal"), Spacer(1, 4), pillar_cards(), Spacer(1, 9)])
    story.extend([section_heading("3", "The monthly decision path"), Spacer(1, 4), monthly_process(), Spacer(1, 8)])
    story.extend([causal_box(audit_pass, audit_total), Spacer(1, 7), probability_note(), PageBreak()])

    story.extend(
        page_header(
            "Two portfolios, one macro engine",
            "Both tracks start from the same monthly season probabilities. Their risk budgets, return sources and implementation costs are different.",
            "Portfolio construction and walk-forward evidence",
        )
    )
    story.extend([portfolio_comparison(), Spacer(1, 9)])
    story.extend([section_heading("4", "Portfolio design and walk-forward discipline"), Spacer(1, 4), selection_and_walkforward(), Spacer(1, 8)])
    story.extend([section_heading("5", "Historical evidence", "after the stated cost rules"), Spacer(1, 4)])
    story.append(evidence_table(data["summary"], data["execution_summary"]))
    story.append(Spacer(1, 4))
    story.append(
        text(
            "The website's 9.21% return-level ensemble represents the target streams before physical netting and "
            "explicit IBKR financing. The holdings-based, costed L/S record begins in January 2007 and earns "
            "7.79% over the frozen sample.",
            6.7,
            MUTED,
            leading=8.2,
        )
    )
    story.extend([
        Spacer(1, 7),
        section_heading("6", "Risk and operating controls"),
        Spacer(1, 4),
        risk_controls(),
        Spacer(1, 7),
        limitations_box(),
        Spacer(1, 5),
    ])
    story.append(
        text(
            "Sources: Federal Reserve FRED and ALFRED, Yahoo Finance adjusted prices, and published IBKR Pro "
            "commission and financing schedules. Full source, immutable V4 inputs, hashes and monthly artifacts "
            "are retained in the release package. Historical simulations are research results, not investment advice.",
            6.4,
            NAVY,
            bold=True,
            leading=7.7,
        )
    )

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    validate_pdf()
    print(f"Wrote {PDF_OUT}")


def validate_pdf() -> None:
    reader = PdfReader(str(PDF_OUT))
    if len(reader.pages) != 2:
        raise RuntimeError(f"Expected 2 pages, found {len(reader.pages)}")
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    required = [
        "The four seasons",
        "What enters the signal",
        "The monthly decision path",
        "LONG-ONLY SEASON PORTFOLIO",
        "L/S PORTFOLIO",
        "What walk-forward means",
        "Material limits",
    ]
    missing = [phrase for phrase in required if phrase not in extracted]
    if missing:
        raise RuntimeError(f"PDF text validation failed; missing: {missing}")
    prohibited = [
        "Live monitor",
        "PENDING",
        "First post-freeze return",
        "Model target, not a broker order",
        "PM pre-trade check",
    ]
    found = [phrase for phrase in prohibited if phrase in extracted]
    if found:
        raise RuntimeError(f"PDF contains retired copy: {found}")


def main() -> None:
    build_pdf(load_outputs())


if __name__ == "__main__":
    main()
