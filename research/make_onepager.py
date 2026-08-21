"""Build the two-page public Macro Seasons methodology PDF."""

from __future__ import annotations

from pathlib import Path
import sys

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
sys.path.insert(0, str(ROOT / "research"))

import macro_seasons_v3 as V3  # noqa: E402

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
        ["Season", "Growth and inflation", "Economic reading", "Fixed allocation emphasis"],
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
            "Data and source",
            "<b>ALFRED:</b> Industrial Production (INDPRO) and Nonfarm Payrolls (PAYEMS).<br/>"
            "<b>FRED:</b> Initial Jobless Claims (ICSA).<br/>"
            "<b>Yahoo Finance:</b> XLI, XLP, SPY and IEF adjusted prices.",
            "Changes used: industrial production over 6m; payrolls over 3m; claims over 3m (falling claims count as stronger growth); XLI versus XLP and SPY versus IEF over 6m.",
        ),
        (
            "INFLATION",
            "Data and source",
            "<b>ALFRED:</b> Consumer Price Index (CPIAUCSL).<br/>"
            "<b>FRED:</b> Five-Year Breakeven Inflation (T5YIE) and WTI crude oil (DCOILWTICO).<br/>"
            "<b>Yahoo Finance:</b> TIP and IEF adjusted prices.",
            "Changes used: change in 12-month CPI inflation over 3m; breakeven inflation over 3m; WTI and TIP versus IEF over 6m.",
        ),
        (
            "LIQUIDITY",
            "Data and source",
            "<b>ALFRED:</b> M2 money stock (M2SL).<br/>"
            "<b>FRED:</b> NFCI, high-yield OAS, 2-year Treasury yield, Fed balance sheet, reverse repos and the Treasury General Account.",
            "Changes used: NFCI level and 3m change; OAS 3m; M2 6m; 2-year yield 12m; Fed net liquidity 3m.<br/>"
            "Portfolio effect: easier liquidity can move up to 15 points from defensive ETFs to risk ETFs; tighter liquidity can move up to 25 points the other way.",
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
        (
            "1",
            "Select the month-end data",
            "Use ALFRED decision-date values for CPI, industrial production, payrolls and M2. Apply the assigned publication delay to the other FRED series, which use latest-revised history. Use Yahoo adjusted prices through month-end.",
        ),
        (
            "2",
            "Calculate three scores",
            "Calculate the listed trailing changes and set their signs so positive means stronger growth, higher inflation or easier liquidity. Standardize each reading against up to 120 prior months, with at least 36 months required and values capped at three standard deviations. Average the available readings and apply a two-month exponential average.",
        ),
        (
            "3",
            "Calculate four season weights",
            "Stronger growth raises Spring and Summer; weaker growth raises Fall and Winter. Higher inflation raises Summer and Fall; lower inflation raises Spring and Winter. The four values total 100% and determine how much of each season allocation enters the portfolio. They classify current conditions rather than forecasting event probabilities.",
        ),
        (
            "4",
            "Build next month's holdings",
            "Average each fixed season allocation with a version adjusted for trailing 36-month volatility. Blend the four results using the current season percentages, then apply the liquidity, real-yield, credit, momentum, trend and portfolio-volatility rules.",
        ),
        (
            "5",
            "Record next month's return",
            "Apply the month-end weights to the following month's returns. The long-only backtest charges 10 basis points per unit of turnover. The later physical L/S simulation applies its separate trading and financing assumptions.",
        ),
    ]
    rows = [
        [
            text(number, 7.2, WHITE, bold=True, align=TA_CENTER, leading=8.0),
            text(title, 7.5, NAVY, bold=True, leading=8.8),
            text(body, 6.9, MUTED, leading=8.5),
        ]
        for number, title, body in steps
    ]
    table = Table(rows, colWidths=[0.30 * inch, 1.65 * inch, CONTENT_WIDTH - 1.95 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WHITE),
                ("BACKGROUND", (0, 0), (0, -1), TEAL),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (0, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 0),
            ]
        )
    )
    return table


def probability_note() -> Table:
    reading = (
        "<b>Reading the dashboard</b><br/>"
        "The four displayed values come from the Growth and Inflation scores and sum to 100%. The portfolio uses them directly to blend the four season allocations. "
        "A 31% Summer value assigns 31% of the pre-overlay blend to the Summer allocation; the other season allocations receive the remaining 69%. "
        "The values describe the current classification and are not calibrated forecasts of future events."
    )
    table = Table(
        [[text(reading, 7.2, INK, leading=9.0)]],
        colWidths=[CONTENT_WIDTH],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FBF7EE")),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def historical_test_design() -> Table:
    cells = [
        (
            "MONTHLY WALK-FORWARD",
            "At each month-end, calculate the signals and holdings with data dated on or before that decision. Apply those holdings only to the following month's returns, then record trading and financing costs.",
        ),
        (
            "CUT-OFF TESTS",
            "Rerun the model through earlier cut-off dates. Signals, holdings and returns through each cut-off must match the same dates in the full run. Any mismatch fails the test.",
        ),
        (
            "RESEARCH WINDOWS",
            "Portfolio rules and overlays were compared using returns through December 2018. The selected combination was then evaluated from January 2019 through May 2026. Both samples contain simulated monthly returns.",
        ),
        (
            "POINT-IN-TIME RERUN",
            "The macro history was rebuilt with decision-date vintages for CPI, industrial production, payrolls and M2. The rerun through July 2026 and the L/S execution history remain retrospective simulations.",
        ),
    ]
    formatted = []
    for heading, body in cells:
        formatted.append(
            [
                text(heading, 6.6, TEAL, bold=True, leading=7.6),
                Spacer(1, 2),
                text(body, 6.55, INK, leading=8.05),
            ]
        )
    table = Table([formatted], colWidths=[CONTENT_WIDTH / 4.0] * 4)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
                ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def portfolio_comparison() -> Table:
    long_only = [
        text("LONG-ONLY PORTFOLIO", 7.1, GOLD, bold=True, leading=8.2),
        Spacer(1, 3),
        text("Weights sum to 100%. The portfolio holds long ETF positions and BIL. It has no shorts or borrowing.", 8.0, NAVY, bold=True, leading=9.7),
        Spacer(1, 5),
        text("1. <b>Season allocation.</b> Start with the fixed percentages in section 4. Divide each ETF's starting weight by its trailing 36-month volatility, rescale the result to 100%, and average it equally with the fixed allocation.", 7.0, INK, leading=8.6),
        Spacer(1, 2),
        text("2. <b>Season and liquidity blend.</b> Combine all four adjusted season allocations using the Growth/Inflation season weights. The Liquidity score can move up to 15 percentage points from defensive ETFs to risk ETFs, or up to 25 points from risk to defense.", 7.0, INK, leading=8.6),
        Spacer(1, 2),
        text("3. <b>Real yields and credit.</b> The six-month change in the 10-year Treasury yield less five-year breakeven inflation can move up to half of the available allocation between GLD and TLT/IEF. Credit stress turns on when high-yield OAS, or the earlier-history Baa yield spread proxy, exceeds 110% of its trailing 36-month median and has widened over three months; the model then halves the risk-asset allocation.", 7.0, INK, leading=8.6),
        Spacer(1, 2),
        text("4. <b>Momentum and trend.</b> Rank current non-BIL holdings by return from 12 months ago to one month ago; the rank multiplier ranges from 0.75 to 1.25. Move each ETF below its 200-day moving average to BIL, using the 10-month average only when daily history is insufficient.", 7.0, INK, leading=8.6),
        Spacer(1, 2),
        text("5. <b>Portfolio volatility and cost.</b> Estimate the current weight vector's volatility from the trailing 24 monthly returns. Scale can range from 0.50 to 1.00 around the 10% target; unused weight goes to BIL. Charge 10 basis points per unit of turnover.", 7.0, INK, leading=8.6),
        Spacer(1, 5),
        text("This is the funded portfolio shown in the allocation table.", 6.5, GOLD, bold=True, leading=7.6),
    ]
    long_short = [
        text("L/S PORTFOLIO", 7.1, TEAL, bold=True, leading=8.2),
        Spacer(1, 3),
        text("The portfolio combines two macro-season streams with a 13-ETF trend stream. Duplicate and opposing positions are netted before trading and financing costs are applied.", 8.0, NAVY, bold=True, leading=9.7),
        Spacer(1, 5),
        text("1. <b>Two season streams.</b> The core stream uses the season blend and Liquidity overlay. The enhanced stream uses the complete long-only process. Each stream is separately scaled between 0.50 and 1.50 using volatility through the previous month.", 7.0, INK, leading=8.6),
        Spacer(1, 2),
        text("2. <b>Time-series momentum stream.</b> The 13 ETFs are SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, UUP, FXY, FXF and HYG. Each is long when its trailing 12-month return exceeds BIL and short otherwise.", 7.0, INK, leading=8.6),
        Spacer(1, 2),
        text("3. <b>Momentum sizing.</b> Use trailing 36-month volatility so a 10%-volatility ETF receives roughly one-thirteenth of the stream. Cap each position at 20% of the stream, then scale the stream between 0.50 and 1.50 toward 10% volatility.", 7.0, INK, leading=8.6),
        Spacer(1, 2),
        text("4. <b>Combine streams.</b> Weight the three streams inversely to their trailing 24-month volatility, measured through the previous month. Scale the combined return stream between 0.50 and 1.50 toward 10% volatility.", 7.0, INK, leading=8.6),
        Spacer(1, 2),
        text("5. <b>Execution and costs.</b> Add duplicate positions, cancel opposing exposures, reduce BIL before recording a USD debit, and apply the gross, net, short, asset and borrowing limits. Charge commissions, regulatory fees, slippage, margin interest and short-position costs.", 7.0, INK, leading=8.6),
        Spacer(1, 5),
        text("Long-only and L/S returns are reported separately.", 6.5, TEAL, bold=True, leading=7.6),
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


def _template_text(season: str) -> str:
    parts = []
    for ticker, weight in V3.TEMPLATES[season].items():
        pct = weight * 100.0
        number = f"{pct:.1f}".rstrip("0").rstrip(".")
        parts.append(f"{ticker} {number}%")
    return ", ".join(parts)


def season_allocations() -> Table:
    rationale = {
        "SPRING": (
            "SPY supplies broad equity exposure; QQQ and SMH add growth and technology; "
            "IWM and XHB add domestic cyclical and housing exposure; HYG and LQD add credit; "
            "IEF adds duration; GLD diversifies the allocation."
        ),
        "SUMMER": (
            "XLE, XLB, XLI, VLUE, IWM and EEM target a reflationary expansion. DBC, CPER "
            "and GLD add commodity exposure, while TIP adds inflation-linked bonds."
        ),
        "FALL": (
            "GLD, DBC, XLE and TIP target inflation and real assets; UUP adds dollar exposure; "
            "XLP, XLU and XLV add defensive equities; SHY and BIL keep duration short."
        ),
        "WINTER": (
            "TLT, IEF and SHY add Treasury duration; USMV, XLP and XLV favor defensive "
            "equities; GLD, FXY and FXF add safe-haven exposure; BIL holds cash."
        ),
    }
    intro = (
        "<b>How the ETFs and starting weights were set</b><br/>"
        "ETF choice follows each asset's economic role. The exact percentages are fixed portfolio-design "
        "choices rather than model estimates. The main expression of a season carries 12.5%-20%, most "
        "supporting exposures carry 10%, and narrower diversifiers carry 5%-7.5%."
    )
    season_colors = {
        "SPRING": SPRING,
        "SUMMER": SUMMER,
        "FALL": FALL,
        "WINTER": WINTER,
    }
    cards = {}
    for season in V3.SEASONS:
        cards[season] = [
            text(season.title(), 6.8, season_colors[season], bold=True, leading=8.0),
            Spacer(1, 1.5),
            text(rationale[season], 6.15, INK, leading=7.35),
            Spacer(1, 2),
            text(
                f"<b>Starting weights:</b> {_template_text(season)}",
                6.0,
                MUTED,
                leading=7.2,
            ),
        ]
    rows = [
        [text(intro, 6.55, INK, leading=8.0), ""],
        [cards["SPRING"], cards["SUMMER"]],
        [cards["FALL"], cards["WINTER"]],
    ]
    table = Table(
        rows,
        colWidths=[CONTENT_WIDTH / 2.0] * 2,
    )
    style = [
        ("SPAN", (0, 0), (-1, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), SOFT_BLUE),
        ("BACKGROUND", (0, 1), (-1, -1), SOFT),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 1), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 5),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
        ("LINEBEFORE", (0, 1), (0, 1), 2.0, SPRING),
        ("LINEBEFORE", (1, 1), (1, 1), 2.0, SUMMER),
        ("LINEBEFORE", (0, 2), (0, 2), 2.0, FALL),
        ("LINEBEFORE", (1, 2), (1, 2), 2.0, WINTER),
    ]
    table.setStyle(TableStyle(style))
    return table


def risk_controls() -> Table:
    long_only = (
        "<b>Long-only limits</b><br/>"
        "ETF and BIL weights total 100%. Exposure can fall as low as 50% when volatility is high; the remainder goes to BIL. "
        "Exposure cannot exceed 100%. Shorts and margin borrowing are disabled."
    )
    long_short = (
        "<b>L/S limits</b><br/>"
        "Maximum gross exposure: 175%. Maximum net exposure: 150%. Maximum gross shorts: 35%. Maximum absolute weight per non-cash ETF: 25%. "
        "Maximum USD margin debit: 50%. IBKR calculates the final Portfolio Margin requirement on the live account."
    )
    operations = (
        "<b>Execution cost assumptions</b><br/>"
        "Assumptions reviewed August 20, 2026: $0.005 per share, a $1 order minimum, US regulatory-fee rates, 1 basis point of slippage and the applicable margin and short-proceeds tiers. "
        "It assumes 1% annual borrow for every short ETF because historical borrow data is unavailable. Taxes and market impact are excluded."
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
    }


def draw_footer(canvas, doc) -> None:
    canvas.saveState()
    y = 0.26 * inch
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.45)
    canvas.line(LEFT_MARGIN, y + 0.13 * inch, PAGE_WIDTH - RIGHT_MARGIN, y + 0.13 * inch)
    canvas.setFont(FONT, 6.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(LEFT_MARGIN, y, "newsixtyforty.com | Macro Seasons methodology | research only")
    canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, y, f"Page {doc.page} of 2")
    canvas.restoreState()


def build_pdf(data: dict[str, pd.DataFrame]) -> None:
    timeline = data["timeline"].dropna(subset=["modal_season"])
    as_of = timeline.index[-1]

    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=letter,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="Macro Seasons - Methodology",
        author="New Sixty Forty",
        subject="Point-in-time macro seasons and portfolio construction methodology",
    )

    story = []
    story.extend(
        page_header(
            "Macro Seasons",
            "Data sources, season rules and monthly portfolio construction.",
            f"Methodology | latest completed signal {as_of:%B %d, %Y}",
        )
    )

    intro = Table(
        [[text(
            "At each month-end, the model calculates one Growth score, one Inflation score and one Liquidity "
            "score. Growth and Inflation set the four season weights. Liquidity changes the final mix between "
            "risk and defensive ETFs.",
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

    story.extend([section_heading("1", "The four seasons"), Spacer(1, 4), season_table(), Spacer(1, 6)])
    story.extend([section_heading("2", "Data used for each pillar"), Spacer(1, 4), pillar_cards(), Spacer(1, 6)])
    story.extend([section_heading("3", "From month-end data to next month's portfolio"), Spacer(1, 4), monthly_process(), Spacer(1, 4)])
    story.extend([probability_note(), Spacer(1, 4)])
    story.append(PageBreak())

    story.extend(
        page_header(
            "Portfolio construction and validation",
            "How the long-only and L/S portfolios are built, tested and constrained.",
            "Methodology",
        )
    )
    story.extend([
        section_heading("4", "Season allocations"),
        Spacer(1, 4),
        season_allocations(),
        Spacer(1, 6),
        section_heading("5", "Portfolio construction"),
        Spacer(1, 4),
        portfolio_comparison(),
        Spacer(1, 6),
        section_heading("6", "Walk-forward research design"),
        Spacer(1, 4),
        historical_test_design(),
        Spacer(1, 5),
        section_heading("7", "Risk limits and trading costs"),
        Spacer(1, 4),
        risk_controls(),
        Spacer(1, 4),
    ])
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
        "Data used for each pillar",
        "From month-end data to next month's portfolio",
        "Season allocations",
        "Portfolio construction",
        "LONG-ONLY PORTFOLIO",
        "L/S PORTFOLIO",
        "Walk-forward research design",
        "Risk limits and trading costs",
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
        "Starting template weights in V3/V4",
        "What the research record supports",
        "June prototype",
        "V2 REDESIGN",
        "V3 RESEARCH",
        "V4 AND EXECUTION",
        "Config A",
        "Git commit",
        "signal freeze",
        "Methodology provenance",
        "Version history",
        "selection record",
        "repository",
        "freeze document",
        "V2 ",
        "V3 ",
        "V4 ",
        "H1 through H8",
        "sleeve",
        "Historical inputs",
        "Liquidity data limitation",
    ]
    found = [phrase for phrase in prohibited if phrase.lower() in extracted.lower()]
    if found:
        raise RuntimeError(f"PDF contains retired copy: {found}")


def main() -> None:
    build_pdf(load_outputs())


if __name__ == "__main__":
    main()
