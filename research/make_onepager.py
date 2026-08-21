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
            "Use macro releases available by that date and ETF adjusted prices through that date.",
        ),
        (
            "2",
            "Calculate three scores",
            "Calculate each listed change. Set the sign so positive means stronger growth, higher inflation or easier liquidity. Compare each result with that series' previous 10 years, average the readings within each pillar and lightly smooth the average.",
        ),
        (
            "3",
            "Calculate four season weights",
            "Stronger growth raises Spring and Summer; weaker growth raises Fall and Winter. Higher inflation raises Summer and Fall; lower inflation raises Spring and Winter. The four weights total 100%.",
        ),
        (
            "4",
            "Build next month's holdings",
            "Blend the four fixed ETF sleeves using the season weights. Then apply the liquidity shift, gold-versus-Treasury adjustment, credit-spread cut, momentum tilt and 200-day trend check.",
        ),
        (
            "5",
            "Record next month's return",
            "Keep those holdings for the following calendar month, charge turnover and financing costs, then repeat the process after the next month-end.",
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
        "The dashboard calls the four numbers season probabilities. The portfolio uses them as blend weights. "
        "A 31% Summer reading puts 31% of the base allocation in the Summer ETF sleeve. Spring, Fall and Winter "
        "receive the remaining 69% in their displayed proportions. The new weights take effect for the next calendar month."
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


def portfolio_comparison() -> Table:
    long_only = [
        text("LONG-ONLY PORTFOLIO", 7.1, GOLD, bold=True, leading=8.2),
        Spacer(1, 3),
        text("Weights sum to 100%. The portfolio holds long ETF positions and BIL. It has no shorts or borrowing.", 8.0, NAVY, bold=True, leading=9.7),
        Spacer(1, 5),
        text("1. <b>Season blend.</b> Combine the fixed Spring, Summer, Fall and Winter ETF weights using the four season weights.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("2. <b>Within each season.</b> Use Yahoo Finance adjusted prices to calculate each ETF's volatility over the previous 36 months. Average the fixed weights with weights that give less to high-volatility ETFs and more to low-volatility ETFs.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("3. <b>Liquidity.</b> Easier liquidity can move up to 15 percentage points from defensive ETFs to risk ETFs. Tighter liquidity can move up to 25 points the other way.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("4. <b>Real yields.</b> From FRED, use the 10-year Treasury yield (DGS10) and five-year breakeven inflation (T5YIE). When the six-month real-yield change is negative and breakeven inflation is flat or rising, move part of TLT and IEF to GLD. Every other non-zero real-yield change moves part of GLD to TLT and IEF. The move cannot exceed half of the source sleeve.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("5. <b>Credit.</b> Use FRED high-yield OAS (BAMLH0A0HYM2); before its usable history, use Moody's Baa yield minus the 10-year Treasury yield. When the spread exceeds 110% of its previous 36-month median and has widened over three months, cut each risk ETF weight in half and move the released weight to defensive ETFs.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("6. <b>Momentum and trend.</b> Use Yahoo Finance adjusted prices. Rank non-cash holdings by return from 12 months ago to one month ago. Adjust each weight between 0.75 and 1.25 times its prior weight. Move any ETF below its 200-day moving average entirely to BIL.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("7. <b>Portfolio volatility.</b> Estimate volatility from the previous 24 months. When it exceeds 10%, reduce all ETF weights and place the difference in BIL. Exposure never rises above 100%.", 7.3, INK, leading=9.1),
        Spacer(1, 5),
        text("Backtest trading cost: 10 basis points per unit of monthly turnover.", 6.5, GOLD, bold=True, leading=7.6),
    ]
    long_short = [
        text("L/S PORTFOLIO", 7.1, TEAL, bold=True, leading=8.2),
        Spacer(1, 3),
        text("The model combines three return streams, converts them into ETF positions, nets duplicate exposures, and charges modeled IBKR trading and financing costs.", 8.0, NAVY, bold=True, leading=9.7),
        Spacer(1, 5),
        text("1. <b>Core season stream.</b> Use the season blend and liquidity adjustment. Scale exposure between 0.5 and 1.5 times to target 10% volatility from the previous 24 months.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("2. <b>Enhanced season stream.</b> Use the complete long-only process on the left, then allow the same 0.5 to 1.5 exposure range around a 10% volatility target.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("3. <b>Trend stream.</b> Use Yahoo Finance adjusted prices for SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, DBC, UUP, FXY, FXF and HYG. Hold an ETF long when its previous 12-month return beats BIL and short when it trails BIL.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("4. <b>Trend position size.</b> Use the previous 36 months of volatility. Lower-volatility ETFs receive larger positions. Each position is capped at 20% of the trend stream.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("5. <b>Combine the streams.</b> Give more weight to streams with lower volatility over the previous 24 months. The volatility window ends one month before the new weights. Scale the combined portfolio between 0.5 and 1.5 times toward 10% volatility.", 7.3, INK, leading=9.1),
        Spacer(1, 2),
        text("6. <b>Create one position per ETF.</b> Add duplicate exposures from the three streams, cancel opposing long and short positions, and sell BIL before creating a USD margin debit.", 7.3, INK, leading=9.1),
        Spacer(1, 5),
        text("Costs: published IBKR Pro commissions, regulatory fees and financing tiers; 1 basis point of slippage; 1% annual short borrow; IBKR interest on short-sale collateral.", 6.5, TEAL, bold=True, leading=7.6),
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
        "<b>Fixed ETF templates</b><br/>"
        "<b>Spring:</b> SPY, QQQ, SMH, IWM, XHB, HYG, LQD, IEF, GLD.<br/>"
        "<b>Summer:</b> XLE, XLB, XLI, VLUE, IWM, EEM, DBC, CPER, GLD, TIP.<br/>"
        "<b>Fall:</b> GLD, DBC, XLE, TIP, UUP, XLP, XLU, XLV, SHY, BIL.<br/>"
        "<b>Winter:</b> TLT, IEF, SHY, GLD, USMV, XLP, XLV, FXY, FXF, BIL.<br/>"
        "The base weights were set from each ETF's economic exposure before the backtest. The code did not optimize those weights on season returns."
    )
    walkforward = (
        "<b>Backtest timing</b><br/>"
        "1. Use data available by the historical month-end.<br/>"
        "2. Calculate the season scores and ETF weights.<br/>"
        "3. Apply those weights to the following month's ETF returns.<br/>"
        "4. Charge trading, borrowing and short-position costs for that month.<br/>"
        "5. Repeat at the next month-end.<br/>"
        "A separate check reruns the model through older cutoff dates. The signals and weights through each cutoff must match the full run."
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


def backtest_coverage(summary: pd.DataFrame, execution_summary: pd.DataFrame) -> Table:
    long_row = summary.loc["Macro Seasons v4 long-only season portfolio"]
    ls_row = execution_summary.loc["IBKR-costed executable ensemble"]
    long_only = (
        "<b>Long-only</b><br/>"
        "Monthly from January 2000 through July 2026. Month-end weights earn the following month's return. "
        f"After the 10 bp turnover cost: CAGR {long_row['cagr_pct']:.2f}%, volatility {long_row['ann_vol_pct']:.2f}%, maximum drawdown {long_row['max_dd_pct']:.2f}%."
    )
    long_short = (
        "<b>L/S physical portfolio</b><br/>"
        "Monthly from January 2007 through July 2026. Includes the IBKR costs listed below. "
        f"CAGR {ls_row['cagr_pct']:.2f}%, volatility {ls_row['ann_vol_pct']:.2f}%, maximum drawdown {ls_row['max_dd_pct']:.2f}%."
    )
    comparison = (
        "<b>Why the website also shows 9.21%</b><br/>"
        "That figure combines the three return streams before physical ETF netting and full IBKR costs. "
        "The 7.79% L/S figure comes from the physical, costed simulation."
    )
    table = Table(
        [[
            text(long_only, 6.8, INK, leading=8.4),
            text(long_short, 6.8, INK, leading=8.4),
            text(comparison, 6.8, INK, leading=8.4),
        ]],
        colWidths=[CONTENT_WIDTH / 3.0] * 3,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SOFT),
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


def limitations_box() -> Table:
    left = (
        "<b>How to read the backtest</b><br/>"
        "The 2019-2026 results influenced later research decisions, so that period belongs to the reviewed sample. "
        "The reported returns include the cost rules described above. They remain historical simulations rather than live account returns."
    )
    right = (
        "<b>Data and execution limits</b><br/>"
        "Full ALFRED histories cover CPI, industrial production, payrolls and M2. Other FRED series use their release delay with latest-revised history. "
        "Yahoo Finance supplies adjusted prices. Mutual funds and futures extend selected ETF histories before inception. "
        "Actual taxes, borrow availability and market impact can differ from the model."
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
        "<b>Cost inputs</b><br/>"
        "IBKR Pro Fixed commission: $0.005 per share, $1 minimum and 1% order-value cap. The model adds current US regulatory fees, "
        "1 basis point of slippage, published margin-rate tiers, 1% annual ETF short borrow and IBKR short-proceeds interest tiers."
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
    canvas.drawString(LEFT_MARGIN, y, "newsixtyforty.com | Macro Seasons v4 methodology | research only")
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
        title="Macro Seasons v4 - Methodology",
        author="New Sixty Forty",
        subject="Point-in-time macro seasons and portfolio construction methodology",
    )

    story = []
    story.extend(
        page_header(
            "Macro Seasons v4",
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

    story.extend([section_heading("1", "The four seasons"), Spacer(1, 4), season_table(), Spacer(1, 9)])
    story.extend([section_heading("2", "Data used for each pillar"), Spacer(1, 4), pillar_cards(), Spacer(1, 9)])
    story.extend([section_heading("3", "From month-end data to next month's portfolio"), Spacer(1, 4), monthly_process(), Spacer(1, 8)])
    story.extend([
        probability_note(),
        Spacer(1, 8),
        section_heading("4", "ETF templates and backtest timing"),
        Spacer(1, 4),
        selection_and_walkforward(),
        PageBreak(),
    ])

    story.extend(
        page_header(
            "Two portfolios, one macro engine",
            "Long-only holds a funded ETF portfolio. L/S combines three return streams, shorts selected ETFs and models IBKR financing.",
            "Portfolio rules and backtest timing",
        )
    )
    story.extend([portfolio_comparison(), Spacer(1, 9)])
    story.extend([section_heading("5", "Backtest coverage"), Spacer(1, 4)])
    story.append(backtest_coverage(data["summary"], data["execution_summary"]))
    story.extend([
        Spacer(1, 7),
        section_heading("6", "Limits and cost assumptions"),
        Spacer(1, 4),
        risk_controls(),
        Spacer(1, 7),
        limitations_box(),
        Spacer(1, 5),
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
        "LONG-ONLY PORTFOLIO",
        "L/S PORTFOLIO",
        "Backtest timing",
        "Data and execution limits",
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
