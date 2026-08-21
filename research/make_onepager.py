"""Build the Macro Seasons v4 point-in-time method summary PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / "exports"
CHART_PNG = ROOT / "tmp" / "pdfs" / "macro_seasons_v4_chart.png"
PDF_OUT = EXPORTS / "macro_seasons_v4_onepager.pdf"

NAVY = colors.HexColor("#12233a")
TEAL = colors.HexColor("#168f78")
GOLD = colors.HexColor("#c99931")
GRAY = colors.HexColor("#5a6472")
LIGHT = colors.HexColor("#eef1f5")
LINE = colors.HexColor("#cbd2db")
WHITE = colors.white
SEASON_COLORS = {
    "SPRING": "#3bb273", "SUMMER": "#e0a62b",
    "FALL": "#db6b4f", "WINTER": "#4d8ad8",
}
TICKER_NAMES = {
    "BIL": "T-Bills", "HYG": "High yield", "XLE": "Energy", "DBC": "Commodities",
    "IWM": "Small caps", "SPY": "S&amp;P 500", "QQQ": "Nasdaq 100", "EEM": "Emerging markets",
    "SHY": "1-3y Treasuries", "XLI": "Industrials", "VLUE": "Value", "XLB": "Materials",
    "XLP": "Staples", "XLV": "Healthcare", "SMH": "Semiconductors", "UUP": "US dollar",
    "CPER": "Copper", "USMV": "Min volatility",
}


def para(text: str, size: float = 7.0, color=GRAY, bold: bool = False,
         leading: float | None = None) -> Paragraph:
    style = ParagraphStyle(
        "body", fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size, leading=leading or size * 1.27, textColor=color,
        alignment=TA_LEFT, spaceAfter=0,
    )
    return Paragraph(text, style)


def heading(text: str) -> Paragraph:
    return para(text.upper(), 7.1, NAVY, True, 8.1)


def load_outputs() -> dict[str, pd.DataFrame]:
    return {
        "long": pd.read_csv(
            EXPORTS / "macro_seasons_v4_long_only_ledger.csv", parse_dates=["return_date"]
        ).set_index("return_date"),
        "ensemble": pd.read_csv(
            EXPORTS / "macro_seasons_v4_ensemble_ledger.csv", parse_dates=["return_date"]
        ).set_index("return_date"),
        "timeline": pd.read_csv(
            EXPORTS / "macro_seasons_v4_season_timeline.csv", parse_dates=["date"]
        ).set_index("date"),
        "summary": pd.read_csv(EXPORTS / "macro_seasons_v4_summary.csv").set_index("series"),
        "allocation": pd.read_csv(EXPORTS / "macro_seasons_v4_current_allocation.csv"),
        "monitor": pd.read_csv(EXPORTS / "macro_seasons_v4_monitor.csv"),
        "audit": pd.read_csv(ROOT / "data" / "macro_seasons_v4_data_audit.csv"),
    }


def build_chart(data: dict[str, pd.DataFrame]) -> None:
    CHART_PNG.parent.mkdir(parents=True, exist_ok=True)
    long_only, ensemble, timeline = data["long"], data["ensemble"], data["timeline"]
    curves = {
        "Leveraged ensemble": np.exp(ensemble["strategy_return"].cumsum()),
        "Long-only season portfolio": np.exp(long_only["strategy_return"].cumsum()),
        "S&P 500": np.exp(long_only["spy_return"].cumsum()),
        "60/40": np.exp(long_only["sixty_forty_return"].cumsum()),
    }
    palette = {
        "Leveraged ensemble": "#168f78", "Long-only season portfolio": "#c99931",
        "S&P 500": "#4d8ad8", "60/40": "#8792a2",
    }
    fig, (ax, strip_ax) = plt.subplots(
        2, 1, figsize=(7.6, 2.45), dpi=210, sharex=True,
        gridspec_kw={"height_ratios": [6.0, 0.55], "hspace": 0.04},
    )
    for name, curve in curves.items():
        width = 1.8 if name == "Leveraged ensemble" else 1.05
        style = "--" if name == "60/40" else "-"
        ax.plot(curve.index, curve, label=name, color=palette[name], lw=width, ls=style)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 4, 8])
    ax.set_yticklabels(["1x", "2x", "4x", "8x"], fontsize=6.4)
    ax.grid(True, axis="y", lw=0.35, alpha=0.35)
    ax.legend(fontsize=6.2, loc="upper left", frameon=False, ncol=4, handlelength=1.7)
    ax.tick_params(axis="x", labelsize=6.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Growth of $1 - point-in-time historical simulation after costs (log scale)",
                 fontsize=7.5, loc="left", color="#12233a", fontweight="bold")

    strip = timeline.reindex(curves["Long-only season portfolio"].index, method="ffill")
    for date, row in strip.dropna(subset=["modal_season"]).iterrows():
        strip_ax.axvspan(
            date, date + pd.offsets.MonthEnd(1),
            color=SEASON_COLORS.get(str(row["modal_season"]), "#999"), lw=0,
        )
    strip_ax.set_yticks([])
    strip_ax.set_ylabel("Season", fontsize=6, rotation=0, labelpad=19, color="#5a6472")
    strip_ax.xaxis.set_major_locator(mdates.YearLocator(4))
    strip_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    strip_ax.tick_params(axis="x", labelsize=6.2, length=0)
    for spine in ("top", "right", "left", "bottom"):
        strip_ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.88, bottom=0.12)
    fig.savefig(CHART_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def performance_table(summary: pd.DataFrame) -> Table:
    rows = [["Portfolio", "CAGR", "Vol", "Excess Sharpe", "Max loss", "Calmar"]]
    display = [
        ("Macro Seasons v4 long-only season portfolio", "Long-only season portfolio"),
        ("Macro Seasons v4 multi-strategy ensemble", "Leveraged ensemble"),
        ("SPY", "S&amp;P 500"),
        ("60/40 SPY/AGG", "60/40"),
    ]
    for key, label in display:
        row = summary.loc[key]
        rows.append([
            label, f"{row['cagr_pct']:.1f}%", f"{row['ann_vol_pct']:.1f}%",
            f"{row['excess_sharpe']:.2f}", f"{row['max_dd_pct']:.1f}%", f"{row['calmar']:.2f}",
        ])
    table = Table(
        [[para(str(cell), 6.6, WHITE if i == 0 else GRAY, bold=(i == 0 or j == 0))
          for j, cell in enumerate(row)] for i, row in enumerate(rows)],
        colWidths=[2.2 * inch, .72 * inch, .68 * inch, 1.05 * inch, .8 * inch, .72 * inch],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT]),
        ("GRID", (0, 0), (-1, -1), .35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def allocation_table(allocation: pd.DataFrame) -> Table:
    ordered = allocation.sort_values("weight_pct", ascending=False).reset_index(drop=True)
    left = ordered.iloc[:9]
    right = ordered.iloc[9:].reset_index(drop=True)
    rows = [["Ticker", "Asset", "Weight", "", "Ticker", "Asset", "Weight"]]
    for idx in range(9):
        lrow = left.iloc[idx]
        rrow = right.iloc[idx] if idx < len(right) else None
        rows.append([
            str(lrow["ticker"]), TICKER_NAMES.get(str(lrow["ticker"]), str(lrow["ticker"])),
            f"{float(lrow['weight_pct']):.2f}%", "",
            str(rrow["ticker"]) if rrow is not None else "",
            TICKER_NAMES.get(str(rrow["ticker"]), str(rrow["ticker"])) if rrow is not None else "",
            f"{float(rrow['weight_pct']):.2f}%" if rrow is not None else "",
        ])
    table = Table(
        [[para(str(cell), 6.2, WHITE if i == 0 and j != 3 else GRAY,
               bold=(i == 0 or j in (0, 4))) for j, cell in enumerate(row)]
         for i, row in enumerate(rows)],
        colWidths=[.5 * inch, 1.42 * inch, .58 * inch, .18 * inch, .5 * inch, 1.42 * inch, .58 * inch],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (2, 0), NAVY),
        ("BACKGROUND", (4, 0), (6, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (2, -1), [WHITE, LIGHT]),
        ("ROWBACKGROUNDS", (4, 1), (6, -1), [WHITE, LIGHT]),
        ("GRID", (0, 0), (2, -1), .3, LINE),
        ("GRID", (4, 0), (6, -1), .3, LINE),
        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
        ("ALIGN", (6, 1), (6, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.6),
    ]))
    return table


def build_pdf(data: dict[str, pd.DataFrame]) -> None:
    summary, timeline = data["summary"], data["timeline"]
    allocation, monitor, audit = data["allocation"], data["monitor"], data["audit"]
    latest = timeline.dropna(subset=["modal_season"]).iloc[-1]
    as_of = timeline.dropna(subset=["modal_season"]).index[-1]
    modal = str(latest["modal_season"])
    effective_month = pd.Timestamp(allocation["effective_month"].iloc[0]).strftime("%B %Y")
    ranked = sorted(["SPRING", "SUMMER", "FALL", "WINTER"], key=lambda key: -float(latest[key]))
    probabilities = " | ".join(f"{season.title()} {float(latest[season]):.1%}" for season in ranked)
    audit_pass = int(audit["status"].eq("PASS").sum())
    alfred_pass = int(((audit["component"] == "alfred_vintage") & audit["status"].eq("PASS")).sum())
    monitor_status = "PENDING" if monitor["status"].astype(str).str.startswith("PENDING").all() else "REVIEW"

    doc = SimpleDocTemplate(
        str(PDF_OUT), pagesize=letter, leftMargin=.46 * inch, rightMargin=.46 * inch,
        topMargin=.35 * inch, bottomMargin=.3 * inch,
        title="Macro Seasons v4 - Point-in-Time Model Summary",
        author="Macro Scenario Dashboard",
    )
    story = [
        para("MACRO SEASONS v4", 15.0, NAVY, True, 16.0),
        para("Point-in-time regime portfolio and separate multi-strategy research ensemble", 8.1, TEAL, True),
        Spacer(1, 3),
        para(
            "Frozen August 20, 2026. Monthly decisions use data available at each historical month-end. "
            "CPI, industrial production, payrolls and M2 are reconstructed from ALFRED vintages. "
            "Yahoo and FRED inputs are refreshed and freshness-gated before each production run.",
            6.8, GRAY,
        ),
        Spacer(1, 5),
    ]

    current = Table([[
        para(f"<b>{effective_month} season call</b><br/><font size='13'>{modal.title()}</font><br/>{probabilities}", 7.0, NAVY),
        para(
            f"<b>Production integrity</b><br/>{audit_pass}/{len(audit)} freshness checks PASS; "
            f"{alfred_pass}/4 ALFRED series PASS.<br/>Live monitor: <b>{monitor_status}</b>. "
            "First post-freeze return due September 30, 2026.", 6.8, GRAY,
        ),
    ]], colWidths=[3.68 * inch, 3.38 * inch])
    current.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("LINEBEFORE", (0, 0), (0, 0), 2.2, GOLD),
        ("LINEBEFORE", (1, 0), (1, 0), 1.0, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([current, Spacer(1, 5)])

    seasons_text = (
        "<b>Spring:</b> growth rising, inflation cooling. <b>Summer:</b> growth rising, inflation heating. "
        "<b>Fall:</b> growth falling, inflation heating. <b>Winter:</b> growth falling, inflation cooling. "
        "Liquidity changes probabilities and risk intensity, but not the quadrant labels."
    )
    construction_text = (
        "<b>Long-only:</b> a 100% ETF portfolio blending all four season sleeves, with real-rate, credit, "
        "momentum, trend and volatility controls. <b>Ensemble:</b> a distinct leveraged research strategy "
        "combining levered core, levered long-only and time-series momentum. Its return must not be read as "
        "the return of the ETF weights shown below."
    )
    methods = Table([[
        [heading("Season definition"), Spacer(1, 2), para(seasons_text, 6.55)],
        [heading("Two portfolio tracks"), Spacer(1, 2), para(construction_text, 6.55)],
    ]], colWidths=[3.54 * inch, 3.54 * inch])
    methods.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 8),
        ("LEFTPADDING", (1, 0), (1, 0), 8),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
    ]))
    story.extend([methods, Spacer(1, 4), Image(str(CHART_PNG), width=7.05 * inch, height=2.28 * inch), Spacer(1, 4)])

    years = float(summary.loc["Macro Seasons v4 long-only season portfolio", "months"]) / 12.0
    story.extend([
        heading(f"Historical results - {years:.1f} years, after modeled costs"),
        Spacer(1, 2), performance_table(summary), Spacer(1, 4),
        para(
            "The table is a historical PIT backtest, not a live track record. Excess Sharpe is return above "
            "cash per unit of volatility. Calmar is CAGR divided by the absolute maximum drawdown. The "
            "2019-2026 period has already informed research review and is not claimed as a new untouched holdout.",
            6.15, GRAY,
        ),
        Spacer(1, 4),
        heading(f"Exact long-only target weights - {effective_month}"), Spacer(1, 2),
        allocation_table(allocation), Spacer(1, 4),
        para(
            f"Signal date: {as_of:%B %d, %Y}. Exact weights sum to {allocation['weight'].sum():.6f}. "
            "No sub-2% positions are hidden or redistributed. The 40.81% BIL allocation reflects the fresh "
            "daily trend gate; TIP, XHB and XLU did not pass that gate for this rebalance.",
            6.15, GRAY,
        ),
        Spacer(1, 4),
        para(
            "Sources: Federal Reserve FRED/ALFRED and Yahoo Finance. Frozen inputs and SHA-256 hashes are "
            "stored with the V4 release. Research only; not investment advice.",
            6.1, NAVY, True,
        ),
    ])
    doc.build(story)
    print(f"Wrote {PDF_OUT}")


def main() -> None:
    data = load_outputs()
    build_chart(data)
    build_pdf(data)


if __name__ == "__main__":
    main()
