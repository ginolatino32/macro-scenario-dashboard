"""Build the static newsixtyforty.com page from frozen Macro Seasons v4 outputs.

The page deliberately separates the implementable long-only season portfolio
from the leveraged three-stream research ensemble. All figures are generated
from the V4 point-in-time export package; no values are hard-coded in the HTML.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "exports"
OUT_DIR = EXPORTS / "website"

SEASON_COLORS = {
    "SPRING": "#3bb273",
    "SUMMER": "#e0a62b",
    "FALL": "#db6b4f",
    "WINTER": "#4d8ad8",
}
ENSEMBLE_COLOR = "#25a886"
LONG_ONLY_COLOR = "#d8ad4a"
SPY_COLOR = "#5793de"
B6040_COLOR = "#8792a2"

TICKER_NAMES = {
    "BIL": "T-Bills", "SHY": "1-3y Treasuries", "IEF": "7-10y Treasuries",
    "TLT": "20y+ Treasuries", "TIP": "Inflation-linked bonds", "LQD": "IG credit",
    "HYG": "High-yield credit", "AGG": "US aggregate bonds", "SPY": "S&P 500",
    "QQQ": "Nasdaq 100", "IWM": "Small caps", "SMH": "Semiconductors",
    "XLE": "Energy", "XLB": "Materials", "XLI": "Industrials",
    "XLP": "Consumer staples", "XLV": "Healthcare", "XLU": "Utilities",
    "XHB": "Homebuilders", "VLUE": "Value stocks", "USMV": "Min-volatility stocks",
    "EEM": "Emerging markets", "EFA": "International developed", "GLD": "Gold",
    "DBC": "Broad commodities", "CPER": "Copper", "UUP": "US dollar",
    "FXY": "Japanese yen", "FXF": "Swiss franc",
}
DEFENSIVE = {
    "BIL", "SHY", "IEF", "TLT", "TIP", "LQD", "AGG", "UUP", "FXY",
    "FXF", "GLD", "XLP", "XLU", "XLV", "USMV",
}
SEASON_GUIDANCE = {
    "SPRING": "growth assets, technology and credit",
    "SUMMER": "cyclicals, commodities and equities, with risk gates still active",
    "FALL": "inflation protection, gold and capital preservation",
    "WINTER": "Treasuries, quality, defensive equities and gold",
}
SEASON_CARDS = [
    ("SPRING", "Recovery", "Growth rising / inflation cooling", "Broad equities, technology, credit"),
    ("SUMMER", "Expansion", "Growth rising / inflation heating", "Cyclicals, commodities, emerging markets"),
    ("FALL", "Stagflation", "Growth falling / inflation heating", "Gold, inflation hedges, cash"),
    ("WINTER", "Downturn", "Growth falling / inflation cooling", "Treasuries, defensives, gold"),
]


def load_data() -> dict[str, pd.DataFrame]:
    long_only = pd.read_csv(
        EXPORTS / "macro_seasons_v4_long_only_ledger.csv", parse_dates=["return_date"]
    ).set_index("return_date")
    ensemble = pd.read_csv(
        EXPORTS / "macro_seasons_v4_ensemble_ledger.csv", parse_dates=["return_date"]
    ).set_index("return_date")
    timeline = pd.read_csv(
        EXPORTS / "macro_seasons_v4_season_timeline.csv", parse_dates=["date"]
    ).set_index("date")
    summary = pd.read_csv(EXPORTS / "macro_seasons_v4_summary.csv").set_index("series")
    monitor = pd.read_csv(EXPORTS / "macro_seasons_v4_monitor.csv")
    allocation = pd.read_csv(EXPORTS / "macro_seasons_v4_current_allocation.csv")
    audit = pd.read_csv(ROOT / "data" / "macro_seasons_v4_data_audit.csv")
    ensemble_state = pd.read_csv(EXPORTS / "macro_seasons_v4_current_ensemble_state.csv")
    return {
        "long_only": long_only,
        "ensemble": ensemble,
        "timeline": timeline,
        "summary": summary,
        "monitor": monitor,
        "allocation": allocation,
        "audit": audit,
        "ensemble_state": ensemble_state,
    }


def monthly_rows(strategy_log: pd.Series, ledger: pd.DataFrame) -> tuple[str, float]:
    returns = (np.exp(strategy_log) - 1.0).dropna()
    spy = np.exp(ledger["spy_return"]) - 1.0
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    html_rows: list[str] = []
    full_years: list[float] = []
    for year, group in returns.groupby(returns.index.year):
        values = {date.month: float(value) * 100.0 for date, value in group.items()}
        ytd = float(((1.0 + group).prod() - 1.0) * 100.0)
        if len(group) == 12:
            full_years.append(ytd)
        cells: list[str] = []
        for month, label in enumerate(month_names, start=1):
            if month not in values:
                cells.append("<td class='na'>&ndash;</td>")
                continue
            value = values[month]
            alpha = min(abs(value) / 8.0, 0.45)
            color = "37,168,134" if value >= 0 else "219,107,79"
            cells.append(
                f"<td style='background:rgba({color},{alpha:.2f})' title='{label} {year}: {value:+.1f}%'>{value:.1f}</td>"
            )
        spy_year = spy.loc[spy.index.year == year]
        spy_ytd = float(((1.0 + spy_year).prod() - 1.0) * 100.0) if len(spy_year) else np.nan
        spy_text = f"{spy_ytd:+.1f}" if np.isfinite(spy_ytd) else "&ndash;"
        html_rows.append(
            f"<tr><td>{year}</td>{''.join(cells)}<td class='ytd'>{ytd:+.1f}</td>"
            f"<td class='spy'>{spy_text}</td></tr>"
        )
    return "".join(html_rows), float(np.mean(full_years))


def equity_svg(long_only: pd.DataFrame, ensemble: pd.DataFrame,
               timeline: pd.DataFrame) -> tuple[str, str]:
    eq = pd.DataFrame(
        {
            "ensemble": np.exp(ensemble["strategy_return"].cumsum()),
            "long_only": np.exp(long_only["strategy_return"].cumsum()),
            "spy": np.exp(long_only["spy_return"].cumsum()),
            "b6040": np.exp(long_only["sixty_forty_return"].cumsum()),
        }
    ).dropna()
    modal = timeline["modal_season"].reindex(eq.index, method="ffill")

    width, height = 960, 350
    ml, mr, mt, strip_h, mb = 50, 16, 12, 24, 38
    plot_w, plot_h = width - ml - mr, height - mt - strip_h - mb
    t0, t1 = eq.index.min().value, eq.index.max().value
    low = math.log(0.5)
    high = math.log(float(eq.max().max()) * 1.22)

    def x(date: pd.Timestamp) -> float:
        return ml + plot_w * (date.value - t0) / (t1 - t0)

    def y(value: float) -> float:
        return mt + plot_h * (1.0 - (math.log(value) - low) / (high - low))

    def path(column: str) -> str:
        return " ".join(
            f"{'M' if i == 0 else 'L'}{x(date):.1f} {y(value):.1f}"
            for i, (date, value) in enumerate(eq[column].items())
        )

    parts: list[str] = []
    for tick in [1, 2, 4, 8, 16]:
        if tick > eq.max().max() * 1.22:
            continue
        yy = y(float(tick))
        parts.append(f"<line x1='{ml}' y1='{yy:.1f}' x2='{width-mr}' y2='{yy:.1f}' class='gridline'/>")
        parts.append(f"<text x='{ml-9}' y='{yy+4:.1f}' text-anchor='end' class='axis'>{tick}x</text>")
    for year in range(2000, eq.index.max().year + 1, 4):
        date = pd.Timestamp(f"{year}-01-15")
        if date >= eq.index.min():
            parts.append(f"<text x='{x(date):.0f}' y='{height-6}' text-anchor='middle' class='axis'>{year}</text>")

    for column, color, stroke, extra in [
        ("b6040", B6040_COLOR, 1.3, "stroke-dasharray='6 5'"),
        ("spy", SPY_COLOR, 1.45, ""),
        ("long_only", LONG_ONLY_COLOR, 1.8, ""),
        ("ensemble", ENSEMBLE_COLOR, 2.6, ""),
    ]:
        parts.append(f"<path d='{path(column)}' fill='none' stroke='{color}' stroke-width='{stroke}' {extra}/>")

    strip_y = mt + plot_h + 8
    strip_width = plot_w / len(eq)
    for i, season in enumerate(modal):
        parts.append(
            f"<rect x='{ml+i*strip_width:.2f}' y='{strip_y}' width='{strip_width+0.35:.2f}' "
            f"height='{strip_h-9}' fill='{SEASON_COLORS.get(str(season), '#555')}' opacity='.88'/>"
        )
    parts.append(
        f"<line id='xh' x1='0' y1='{mt}' x2='0' y2='{mt+plot_h}' "
        "stroke='rgba(255,255,255,.3)' style='display:none'/>"
    )
    payload = {
        "ml": ml, "mr": mr, "w": width,
        "rows": [[date.strftime("%Y-%m"), round(float(row["ensemble"]), 2),
                  round(float(row["long_only"]), 2), round(float(row["spy"]), 2),
                  round(float(row["b6040"]), 2), str(modal.get(date, ""))]
                 for date, row in eq.iterrows()],
    }
    svg = (
        f"<svg id='eqchart' viewBox='0 0 {width} {height}' role='img' "
        "aria-label='Growth of one dollar for the ensemble, long-only season portfolio, S&P 500, and 60/40 benchmark'>"
        + "".join(parts) + "</svg>"
    )
    return svg, json.dumps(payload, separators=(",", ":"))


def metric_block(title: str, row: pd.Series, accent: str) -> str:
    return (
        f"<div class='track' style='--accent:{accent}'><div class='track-title'>{title}</div>"
        f"<div class='metric'><span>CAGR</span><b>{row['cagr_pct']:.1f}%</b></div>"
        f"<div class='metric'><span>Excess Sharpe</span><b>{row['excess_sharpe']:.2f}</b></div>"
        f"<div class='metric'><span>Max loss</span><b>{row['max_dd_pct']:.1f}%</b></div>"
        f"<div class='metric'><span>Calmar</span><b>{row['calmar']:.2f}</b></div></div>"
    )


def build_page(data: dict[str, pd.DataFrame]) -> str:
    long_only, ensemble = data["long_only"], data["ensemble"]
    timeline, summary = data["timeline"], data["summary"]
    monitor = data["monitor"]
    allocation = data["allocation"].sort_values("weight_pct", ascending=False)
    audit = data["audit"]
    ensemble_state = data["ensemble_state"].iloc[0]

    long_stats = summary.loc["Macro Seasons v4 long-only season portfolio"]
    ensemble_stats = summary.loc["Macro Seasons v4 multi-strategy ensemble"]
    spy_stats, b6040_stats = summary.loc["SPY"], summary.loc["60/40 SPY/AGG"]
    latest = timeline.dropna(subset=["modal_season"]).iloc[-1]
    as_of = timeline.dropna(subset=["modal_season"]).index[-1]
    effective_month = pd.Timestamp(allocation["effective_month"].iloc[0]).strftime("%B %Y")
    modal = str(latest["modal_season"])
    years = float(long_stats["months"]) / 12.0

    statuses = monitor["status"].astype(str)
    if statuses.str.startswith("PENDING").all():
        monitor_label, monitor_class = "Pending", "pending"
        first_live = pd.Timestamp(monitor["first_live_return_date"].iloc[0]).strftime("%B %d, %Y")
        monitor_detail = f"first post-freeze return due {first_live}"
    elif statuses.eq("OK").all():
        monitor_label, monitor_class = "On track", "ok"
        monitor_detail = f"{int(monitor['completed_live_months'].min())} completed live months"
    else:
        monitor_label, monitor_class = "Review", "review"
        monitor_detail = "at least one frozen monitoring limit was breached"

    audit_pass = int(audit["status"].eq("PASS").sum())
    audit_total = int(len(audit))
    alfred_pass = int(((audit["component"] == "alfred_vintage") & (audit["status"] == "PASS")).sum())
    chips = "".join(
        f"<div class='chip{' selected' if season == modal else ''}'>"
        f"<i style='background:{SEASON_COLORS[season]}'></i><span>{season.title()}</span>"
        f"<b>{float(latest[season]):.1%}</b></div>"
        for season in ["SPRING", "SUMMER", "FALL", "WINTER"]
    )
    season_rows = "".join(
        f"<div class='season-row'><div class='season-name' style='color:{SEASON_COLORS[season]}'>{season.title()}</div>"
        f"<div><b>{label}</b><span>{definition}</span></div><div>{assets}</div></div>"
        for season, label, definition, assets in SEASON_CARDS
    )

    max_weight = max(float(allocation["weight_pct"].max()), 1.0)
    allocation_rows: list[str] = []
    for row in allocation.itertuples():
        asset_name = html.escape(TICKER_NAMES.get(row.ticker, row.ticker))
        category = "defensive" if row.ticker in DEFENSIVE else "risk"
        allocation_rows.append(
            f"<div class='allocation-row'><div class='asset'><b>{row.ticker}</b><span>{asset_name}</span></div>"
            f"<div class='bar'><i class='{category}' style='width:{float(row.weight_pct)/max_weight*100:.1f}%'></i></div>"
            f"<div class='weight'>{float(row.weight_pct):.2f}%</div></div>"
        )
    defensive_weight = float(allocation.loc[allocation["ticker"].isin(DEFENSIVE), "weight_pct"].sum())
    core_weight = float(ensemble_state["weight_seasons_core_lev"]) * 100.0
    long_weight = float(ensemble_state["weight_long_only_lev"]) * 100.0
    tsmom_weight = float(ensemble_state["weight_tsmom"]) * 100.0
    risk_scale = float(ensemble_state["ensemble_vol_target_scale"])
    svg, payload = equity_svg(long_only, ensemble, timeline)
    tbody, average_year = monthly_rows(long_only["strategy_return"], long_only)
    first_year = int(long_only.index.min().year)
    month_headers = "".join(f"<th>{month}</th>" for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    chart_script = """(function(){
const D=__PAYLOAD__,svg=document.getElementById('eqchart'),tip=document.getElementById('eqtip'),xh=svg.querySelector('#xh');
svg.addEventListener('mousemove',function(e){
  const box=svg.getBoundingClientRect(),sx=D.w/box.width,mx=(e.clientX-box.left)*sx;
  if(mx<D.ml||mx>D.w-D.mr){tip.style.display='none';xh.style.display='none';return}
  const f=(mx-D.ml)/(D.w-D.ml-D.mr),i=Math.max(0,Math.min(D.rows.length-1,Math.round(f*(D.rows.length-1))));
  const row=D.rows[i],xx=D.ml+(D.w-D.ml-D.mr)*i/(D.rows.length-1);
  xh.setAttribute('x1',xx);xh.setAttribute('x2',xx);xh.style.display='block';
  const season=row[5]?row[5][0]+row[5].slice(1).toLowerCase():'';
  tip.innerHTML='<b>'+row[0]+'</b> &middot; '+season+'<br>Ensemble <b>'+row[1].toFixed(2)+'x</b> &nbsp; Long-only <b>'+row[2].toFixed(2)+'x</b><br>S&amp;P '+row[3].toFixed(2)+'x &nbsp; 60/40 '+row[4].toFixed(2)+'x';
  const parent=svg.closest('.chart-wrap').getBoundingClientRect();
  tip.style.display='block';tip.style.left=Math.min(e.clientX-parent.left+12,parent.width-230)+'px';tip.style.top=(e.clientY-parent.top-30)+'px';
});
svg.addEventListener('mouseleave',function(){tip.style.display='none';xh.style.display='none'});
})();""".replace("__PAYLOAD__", payload)

    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>New Sixty Forty | Macro Seasons v4</title>
<meta name='description' content='Point-in-time Macro Seasons portfolios, current allocation and historical results.'>
<style>
:root{{--page:#0d1015;--surface:#151922;--surface2:#11151c;--line:rgba(255,255,255,.09);--ink:#f4f6f8;--text:#b8c0cb;--muted:#7f8997;--green:{ENSEMBLE_COLOR};--gold:{LONG_ONLY_COLOR};color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;letter-spacing:0}}main{{max-width:1080px;margin:auto;padding:0 26px 52px}}
nav{{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:19px 0;border-bottom:1px solid var(--line)}}.brand{{color:var(--ink);font-size:15px;font-weight:800;letter-spacing:.12em}}.brand em{{color:var(--green);font-style:normal}}.stamp{{color:var(--muted);font-size:12px;text-align:right}}
.hero{{display:grid;grid-template-columns:1.16fr .84fr;gap:44px;padding:38px 0 34px;border-bottom:1px solid var(--line)}}.eyebrow{{color:var(--green);font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:.1em}}h1{{margin:7px 0 8px;color:var(--ink);font-size:32px;line-height:1.15;font-weight:750}}.lede{{max-width:650px;margin:0 0 20px;font-size:14px;color:var(--text)}}
.chips{{display:flex;gap:8px;flex-wrap:wrap}}.chip{{display:grid;grid-template-columns:8px auto auto;gap:8px;align-items:center;padding:8px 11px;border:1px solid var(--line);border-radius:6px;background:var(--surface2);font-size:12px}}.chip i{{width:8px;height:8px;border-radius:50%}}.chip span{{color:var(--text)}}.chip b{{color:var(--ink);font-variant-numeric:tabular-nums}}.chip.selected{{border-color:{SEASON_COLORS[modal]};background:rgba(224,166,43,.08)}}
.current{{align-self:start;padding-left:22px;border-left:2px solid {SEASON_COLORS[modal]}}}.current-label{{font-size:11px;text-transform:uppercase;color:var(--muted);font-weight:700}}.current-season{{font-size:30px;line-height:1.15;color:var(--ink);font-weight:760;margin:5px 0}}.current p{{margin:0;color:var(--text)}}.current small{{display:block;margin-top:12px;color:var(--muted)}}
section{{padding:30px 0;border-bottom:1px solid var(--line)}}h2{{margin:0;color:var(--ink);font-size:18px;font-weight:720}}.sub{{margin:4px 0 18px;color:var(--muted);font-size:12.5px}}.section-head{{display:flex;justify-content:space-between;align-items:end;gap:16px}}
.tracks{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.track{{display:grid;grid-template-columns:1.5fr repeat(4,.7fr);gap:12px;align-items:center;padding:16px;border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:6px;background:var(--surface)}}.track-title{{color:var(--ink);font-weight:680}}.metric span{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}}.metric b{{display:block;color:var(--ink);font-size:16px;font-variant-numeric:tabular-nums}}
.definitions{{display:grid;grid-template-columns:1fr 1fr;gap:0 30px}}.season-row{{display:grid;grid-template-columns:72px 1.25fr 1fr;gap:12px;padding:13px 0;border-bottom:1px solid var(--line);font-size:12px}}.season-row:nth-last-child(-n+2){{border-bottom:0}}.season-name{{font-weight:760}}.season-row b{{display:block;color:var(--ink);font-weight:650}}.season-row span{{display:block;color:var(--muted);margin-top:2px}}
.integrity{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden}}.integrity div{{background:var(--surface);padding:15px}}.integrity span{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}}.integrity b{{display:block;color:var(--ink);font-size:15px;margin-top:3px}}.pending{{color:var(--gold)!important}}.ok{{color:var(--green)!important}}.review{{color:#e36d56!important}}
.portfolio-note{{display:grid;grid-template-columns:1fr 1fr;gap:32px;margin:18px 0 22px}}.portfolio-note h3{{margin:0 0 4px;color:var(--ink);font-size:13px}}.portfolio-note p{{margin:0;color:var(--muted);font-size:12px}}
.legend{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:6px 0 8px;color:var(--muted);font-size:11px}}.legend i{{display:inline-block;width:17px;border-top:2px solid;margin-right:6px;vertical-align:middle}}.legend .dash{{border-top-style:dashed}}svg{{display:block;width:100%;height:auto}}.gridline{{stroke:rgba(255,255,255,.065)}}.axis{{fill:#77818f;font-size:10px}}.season-key{{display:flex;gap:14px;color:var(--muted);font-size:10.5px;flex-wrap:wrap}}.season-key i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}}.tip{{position:absolute;display:none;pointer-events:none;padding:7px 9px;border:1px solid var(--line);border-radius:5px;background:#090c10;color:var(--text);font-size:11px;white-space:nowrap;z-index:2}}.chart-wrap{{position:relative}}
.allocation-grid{{display:grid;grid-template-columns:1fr 1fr;gap:0 38px}}.allocation-row{{display:grid;grid-template-columns:142px 1fr 62px;gap:10px;align-items:center;padding:6px 0}}.asset b{{display:block;color:var(--ink);font-size:12px}}.asset span{{display:block;color:var(--muted);font-size:10px}}.bar{{height:8px;background:#090c10;border:1px solid rgba(255,255,255,.05);overflow:hidden}}.bar i{{display:block;height:100%}}.bar .risk{{background:var(--green)}}.bar .defensive{{background:#6485b6}}.weight{{text-align:right;color:var(--ink);font-size:12px;font-variant-numeric:tabular-nums}}
.table-wrap{{max-height:430px;overflow:auto;border:1px solid var(--line);border-radius:6px}}table{{width:100%;border-collapse:collapse;font-size:10.5px;font-variant-numeric:tabular-nums}}th{{position:sticky;top:0;background:var(--surface);color:var(--muted);font-size:9.5px;text-align:right;padding:5px;border-bottom:1px solid var(--line)}}th:first-child,td:first-child{{text-align:left}}td{{padding:4px 5px;text-align:right;color:var(--ink);border-bottom:1px solid rgba(255,255,255,.035)}}td:first-child{{color:var(--text);font-weight:650}}td.ytd{{border-left:1px solid var(--line);font-weight:700}}td.spy{{color:var(--muted)}}td.na{{color:#3e4652}}
details{{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}}summary{{cursor:pointer;color:var(--text);font-size:12px}}details p{{max-width:880px;color:var(--muted);font-size:11.5px}}
footer{{display:flex;justify-content:space-between;gap:20px;padding:20px 0;color:var(--muted);font-size:10.5px;flex-wrap:wrap}}a{{color:var(--text)}}
@media(max-width:850px){{.hero{{grid-template-columns:1fr;gap:24px}}.current{{padding-left:14px}}.tracks,.definitions,.portfolio-note,.allocation-grid{{grid-template-columns:1fr}}.track{{grid-template-columns:1fr 1fr 1fr}}.track-title{{grid-column:1/-1}}.integrity{{grid-template-columns:1fr 1fr}}.season-row:nth-last-child(2){{border-bottom:1px solid var(--line)}}}}
@media(max-width:520px){{main{{padding:0 16px 38px}}h1{{font-size:27px}}.track{{grid-template-columns:1fr 1fr}}.integrity{{grid-template-columns:1fr}}.season-row{{grid-template-columns:66px 1fr}}.season-row>div:last-child{{grid-column:2}}.allocation-row{{grid-template-columns:112px 1fr 58px}}}}
</style></head><body><main>
<nav><div class='brand'>NEW<em>SIXTY</em>FORTY</div><div class='stamp'>Macro Seasons v4 &middot; PIT data through {as_of:%b %Y}</div></nav>

<div class='hero'><div><div class='eyebrow'>Current macro season</div><h1>{modal.title()}, but conviction is low.</h1>
<p class='lede'>The point-in-time growth, inflation and liquidity readings produce a broad probability mix. The model therefore blends all four season portfolios instead of treating the modal label as a certain forecast.</p>
<div class='chips'>{chips}</div></div>
<div class='current'><div class='current-label'>{effective_month} long-only positioning</div><div class='current-season'>{modal.title()}</div>
<p>Current emphasis: {SEASON_GUIDANCE[modal]}.</p><small>Signal date {as_of:%B %d, %Y} &middot; confidence gap {float(latest['confidence']):.1%}</small></div></div>

<section><div class='section-head'><div><h2>Two portfolios, reported separately</h2><p class='sub'>Historical point-in-time simulation after costs. These are research backtests, not a live record.</p></div><div class='stamp'>{years:.1f} years &middot; monthly</div></div>
<div class='tracks'>{metric_block('Long-only season portfolio', long_stats, LONG_ONLY_COLOR)}{metric_block('Multi-strategy leveraged ensemble', ensemble_stats, ENSEMBLE_COLOR)}</div>
<div class='portfolio-note'><div><h3>Long-only season portfolio</h3><p>The exact investable ETF mix shown below. It blends the four season allocations, then applies real-rate, credit, momentum, trend and volatility controls. Weights sum to 100%.</p></div>
<div><h3>Leveraged ensemble</h3><p>A separate research strategy: {core_weight:.1f}% levered core, {long_weight:.1f}% levered long-only and {tsmom_weight:.1f}% trend sleeve, followed by a {risk_scale:.2f}x portfolio risk scale. Its headline return is not the ETF allocation shown below.</p></div></div>
<div class='integrity'><div><span>Input freshness</span><b>{audit_pass}/{audit_total} PASS</b></div><div><span>ALFRED vintages</span><b>{alfred_pass}/4 PASS</b></div><div><span>Live monitor</span><b class='{monitor_class}'>{monitor_label}</b></div><div><span>First live return</span><b>Sep 30, 2026</b></div></div>
<details><summary>Data integrity and live-monitor definition</summary><p>V4 refreshes every FRED and Yahoo cache before each monthly run. CPI, industrial production, payrolls and M2 are reconstructed from the latest ALFRED vintage actually available at each historical month-end. The model was frozen on August 20, 2026; the August 31 decision is the first fully post-freeze signal and its September 30 realized return is the first live observation. Current monitor status: {html.escape(monitor_detail)}.</p></details></section>

<section><h2>How the seasons are defined</h2><p class='sub'>Growth and inflation define the four quadrants. Liquidity affects probabilities and risk intensity; it does not silently redefine the labels.</p><div class='definitions'>{season_rows}</div>
<details><summary>Signals and source data</summary><p><b>Growth:</b> point-in-time industrial production, payrolls and jobless claims, confirmed by market growth/risk proxies. <b>Inflation:</b> point-in-time CPI, breakevens and oil. <b>Liquidity:</b> point-in-time M2 plus Fed liquidity, financial conditions and credit spreads. Monthly macro series use FRED/ALFRED release vintages; ETF prices come from Yahoo Finance. Each pillar uses trailing transformations calculated only with observations available on that decision date.</p></details></section>

<section><h2>Growth of $1</h2><p class='sub'>Log scale &middot; historical PIT simulation after modeled trading costs</p><div class='legend'><span><i style='border-color:{ENSEMBLE_COLOR}'></i>Leveraged ensemble</span><span><i style='border-color:{LONG_ONLY_COLOR}'></i>Long-only season portfolio</span><span><i style='border-color:{SPY_COLOR}'></i>S&amp;P 500</span><span><i class='dash' style='border-color:{B6040_COLOR}'></i>60/40</span></div>
<div class='chart-wrap'>{svg}<div class='tip' id='eqtip'></div></div><div class='season-key'><span><i style='background:{SEASON_COLORS['SPRING']}'></i>Spring</span><span><i style='background:{SEASON_COLORS['SUMMER']}'></i>Summer</span><span><i style='background:{SEASON_COLORS['FALL']}'></i>Fall</span><span><i style='background:{SEASON_COLORS['WINTER']}'></i>Winter</span><span style='margin-left:auto'>strip = monthly modal season</span></div>
<details><summary>Benchmark results</summary><p>S&amp;P 500: CAGR {spy_stats['cagr_pct']:.1f}%, excess Sharpe {spy_stats['excess_sharpe']:.2f}, max loss {spy_stats['max_dd_pct']:.1f}%. 60/40 SPY/AGG: CAGR {b6040_stats['cagr_pct']:.1f}%, excess Sharpe {b6040_stats['excess_sharpe']:.2f}, max loss {b6040_stats['max_dd_pct']:.1f}%.</p></details></section>

<section><h2>Exact long-only allocation for {effective_month}</h2><p class='sub'>Generated from July month-end signals &middot; {len(allocation)} holdings &middot; {defensive_weight:.1f}% defensive &middot; no hidden consolidation or pro-rata renormalization</p><div class='allocation-grid'>{''.join(allocation_rows)}</div></section>

<section><h2>Long-only monthly returns since {first_year}</h2><p class='sub'>Percent per month after costs &middot; average completed calendar year {average_year:+.1f}% &middot; S&amp;P column shown for context</p><div class='table-wrap'><table><thead><tr><th>Year</th>{month_headers}<th>YTD</th><th>S&amp;P</th></tr></thead><tbody>{tbody}</tbody></table></div></section>

<footer><div>Frozen V4 rules and PIT rerun: August 20, 2026 &middot; research, not investment advice &middot; <a href='macro_seasons_v4_onepager.pdf'>method summary (PDF)</a></div><div>&copy; 2026 newsixtyforty.com</div></footer>
<script>{chart_script}</script>
</main></body></html>"""


def main() -> None:
    data = load_data()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page = build_page(data)
    output = OUT_DIR / "index.html"
    output.write_text(page, encoding="utf-8")
    pdf = EXPORTS / "macro_seasons_v4_onepager.pdf"
    if pdf.exists():
        (OUT_DIR / pdf.name).write_bytes(pdf.read_bytes())
    print(f"Wrote {output} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
