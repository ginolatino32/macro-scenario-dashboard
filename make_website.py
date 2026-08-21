"""Build newsixtyforty.com from frozen V4 and its IBKR execution overlay.

The page deliberately separates the implementable long-only season portfolio
from the leveraged three-stream research ensemble and the costed executable
overlay. All figures are generated from export artifacts, not hard-coded.
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
    (
        "SPRING", "Recovery", "Growth up, inflation cooling",
        "Broad stocks, tech, corporate credit, housing", "2003, 2023",
    ),
    (
        "SUMMER", "Boom", "Growth up, inflation heating",
        "Energy, industrials, commodities, emerging markets", "2006-07, 2021",
    ),
    (
        "FALL", "Stagflation", "Growth down, inflation heating",
        "Gold, commodities, inflation-linked bonds, cash", "early 2008, 2022",
    ),
    (
        "WINTER", "Downturn", "Growth down, inflation cooling",
        "Treasury bonds, defensive stocks, gold", "2001-02, 2008, 2020",
    ),
]


def load_data() -> dict[str, object]:
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
    execution_positions = pd.read_csv(
        EXPORTS / "macro_seasons_v4_execution_current_positions.csv"
    )
    execution_tsmom = pd.read_csv(
        EXPORTS / "macro_seasons_v4_execution_current_tsmom.csv"
    )
    execution_summary = pd.read_csv(
        EXPORTS / "macro_seasons_v4_execution_summary.csv"
    ).set_index("series")
    execution_ledger = pd.read_csv(
        EXPORTS / "macro_seasons_v4_execution_ledger.csv", parse_dates=["return_date"]
    ).set_index("return_date")
    execution_costs = pd.read_csv(
        EXPORTS / "macro_seasons_v4_execution_cost_summary.csv"
    ).set_index("cost_component")
    execution_manifest = json.loads(
        (EXPORTS / "macro_seasons_v4_execution_manifest.json").read_text()
    )
    return {
        "long_only": long_only,
        "ensemble": ensemble,
        "timeline": timeline,
        "summary": summary,
        "monitor": monitor,
        "allocation": allocation,
        "audit": audit,
        "ensemble_state": ensemble_state,
        "execution_positions": execution_positions,
        "execution_tsmom": execution_tsmom,
        "execution_summary": execution_summary,
        "execution_ledger": execution_ledger,
        "execution_costs": execution_costs,
        "execution_manifest": execution_manifest,
    }


def monthly_rows(
    strategy: pd.Series,
    spy_log: pd.Series,
    *,
    strategy_is_log: bool,
    start_year: int,
) -> tuple[str, float]:
    returns = (np.exp(strategy) - 1.0 if strategy_is_log else strategy).dropna()
    spy = np.exp(spy_log) - 1.0
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    html_rows: list[str] = []
    full_years: list[float] = []
    for year in range(start_year, int(returns.index.max().year) + 1):
        group = returns.loc[returns.index.year == year]
        values = {date.month: float(value) * 100.0 for date, value in group.items()}
        ytd = float(((1.0 + group).prod() - 1.0) * 100.0) if len(group) else np.nan
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
        ytd_text = f"{ytd:+.1f}" if np.isfinite(ytd) else "&ndash;"
        html_rows.append(
            f"<tr><td>{year}</td>{''.join(cells)}<td class='ytd'>{ytd_text}</td>"
            f"<td class='spy'>{spy_text}</td></tr>"
        )
    average_year = float(np.mean(full_years)) if full_years else np.nan
    return "".join(html_rows), average_year


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

    for column, series_id, color, stroke, extra in [
        ("b6040", "6040", B6040_COLOR, 1.3, "stroke-dasharray='6 5'"),
        ("spy", "spy", SPY_COLOR, 1.45, ""),
        ("long_only", "long-only", LONG_ONLY_COLOR, 1.8, ""),
        ("ensemble", "ls", ENSEMBLE_COLOR, 2.6, ""),
    ]:
        parts.append(
            f"<path data-series='{series_id}' d='{path(column)}' fill='none' "
            f"stroke='{color}' stroke-width='{stroke}' {extra}/>"
        )

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
        "aria-label='Growth of one dollar for the L/S portfolio, long-only season portfolio, S&P 500, and 60/40 benchmark'>"
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
    execution_positions = data["execution_positions"].copy()
    execution_tsmom = data["execution_tsmom"].copy()
    execution_summary = data["execution_summary"]
    execution_ledger = data["execution_ledger"]
    execution_costs = data["execution_costs"]
    execution_manifest = data["execution_manifest"]

    long_stats = summary.loc["Macro Seasons v4 long-only season portfolio"]
    ensemble_stats = summary.loc["Macro Seasons v4 multi-strategy ensemble"]
    execution_stats = execution_summary.loc["IBKR-costed executable ensemble"]
    execution_window_stats = execution_summary.loc[
        "Frozen return-level ensemble on executable window"
    ]
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
        f"<div><b>{label}</b><span>{definition}</span></div>"
        f"<div class='season-assets'>{assets}<span>e.g. {examples}</span></div></div>"
        for season, label, definition, assets, examples in SEASON_CARDS
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
    long_only_bil_weight = float(
        allocation.loc[allocation["ticker"].eq("BIL"), "weight_pct"].sum()
    )
    core_weight = float(ensemble_state["weight_seasons_core_lev"]) * 100.0
    long_weight = float(ensemble_state["weight_long_only_lev"]) * 100.0
    tsmom_weight = float(ensemble_state["weight_tsmom"]) * 100.0
    risk_scale = float(ensemble_state["ensemble_vol_target_scale"])
    execution_meta = execution_manifest["current_metadata"]
    gross_exposure = float(execution_meta["gross_exposure"]) * 100.0
    net_exposure = float(execution_meta["net_exposure"]) * 100.0
    short_exposure = float(execution_meta["short_gross_exposure"]) * 100.0
    cash_weight = float(execution_meta["cash_weight"]) * 100.0
    margin_debit = float(execution_meta["margin_debit_weight"]) * 100.0
    short_collateral = float(execution_meta["short_collateral_weight"]) * 100.0
    regt_reference_buffer = float(execution_meta["regt_reference_buffer"]) * 100.0
    netted_bil_weight = float(execution_meta["netted_bil_weight"]) * 100.0

    execution_rows: list[str] = []
    for row in execution_positions.sort_values("target_weight", ascending=False).itertuples():
        ticker = str(row.ticker)
        name = "USD cash balance" if ticker == "USD_CASH" else TICKER_NAMES.get(ticker, ticker)
        side_class = "short" if float(row.target_weight) < 0 else "long"
        if ticker == "USD_CASH":
            side_class = "borrow" if float(row.target_weight) < 0 else "cash"
        execution_rows.append(
            f"<div class='exec-row'><div><b>{html.escape(ticker)}</b><span>{html.escape(name)}</span></div>"
            f"<span class='side {side_class}'>{html.escape(str(row.side))}</span>"
            f"<strong>{float(row.target_weight_pct):+.2f}%</strong></div>"
        )

    tsmom_rows = "".join(
        f"<tr><td>{html.escape(str(row.ticker))}</td><td class='{str(row.side).lower()}'>{html.escape(str(row.side))}</td>"
        f"<td>{float(row.excess_12m_return):+.1%}</td><td>{float(row.trailing_36m_vol):.1%}</td>"
        f"<td>{float(row.sleeve_weight_pct):+.2f}%</td></tr>"
        for row in execution_tsmom.sort_values("sleeve_weight", ascending=False).itertuples()
    )
    tsmom_gross = float(execution_tsmom["sleeve_weight"].abs().sum()) * 100.0
    tsmom_net = float(execution_tsmom["sleeve_weight"].sum()) * 100.0
    trading_bps = float(
        execution_costs.loc[
            ["commission_cost", "regulatory_fee_cost", "slippage_cost"],
            "average_annual_bps",
        ].sum()
    )
    financing_bps = float(
        execution_costs.loc[["margin_benchmark_cost", "margin_spread_cost"], "average_annual_bps"].sum()
    )
    borrow_bps = float(execution_costs.loc["short_borrow_cost", "average_annual_bps"])
    short_credit_bps = -float(
        execution_costs.loc["short_proceeds_interest_credit", "average_annual_bps"]
    )
    short_net_bps = borrow_bps - short_credit_bps
    roll_bps = float(execution_costs.loc["futures_roll_cost", "average_annual_bps"])
    svg, payload = equity_svg(long_only, ensemble, timeline)
    returns_start_year = 2007
    long_only_tbody, long_only_average_year = monthly_rows(
        long_only["strategy_return"],
        long_only["spy_return"],
        strategy_is_log=True,
        start_year=returns_start_year,
    )
    ls_tbody, ls_average_year = monthly_rows(
        execution_ledger["net_return"],
        long_only["spy_return"],
        strategy_is_log=False,
        start_year=returns_start_year,
    )
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
  tip.innerHTML='<b>'+row[0]+'</b> &middot; '+season+'<br>L/S <b>'+row[1].toFixed(2)+'x</b> &nbsp; Long-only <b>'+row[2].toFixed(2)+'x</b><br>S&amp;P '+row[3].toFixed(2)+'x &nbsp; 60/40 '+row[4].toFixed(2)+'x';
  const parent=svg.closest('.chart-wrap').getBoundingClientRect();
  tip.style.display='block';tip.style.left=Math.min(e.clientX-parent.left+12,parent.width-230)+'px';tip.style.top=(e.clientY-parent.top-30)+'px';
});
svg.addEventListener('mouseleave',function(){tip.style.display='none';xh.style.display='none'});
})();""".replace("__PAYLOAD__", payload)
    returns_script = """(function(){
const buttons=Array.from(document.querySelectorAll('[data-return-view]'));
const panels=Array.from(document.querySelectorAll('[data-return-panel]'));
buttons.forEach(function(button){button.addEventListener('click',function(){
  const target=button.getAttribute('data-return-view');
  buttons.forEach(function(item){
    const selected=item===button;
    item.classList.toggle('selected',selected);
    item.setAttribute('aria-selected',selected?'true':'false');
  });
  panels.forEach(function(panel){
    const selected=panel.getAttribute('data-return-panel')===target;
    panel.hidden=!selected;
  });
});});
})();"""

    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'>
<meta http-equiv='Pragma' content='no-cache'>
<meta http-equiv='Expires' content='0'>
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
.exec-summary{{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;margin:15px 0 20px;background:var(--line);border:1px solid var(--line);border-radius:6px;overflow:hidden}}.exec-summary div{{background:var(--surface);padding:14px}}.exec-summary span,.cost-item span{{display:block;color:var(--muted);font-size:9.5px;text-transform:uppercase}}.exec-summary b,.cost-item b{{display:block;color:var(--ink);font-size:16px;margin-top:2px;font-variant-numeric:tabular-nums}}.exec-grid{{display:grid;grid-template-columns:1fr 1fr;gap:0 34px}}.exec-row{{display:grid;grid-template-columns:1fr 58px 65px;gap:9px;align-items:center;padding:6px 0;border-bottom:1px solid rgba(255,255,255,.045)}}.exec-row div b{{display:block;color:var(--ink);font-size:11.5px}}.exec-row div span{{display:block;color:var(--muted);font-size:9.5px}}.exec-row strong{{text-align:right;color:var(--ink);font-size:11.5px;font-variant-numeric:tabular-nums}}.side{{font-size:8.5px;font-weight:750;text-align:center;padding:2px 4px;border:1px solid var(--line);border-radius:3px}}.side.long{{color:#65c99a}}.side.short,.side.borrow{{color:#eb846c}}.side.cash{{color:#82a9df}}.cost-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:20px 0 4px}}.cost-item{{padding:12px;border:1px solid var(--line);border-radius:5px;background:var(--surface2)}}.cost-item small{{display:block;color:var(--muted);font-size:9.5px;margin-top:2px}}td.long{{color:#65c99a}}td.short{{color:#eb846c}}
.netting-note{{display:grid;grid-template-columns:150px 1fr;gap:20px;margin:-4px 0 18px;padding:13px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}.netting-note b{{color:var(--ink);font-size:12px}}.netting-note p{{margin:0;color:var(--muted);font-size:11.5px}}
.table-wrap{{max-height:430px;overflow:auto;border:1px solid var(--line);border-radius:6px}}table{{width:100%;border-collapse:collapse;font-size:10.5px;font-variant-numeric:tabular-nums}}th{{position:sticky;top:0;background:var(--surface);color:var(--muted);font-size:9.5px;text-align:right;padding:5px;border-bottom:1px solid var(--line)}}th:first-child,td:first-child{{text-align:left}}td{{padding:4px 5px;text-align:right;color:var(--ink);border-bottom:1px solid rgba(255,255,255,.035)}}td:first-child{{color:var(--text);font-weight:650}}td.ytd{{border-left:1px solid var(--line);font-weight:700}}td.spy{{color:var(--muted)}}td.na{{color:#3e4652}}
.returns-head{{display:flex;align-items:end;justify-content:space-between;gap:18px;margin-bottom:14px}}.segmented{{display:inline-grid;grid-template-columns:1fr 1fr;padding:3px;border:1px solid var(--line);border-radius:6px;background:var(--surface2)}}.segmented button{{min-width:104px;border:0;border-radius:4px;padding:7px 11px;background:transparent;color:var(--muted);font:600 11px/1.2 system-ui,-apple-system,"Segoe UI",sans-serif;cursor:pointer}}.segmented button.selected{{background:var(--surface);color:var(--ink);box-shadow:0 0 0 1px var(--line)}}.return-panel[hidden]{{display:none}}
details{{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}}summary{{cursor:pointer;color:var(--text);font-size:12px}}details p{{max-width:880px;color:var(--muted);font-size:11.5px}}
footer{{display:flex;justify-content:space-between;gap:20px;padding:20px 0;color:var(--muted);font-size:10.5px;flex-wrap:wrap}}a{{color:var(--text)}}
@media(max-width:850px){{.hero{{grid-template-columns:1fr;gap:24px}}.current{{padding-left:14px}}.tracks,.definitions,.portfolio-note,.allocation-grid,.exec-grid{{grid-template-columns:1fr}}.track{{grid-template-columns:1fr 1fr 1fr}}.track-title{{grid-column:1/-1}}.integrity{{grid-template-columns:1fr 1fr}}.exec-summary{{grid-template-columns:1fr 1fr}}.cost-grid{{grid-template-columns:1fr 1fr}}.season-row:nth-last-child(2){{border-bottom:1px solid var(--line)}}}}
@media(max-width:520px){{main{{padding:0 16px 38px}}h1{{font-size:27px}}.track{{grid-template-columns:1fr 1fr}}.integrity,.exec-summary,.cost-grid{{grid-template-columns:1fr}}.season-row{{grid-template-columns:66px 1fr}}.season-row>div:last-child{{grid-column:2}}.allocation-row{{grid-template-columns:112px 1fr 58px}}.netting-note{{grid-template-columns:1fr;gap:4px}}.returns-head{{align-items:stretch;flex-direction:column}}.segmented{{width:100%}}.segmented button{{min-width:0}}}}
</style></head><body><main>
<nav><div class='brand'>NEW<em>SIXTY</em>FORTY</div><div class='stamp'>Macro Seasons v4 &middot; PIT data through {as_of:%b %Y}</div></nav>

<div class='hero'><div><div class='eyebrow'>Current macro season</div><h1>{modal.title()}, but conviction is low.</h1>
<p class='lede'>The point-in-time growth, inflation and liquidity readings produce a broad probability mix. The model therefore blends all four season portfolios instead of treating the modal label as a certain forecast.</p>
<div class='chips'>{chips}</div></div>
<div class='current'><div class='current-label'>{effective_month} long-only positioning</div><div class='current-season'>{modal.title()}</div>
<p>Current emphasis: {SEASON_GUIDANCE[modal]}.</p><small>Signal date {as_of:%B %d, %Y} &middot; confidence gap {float(latest['confidence']):.1%}</small></div></div>

<section><h2>How the seasons are defined</h2><p class='sub'>Growth and inflation define the four quadrants. Liquidity affects probabilities and risk intensity; it does not silently redefine the labels. Historical examples are illustrative episodes, not training labels.</p><div class='definitions'>{season_rows}</div>
<details><summary>Signals and source data</summary><p><b>Growth:</b> point-in-time industrial production, payrolls and jobless claims, confirmed by market growth/risk proxies. <b>Inflation:</b> point-in-time CPI, breakevens and oil. <b>Liquidity:</b> point-in-time M2 plus Fed liquidity, financial conditions and credit spreads. Monthly macro series use FRED/ALFRED release vintages; ETF prices come from Yahoo Finance. Each pillar uses trailing transformations calculated only with observations available on that decision date.</p></details></section>

<section><div class='section-head'><div><h2>Long Only and L/S Portfolio</h2></div><div class='stamp'>{years:.1f} years &middot; monthly</div></div>
<div class='tracks'>{metric_block('Long-only season portfolio', long_stats, LONG_ONLY_COLOR)}{metric_block('L/S portfolio', ensemble_stats, ENSEMBLE_COLOR)}</div>
<div class='portfolio-note'><div><h3>Long-only season portfolio</h3><p>The exact investable ETF mix shown below. It blends the four season allocations, then applies real-rate, credit, momentum, trend and volatility controls. Weights sum to 100%.</p></div>
<div><h3>L/S portfolio</h3><p>A separate strategy: {core_weight:.1f}% levered core, {long_weight:.1f}% levered long-only and {tsmom_weight:.1f}% long/short trend sleeve, followed by a {risk_scale:.2f}x portfolio risk scale. Its physical, costed IBKR implementation is reported separately below.</p></div></div>
<div class='integrity'><div><span>Input freshness</span><b>{audit_pass}/{audit_total} PASS</b></div><div><span>ALFRED vintages</span><b>{alfred_pass}/4 PASS</b></div><div><span>Live monitor</span><b class='{monitor_class}'>{monitor_label}</b></div><div><span>First live return</span><b>Sep 30, 2026</b></div></div>
<details><summary>Data integrity and live-monitor definition</summary><p>V4 refreshes every FRED and Yahoo cache before each monthly run. CPI, industrial production, payrolls and M2 are reconstructed from the latest ALFRED vintage actually available at each historical month-end. The model was frozen on August 20, 2026; the August 31 decision is the first fully post-freeze signal and its September 30 realized return is the first live observation. Current monitor status: {html.escape(monitor_detail)}.</p></details></section>

<section><h2>Growth of $1</h2><p class='sub'>Log scale &middot; historical point-in-time simulation after modeled trading costs</p><div class='legend'><span><i style='border-color:{ENSEMBLE_COLOR}'></i>L/S portfolio</span><span><i style='border-color:{LONG_ONLY_COLOR}'></i>Long-only season portfolio</span><span><i style='border-color:{SPY_COLOR}'></i>S&amp;P 500</span><span><i class='dash' style='border-color:{B6040_COLOR}'></i>60/40</span></div>
<div class='chart-wrap'>{svg}<div class='tip' id='eqtip'></div></div><div class='season-key'><span><i style='background:{SEASON_COLORS['SPRING']}'></i>Spring</span><span><i style='background:{SEASON_COLORS['SUMMER']}'></i>Summer</span><span><i style='background:{SEASON_COLORS['FALL']}'></i>Fall</span><span><i style='background:{SEASON_COLORS['WINTER']}'></i>Winter</span></div>
<details><summary>Benchmark results</summary><p>S&amp;P 500: CAGR {spy_stats['cagr_pct']:.1f}%, excess Sharpe {spy_stats['excess_sharpe']:.2f}, max loss {spy_stats['max_dd_pct']:.1f}%. 60/40 SPY/AGG: CAGR {b6040_stats['cagr_pct']:.1f}%, excess Sharpe {b6040_stats['excess_sharpe']:.2f}, max loss {b6040_stats['max_dd_pct']:.1f}%.</p></details></section>

<section data-allocation='long-only'><h2>Long-only allocation for {effective_month}</h2><div class='allocation-grid'>{''.join(allocation_rows)}</div></section>

<section id='execution' data-allocation='ls'><h2>L/S allocation for {effective_month}</h2>
<div class='exec-summary'><div><span>Gross exposure</span><b>{gross_exposure:.1f}%</b></div><div><span>Net exposure</span><b>{net_exposure:.1f}%</b></div><div><span>Gross shorts</span><b>{short_exposure:.1f}%</b></div><div><span>Economic cash / borrow</span><b>{cash_weight:+.1f}%</b></div><div><span>Margin debit</span><b>{margin_debit:.1f}%</b></div><div><span>Short collateral</span><b>{short_collateral:.1f}%</b></div></div>
<div class='netting-note'><b>Why BIL is absent</b><p>BIL remains {long_only_bil_weight:.1f}% of the separate long-only portfolio. In the aggregated L/S book, {netted_bil_weight:.1f}% of raw BIL exposure is sold first and used to reduce the margin debit. Holding BIL while borrowing at IBKR's higher margin rate would be a negative-carry round trip. This is financing netting, not a change to the Summer season allocation.</p></div>
<div class='exec-grid'>{''.join(execution_rows)}</div>
<div class='cost-grid'><div class='cost-item'><span>Costed CAGR</span><b>{execution_stats['cagr_pct']:.2f}%</b><small>2008&ndash;2026 executable window</small></div><div class='cost-item'><span>Trading</span><b>{trading_bps:.1f} bp/yr</b><small>commission, regulatory fees and 1 bp slippage</small></div><div class='cost-item'><span>Margin financing</span><b>{financing_bps:.1f} bp/yr</b><small>benchmark plus IBKR tiers</small></div><div class='cost-item'><span>Net short carry</span><b>{short_net_bps:.1f} bp/yr</b><small>{borrow_bps:.1f} borrow less {short_credit_bps:.1f} proceeds credit</small></div></div>
<p class='sub'>IBKR segregates short-sale proceeds. The {cash_weight:+.1f}% economic cash balance therefore consists of a {margin_debit:.1f}% margin debit and {short_collateral:.1f}% of marked short collateral. The simulation assumes Portfolio Margin. The {regt_reference_buffer:.1f}% Reg T-equivalent buffer is shown for reference; the full allocation still has to pass IBKR's live Check Margin with at least a 20% projected cushion. Over 2008&ndash;2026, the return-level L/S series earned {execution_window_stats['cagr_pct']:.2f}%. The holdings-based simulation earned {execution_summary.loc['Executable physical accounting before explicit IBKR costs', 'cagr_pct']:.2f}% before IBKR costs and {execution_stats['cagr_pct']:.2f}% after commissions, regulatory fees, slippage, margin interest, stock borrow and interest on short proceeds. After-cost volatility was {execution_stats['ann_vol_pct']:.2f}%, and maximum drawdown was {execution_stats['max_dd_pct']:.2f}%. Futures roll cost is {roll_bps:.1f} bp because the current portfolio holds ETFs, not futures.</p>
<details><summary>Exact TSMOM long and short sleeve</summary><p>The unaggregated trend sleeve is {tsmom_gross:.1f}% gross and {tsmom_net:+.1f}% net before its ensemble weight. A positive position means the ETF beat BIL over the trailing 12 months; a negative position means it lagged BIL.</p><div class='table-wrap'><table><thead><tr><th>Ticker</th><th>Side</th><th>12m excess</th><th>36m vol</th><th>Sleeve weight</th></tr></thead><tbody>{tsmom_rows}</tbody></table></div></details>
<details><summary>IBKR assumptions, limits and downloadable artifacts</summary><p>IBKR Pro Fixed: USD 0.005/share, USD 1 minimum and 1% order-value cap, plus published US regulatory fees. Positive free USD cash: benchmark minus 0.50%, with no interest on the first USD 10,000. Long financing: published benchmark-plus tiers starting at +1.50%. Short collateral is marked at 102%; the first USD 100,000 earns no proceeds interest and higher tiers earn benchmark minus the published spread. Historical borrow data is unavailable, so every liquid ETF short is conservatively charged 1.00% annually and must pass a live SLB check before trading. Hard limits: 175% gross, 150% net, 35% gross shorts, 25% per non-cash ETF and 50% maximum margin debit. Portfolio Margin is risk-based and broker-calculated; no public static formula can replace its live whole-book margin preview. <a href='macro_seasons_v4_execution_current_positions.csv'>positions CSV</a> &middot; <a href='macro_seasons_v4_execution_current_tsmom.csv'>TSMOM CSV</a> &middot; <a href='macro_seasons_v4_execution_summary.csv'>performance CSV</a> &middot; <a href='macro_seasons_v4_execution_pm_pretrade_check.csv'>PM check CSV</a> &middot; <a href='macro_seasons_v4_execution_assumptions.csv'>assumptions CSV</a>.</p></details></section>

<section id='monthly-returns'><div class='returns-head'><div><h2>Monthly returns from {returns_start_year}</h2></div><div class='segmented' role='tablist' aria-label='Monthly return series'><button class='selected' type='button' role='tab' aria-selected='true' data-return-view='long-only'>Long only</button><button type='button' role='tab' aria-selected='false' data-return-view='ls'>L/S portfolio</button></div></div>
<div class='return-panel' data-return-panel='long-only'><p class='sub'>Long-only portfolio &middot; percent per month after costs &middot; average completed calendar year {long_only_average_year:+.1f}% &middot; S&amp;P column shown for context</p><div class='table-wrap'><table><thead><tr><th>Year</th>{month_headers}<th>YTD</th><th>S&amp;P</th></tr></thead><tbody>{long_only_tbody}</tbody></table></div></div>
<div class='return-panel' data-return-panel='ls' hidden><p class='sub'>L/S portfolio after modeled IBKR costs &middot; history begins January 2008, so 2007 is blank &middot; average completed calendar year {ls_average_year:+.1f}% &middot; S&amp;P column shown for context</p><div class='table-wrap'><table><thead><tr><th>Year</th>{month_headers}<th>YTD</th><th>S&amp;P</th></tr></thead><tbody>{ls_tbody}</tbody></table></div></div></section>

<footer><div>Frozen V4 rules and PIT rerun: August 20, 2026 &middot; research, not investment advice &middot; <a href='macro_seasons_v4_onepager.pdf'>method summary (PDF)</a></div><div>&copy; 2026 newsixtyforty.com</div></footer>
<script>{chart_script}{returns_script}</script>
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
    for name in [
        "macro_seasons_v4_execution_current_positions.csv",
        "macro_seasons_v4_execution_current_tsmom.csv",
        "macro_seasons_v4_execution_summary.csv",
        "macro_seasons_v4_execution_pm_pretrade_check.csv",
        "macro_seasons_v4_execution_assumptions.csv",
    ]:
        source = EXPORTS / name
        if source.exists():
            (OUT_DIR / name).write_bytes(source.read_bytes())
    print(f"Wrote {output} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
