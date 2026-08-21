#!/bin/zsh
# Reproducible Macro Seasons v4 refresh and deployment. The execution step
# always marks the current month through the prior New York calendar day.

set -u
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$HOME/Library/Logs/macro_seasons_update_$(date +%Y%m%d_%H%M%S).log"
PY="${PYTHON:-$(command -v python3)}"

if [ -z "${FRED_API_KEY:-}" ] && [ -f "$HOME/.fred_api_key" ]; then
  export FRED_API_KEY="$(tr -d '[:space:]' < "$HOME/.fred_api_key")"
fi
if [ -z "${FRED_API_KEY:-}" ]; then
  echo "ERROR: FRED_API_KEY is required. V4 does not fall back to revised macro history."
  exit 1
fi

cd "$PROJECT" || exit 1
FAILURES=0

run_step() {
  local desc="$1"
  shift
  echo "==> $desc"
  if "$@" >>"$LOG" 2>&1; then
    echo "    OK"
  else
    echo "    FAILED - see $LOG"
    FAILURES=$((FAILURES + 1))
  fi
}

echo "Macro Seasons v4 monthly update - $(date)"
echo "Log: $LOG"

run_step "1/9 Refresh broad price and dashboard data" "$PY" update_macro_seasons_research.py --append-only
run_step "2/9 Refresh and gate V4 FRED/Yahoo/ALFRED" "$PY" refresh_macro_seasons_v4_data.py
run_step "3/9 Regenerate Black-Litterman outputs" "$PY" generate_bl_outputs.py
run_step "4/9 Run frozen Macro Seasons v4 PIT model" "$PY" run_macro_seasons_v4.py --no-network
run_step "5/9 Refresh latest closes and rebuild V4 execution overlay" "$PY" run_macro_seasons_v4_execution.py
run_step "6/9 Run regression tests" "$PY" -m pytest -q
run_step "7/9 Rebuild V4 method PDF" "$PY" research/make_onepager.py
run_step "8/9 Regenerate static website" "$PY" make_website.py

DEPLOY_HOST="${MACRO_DEPLOY_HOST:-Administrator@newsixtyforty.com}"
DEPLOY_PATH="${MACRO_DEPLOY_PATH:-/C:/inetpub/wwwroot/newsixtyforty}"
if [ "$FAILURES" -eq 0 ]; then
  echo "==> 9/9 Deploy website and execution artifacts"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$DEPLOY_HOST" \
    "cmd /c copy /Y C:\\inetpub\\wwwroot\\newsixtyforty\\index.html C:\\inetpub\\wwwroot\\newsixtyforty\\index_prev.html" \
    >>"$LOG" 2>&1 || true
  if scp -o BatchMode=yes -o ConnectTimeout=15 \
    exports/website/index.html \
    exports/website/macro_seasons_v4_onepager.pdf \
    exports/website/macro_seasons_v4_execution_current_positions.csv \
    exports/website/macro_seasons_v4_execution_current_tsmom.csv \
    exports/website/macro_seasons_v4_execution_summary.csv \
    exports/website/macro_seasons_v4_live_mtd.csv \
    exports/website/macro_seasons_v4_execution_pm_pretrade_check.csv \
    exports/website/macro_seasons_v4_execution_assumptions.csv \
    "$DEPLOY_HOST:$DEPLOY_PATH/" >>"$LOG" 2>&1; then
    echo "    OK - https://newsixtyforty.com/"
  else
    echo "    FAILED - see $LOG"
    FAILURES=$((FAILURES + 1))
  fi
else
  echo "==> 9/9 Deploy skipped because an earlier step failed."
fi

exit "$FAILURES"
