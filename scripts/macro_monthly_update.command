#!/bin/zsh
# Reproducible Macro Seasons v4 monthly refresh and deployment.

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

run_step "1/7 Refresh broad price and dashboard data" "$PY" update_macro_seasons_research.py --append-only
run_step "2/7 Refresh and gate V4 FRED/Yahoo/ALFRED" "$PY" refresh_macro_seasons_v4_data.py
run_step "3/7 Regenerate Black-Litterman outputs" "$PY" generate_bl_outputs.py
run_step "4/7 Run frozen Macro Seasons v4 PIT model" "$PY" run_macro_seasons_v4.py --no-network
run_step "5/7 Rebuild V4 method PDF" "$PY" research/make_onepager.py
run_step "6/7 Regenerate static website" "$PY" make_website.py

DEPLOY_HOST="${MACRO_DEPLOY_HOST:-Administrator@newsixtyforty.com}"
DEPLOY_PATH="${MACRO_DEPLOY_PATH:-/C:/inetpub/wwwroot/newsixtyforty}"
if [ "$FAILURES" -eq 0 ]; then
  echo "==> 7/7 Deploy website"
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$DEPLOY_HOST" \
    "cmd /c copy /Y C:\\inetpub\\wwwroot\\newsixtyforty\\index.html C:\\inetpub\\wwwroot\\newsixtyforty\\index_prev.html" \
    >>"$LOG" 2>&1 || true
  if scp -o BatchMode=yes -o ConnectTimeout=15 \
    exports/website/index.html exports/website/macro_seasons_v4_onepager.pdf \
    "$DEPLOY_HOST:$DEPLOY_PATH/" >>"$LOG" 2>&1; then
    echo "    OK - https://newsixtyforty.com/"
  else
    echo "    FAILED - see $LOG"
    FAILURES=$((FAILURES + 1))
  fi
else
  echo "==> 7/7 Deploy skipped because an earlier step failed."
fi

exit "$FAILURES"
