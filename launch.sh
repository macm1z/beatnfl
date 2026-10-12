#!/bin/bash
# Start BeatNFL in the background (if it isn't already running) and open it in the browser.
#   launch.sh        start / open
#   launch.sh stop   stop the background app
DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8501
URL="http://localhost:$PORT"
LOG="$HOME/BeatNFL/app.log"
mkdir -p "$HOME/BeatNFL"
running() { curl -sf "$URL/_stcore/health" >/dev/null 2>&1; }

if [ "$1" = "stop" ]; then
  pkill -f "streamlit run $DIR/app.py" && echo "BeatNFL stopped." || echo "BeatNFL was not running."
  exit 0
fi

if ! running; then
  if [ ! -x "$DIR/.venv/bin/python" ]; then
    osascript -e 'display alert "BeatNFL is not installed yet" message "Run Install BeatNFL.command in the BeatNFL folder first."' >/dev/null 2>&1
    exit 1
  fi
  # one-time Streamlit email prompt off
  [ -f "$HOME/.streamlit/credentials.toml" ] || { mkdir -p "$HOME/.streamlit"; printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"; }
  cd "$DIR" || exit 1
  nohup "$DIR/.venv/bin/python" -m streamlit run "$DIR/app.py" --server.headless=true --server.port=$PORT \
        --browser.gatherUsageStats=false >"$LOG" 2>&1 &
  for _ in $(seq 1 60); do running && break; sleep 0.5; done
  if ! running; then
    osascript -e "display alert \"BeatNFL didn't start\" message \"Details are in $LOG — send the last lines to Claude.\"" >/dev/null 2>&1
    exit 1
  fi
fi
open "$URL"
