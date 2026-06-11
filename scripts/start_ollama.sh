#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/run}"
OLLAMA_HOST_VALUE="${OLLAMA_HOST_VALUE:-127.0.0.1:11435}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/local/ollama/bin/ollama}"
PID_FILE="$RUN_DIR/ollama.pid"
LOG_FILE="$LOG_DIR/ollama.log"

mkdir -p "$LOG_DIR" "$RUN_DIR"

if [ ! -x "$OLLAMA_BIN" ]; then
  echo "ollama binary not found: $OLLAMA_BIN" >&2
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Ollama already running with PID $(cat "$PID_FILE")"
  exit 0
fi

export OLLAMA_HOST="$OLLAMA_HOST_VALUE"
nohup "$OLLAMA_BIN" serve >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
sleep 2

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Ollama started on $OLLAMA_HOST_VALUE"
else
  echo "Ollama failed to start. Check $LOG_FILE" >&2
  exit 1
fi
