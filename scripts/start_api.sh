#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/run}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
PID_FILE="$RUN_DIR/api.pid"
LOG_FILE="$LOG_DIR/api.log"

mkdir -p "$LOG_DIR" "$RUN_DIR"

if [ ! -f "$ROOT_DIR/.env" ]; then
  echo ".env not found under $ROOT_DIR" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "python binary not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "API already running with PID $(cat "$PID_FILE")"
  exit 0
fi

cd "$ROOT_DIR"
set -a
. "$ROOT_DIR/.env"
set +a

nohup "$PYTHON_BIN" -m uvicorn app.main:app --host "${APP_HOST:-0.0.0.0}" --port "${APP_PORT:-8000}" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
sleep 2

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "API started on ${APP_HOST:-0.0.0.0}:${APP_PORT:-8000}"
else
  echo "API failed to start. Check $LOG_FILE" >&2
  exit 1
fi
