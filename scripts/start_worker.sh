#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/run}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PID_FILE="$RUN_DIR/worker.pid"
LOG_FILE="$LOG_DIR/worker.log"

mkdir -p "$LOG_DIR" "$RUN_DIR"

if [ ! -f "$ROOT_DIR/.env" ]; then
  echo ".env not found under $ROOT_DIR" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "virtualenv not found: $VENV_DIR" >&2
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Worker already running with PID $(cat "$PID_FILE")"
  exit 0
fi

cd "$ROOT_DIR"
set -a
. "$ROOT_DIR/.env"
set +a
. "$VENV_DIR/bin/activate"

nohup python -m app.workers.runner >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
sleep 2

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Worker started"
else
  echo "Worker failed to start. Check $LOG_FILE" >&2
  exit 1
fi
