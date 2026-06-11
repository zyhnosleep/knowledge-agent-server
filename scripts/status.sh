#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/run}"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  . "$ROOT_DIR/.env"
  set +a
fi

APP_PORT="${APP_PORT:-8000}"

check_pid() {
  name="$1"
  pid_file="$2"

  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name: running (PID $(cat "$pid_file"))"
  else
    echo "$name: stopped"
  fi
}

check_pid "redis" "$RUN_DIR/redis.pid"
check_pid "ollama" "$RUN_DIR/ollama.pid"
check_pid "api" "$RUN_DIR/api.pid"
check_pid "worker" "$RUN_DIR/worker.pid"

if command -v curl >/dev/null 2>&1; then
  echo
  curl -fsS "http://127.0.0.1:${APP_PORT}/api/health" || true
  echo
fi
