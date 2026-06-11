#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/run}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_BIND="${REDIS_BIND:-127.0.0.1}"
REDIS_CLI_BIN="${REDIS_CLI_BIN:-$HOME/local/redis/bin/redis-cli}"
OLLAMA_HOST_VALUE="${OLLAMA_HOST_VALUE:-127.0.0.1:11435}"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  . "$ROOT_DIR/.env"
  set +a
fi

APP_PORT="${APP_PORT:-8000}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://$OLLAMA_HOST_VALUE}"

check_pid() {
  name="$1"
  pid_file="$2"

  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name: running (PID $(cat "$pid_file"))"
  else
    echo "$name: stopped"
  fi
}

check_redis() {
  if [ -x "$REDIS_CLI_BIN" ] && "$REDIS_CLI_BIN" -h "$REDIS_BIND" -p "$REDIS_PORT" ping 2>/dev/null | grep -q '^PONG$'; then
    echo "redis: running (${REDIS_BIND}:${REDIS_PORT})"
  elif command -v redis-cli >/dev/null 2>&1 && redis-cli -h "$REDIS_BIND" -p "$REDIS_PORT" ping 2>/dev/null | grep -q '^PONG$'; then
    echo "redis: running (${REDIS_BIND}:${REDIS_PORT})"
  else
    check_pid "redis" "$RUN_DIR/redis.pid"
  fi
}

check_ollama() {
  if command -v curl >/dev/null 2>&1 && curl -fsS "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "ollama: running ($OLLAMA_BASE_URL)"
  else
    check_pid "ollama" "$RUN_DIR/ollama.pid"
  fi
}

check_api() {
  if command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:${APP_PORT}/api/health" >/dev/null 2>&1; then
    echo "api: running (http://127.0.0.1:${APP_PORT})"
  else
    check_pid "api" "$RUN_DIR/api.pid"
  fi
}

check_redis
check_ollama
check_api
check_pid "worker" "$RUN_DIR/worker.pid"

if command -v curl >/dev/null 2>&1; then
  echo
  curl -fsS "http://127.0.0.1:${APP_PORT}/api/health" || true
  echo
fi
