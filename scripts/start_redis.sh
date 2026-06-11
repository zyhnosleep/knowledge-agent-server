#!/usr/bin/env sh
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/run}"
DATA_DIR="${REDIS_DATA_DIR:-$ROOT_DIR/data/redis}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_BIND="${REDIS_BIND:-127.0.0.1}"
REDIS_SERVER_BIN="${REDIS_SERVER_BIN:-$HOME/local/redis/bin/redis-server}"
REDIS_CONFIG="$RUN_DIR/redis.conf"
PID_FILE="$RUN_DIR/redis.pid"
LOG_FILE="$LOG_DIR/redis.log"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$DATA_DIR"

if [ ! -x "$REDIS_SERVER_BIN" ]; then
  echo "redis-server not found: $REDIS_SERVER_BIN" >&2
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Redis already running with PID $(cat "$PID_FILE")"
  exit 0
fi

cat >"$REDIS_CONFIG" <<EOF
bind $REDIS_BIND
port $REDIS_PORT
dir $DATA_DIR
pidfile $PID_FILE
logfile $LOG_FILE
daemonize no
save 900 1
save 300 10
save 60 10000
appendonly no
protected-mode yes
EOF

nohup "$REDIS_SERVER_BIN" "$REDIS_CONFIG" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
sleep 1

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Redis started on ${REDIS_BIND}:${REDIS_PORT}"
else
  echo "Redis failed to start. Check $LOG_FILE" >&2
  exit 1
fi
