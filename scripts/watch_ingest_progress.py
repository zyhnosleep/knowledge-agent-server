from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


TERMINAL_STATES = {"completed", "failed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch the latest ingest run progress.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Knowledge Agent API base URL.")
    parser.add_argument("--run-id", default="", help="Optional run id to watch. Defaults to latest ingest run.")
    parser.add_argument("--interval", type=float, default=3.0, help="Polling interval in seconds.")
    return parser.parse_args()


def fetch_runs(base_url: str) -> list[dict]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/runs", timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def select_run(runs: list[dict], run_id: str) -> dict | None:
    ingest_runs = [run for run in runs if run.get("run_type") == "ingest"]
    if run_id:
        return next((run for run in ingest_runs if run.get("id") == run_id), None)
    return ingest_runs[0] if ingest_runs else None


def progress_from_run(run: dict) -> tuple[int, str, str]:
    report = run.get("provider_report") or {}
    progress = report.get("progress") or {}
    status = run.get("status") or "unknown"
    percent = int(progress.get("percent") or (100 if status in TERMINAL_STATES else 0))
    stage = progress.get("stage") or status
    message = progress.get("message") or run.get("notes") or ""
    return max(0, min(percent, 100)), stage, message


def render_bar(percent: int, width: int = 30) -> str:
    filled = round(width * percent / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main() -> int:
    args = parse_args()
    last_line_length = 0
    while True:
        try:
            run = select_run(fetch_runs(args.base_url), args.run_id)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            line = f"waiting for API: {exc}"
            sys.stdout.write("\r" + line + " " * max(0, last_line_length - len(line)))
            sys.stdout.flush()
            last_line_length = len(line)
            time.sleep(args.interval)
            continue

        if run is None:
            line = "waiting for ingest run..."
            sys.stdout.write("\r" + line + " " * max(0, last_line_length - len(line)))
            sys.stdout.flush()
            last_line_length = len(line)
            time.sleep(args.interval)
            continue

        percent, stage, message = progress_from_run(run)
        line = f"{render_bar(percent)} {percent:3d}% {stage} run={run.get('id')} doc={run.get('document_id')} {message}"
        sys.stdout.write("\r" + line + " " * max(0, last_line_length - len(line)))
        sys.stdout.flush()
        last_line_length = len(line)

        if run.get("status") in TERMINAL_STATES:
            sys.stdout.write("\n")
            return 0 if run.get("status") == "completed" else 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
