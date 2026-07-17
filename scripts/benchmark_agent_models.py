from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx


def benchmark(
    *, api_url: str, project_slug: str, question: str, answer_mode: str
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at: float | None = None
    token_events = 0
    final: dict[str, Any] = {}
    payload = {
        "project_slug": project_slug,
        "query": question,
        "answer_mode": answer_mode,
        "constraints": {"timeout_seconds": 600},
    }
    with httpx.Client(timeout=660) as client:
        with client.stream(
            "POST",
            f"{api_url.rstrip('/')}/api/agent/query/stream",
            json=payload,
        ) as response:
            response.raise_for_status()
            event_name = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event_name = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if event_name == "token":
                        token_events += 1
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                    elif event_name == "final":
                        final = data

    finished = time.perf_counter()
    usage = final.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or token_events)
    generation_seconds = max(
        0.001, finished - (first_token_at or finished)
    )
    return {
        "answer_mode": answer_mode,
        "selected_model": final.get("model") or final.get("answer_model"),
        "ttft_ms": (
            round((first_token_at - started) * 1000, 1)
            if first_token_at is not None
            else None
        ),
        "total_latency_ms": round((finished - started) * 1000, 1),
        "completion_tokens": completion_tokens,
        "tokens_per_second": round(completion_tokens / generation_seconds, 2),
        "citation_count": len(final.get("citations") or []),
        "status": final.get("status", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fast/deep Agent routes")
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--project", default="internal-research")
    parser.add_argument(
        "--question",
        default="请根据知识库总结核心结论并给出来源。",
    )
    args = parser.parse_args()
    results = [
        benchmark(
            api_url=args.api_url,
            project_slug=args.project,
            question=args.question,
            answer_mode=mode,
        )
        for mode in ("fast", "deep")
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
