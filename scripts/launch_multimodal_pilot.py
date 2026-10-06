"""Launch isolated pilot processes with durable PIDs/logs and detached sessions."""
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=["model", "api", "eval"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True)
    pid_path = runtime / (args.component + ".pid")
    if pid_path.exists():
        pid = int(pid_path.read_text())
        process_dir = Path(f"/proc/{pid}")
        if process_dir.exists() and (process_dir / "cwd").resolve() == root:
            raise RuntimeError(f"Recorded {args.component} process {pid} is still running")
    commands = {
        "model": [
            "/root/autodl-tmp/qwen3vl_baseline_20260922/venv/bin/python", "-u",
            "scripts/serve_qwen_multimodal.py", "--cache", "/root/autodl-tmp/model_cache",
            "--generator-path", "/root/autodl-tmp/model_cache/models/Qwen--Qwen3-VL-8B-Instruct/snapshots/master",
            "--image-root", str(runtime / "data/cache/multimodal_pages")],
        "api": [str(root / ".venv/bin/python"), "-u", "scripts/serve_multimodal_app.py"],
        "eval": [str(root / ".venv/bin/python"), "-u", "scripts/run_multimodal_pilot.py",
                 "--corpus", str(runtime / "corpus"),
                 "--questions", "/root/autodl-tmp/qwen3vl_baseline_20260922/pilot_v2_reviewed/dev.jsonl",
                 "--output", str(runtime / "multimodal_dev_v1")],
    }
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               MODELSCOPE_DOWNLOAD_PARALLELS="8")
    log_path = runtime / (args.component + ".log")
    with log_path.open("ab") as log:
        child = subprocess.Popen(commands[args.component], cwd=root, env=env,
                                 stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
    pid_path.write_text(str(child.pid) + "\n")
    print(json.dumps({"component": args.component, "pid": child.pid,
                      "directory": str(root), "log": str(log_path)}))


if __name__ == "__main__":
    main()
