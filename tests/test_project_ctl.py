"""Lifecycle safety at the process/filesystem boundaries; never signal foreign PIDs."""
import json
import threading
from pathlib import Path

import pytest

from scripts.project_ctl import ProjectController, ProcInspector


@pytest.fixture
def controller(tmp_path, monkeypatch):
    import scripts.project_ctl as module
    # Fake PIDs are outside the OS boundary; the dedicated pidfd test below
    # installs a controlled kernel double that recycles a PID while opening it.
    monkeypatch.delattr(module.os, "pidfd_open", raising=False)
    for folder in ("runtime/data/parsed", "runtime/data/cache", "models/embed", "models/chat", ".venv/bin", "modelenv/bin", "scripts"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    for path in (".venv/bin/python", "modelenv/bin/python", "scripts/serve_local_models.py"):
        (tmp_path / path).write_text("fixture")
    env = {"APP_HOST": "127.0.0.1", "APP_PORT": "18002", "AUTH_ENABLED": "false",
        "DATABASE_URL": "postgresql+psycopg:///test", "VECTOR_STORE_BACKEND": "pgvector",
        "VECTOR_STORE_ENABLED": "true", "VECTOR_STORE_STRICT": "true", "OLLAMA_EMBEDDING_DIMENSIONS": "2048",
        "OLLAMA_EMBEDDING_MODEL": "qwen3-vl-embedding:2b", "EMBEDDING_REVISION": "weights", "EMBEDDING_PROCESSOR_HASH": "processor",
        "OLLAMA_GENERATION_BASE_URL": "http://127.0.0.1:18080", "OLLAMA_EMBEDDING_BASE_URL": "http://127.0.0.1:18080",
        "LOCAL_MODEL_HOST": "127.0.0.1", "LOCAL_MODEL_PORT": "18080", "LOCAL_MODEL_EMBED_BACKEND": "vl",
        "LOCAL_MODEL_IMAGE_EMBED": str(tmp_path / "models/embed"), "LOCAL_MODEL_CHAT": str(tmp_path / "models/chat"),
        "LOCAL_MODEL_PYTHON": str(tmp_path / "modelenv/bin/python"),
        "LOCAL_MODEL_IMAGE_ROOTS": str(tmp_path / "runtime/data/parsed") + "," + str(tmp_path / "runtime/data/cache"),
        "CANONICAL_ARTIFACTS_DIR": str(tmp_path / "runtime/data/parsed"), "CACHE_DIR": str(tmp_path / "runtime/data/cache")}
    ctl = ProjectController(tmp_path, env=env)
    processes, listeners, launched, killed = {}, {}, [], []
    monkeypatch.setattr(ctl.inspector, "snapshot", lambda pid: processes.get(pid))
    monkeypatch.setattr(ctl.inspector, "listeners", lambda port: listeners.get(port, set()))
    monkeypatch.setattr(ctl, "_database_check", lambda: {"status": "ready", "backend": "pgvector"})
    monkeypatch.setattr(ctl, "_ensure_postgres", lambda: None)
    monkeypatch.setattr(ctl, "_health", lambda name: True)
    def spawn(name):
        launched.append(name)
        pid = 100 + len(launched)
        argv = ctl._argv(name)
        inode = str(pid + 1000)
        processes[pid] = {"pid": pid, "start_time": "123", "cwd": str(tmp_path.resolve()),
            "exe": str(Path(argv[0]).resolve()), "argv": argv, "sockets": {inode}}
        listeners[ctl._port(name)] = {inode}
        return pid
    monkeypatch.setattr(ctl, "_spawn", spawn)
    def signal(pid, sig):
        killed.append(pid)
        processes.pop(pid, None)
    monkeypatch.setattr(ctl, "_signal", signal)
    return ctl, processes, listeners, launched, killed


def record(ctl, name, pid=999, start="123"):
    argv = ctl._argv(name)
    ctl._save_record(name, {"pid": pid, "start_time": start, "cwd": str(ctl.root),
        "exe": str(Path(argv[0]).resolve()), "argv": argv})


def test_reused_pid_is_never_killed(controller):
    ctl, processes, listeners, launched, killed = controller
    record(ctl, "api")
    processes[999] = {"pid": 999, "start_time": "999", "cwd": str(ctl.root),
        "exe": ctl._argv("api")[0], "argv": ctl._argv("api"), "sockets": {"12"}}
    listeners[18002] = {"12"}
    result = ctl.stop()
    assert killed == [] and result["refused"] == ["api"]


@pytest.mark.parametrize("changed", ["cwd", "argv", "exe"])
def test_process_identity_mismatch_cannot_be_stopped(controller, changed):
    ctl, processes, listeners, launched, killed = controller
    ctl.start()
    pid = 101
    processes[pid][changed] = ["other-program"] if changed == "argv" else "other"
    ctl.stop()
    assert pid not in killed


def test_foreign_port_blocks_start_even_if_http_is_healthy(controller):
    ctl, _, listeners, launched, killed = controller
    listeners[18080] = {"foreign-inode"}
    result = ctl.start()
    assert result["status"] == "blocked" and result["reason"] == "foreign_port"
    assert launched == [] and killed == []


def test_owned_pid_with_foreign_listener_is_not_ready(controller):
    ctl, _, listeners, _, killed = controller
    assert ctl.start()["status"] == "ready"
    listeners[18002] = {"someone-elses-socket"}
    result = ctl.status()
    assert result["status"] != "ready" and result["api"]["owned"] is False


def test_two_start_calls_launch_each_service_once(controller):
    ctl, _, _, launched, _ = controller
    results = []
    first = threading.Thread(target=lambda: results.append(ctl.start()))
    second = threading.Thread(target=lambda: results.append(ctl.start()))
    first.start(); second.start()
    first.join(3); second.join(3)
    assert not first.is_alive() and not second.is_alive()
    assert launched == ["model", "api"]
    assert len(results) == 2 and all(r["status"] == "ready" for r in results)


def test_status_is_read_only_and_cannot_launch(controller):
    ctl, _, _, launched, _ = controller
    before = set(ctl.root.rglob("*"))
    assert ctl.status()["status"] != "ready"
    assert set(ctl.root.rglob("*")) == before and launched == []


def test_stop_only_owned_services_and_never_database(controller):
    ctl, processes, _, launched, killed = controller
    assert ctl.start()["status"] == "ready"
    processes[700] = {"argv": ["postgres"]}
    result = ctl.stop()
    assert result["stopped"] == ["api", "model"]
    assert killed == [102, 101] and 700 in processes


def test_anonymous_public_listen_is_rejected(controller):
    ctl, _, _, launched, _ = controller
    ctl.env["APP_HOST"] = "0.0.0.0"
    result = ctl.preflight()
    assert result["status"] == "blocked" and result["reason"] == "non_loopback_listener"
    assert launched == []


def test_image_root_cannot_expand_to_workspace(controller):
    ctl, _, _, _, _ = controller
    ctl.env["LOCAL_MODEL_IMAGE_ROOTS"] = str(ctl.root)
    assert ctl.preflight()["reason"] == "image_root_invalid"


def test_image_symlink_outside_project_is_rejected(controller, tmp_path):
    ctl, _, _, _, _ = controller
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    link = tmp_path / "runtime/data/escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("host does not permit test symlinks; Linux acceptance must cover this")
    ctl.env["CANONICAL_ARTIFACTS_DIR"] = str(link)
    ctl.env["LOCAL_MODEL_IMAGE_ROOTS"] = str(link)
    assert ctl.preflight()["reason"] == "image_root_invalid"


def test_preflight_reports_contract_error_without_credentials(controller, monkeypatch):
    from app.services.runtime_contract import RuntimeContractError
    ctl, _, _, _, _ = controller
    def fail():
        raise RuntimeContractError("embedding_identity_unverified")
    monkeypatch.setattr(ctl, "_database_check", fail)
    result = ctl.preflight()
    assert result["status"] == "blocked" and result["reason"] == "embedding_identity_unverified"
    assert "DATABASE_URL" not in json.dumps(result)


def test_api_cannot_launch_before_model_health_and_database_contract(controller, monkeypatch):
    ctl, _, _, launched, _ = controller
    monkeypatch.setattr(ctl, "_wait_ready", lambda name: False)
    result = ctl.start()
    assert result["status"] == "blocked" and launched == ["model"]


def test_proc_listener_parse_filters_state_and_port(tmp_path):
    net = tmp_path / "net"
    net.mkdir()
    (net / "tcp").write_text("header\n 0: 0100007F:4652 00000000:0000 0A 0:0 00:0 0 0 0 123\n"
                            " 1: 0100007F:4652 00000000:0000 01 0:0 00:0 0 0 0 456\n")
    assert ProcInspector(tmp_path).listeners(18002) == {"123"}


def test_start_waits_for_exec_before_recording_process(controller, monkeypatch):
    ctl, processes, _, launched, _ = controller
    snapshot = ctl.inspector.snapshot
    seen = set()
    def transitioning(pid):
        if pid not in seen and pid in processes:
            seen.add(pid)
            return {**processes[pid], "argv": ["parent-before-exec"]}
        return snapshot(pid)
    monkeypatch.setattr(ctl.inspector, "snapshot", transitioning)
    assert ctl.start()["status"] == "ready"
    assert launched == ["model", "api"]
    assert ctl._record("api")["argv"] == ctl._argv("api")


def test_port_taken_during_model_start_blocks_api_launch(controller, monkeypatch):
    ctl, _, listeners, launched, _ = controller
    ready = ctl._wait_ready
    def wait(name):
        result = ready(name)
        if name == "model":
            listeners[18002] = {"foreign-inode"}
        return result
    monkeypatch.setattr(ctl, "_wait_ready", wait)
    assert ctl.start()["reason"] == "foreign_port"
    assert launched == ["model"]


def test_stop_does_not_claim_success_before_exit(controller, monkeypatch):
    import scripts.project_ctl as module
    ctl, _, _, _, _ = controller
    assert ctl.start()["status"] == "ready"
    monkeypatch.setattr(ctl, "_signal", lambda *args: None)
    monkeypatch.setattr(module.signal, "SIGKILL", 9, raising=False)
    tick = [0]
    def clock():
        tick[0] += 1
        return tick[0]
    monkeypatch.setattr(module.time, "monotonic", clock)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = ctl.stop()
    assert result["status"] == "partial"
    assert result["stopped"] == []
    assert result["refused"] == ["api", "model"]


@pytest.mark.parametrize("filename", ["lock", "model.log", "api.json"])
def test_control_symlinks_cannot_write_or_read_another_file(controller, filename, monkeypatch):
    ctl, _, _, launched, killed = controller
    ctl.control.mkdir(parents=True)
    outside = ctl.root / "must-preserve.txt"
    outside.write_text("preserve")
    try:
        (ctl.control / filename).symlink_to(outside)
    except OSError:
        pytest.skip("Linux acceptance covers symlink creation")
    if filename == "model.log":
        monkeypatch.setattr(ctl, "_spawn", ProjectController._spawn.__get__(ctl))
    result = ctl.start()
    assert result["status"] == "blocked"
    assert outside.read_text() == "preserve"
    assert killed == []


def test_wrong_model_alias_is_blocked_before_launch(controller):
    ctl, _, _, launched, _ = controller
    ctl.env["LOCAL_MODEL_IMAGE_EMBED_NAME"] = "other-space"
    assert ctl.preflight()["status"] == "blocked"
    assert launched == []


def test_pid_recycled_between_validation_and_signal_is_not_signalled(controller, monkeypatch):
    import scripts.project_ctl as module
    ctl, processes, _, _, killed = controller
    assert ctl.start()["status"] == "ready"
    closed, delivered = [], []
    def open_pidfd(pid):
        processes[pid] = {**processes[pid], "start_time": "recycled"}
        return 555
    monkeypatch.setattr(module.os, "pidfd_open", open_pidfd, raising=False)
    monkeypatch.setattr(module.os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(module.signal, "pidfd_send_signal", lambda *args: delivered.append(args), raising=False)
    result = ctl.stop()
    assert delivered == [] and killed == []
    assert result["status"] == "partial"
    assert set(result["refused"]) == {"api", "model"}
    assert closed == [555, 555]
