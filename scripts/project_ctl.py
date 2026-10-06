#!/usr/bin/env python
"""Linux/container lifecycle control, with exact process and listener ownership."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ipaddress
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dotenv import dotenv_values


class ControlError(RuntimeError):
    pass


class ProcInspector:
    def __init__(self, root=Path('/proc')):
        self.root = Path(root)

    def snapshot(self, pid):
        try:
            proc = self.root / str(pid)
            stat = (proc / 'stat').read_text().rpartition(')')[2].split()
            if stat[0] == 'Z':
                return None
            sockets = set()
            for fd in (proc / 'fd').iterdir():
                try:
                    link = os.readlink(fd)
                    if link.startswith('socket:['):
                        sockets.add(link[8:-1])
                except OSError:
                    continue
            return {'pid': pid, 'start_time': stat[19], 'cwd': str((proc / 'cwd').resolve(strict=True)),
                'exe': str((proc / 'exe').resolve(strict=True)), 'sockets': sockets,
                'argv': (proc / 'cmdline').read_bytes().decode().strip('\0').split('\0')}
        except (OSError, ValueError, IndexError, UnicodeError):
            return None

    def listeners(self, port):
        inodes = set()
        for name in ('tcp', 'tcp6'):
            path = self.root / 'net' / name
            if not path.exists():
                continue
            for line in path.read_text().splitlines()[1:]:
                fields = line.split()
                if len(fields) >= 10 and fields[3] == '0A' and int(fields[1].split(':')[1], 16) == port:
                    inodes.add(fields[9])
        return inodes


class ProjectController:
    def __init__(self, root: Path, *, env=None):
        self.root = Path(root).resolve(strict=True)
        self.env = dict(env) if env is not None else {k: v for k, v in dotenv_values(self.root / '.env').items() if v is not None}
        self.inspector = ProcInspector()
        self.control = self.root / 'runtime/control'

    def _safe_control(self):
        if (not self.control.resolve().is_relative_to(self.root)
                or (self.root / 'runtime').is_symlink() or self.control.is_symlink()):
            raise ControlError('control_path_invalid')

    def _open_control(self, path, flags, mode):
        self._safe_control()
        if path.is_symlink() or path.parent != self.control:
            raise ControlError('control_path_invalid')
        fd = os.open(path, flags | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ControlError('control_path_invalid')
        return os.fdopen(fd, mode)

    def _port(self, name):
        return int(self.env.get('LOCAL_MODEL_PORT' if name == 'model' else 'APP_PORT', 18080 if name == 'model' else 18002))

    def _argv(self, name):
        if name == 'model':
            return [self.env.get('LOCAL_MODEL_PYTHON', ''), str(self.root / 'scripts/serve_local_models.py')]
        python = self.env.get('PROJECT_API_PYTHON', str(self.root / '.venv/bin/python'))
        return [python, '-m', 'uvicorn', 'app.main:app', '--host', self.env.get('APP_HOST', '127.0.0.1'),
                '--port', str(self._port('api'))]

    def _validate(self):
        self._safe_control()
        for key in ('APP_HOST', 'LOCAL_MODEL_HOST'):
            try:
                if not ipaddress.ip_address(self.env.get(key, '127.0.0.1')).is_loopback:
                    raise ControlError('non_loopback_listener')
            except ValueError:
                raise ControlError('non_loopback_listener') from None
        model_port, api_port = self._port('model'), self._port('api')
        if model_port == api_port or any(not 1 <= port <= 65535 for port in (model_port, api_port)):
            raise ControlError('port_invalid')
        for key in ('OLLAMA_GENERATION_BASE_URL', 'OLLAMA_EMBEDDING_BASE_URL'):
            parsed = urlparse(self.env.get(key, ''))
            if parsed.scheme != 'http' or parsed.hostname != self.env.get('LOCAL_MODEL_HOST', '127.0.0.1') or parsed.port != model_port:
                raise ControlError('model_endpoint_invalid')
        if (self.env.get('VECTOR_STORE_BACKEND') != 'pgvector'
                or self.env.get('VECTOR_STORE_STRICT', '').lower() != 'true'
                or self.env.get('VECTOR_STORE_ENABLED', '').lower() != 'true'
                or int(self.env.get('OLLAMA_EMBEDDING_DIMENSIONS', 0)) != 2048
                or not self.env.get('DATABASE_URL', '').startswith('postgresql+psycopg:')
                or self.env.get('LOCAL_MODEL_EMBED_BACKEND', 'vl') != 'vl'):
            raise ControlError('pgvector_config_invalid')
        if (self.env.get('EMBEDDING_PROVIDER', 'ollama') != 'ollama'
                or self.env.get('LOCAL_MODEL_IMAGE_EMBED_NAME', 'qwen3-vl-embedding:2b')
                    != self.env.get('OLLAMA_EMBEDDING_MODEL')
                or self.env.get('LOCAL_MODEL_CHAT_ALIAS', 'qwen3-vl:4b')
                    != self.env.get('OLLAMA_GENERATION_MODEL', 'qwen3-vl:4b')
                or self.env.get('OLLAMA_VISION_MODEL', 'qwen3-vl:4b')
                    != self.env.get('OLLAMA_GENERATION_MODEL', 'qwen3-vl:4b')):
            raise ControlError('model_alias_invalid')
        for name in ('model', 'api'):
            if not Path(self._argv(name)[0]).is_absolute() or not Path(self._argv(name)[0]).is_file():
                raise ControlError('python_missing')
        for key in ('LOCAL_MODEL_IMAGE_EMBED', 'LOCAL_MODEL_CHAT'):
            path = Path(self.env.get(key, ''))
            if not path.is_absolute() or not path.is_dir():
                raise ControlError('model_assets_missing')
        adapter = self.env.get('LOCAL_MODEL_CHAT_ADAPTER')
        if adapter and (not Path(adapter).is_absolute() or not Path(adapter).is_dir()):
            raise ControlError('adapter_missing')
        allowed = set()
        for key, default in (('CANONICAL_ARTIFACTS_DIR', 'runtime/data/parsed'), ('CACHE_DIR', 'runtime/data/cache')):
            path = (self.root / self.env.get(key, default)).resolve(strict=True)
            if not path.is_relative_to((self.root / 'runtime/data').resolve()) or path == (self.root / 'runtime/data').resolve():
                raise ControlError('image_root_invalid')
            if not path.is_relative_to(self.root):
                raise ControlError('image_root_invalid')
            allowed.add(path)
        roots = {Path(value.strip()).resolve(strict=True) for value in self.env.get('LOCAL_MODEL_IMAGE_ROOTS', '').split(',') if value.strip()}
        if not roots or not roots <= allowed:
            raise ControlError('image_root_invalid')

    @contextmanager
    def _lock(self):
        self._safe_control()
        self.control.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.control.chmod(0o700)
        with self._open_control(self.control / 'lock', os.O_RDWR | os.O_CREAT, 'r+b') as handle:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                deadline = time.monotonic() + 10
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise ControlError('control_busy') from None
                        time.sleep(.05)
                try:
                    yield
                finally:
                    handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                deadline = time.monotonic() + 10
                while True:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise ControlError('control_busy') from None
                        time.sleep(.05)
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _record(self, name):
        self._safe_control()
        path = self.control / f'{name}.json'
        if path.is_symlink():
            raise ControlError('control_path_invalid')
        try:
            with self._open_control(path, os.O_RDONLY, 'r') as handle:
                record = json.load(handle)
                return record if isinstance(record, dict) else None
        except (OSError, ValueError):
            return None

    def _save_record(self, name, record):
        self._safe_control()
        self.control.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination = self.control / f'{name}.json'
        if destination.is_symlink():
            raise ControlError('control_path_invalid')
        fd, raw = tempfile.mkstemp(prefix=f'{name}.', suffix='.tmp', dir=self.control)
        temporary = Path(raw)
        with os.fdopen(fd, 'w') as handle:
            json.dump(record, handle)
        temporary.chmod(0o600)
        os.replace(temporary, destination)

    def _owned_pid(self, name):
        record = self._record(name)
        if not record or type(record.get('pid')) is not int or record['pid'] <= 1:
            return None
        actual = self.inspector.snapshot(record['pid'])
        expected = {'cwd': str(self.root), 'exe': str(Path(self._argv(name)[0]).resolve()), 'argv': self._argv(name)}
        if (not actual or any(actual.get(k) != record.get(k) or actual.get(k) != v for k, v in expected.items())
                or actual.get('start_time') != record.get('start_time')):
            return None
        return actual

    def _listener_owned(self, name, actual):
        listeners = self.inspector.listeners(self._port(name))
        return bool(actual and listeners and listeners <= actual.get('sockets', set()))

    def _database_check(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from app.core.config import Settings
        from app.services.runtime_contract import check_pgvector_contract
        config = Settings(_env_file=None, **self.env)
        engine = create_engine(config.database_url, connect_args={'connect_timeout': 5})
        try:
            with Session(engine) as db:
                return check_pgvector_contract(db, config)
        finally:
            engine.dispose()

    def _ensure_postgres(self):
        result = subprocess.run(['pg_ctlcluster', '16', 'main', 'status'], capture_output=True, timeout=10)
        if result.returncode:
            result = subprocess.run(['pg_ctlcluster', '16', 'main', 'start'], capture_output=True, timeout=45)
            if result.returncode:
                raise ControlError('postgres_start_failed')

    def _health(self, name):
        path = '/api/embedding_identity' if name == 'model' else '/api/health'
        try:
            with urlopen(f"http://{self.env.get('LOCAL_MODEL_HOST' if name == 'model' else 'APP_HOST', '127.0.0.1')}:{self._port(name)}{path}", timeout=2) as response:
                body = json.load(response)
            if name == 'model':
                from app.core.config import Settings
                from app.services.runtime_contract import EmbeddingIdentity
                return EmbeddingIdentity.from_mapping(body) == EmbeddingIdentity.from_settings(Settings(_env_file=None, **self.env))
            return body.get('status') == 'ok' and body.get('models', {}).get('vector_store', {}).get('status') == 'ready'
        except Exception:
            return False

    def _spawn(self, name):
        child_env = {**os.environ, **self.env, 'PYTHONPATH': str(self.root / 'src')}
        with self._open_control(self.control / f'{name}.log', os.O_WRONLY | os.O_APPEND | os.O_CREAT, 'ab') as log:
            process = subprocess.Popen(self._argv(name), cwd=self.root, env=child_env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        return process.pid

    def _wait_exec(self, name, pid):
        expected = {'cwd': str(self.root), 'exe': str(Path(self._argv(name)[0]).resolve()), 'argv': self._argv(name)}
        deadline = time.monotonic() + 5
        first_start = None
        while time.monotonic() < deadline:
            actual = self.inspector.snapshot(pid)
            if actual:
                first_start = first_start or actual['start_time']
                if actual['start_time'] != first_start:
                    break  # The child PID was recycled; do not adopt its replacement.
                if all(actual.get(k) == v for k, v in expected.items()):
                    return actual
            time.sleep(.02)
        raise ControlError(f'{name}_start_failed')

    def _wait_ready(self, name):
        deadline = time.monotonic() + int(self.env.get('PROJECT_START_TIMEOUT', 300))
        while time.monotonic() < deadline:
            actual = self._owned_pid(name)
            if actual is None:
                return False
            if self._listener_owned(name, actual) and self._health(name):
                return True
            time.sleep(.2)
        return False

    @staticmethod
    def _failure(exc):
        from app.services.runtime_contract import RuntimeContractError
        return {'status': 'blocked', 'reason': str(exc) if isinstance(exc, (ControlError, RuntimeContractError)) else 'control_check_failed'}

    def preflight(self):
        try:
            self._validate()
            self._database_check()
            return {'status': 'ready', 'backend': 'pgvector', 'embedding_dimensions': 2048}
        except Exception as exc:
            return self._failure(exc)

    def status(self):
        try:
            self._validate()
            report = {}
            for name in ('model', 'api'):
                actual = self._owned_pid(name)
                owned = self._listener_owned(name, actual)
                report[name] = {'owned': owned, 'healthy': bool(owned and self._health(name)),
                                'pid': actual['pid'] if owned else None}
            report['status'] = 'ready' if all(report[n]['healthy'] for n in ('model', 'api')) else 'not_ready'
            return report
        except Exception as exc:
            return self._failure(exc)

    def start(self):
        try:
            self._validate()
            with self._lock():
                # Check both ports before starting anything. A healthy HTTP response is not ownership.
                for name in ('model', 'api'):
                    listeners = self.inspector.listeners(self._port(name))
                    if listeners and not self._listener_owned(name, self._owned_pid(name)):
                        raise ControlError('foreign_port')
                self._ensure_postgres()
                self._database_check()
                for name in ('model', 'api'):
                    actual = self._owned_pid(name)
                    if self.inspector.listeners(self._port(name)) and not self._listener_owned(name, actual):
                        raise ControlError('foreign_port')
                    if actual is None:
                        pid = self._spawn(name)
                        actual = self._wait_exec(name, pid)
                        self._save_record(name, {key: actual[key] for key in ('pid', 'start_time', 'cwd', 'exe', 'argv')})
                    if not self._wait_ready(name):
                        raise ControlError(f'{name}_not_ready')
                return self.status()
        except Exception as exc:
            return self._failure(exc)

    def _signal(self, pid, sig):
        os.kill(pid, sig)

    def _signal_owned(self, name, actual, sig):
        if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
            try:
                fd = os.pidfd_open(actual['pid'])
            except ProcessLookupError:
                return False
            try:
                current = self._owned_pid(name)
                if not current or current['start_time'] != actual['start_time']:
                    return False
                signal.pidfd_send_signal(fd, sig)
                return True
            finally:
                os.close(fd)
        current = self._owned_pid(name)
        if not current or current['start_time'] != actual['start_time']:
            return False
        self._signal(actual['pid'], sig)
        return True

    def stop(self):
        stopped, refused = [], []
        try:
            with self._lock():
                for name in ('api', 'model'):
                    record = self._record(name)
                    if not record:
                        continue
                    actual = self._owned_pid(name)
                    if actual is None:
                        if self.inspector.snapshot(record.get('pid')) is not None:
                            refused.append(name)
                        continue
                    if not self._signal_owned(name, actual, signal.SIGTERM):
                        refused.append(name)
                        continue
                    deadline = time.monotonic() + 5
                    while self._owned_pid(name) is not None and time.monotonic() < deadline:
                        time.sleep(.05)
                    actual = self._owned_pid(name)  # Re-validate even before the final signal.
                    if actual is not None:
                        if not self._signal_owned(name, actual, signal.SIGKILL):
                            refused.append(name)
                            continue
                        deadline = time.monotonic() + 5
                        while self._owned_pid(name) is not None and time.monotonic() < deadline:
                            time.sleep(.05)
                    if self._owned_pid(name) is not None:
                        refused.append(name)
                    else:
                        stopped.append(name)
                return {'status': 'stopped' if not refused else 'partial', 'stopped': stopped, 'refused': refused}
        except Exception as exc:
            return self._failure(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'start', 'status', 'stop'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        result = getattr(ProjectController(args.root), args.action)()
    except Exception:
        result = {'status': 'blocked', 'reason': 'configuration_unavailable'}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['status'] in ('ready', 'stopped') else 1


if __name__ == '__main__':
    raise SystemExit(main())
