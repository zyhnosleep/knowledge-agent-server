"""Model-serving contract tests; skipped in the CPU-only application venv."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('torch')
pytest.importorskip('PIL')
from fastapi.testclient import TestClient


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv('LOCAL_MODEL_PRELOAD', 'none')
    for name in ('encoder', 'chat', 'adapter', 'parsed', 'cache'):
        (tmp_path / name).mkdir()
    for key, name in (('LOCAL_MODEL_IMAGE_EMBED', 'encoder'), ('LOCAL_MODEL_CHAT', 'chat'),
                      ('LOCAL_MODEL_CHAT_ADAPTER', 'adapter')):
        monkeypatch.setenv(key, str(tmp_path / name))
    root = Path(__file__).resolve().parents[1]
    # Configuration path safety is independently tested without importing torch.
    monkeypatch.setattr('app.services.local_model_config.read_local_model_paths', lambda env, root: {
        'text_embed': None, 'image_embed': tmp_path / 'encoder', 'chat': tmp_path / 'chat',
        'adapter': tmp_path / 'adapter', 'image_roots': (tmp_path / 'parsed', tmp_path / 'cache')})
    path = Path(__file__).resolve().parents[1] / 'scripts/serve_local_models.py'
    spec = importlib.util.spec_from_file_location('contract_server', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unknown_alias_fails_without_loading_base(server, monkeypatch):
    calls = []
    monkeypatch.setattr(server.CHAT_MODEL, 'generate', lambda **kw: calls.append(kw) or {
        'content':'{}', 'adapter':False, 'prompt_eval_count':1, 'eval_count':1, 'guided':False})
    with TestClient(server.app) as client:
        result = client.post('/api/chat', json={'model':'typo-ft', 'messages':[{'role':'user','content':'question'}]})
    assert result.status_code == 400
    assert not calls


def test_ft_request_requires_available_adapter(server, monkeypatch):
    server.CHAT_MODEL.adapter_path = None
    monkeypatch.setattr(server.CHAT_MODEL, 'generate', lambda **kw: {
        'content':'{}', 'adapter':False, 'prompt_eval_count':1, 'eval_count':1, 'guided':False})
    with TestClient(server.app) as client:
        result = client.post('/api/chat', json={'model':server.CHAT_FT_ALIAS, 'messages':[{'role':'user','content':'question'}]})
    assert result.status_code == 400


@pytest.mark.parametrize('tokens,eos,want_reason', [([4,99],True,'stop'),([4,5],False,'length')])
def test_generation_reports_real_eos_and_limit(server, monkeypatch, tokens, eos, want_reason):
    import torch
    chat = server.ChatModel(Path('/unused'),None)
    processor = SimpleNamespace(decode=lambda *a,**k:'answer')
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[99]),
        generate=lambda **k:torch.tensor([[1,*tokens]]))
    monkeypatch.setattr(chat,'_ensure',lambda:(processor,model))
    monkeypatch.setattr(chat,'_build_inputs',lambda *a:({},1))
    monkeypatch.setattr(chat,'_schema_logits_processor',lambda *a:None)
    result = chat.generate(messages=[{'role':'user','content':'q'}],use_adapter=False,max_new_tokens=2)
    assert result['eos_reached'] is eos
    assert result['done_reason'] == want_reason


def test_chat_endpoint_preserves_length_reason(server,monkeypatch):
    monkeypatch.setattr(server.CHAT_MODEL,'generate',lambda **k:{'content':'answer','adapter':False,
        'prompt_eval_count':1,'eval_count':2,'guided':False,'eos_reached':False,'done_reason':'length'})
    with TestClient(server.app) as client:
        response=client.post('/api/chat',json={'model':server.CHAT_ALIAS,'messages':[{'role':'user','content':'q'}]})
    assert response.json()['done_reason']=='length'
    assert response.json()['eos_reached'] is False
