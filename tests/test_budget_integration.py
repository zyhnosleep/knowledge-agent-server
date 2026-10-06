"""Real clients with offline HTTP transport, never live model endpoints."""
import json
import threading
import time

import httpx
import pytest

from app.schemas.agent import AgentConstraints
from app.services.execution_budget import ExecutionBudget, BudgetExceeded, execution_budget_scope
from app.services import ai
from app.services.ai import OllamaClient, QueryAnswerPayload, DeepSeekClient, safe_model_call
from app.services.model_runtime import ModelRuntime


def transport_fixture(monkeypatch, handler):
    real_client = httpx.Client
    timeouts = []
    def client(*args, **kwargs):
        timeouts.append(kwargs.get('timeout'))
        return real_client(*args, **kwargs, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(ai.httpx, 'Client', client)
    return timeouts


def test_full_prompt_over_budget_never_posts(monkeypatch):
    calls = []
    transport_fixture(monkeypatch, lambda req: calls.append(req) or httpx.Response(200, json={
        'message':{'content':'answer'}, 'prompt_eval_count':11, 'eval_count':5}))
    budget = ExecutionBudget(AgentConstraints(budget_tokens=1))
    with execution_budget_scope(budget), pytest.raises(BudgetExceeded):
        OllamaClient().generate_chat(messages=[{'role':'system','content':'Only evidence'},
            {'role':'user','content':'Previous answer + retrieved evidence + new question'}],
            model='text', context_length=4096, max_output_tokens=100)
    assert calls == []
    assert budget.model_requests == 0


def test_http_calls_accumulate_usage_and_clamp_timeout_and_output(monkeypatch):
    calls = []
    def handler(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={'message':{'content':'answer'},
            'prompt_eval_count':11 if len(calls)==1 else 13, 'eval_count':5 if len(calls)==1 else 7})
    timeouts = transport_fixture(monkeypatch, handler)
    budget = ExecutionBudget(AgentConstraints(timeout_seconds=1, budget_tokens=1000), clock=lambda: 0)
    with execution_budget_scope(budget):
        for _ in range(2):
            OllamaClient().generate_chat(messages=[{'role':'user','content':'hi'}], model='text',
                                        context_length=4096, max_output_tokens=2000)
    assert (budget.prompt_tokens, budget.completion_tokens, budget.model_requests) == (24,12,2)
    assert all(t <= 1 for t in timeouts)
    assert all(c['options']['num_predict'] < 1000 for c in calls)


def test_visual_format_repair_keeps_images_and_request_wide_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(ai.settings, 'ollama_generation_context_length', 512)
    def handler(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={'message':{'content': 'invalid' if len(calls)!=2 else
            '{"answer_markdown":"Purple [0]","citations":[0]}'}, 'prompt_eval_count':20,'eval_count':10})
    transport_fixture(monkeypatch, handler)
    budget = ExecutionBudget(AgentConstraints(budget_tokens=10000))
    with execution_budget_scope(budget):
        client = OllamaClient()
        result = client.generate_structured_with_images(QueryAnswerPayload, system_prompt='read pixels',
            user_prompt='color?', images=[b'pixels'], model='vision')
        with pytest.raises(ValueError):
            client.generate_structured_with_images(QueryAnswerPayload, system_prompt='read pixels',
                user_prompt='color?', images=[b'pixels'], model='vision')
    assert 'Purple' in result.answer_markdown
    assert len(calls) == 3
    assert calls[0]['messages'][1]['images'] == calls[1]['messages'][1]['images']
    assert budget.format_retries == 1


def test_unknown_image_bound_cannot_be_bypassed_by_removing_pixels(monkeypatch):
    calls = []
    transport_fixture(monkeypatch, lambda req: calls.append(req) or httpx.Response(200, json={'message':{'content':'{}'}}))
    budget = ExecutionBudget(AgentConstraints(budget_tokens=1000))
    with execution_budget_scope(budget), pytest.raises(BudgetExceeded):
        OllamaClient().generate_chat(messages=[{'role':'user','content':'color?', 'images':['pixels']}],
                                    model='unknown', context_length=4096, max_output_tokens=100)
    assert calls == []


def test_safe_model_call_does_not_swallow_budget_stop():
    with pytest.raises(BudgetExceeded):
        safe_model_call(lambda: (_ for _ in ()).throw(BudgetExceeded('deadline')), 'fallback')


def test_embedding_batches_shrink_http_timeout_without_generation_usage(monkeypatch):
    now = [0.0]
    captured = []
    monkeypatch.setattr(ai.settings, 'embedding_provider', 'openai-compatible')
    monkeypatch.setattr(ai.settings, 'embedding_api_base_url', 'http://offline/v1')
    monkeypatch.setattr(ai.settings, 'embedding_api_key', 'offline-key')
    monkeypatch.setattr(ai.settings, 'embedding_api_batch_size', 1)
    monkeypatch.setattr(ai.settings, 'embedding_dimensions', 2)
    monkeypatch.setattr(ai.settings, 'embedding_api_timeout', 100)

    def handler(request):
        captured.append(request.extensions['timeout']['read'])
        now[0] += 0.4
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1.0, 0.0]}]})

    transport_fixture(monkeypatch, handler)
    budget = ExecutionBudget(AgentConstraints(timeout_seconds=1), clock=lambda: now[0])
    with execution_budget_scope(budget):
        assert OllamaClient().embed(['first', 'second']) == [[1.0, 0.0], [1.0, 0.0]]
    assert captured == pytest.approx([1.0, 0.6])
    assert budget.model_requests == 0
    assert (budget.prompt_tokens, budget.completion_tokens) == (0, 0)


def test_deepseek_retry_accounts_every_attempt(monkeypatch):
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(503 if len(calls)==1 else 200, json={
            'choices':[{'message':{'content':'ok'}}], 'usage':{'prompt_tokens':11,'completion_tokens':5}})
    timeouts = transport_fixture(monkeypatch, handler)
    budget = ExecutionBudget(AgentConstraints(timeout_seconds=1))
    with execution_budget_scope(budget):
        client = DeepSeekClient(base_url='http://offline', api_key='offline-key', model='text',
                               timeout=100, max_retries=1, retry_backoff_seconds=0)
        result = client.generate_chat(messages=[{'role':'user','content':'hello'}], max_output_tokens=100)
    assert result['content'] == 'ok'
    assert budget.model_requests == 2
    assert budget.usage_metadata()['usage_source'] == 'mixed'
    assert all(t <= 1 for t in timeouts)


def test_fifo_deadline_removes_ticket_and_next_waiter_acquires():
    runtime = ModelRuntime({'generation':1})
    outcomes = []
    def waiter():
        try:
            with runtime.acquire('generation', deadline=time.monotonic()+0.02):
                outcomes.append('acquired')
        except BudgetExceeded:
            outcomes.append('expired')
    with runtime.acquire('generation'):
        thread = threading.Thread(target=waiter)
        thread.start()
        thread.join(timeout=1)
        assert outcomes == ['expired']
        assert runtime.snapshot()['generation']['queued'] == 0
    with runtime.acquire('generation'):
        assert runtime.snapshot()['generation']['active'] == 1


def test_stream_deadline_closes_response_and_keeps_partial_usage(monkeypatch):
    now = [0.0]
    closed = []
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"message":{"content":"first"},"done":false}\n'
            now[0] = 2.0
            yield b'{"message":{"content":"late"},"done":false}\n'
        def close(self):
            closed.append(True)
    transport_fixture(monkeypatch, lambda req: httpx.Response(200, stream=Stream()))
    budget = ExecutionBudget(AgentConstraints(timeout_seconds=1), clock=lambda: now[0])
    with execution_budget_scope(budget):
        stream = OllamaClient().stream_chat(messages=[{'role':'user','content':'hi'}],
                                           model='text', context_length=4096, max_output_tokens=50)
        assert next(stream)['content'] == 'first'
        with pytest.raises(BudgetExceeded, match='deadline'):
            next(stream)
    assert closed == [True]
    assert budget.model_requests == 1
    assert budget.usage_metadata()['unsettled_requests'] == 0


@pytest.mark.parametrize('reason', ['deadline', 'cancelled'])
@pytest.mark.parametrize('transport', ['close_interrupt', 'native_socket'])
def test_stalled_partial_stream_is_interrupted_without_waiting_for_next_line(monkeypatch, reason, transport):
    import socket
    now = [0.0]
    cancel = threading.Event()
    unblock = threading.Event()
    closed = []
    read_socket, peer_socket = socket.socketpair()
    read_socket.settimeout(2)
    class StalledStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"message":{"content":"partial"},"done":false}\n'
            if reason == 'deadline':
                now[0] = 2.0
            else:
                cancel.set()
            yield b'{"message":{"content":"unfinished'
            if transport == 'native_socket':
                read_socket.recv(1)  # close alone cannot reliably wake recv on Windows
            elif not unblock.wait(2):
                yield b'"},"done":false}\n'
        def close(self):
            closed.append(True)
            unblock.set()
            read_socket.close()
    class NativeNetworkStream:
        def get_extra_info(self, key):
            return read_socket if key == 'socket' else None
    transport_fixture(monkeypatch, lambda req: httpx.Response(200, stream=StalledStream(),
        extensions={'network_stream': NativeNetworkStream()} if transport == 'native_socket' else {}))
    budget = ExecutionBudget(AgentConstraints(timeout_seconds=1), clock=lambda: now[0], cancel_event=cancel)
    runtime = ModelRuntime({'generation': 1})
    try:
        with execution_budget_scope(budget), runtime.acquire('generation'):
            stream = OllamaClient().stream_chat(messages=[{'role':'user','content':'hi'}],
                model='text', context_length=4096, max_output_tokens=50, cancel_event=cancel)
            assert next(stream)['content'] == 'partial'
            started = time.monotonic()
            with pytest.raises(BudgetExceeded, match=reason):
                next(stream)
            elapsed = time.monotonic() - started
    finally:
        read_socket.close()
        peer_socket.close()
    assert elapsed < 0.6
    assert closed == [True]
    assert budget.model_requests == 1
    assert budget.usage_metadata()['unsettled_requests'] == 0
    assert runtime.snapshot()['generation']['active'] == 0


@pytest.mark.parametrize('provider', ['local', 'ollama', 'external_api'])
def test_synthesizer_propagates_budget_termination(monkeypatch, provider):
    from app.services.agent_synthesizer import AgentSynthesizer
    from test_agent_synthesizer import _fake_settings_local, _fake_settings_ollama, _fake_settings_external
    factories = {'local':_fake_settings_local, 'ollama':_fake_settings_ollama, 'external_api':_fake_settings_external}
    monkeypatch.setattr('app.services.agent_synthesizer.get_settings', factories[provider])
    calls = []
    transport_fixture(monkeypatch, lambda req: calls.append(req) or httpx.Response(200, json={
        'message':{'content':'ok'}, 'choices':[{'message':{'content':'{}'}}]}))
    budget = ExecutionBudget(AgentConstraints(budget_tokens=1))
    with execution_budget_scope(budget), pytest.raises(BudgetExceeded):
        AgentSynthesizer().synthesize(query='Explain evidence', route='simple_rag', rag_answer='draft',
                                     citations=[{'document_id':'d1','excerpt':'Source'}])
    assert calls == []


def test_synthesis_coverage_repair_shares_answer_allowance(monkeypatch):
    from app.services.agent_synthesizer import AgentSynthesizer
    from test_agent_synthesizer import _fake_settings_local
    monkeypatch.setattr('app.services.agent_synthesizer.get_settings', _fake_settings_local)
    calls = []
    transport_fixture(monkeypatch, lambda req: calls.append(req) or httpx.Response(200, json={
        'message':{'content':'short [0]'}, 'prompt_eval_count':10,'eval_count':5}))
    budget = ExecutionBudget(AgentConstraints(), max_answer_retries=1)
    assert budget.consume_answer_retry() is True
    with execution_budget_scope(budget):
        result = AgentSynthesizer().synthesize(query='Compare accuracy', route='table_or_metric', rag_answer='draft',
            citations=[{'document_id':'d1','excerpt':'Accuracy 95% AUC 0.9'}],
            evidence_pack={'items':[{'excerpt':'Accuracy 95% AUC 0.9'}]})
    assert len(calls) == 1
    assert 'short' in result['answer_markdown']
