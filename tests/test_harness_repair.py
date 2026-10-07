"""Request-local evidence and multimodal workflow regression contracts."""
import base64
from unittest.mock import Mock

import pytest
from app.models.records import Document, Project
from app.schemas.agent import EvidenceItem, EvidencePack
from app.services.search import QueryService
from app.services.rag_adapter import RAGAdapter
from app.schemas.agent import AgentQueryRequest, AgentConstraints
from test_agent_executor import build_executor_with_rag
from test_visual_routing import evidence


def test_plain_answer_exposes_actual_pixels_and_backend_without_paths(evidence, monkeypatch):
    _, service, _ = evidence
    service.parse_version_map = {'d1': 'v4'}
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw:
        service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0]))
    def post(payload):
        assert payload['messages'][1]['images']
        return {'message': {'content': '{"answer_markdown":"Purple. [0]","citations":[0]}'}}
    monkeypatch.setattr(service.ollama, '_post_chat', post)
    response = service.answer('pilot', 'In Figure 1 what color is the legend?', save_answer=False)
    assert response.metadata['retrieval_backend'] == 'lexical_or_json'
    assert response.metadata['visual_evidence']['sent'][0]['chunk_id'] == 'v4-fig'
    assert 'path' not in response.metadata['visual_evidence']['sent'][0]


def test_plain_empty_answer_does_not_reuse_previous_pixel_trace(evidence, monkeypatch):
    _, service, _ = evidence
    service.visual_evidence_trace = {'sent': [{'path': '/private/previous.png'}]}
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: [])
    monkeypatch.setattr(service.ollama, '_post_chat', lambda payload: {
        'message': {'content': '{"answer_markdown":"No evidence.","citations":[]}'}})
    prepared = service.prepare_evidence('pilot', 'Unknown source?')
    prepared.contexts = []
    prepared.pack.items = []
    prepared.retrieval_backend = 'not_queried'
    response = service.answer_from_evidence('pilot', 'Unknown source?', prepared)
    assert response.metadata['visual_evidence']['sent'] == []
    assert response.metadata['retrieval_backend'] == 'not_queried'


def test_answer_from_prepared_does_not_repeat_retrieval(evidence, monkeypatch):
    db, service, _ = evidence
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: contexts)
    prepared = service.prepare_evidence('pilot', 'In Figure 1 what color is the legend?')
    def forbidden(*args, **kwargs):
        raise AssertionError('generation repeated retrieval')
    monkeypatch.setattr(service, '_route_papers', forbidden)
    monkeypatch.setattr(service, '_build_rag_contexts', forbidden)
    monkeypatch.setattr(service.ollama, '_post_chat', lambda payload: {
        'message': {'content': '{"answer_markdown":"Purple. [0]","citations":[0]}'}})
    answer = service.answer_from_evidence('pilot', 'In Figure 1 what color is the legend?', prepared)
    assert 'Purple.' in answer.answer_markdown
    assert [c.chunk_id for c in answer.citations] == ['v5-fig']


def test_prepared_snapshot_pins_parse_versions(evidence, monkeypatch):
    db, service, root = evidence
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: contexts)
    prepared = service.prepare_evidence('pilot', 'In Figure 1 what color is the legend?')
    db.get(Document, 'd1').active_parse_version = 'v4'
    service.parse_version_map = None
    calls = []
    def post(payload):
        calls.append(payload)
        return {'message': {'content': '{"answer_markdown":"Purple. [0]","citations":[0]}'}}
    monkeypatch.setattr(service.ollama, '_post_chat', post)
    answer = service.answer_from_evidence('pilot', 'In Figure 1 what color is the legend?', prepared)
    assert {c.parse_version for c in answer.citations} == {'v5'}
    assert base64.b64decode(calls[0]['messages'][1]['images'][0]) == (root/'d1/v5/assets/figure.png').read_bytes()
    assert service.parse_version_map is None


def test_prepared_snapshot_rejects_other_project(evidence, monkeypatch):
    db, service, _ = evidence
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: [])
    prepared = service.prepare_evidence('pilot', 'legend')
    db.add(Project(id='p2', slug='other', name='Other'))
    db.commit()
    post = Mock(side_effect=AssertionError('out of scope generation'))
    monkeypatch.setattr(service.ollama, '_post_chat', post)
    with pytest.raises(ValueError, match='scope|project'):
        service.answer_from_evidence('other', 'legend', prepared)
    assert post.call_count == 0


def test_prepared_backend_is_current_retrieval_not_previous_call(evidence, monkeypatch):
    _, service, _ = evidence
    contexts = service._search_source_chunks("legend", "p1", ["d1"], question_vector=[1, 0])
    service.retrieval_backend = "pgvector"
    monkeypatch.setattr(service, "_route_papers", lambda *a, **kw: [])
    monkeypatch.setattr(service, "_build_rag_contexts", lambda *a, **kw: contexts)
    prepared = service.prepare_evidence("pilot", "legend")
    assert prepared.retrieval_backend == "canonical_sql"


def test_prepare_freezes_versions_before_routing_and_table_expansion(evidence, monkeypatch):
    from app.models.records import DocumentChunk
    db, service, _ = evidence
    service.parse_version_map = None
    for version, value in [('v4', '94%'), ('v5', '95%')]:
        db.add(DocumentChunk(id=version+'-table', document_id='d1', parse_version=version,
            chunk_role='child', block_type='table', ordinal=1, source_block_ids=['table-1'],
            source_spans=[{'table_id': 'table-1'}], embedding=[1.0, 0.0],
            text='Table 1\n| Model | Accuracy |\n|---|---|\n| A | '+value+' |'))
    db.commit()
    def route(*args, **kwargs):
        db.get(Document, 'd1').active_parse_version = 'v5'
        db.commit()
        return []
    monkeypatch.setattr(service, '_route_papers', route)
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw:
        service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0]) +
        service._search_source_chunks('In Table 1 report Model A accuracy', 'p1', ['d1'], question_vector=[1, 0]))
    prepared = service.prepare_evidence('pilot', 'In Figure 1 and Table 1 what is accuracy?')
    assert prepared.parse_version_map == {'d1': 'v4'}
    assert {c.citation.parse_version for c in prepared.contexts} == {'v4'}
    assert any(f.value == '94%' for f in prepared.pack.table_facts)
    assert all(f.value != '95%' for f in prepared.pack.table_facts)
    assert all(i.parse_version == 'v4' for i in prepared.pack.inventory)
    assert service.parse_version_map is None
    monkeypatch.setattr(service.ollama, '_post_chat', lambda payload: {
        'message': {'content': '{"answer_markdown":"94% [0]","citations":[0]}'}})
    answer = service.answer_from_evidence('pilot', 'What is shown in Figure 1?', prepared)
    assert {c.parse_version for c in answer.citations} == {'v4'}
    assert prepared.visual_evidence_trace['sent'][0]['chunk_id'] == 'v4-fig'


def test_prepare_does_not_overwrite_frozen_map_with_wrong_version(evidence, monkeypatch):
    db, service, _ = evidence
    wrong = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    service.parse_version_map = None  # active is v4, but supplied contexts are v5
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: wrong)
    prepared = service.prepare_evidence('pilot', 'legend')
    assert prepared.parse_version_map == {'d1': 'v4'}
    assert prepared.contexts == []
    assert prepared.pack.items == []


@pytest.mark.parametrize('canonical_now', [False, True])
def test_legacy_history_anchor_normalizes_identity_without_crossing_canonical(evidence, monkeypatch, canonical_now):
    db, service, _ = evidence
    service.parse_version_map = None
    db.get(Document, 'd1').active_parse_version = 'v5' if canonical_now else None
    db.commit()
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_search_source_chunks', lambda *a, **kw: [])
    prepared = service.prepare_evidence('pilot', '第二张表呢？')
    merged = service.merge_prepared_evidence(prepared, EvidencePack(status='ok', items=[
        EvidenceItem(index=0, document_id='d1', parse_version=None, evidence_kind='table',
                     block_type='table', excerpt='Table 2\n| X | Y |\n|---|---|\n| A | 94% |')]))
    assert len(merged.contexts) == (0 if canonical_now else 1)
    if not canonical_now:
        assert '94%' in merged.contexts[0].prompt_text


def test_merge_prepared_rejects_untrusted_attachment(evidence, monkeypatch):
    _, service, _ = evidence
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: [])
    prepared = service.prepare_evidence('pilot', 'question')
    pack = EvidencePack(status='ok', items=[EvidenceItem(index=0, attachment_id='foreign', excerpt='secret')])
    merged = service.merge_prepared_evidence(prepared, pack)
    assert all(c.citation.attachment_id != 'foreign' for c in merged.contexts)
    assert [i.chunk_id for i in merged.pack.items] == ['v5-fig']
    assert all(i.attachment_id != 'foreign' for i in merged.pack.items)


def real_executor(evidence, monkeypatch):
    db, service, _ = evidence
    # This integration adapter uses active versions, not the fixture service's
    # explicit shadow map. Its pre-built contexts must have the same identity.
    db.get(Document, 'd1').active_parse_version = 'v5'
    db.commit()
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    retrieval = []
    posts = []
    syntheses = []
    def route(self, question, project_id, **kwargs):
        retrieval.append(question)
        return []
    monkeypatch.setattr(QueryService, '_route_papers', route)
    monkeypatch.setattr(QueryService, '_build_rag_contexts', lambda *a, **kw: contexts)
    def post(self, payload):
        posts.append(payload)
        return {'message': {'content': '{"answer_markdown":"Purple. [0]","citations":[0]}'}}
    monkeypatch.setattr('app.services.ai.OllamaClient._post_chat', post)
    executor = build_executor_with_rag(db, RAGAdapter())
    def synth(args, ctx):
        syntheses.append(args)
        return {'answer_markdown': args['rag_answer'], 'cited_indexes': [0], 'warnings': [], 'model': 'offline'}
    executor._tools.get('answer.synthesize')['handler'] = synth
    return executor, retrieval, posts, syntheses


def test_agent_reuses_one_evidence_snapshot(evidence, monkeypatch):
    executor, retrieval, posts, _ = real_executor(evidence, monkeypatch)
    response = executor.execute(AgentQueryRequest(project_slug='pilot', query='In Figure 1 what color is the legend?'))
    assert response.status == 'completed'
    assert len(retrieval) == 1
    assert len(posts) == 1
    assert response.citations[0].parse_version == 'v5'


def test_visual_followup_sends_history_and_pixels_once(evidence, monkeypatch):
    executor, retrieval, posts, syntheses = real_executor(evidence, monkeypatch)
    executor.execute(AgentQueryRequest(project_slug='pilot', session_id='visual-session',
                                     query='In Figure 1 what color is the legend?'))
    response = executor.execute(AgentQueryRequest(project_slug='pilot', session_id='visual-session',
                               query='In Figure 1 what color is its second legend?',
                               constraints=AgentConstraints(max_steps=20, max_tool_calls=10)))
    assert 'Purple' in posts[-1]['messages'][1]['content']
    assert len(posts) == 2
    assert posts[-1]['messages'][1]['images']
    assert syntheses == []
    assert response.citations[0].chunk_id == 'v5-fig'


def test_visual_to_text_topic_switch_keeps_text_synthesis(evidence, monkeypatch):
    executor, _, posts, syntheses = real_executor(evidence, monkeypatch)
    executor.execute(AgentQueryRequest(project_slug='pilot', session_id='topic-switch',
                                     query='In Figure 1 what color is the legend?'))
    executor.execute(AgentQueryRequest(project_slug='pilot', session_id='topic-switch',
                                     query='Explain the method objective and its limitations in detail, not the previous figure.',
                                     constraints=AgentConstraints(max_steps=20, max_tool_calls=10)))
    assert len(syntheses) == 1
    assert not posts[-1]['messages'][1].get('images')


@pytest.mark.parametrize('query', ['这篇论文的方法是什么？', 'What is the paper objective?', '表1的准确率是多少？'])
def test_short_explicit_text_topic_switch_does_not_inherit_visual_intent(evidence, monkeypatch, query):
    executor, retrieval, posts, syntheses = real_executor(evidence, monkeypatch)
    executor.execute(AgentQueryRequest(project_slug='pilot', session_id='short-switch',
                                     query='解释图1的图例颜色'))
    response = executor.execute(AgentQueryRequest(project_slug='pilot', session_id='short-switch', query=query,
        constraints=AgentConstraints(max_steps=20, max_tool_calls=10)))
    assert not posts[-1]['messages'][1].get('images')
    assert len(syntheses) == 1
    assert 'Purple' in syntheses[0]['conversation_summary']
    assert retrieval[-1] == query
    assert response.metadata['visual_evidence']['intent'] is False


def test_history_fallback_replaces_snapshot(evidence, monkeypatch):
    executor, retrieval, posts, _ = real_executor(evidence, monkeypatch)
    contexts = executor._rag.prepare_evidence(evidence[0], 'pilot', 'legend').contexts
    retrieval.clear()
    executor._memory.touch_session('fallback', project_slug='pilot', ttl_days=1)
    executor._memory.add_turn('fallback', role='tool', step_type='retrieve', content='Previous',
                              citations=[{'document_id':'d1'}])
    monkeypatch.setattr(QueryService, '_build_rag_contexts',
                        lambda *a, **kw: contexts if kw.get('document_ids') else [])
    monkeypatch.setattr(QueryService, '_search_source_chunks', lambda *a, **kw: [])
    response = executor.execute(AgentQueryRequest(project_slug='pilot', session_id='fallback',
        query='In Figure 1 what color is the legend?',
        constraints=AgentConstraints(max_steps=20, max_tool_calls=10)))
    assert len(retrieval) == 2
    assert len(posts) == 1
    assert response.citations[0].document_id == 'd1'
    assert next(s for s in response.steps if s.tool_name == 'rag.answer').metadata['evidence_snapshot_reused']


def test_visual_weak_followup_keeps_original_question(evidence, monkeypatch):
    executor, _, posts, syntheses = real_executor(evidence, monkeypatch)
    executor.execute(AgentQueryRequest(project_slug='pilot', session_id='weak', query='解释图1的图例颜色'))
    executor.execute(AgentQueryRequest(project_slug='pilot', session_id='weak', query='第二个呢？',
        constraints=AgentConstraints(max_steps=20, max_tool_calls=10)))
    assert 'Question: 第二个呢？' in posts[-1]['messages'][1]['content']
    assert 'Purple' in posts[-1]['messages'][1]['content']
    assert posts[-1]['messages'][1]['images']
    assert syntheses == []


def test_visual_metric_followup_keeps_pixel_citation_with_mixed_table_evidence(evidence, monkeypatch):
    import json
    import re
    from app.models.records import DocumentChunk
    db, service, _ = evidence
    db.add(DocumentChunk(id='v5-table-metric', document_id='d1', parse_version='v5',
        chunk_role='child', block_type='table', ordinal=1, embedding=[1.0, 0.0],
        source_block_ids=['table-1'], source_spans=[{'table_id': 'table-1'}],
        text='Table 1\n| Model | Accuracy |\n|---|---|\n| A | 95% |'))
    db.commit()
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: contexts)
    prepared = service.prepare_evidence('pilot', '上一轮问题：解释图1\n当前追问：准确率指标是多少？')
    posts = []
    def post(payload):
        posts.append(payload)
        image_context = int(re.search(r'Image \d+ -> context \[(\d+)\]', payload['messages'][1]['content'])[1])
        return {'message': {'content': json.dumps({
            'answer_markdown': '97% ['+str(image_context)+']', 'citations': [image_context]})}}
    monkeypatch.setattr(service.ollama, '_post_chat', post)
    answer = service.answer_from_evidence('pilot', '准确率指标是多少？', prepared,
        visual_intent=True, conversation_summary='User asked about Figure 1')
    assert posts[0]['messages'][1]['images']
    assert [c.chunk_id for c in answer.citations] == ['v5-fig']
    assert '97% [0]' in answer.answer_markdown
    assert prepared.visual_evidence_trace['sent'][0]['citation_index'] == 0
    assert prepared.visual_evidence_trace['sent'][0]['context_index'] == 0


def test_visual_tool_trace_does_not_persist_pixels_or_paths(evidence, monkeypatch):
    executor, _, posts, _ = real_executor(evidence, monkeypatch)
    response = executor.execute(AgentQueryRequest(project_slug='pilot', query='In Figure 1 what color is the legend?'))
    trace = response.model_dump_json()
    assert posts[0]['messages'][1]['images'][0] not in trace
    assert str(evidence[2]) not in trace
    sent = next(s for s in response.steps if s.tool_name == 'rag.answer').metadata['visual_evidence']['sent']
    assert sent[0]['citation_index'] == 0
    assert sent[0]['chunk_id'] == 'v5-fig'


def test_legacy_adapter_trace_declares_snapshot_not_reused():
    from test_agent_executor import make_db, make_executor
    db = make_db()
    db.add(Project(id='p1', slug='demo', name='Demo'))
    db.commit()
    response = make_executor(db).execute(AgentQueryRequest(project_slug='demo', query='method'))
    assert next(s for s in response.steps if s.tool_name == 'rag.answer').metadata['evidence_snapshot_reused'] is False


def test_adapter_with_legacy_answer_and_new_prepare_uses_visible_fallback(evidence, monkeypatch):
    from app.schemas.common import QueryResponse
    from app.services.tool_registry import ToolRegistry
    contexts = evidence[1]._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    monkeypatch.setattr(QueryService, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(QueryService, '_build_rag_contexts', lambda *a, **kw: contexts)

    class LegacyAnswerAdapter(RAGAdapter):
        # A deployed extension can inherit new retrieval without upgrading
        # its old answer signature. Do not pass unsupported internal kwargs.
        def answer(self, db, project_slug, question, document_id=None):
            return QueryResponse(answer_markdown=question, citations=[], verification_status='local-only')

    registry = ToolRegistry()
    registry._register_builtins(LegacyAnswerAdapter())
    holder = {}
    args = {'project_slug': 'pilot', 'question': 'contextualized retrieval question'}
    retrieved = registry.call_tool('rag.retrieve_evidence', args,
                                  ctx={'db': evidence[0], 'prepared_evidence_out': holder})
    assert retrieved['ok'] is True
    answered = registry.call_tool('rag.answer', args, ctx={
        'db': evidence[0], 'prepared_evidence': holder.get('prepared'),
        'answer_query': 'original question', 'conversation_summary': 'history',
    })
    assert answered['ok'] is True
    assert answered['result']['answer_markdown'] == 'contextualized retrieval question'
    assert answered['result']['evidence_snapshot_reused'] is False
    assert holder == {}


def test_merge_rejects_old_version_table_anchor(evidence, monkeypatch):
    _, service, _ = evidence
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: [])
    prepared = service.prepare_evidence('pilot', 'question')
    prepared.parse_version_map = {'d1':'v5'}
    merged = service.merge_prepared_evidence(prepared, EvidencePack(status='ok', items=[
        EvidenceItem(index=0, document_id='d1', parse_version='v4', chunk_id='v4-fig',
                     evidence_kind='table', excerpt='OLD SECRET')]))
    assert [i.chunk_id for i in merged.pack.items] == ['v5-fig']
    assert all('OLD SECRET' not in c.prompt_text for c in merged.contexts)


def test_history_table_anchor_preserves_canonical_identity(evidence):
    from app.services.conversation_memory import ConversationMemory
    memory = ConversationMemory(evidence[0])
    memory.touch_session('tables', project_slug='pilot', ttl_days=1)
    memory.add_turn('tables', role='agent', step_type='finalize', content='Table [0]', citations=[{
        'document_id':'d1', 'parse_version':'v5', 'chunk_id':'table-child', 'table_id':'table1',
        'block_type':'table', 'excerpt':'| X | Y |', 'source_spans':[{'page_index':0}]}])
    anchor = memory.get_recent_table_anchors('tables')[0]
    assert anchor['parse_version'] == 'v5'
    assert anchor['chunk_id'] == 'table-child'
    assert anchor['source_spans'] == [{'page_index':0}]


def test_draft_does_not_mutate_shared_client_timeout(evidence, monkeypatch):
    _, service, _ = evidence
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    service.ollama.timeout = 123
    observed = []
    monkeypatch.setattr(service.ollama, '_post_chat', lambda payload: observed.append(service.ollama.timeout) or {
        'message':{'content':'{"answer_markdown":"Purple [0]","citations":[0]}'}})
    service._draft_answer('In Figure 1 what color is the legend?', None, contexts)
    assert observed == [123]
    assert service.ollama.timeout == 123


def test_draft_format_failure_does_not_restart_format_allowance(evidence, monkeypatch):
    import httpx
    from app.services import ai
    from app.services.execution_budget import ExecutionBudget, execution_budget_scope
    from test_budget_integration import transport_fixture
    _, service, _ = evidence
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    calls = []
    transport_fixture(monkeypatch, lambda req: calls.append(req) or httpx.Response(200, json={'message':{'content':'invalid'}}))
    monkeypatch.setattr(ai.time, 'sleep', lambda _: None)
    budget = ExecutionBudget(AgentConstraints(budget_tokens=100000))
    with execution_budget_scope(budget):
        service._draft_answer('What is the method?', None, contexts)
    assert len(calls) == 2
    assert budget.format_retries == 1


def test_numeric_repair_cannot_add_another_shared_retry(evidence, monkeypatch):
    from app.services.ai import QueryAnswerPayload
    from app.services.execution_budget import ExecutionBudget, execution_budget_scope
    _, service, _ = evidence
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    post = Mock(side_effect=AssertionError('unauthorized extra answer'))
    monkeypatch.setattr(service.ollama, '_post_chat', post)
    budget = ExecutionBudget(AgentConstraints(), max_answer_retries=1)
    assert budget.consume_answer_retry()
    with execution_budget_scope(budget):
        answer = service._repair_unsupported_numeric_answer('accuracy?', None, contexts,
            QueryAnswerPayload(answer_markdown='Accuracy 999 [0]', citations=[0]), [0])
    assert post.call_count == 0
    assert '999' not in answer.answer_markdown


def retry_executor():
    from test_agent_executor import make_db, make_executor
    db = make_db()
    db.add(Project(id='p1', slug='demo', name='Demo'))
    db.commit()
    executor = make_executor(db, rag_answer_text='answer [0]')
    from app.schemas.agent import ToolSpec
    executor._tools.register(ToolSpec(name='rag.retrieve_evidence', description='offline evidence', input_schema={}),
        lambda args, ctx: {'status':'empty','items':[]})
    executor._tools.get('answer.verify')['handler'] = lambda args, ctx: {
        'ok':True, 'retry_recommended':True, 'warnings':['quality failed'], 'reason':'retry'}
    return executor


def test_draft_and_final_share_one_retry():
    executor = retry_executor()
    response = executor.execute(AgentQueryRequest(project_slug='demo', query='Give sources for the method',
        constraints=AgentConstraints(max_steps=20, max_tool_calls=20)))
    assert len([s for s in response.steps if s.tool_name == 'rag.answer']) == 2
    assert response.usage.answer_retries == 1
    assert response.metadata['verification_state'] == 'failed'


def test_empty_rescue_does_not_add_second_retry():
    executor = retry_executor()
    executor._tools.get('answer.synthesize')['handler'] = lambda args, ctx: {
        'answer_markdown':'', 'cited_indexes':[], 'warnings':[]}
    response = executor.execute(AgentQueryRequest(project_slug='demo', query='Give sources for the method',
        constraints=AgentConstraints(max_steps=20, max_tool_calls=20)))
    assert len([s for s in response.steps if s.tool_name == 'rag.answer']) == 2
    assert response.usage.answer_retries == 1


@pytest.mark.parametrize('limits', [AgentConstraints(max_tool_calls=2), AgentConstraints(max_steps=3)])
def test_skipped_verify_never_passes_direct_gate(limits):
    executor = retry_executor()
    response = executor.execute(AgentQueryRequest(project_slug='demo', query='Explain the method', constraints=limits))
    assert not [s for s in response.steps if s.tool_name == 'answer.verify']
    assert response.metadata['verification_state'] == 'skipped'
    assert response.metadata['verification_executed'] is False
    assert response.answer_model != 'rag-direct'
    assert any('verification' in w.lower() for w in response.warnings)


def test_executor_real_client_token_rejection_is_visible(evidence, monkeypatch):
    import httpx
    from test_budget_integration import transport_fixture
    evidence[0].get(Document, 'd1').active_parse_version = 'v5'
    evidence[0].commit()
    executor = build_executor_with_rag(evidence[0], RAGAdapter())
    # Isolate retrieval, leaving the HTTP/client accounting real.
    monkeypatch.setattr(QueryService, '_route_papers', lambda *a, **kw: [])
    contexts = evidence[1]._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1,0])
    monkeypatch.setattr(QueryService, '_build_rag_contexts', lambda *a, **kw: contexts)
    calls = []
    transport_fixture(monkeypatch, lambda req: calls.append(req) or httpx.Response(200, json={
        'message':{'content':'{"answer_markdown":"offline [0]","citations":[0]}'}}))
    response = executor.execute(AgentQueryRequest(project_slug='pilot', query='Explain method',
        constraints=AgentConstraints(budget_tokens=1)))
    assert calls == []
    assert response.status == 'error'
    assert response.metadata['budget_stop'] == 'token_limit'
    assert response.usage.model_requests == 0


def test_tool_registry_does_not_convert_budget_stop_to_tool_error():
    from app.services.tool_registry import ToolRegistry
    from app.schemas.agent import ToolSpec
    from app.services.execution_budget import BudgetExceeded
    registry = ToolRegistry()
    registry.register(ToolSpec(name='stop', description='stop', input_schema={}),
                      lambda args, ctx: (_ for _ in ()).throw(BudgetExceeded('deadline')))
    with pytest.raises(BudgetExceeded):
        registry.call_tool('stop', {})


def test_executor_partial_stream_deadline_is_not_completed(evidence, monkeypatch):
    import httpx
    from app.services.execution_budget import ExecutionBudget
    from app.services.model_runtime import ModelRuntime
    from test_budget_integration import transport_fixture
    from test_agent_synthesizer import _fake_settings_local
    evidence[0].get(Document, 'd1').active_parse_version = 'v5'
    evidence[0].commit()
    monkeypatch.setattr('app.services.agent_synthesizer.get_settings', _fake_settings_local)
    now = [0.0]
    monkeypatch.setattr('app.services.agent_executor.ExecutionBudget',
        lambda constraints, **kwargs: ExecutionBudget(constraints, clock=lambda: now[0], **kwargs))
    contexts = evidence[1]._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1,0])
    monkeypatch.setattr(QueryService, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(QueryService, '_build_rag_contexts', lambda *a, **kw: contexts)
    events = []
    closed = []
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"message":{"content":"partial"},"done":false}\n'
            now[0] = 2.0
            yield b'{"message":{"content":"too late"},"done":false}\n'
        def close(self):
            closed.append(True)
    calls = []
    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(200, json={'message':{'content':'{"answer_markdown":"Draft [0]","citations":[0]}'},
                'prompt_eval_count':11, 'eval_count':5})
        return httpx.Response(200, stream=Stream())
    transport_fixture(monkeypatch, handler)
    executor = build_executor_with_rag(evidence[0], RAGAdapter())
    executor._event_sink = lambda name, data: events.append((name, data))
    executor._model_runtime = ModelRuntime({'generation':1})
    executor._memory.touch_session('partial-stream', project_slug='pilot', ttl_days=1)
    executor._memory.add_turn('partial-stream', role='user', content='Previous method question')
    response = executor.execute(AgentQueryRequest(project_slug='pilot', session_id='partial-stream',
        query='Explain the method and all of the limitations in detail', constraints=AgentConstraints(timeout_seconds=1)))
    assert [data['delta'] for name, data in events if name=='token'] == ['partial']
    assert response.status == 'timeout'
    assert response.metadata['budget_stop'] == 'deadline'
    assert response.metadata['verification_state'] == 'skipped'
    assert response.final_answer.startswith('Draft')
    assert len(calls) == 2
    assert closed == [True]
    assert response.usage.model_requests == 2
    assert executor._model_runtime.snapshot()['generation']['active'] == 0


def test_merged_table_facts_are_derived_from_canonical_chunks(evidence, monkeypatch):
    from app.models.records import DocumentChunk
    from app.schemas.agent import TableFactEvidence
    db, service, _ = evidence
    db.add(DocumentChunk(id='v5-table', document_id='d1', parse_version='v5', chunk_role='child',
        block_type='table', ordinal=1, source_block_ids=['table-1'],
        source_spans=[{'table_id':'table-1'}],
        text='Table 1\n| Model | Accuracy |\n|---|---|\n| A | 95% |'))
    db.commit()
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: [])
    prepared = service.prepare_evidence('pilot', 'In Table 1 report Model A accuracy')
    pack = EvidencePack(status='ok', items=[EvidenceItem(index=0, document_id='d1', chunk_id='v5-table',
        parse_version='v5', block_type='table', table_id='table-1', evidence_kind='table', excerpt='WRONG 999%')],
        table_facts=[TableFactEvidence(table_id='foreign', document_id='d1', parse_version='v5',
            column='Accuracy', value='999%', source_chunk_ids=['v4-fig'])])
    merged = service.merge_prepared_evidence(prepared, pack)
    assert all(f.value != '999%' for f in merged.pack.table_facts)
    assert any(f.value == '95%' for f in merged.pack.table_facts)
    table = next(c for c in merged.contexts if c.citation.chunk_id == 'v5-table')
    assert table.table_context is not None
    assert table.citation.excerpt != 'WRONG 999%'


def test_trusted_attachment_identity_survives_pack_projection(evidence, monkeypatch):
    _, service, _ = evidence
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_search_source_chunks', lambda *a, **kw: [])
    prepared = service.prepare_evidence('pilot', 'question')
    merged = service.merge_prepared_evidence(prepared, EvidencePack(status='ok', items=[
        EvidenceItem(index=0, attachment_id='local', excerpt='Local evidence', evidence_kind='session_attachment')]),
        trusted_attachment_ids=frozenset({'local'}))
    assert merged.pack.items[0].attachment_id == 'local'
    assert merged.contexts[0].citation.attachment_id == 'local'


def test_empty_snapshot_still_pins_versions_for_history_additions(evidence, monkeypatch):
    db, service, _ = evidence
    service.parse_version_map = None
    monkeypatch.setattr(service, '_route_papers', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_build_rag_contexts', lambda *a, **kw: [])
    monkeypatch.setattr(service, '_search_source_chunks', lambda *a, **kw: [])
    prepared = service.prepare_evidence('pilot', 'question')
    db.get(Document, 'd1').active_parse_version = 'v5'
    merged = service.merge_prepared_evidence(prepared, EvidencePack(status='ok', items=[
        EvidenceItem(index=0, document_id='d1', chunk_id='v5-fig', parse_version='v5', block_type='figure')]))
    assert merged.pack.items == []
    assert merged.contexts == []
