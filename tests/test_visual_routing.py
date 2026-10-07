"""Ordinary-query visual contracts: real SQLite evidence, network boundary only mocked."""
import base64
import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import pymupdf

from app.db.session import Base
from app.models.records import Document, DocumentChunk, DocumentParseVersion, Project
from app.schemas.common import Citation
from app.services.ai import cosine_similarity
from app.services.search import QueryService, RetrievedContext
from app.services import search as module


@pytest.mark.parametrize('a,b', [([1, 0], [1, 0, 5]), ([], [1])])
def test_cosine_rejects_different_vector_spaces(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


@pytest.mark.parametrize('question,want', [
    ('Which curve is highest in this scatter plot?', True),
    ('In the chart, what color is the legend?', True),
    ('How many boxes are in the left panel?', True),
    ('解释图2中的箭头方向', True), ('这张图里有几条曲线？', True),
    ('用户意图是什么', False), ('作者试图解决什么问题', False),
    ('What is the method and its objective?', False),
    ('为什么直接拟合能量曲线不如室温拟合？', False),
    ('照片中建筑门前有多少根立柱？', True),
    ('左右桑基图的分支数量分别是多少？', True),
    ('Which color is the car in this photograph?', True),
    ('What do the arrows in this image connect?', True),
    ('介绍照片分类算法的训练方法', False),
    ('折线面板中两条线总体上升还是下降？', True),
    ('图的上下两个面板中，权重增大时纵轴怎样变化？', True),
    ('曲线面板显示的训练规模和性能有何变化？', True),
])
def test_visual_intent_without_chinese_substring_false_positives(question, want):
    assert QueryService._is_figure_query(question) is want


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    monkeypatch.setattr(module.settings, 'canonical_artifacts_dir', tmp_path)
    with Session(engine) as db:
        db.add(Project(id='p1', slug='pilot', name='Pilot'))
        db.add(Document(id='d1', project_id='p1', title='Paper', file_name='p.pdf',
            raw_path='raw/p.pdf', sha256='abc', status='ready', active_parse_version='v4'))
        db.flush()
        for version in ('v4', 'v5'):
            bundle = tmp_path / 'd1' / version
            (bundle / 'assets').mkdir(parents=True)
            with pymupdf.open() as pdf:
                pdf.new_page(width=40, height=30).get_pixmap().save(bundle / 'assets/figure.png')
            image_hash = hashlib.sha256((bundle/'assets/figure.png').read_bytes()).hexdigest()
            (bundle/'manifest.json').write_text(json.dumps({'document_id':'d1', 'version':version,
                'assets':[{'path':'assets/figure.png', 'sha256':image_hash}]}))
            (bundle/'figures.json').write_text(json.dumps([{'figure_id':'figure1','asset_path':'assets/figure.png'}]))
            db.add(DocumentParseVersion(document_id='d1', version_key=version,
                artifact_dir=str(bundle), status='active' if version == 'v4' else 'ready_to_activate',
                manifest_json={'ingestion_config':{'embedding':{'provider':module.settings.active_embedding_provider,
                    'model':module.settings.active_embedding_model, 'dimensions':2}}}))
            db.add(DocumentChunk(id=version+'-fig', document_id='d1', parse_version=version,
                chunk_role='child', block_type='figure', ordinal=0,
                text='Figure 1: legend. ![figure](assets/figure.png)',
                embedding=[1.0, 0.0], source_spans=[{'page_index': 0}], page_label='1'))
        db.commit()
        service = QueryService(db, parse_version_map={'d1': 'v5'})
        service._retrieval_token_counter = lambda text: max(1, len(text.split()))
        monkeypatch.setattr(service.ollama, 'embed', lambda texts: [[1.0, 0.0] for _ in texts])
        yield db, service, tmp_path
    engine.dispose()


def test_dimension_mismatch_falls_back_to_lexical_without_crash(evidence):
    db, service, _ = evidence
    db.get(DocumentChunk, 'v5-fig').embedding = [1.0, 0.0, 0.0]
    db.commit()
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    assert contexts
    assert max(context.score for context in contexts) <= 1.2


def test_canonical_figure_channel_uses_shadow_not_active_version(evidence):
    _, service, _ = evidence
    contexts = service._search_document_figure_contexts('what color is the legend in Figure 1?', 'p1', ['d1'])
    assert [context.citation.chunk_id for context in contexts] == ['v5-fig']


def test_same_dimension_wrong_model_is_not_used_for_semantic_ranking(evidence):
    db, service, _ = evidence
    from sqlalchemy import select
    version = db.scalar(select(DocumentParseVersion).where(DocumentParseVersion.version_key == 'v5'))
    version.manifest_json = {'ingestion_config': {'embedding': {
        'provider': 'ollama', 'model': 'unrelated-model', 'dimensions': 2}}}
    db.commit()
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    assert contexts
    assert max(c.score for c in contexts) <= 1.2


def test_reserved_figures_survive_high_scoring_text_and_keep_image_rank(evidence):
    _, service, _ = evidence
    figures = [RetrievedContext(Citation(document_id='d1', excerpt='Figure '+str(i), score=score),
        'Figure '+str(i), score, evidence_kind='figure') for i, score in [(1, 8), (2, 7), (3, 6)]]
    texts = [RetrievedContext(Citation(excerpt='unrelated '+str(i), score=50+i),
        'unrelated '+str(i), 50+i) for i in range(10)]
    result = service._finalize_contexts(figures+texts, question='Which curve in the chart is highest?')
    assert [c.citation.excerpt for c in result[:3]] == ['Figure 1', 'Figure 2', 'Figure 3']


def test_ordinary_draft_sends_real_pixels_not_markdown_paths(evidence, monkeypatch):
    _, service, root = evidence
    calls = []
    def post(payload):
        calls.append(payload)
        return {'message': {'content': '{"answer_markdown":"Purple.","citations":[0],"risk_level":"normal"}'}}
    monkeypatch.setattr(service.ollama, '_post_chat', post)
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    answer = service._draft_answer('In Figure 1, what color is the legend?', None, contexts)
    assert answer.answer_markdown == 'Purple.'
    assert calls
    encoded = calls[0]['messages'][1].get('images', [])
    assert len(encoded) == 1
    assert base64.b64decode(encoded[0]) == (root/'d1/v5/assets/figure.png').read_bytes()


def test_missing_visual_asset_abstains_without_text_only_generation(evidence, monkeypatch):
    _, service, root = evidence
    (root/'d1/v5/assets/figure.png').unlink()
    calls = []
    monkeypatch.setattr(service.ollama, '_post_chat', lambda payload: calls.append(payload))
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    answer = service._draft_answer('In Figure 1, what color is the legend?', None, contexts)
    assert not calls
    assert not answer.citations
    assert '图像' in answer.answer_markdown or 'image' in answer.answer_markdown.lower()


@pytest.mark.parametrize('link', ['../../../../secret.png', '/etc/passwd', 'https://example.org/figure.png'])
def test_prompt_cannot_request_images_outside_its_canonical_bundle(evidence, monkeypatch, link):
    db, service, _ = evidence
    db.get(DocumentChunk, 'v5-fig').text = f'Figure 1: legend. ![figure]({link})'
    db.commit()
    calls = []
    monkeypatch.setattr(service.ollama, '_post_chat', lambda payload: calls.append(payload))
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    answer = service._draft_answer('In Figure 1 what color is the legend?', None, contexts)
    assert not calls
    assert not answer.citations


def test_text_query_does_not_send_figure_pixels(evidence, monkeypatch):
    _, service, _ = evidence
    calls = []
    def post(payload):
        calls.append(payload)
        return {'message': {'content': '{"answer_markdown":"Method summary.","citations":[0]}'}}
    monkeypatch.setattr(service.ollama, '_post_chat', post)
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    service._draft_answer('Explain the method.', None, contexts)
    assert calls and 'images' not in calls[0]['messages'][1]


def test_visual_answer_numbers_are_not_erased_by_text_only_repair(evidence, monkeypatch):
    _, service, _ = evidence
    calls = []
    def post(payload):
        calls.append(payload)
        return {'message': {'content': '{"answer_markdown":"The plot value is 42.7 units [0].","citations":[0]}'}}
    monkeypatch.setattr(service.ollama, '_post_chat', post)
    response = service.answer('pilot', 'In Figure 1 what is the plotted value?', save_answer=False, document_id='d1')
    assert '42.7' in response.answer_markdown
    assert len(calls) == 1
    assert calls[0]['messages'][1].get('images')
    assert response.citations and response.citations[0].chunk_id == 'v5-fig'


def test_visual_citation_is_not_replaced_by_same_number_in_unrelated_text(evidence):
    _, service, _ = evidence
    figure = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])[0]
    unrelated = RetrievedContext(Citation(excerpt='42.7 seconds was the unrelated runtime.', score=1),
        '42.7 seconds was the unrelated runtime.', 1)
    service._visual_sent_context_indexes = {0}
    assert service._supported_citation_indexes('Plot value 42.7 units [0].', [figure, unrelated], [0]) == [0]


def test_missing_canonical_identity_disables_semantic_ranking(evidence):
    db, service, _ = evidence
    from sqlalchemy import select
    version = db.scalar(select(DocumentParseVersion).where(DocumentParseVersion.version_key == 'v5'))
    version.manifest_json = {}
    db.commit()
    result = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])
    assert result and max(c.score for c in result) <= 1.2


@pytest.mark.parametrize('damage', ['unregistered', 'wrong_identity', 'replaced', 'manifest_array'])
def test_visual_pixels_require_manifest_identity_and_registered_hash(evidence, monkeypatch, damage):
    _, service, root = evidence
    bundle = root/'d1/v5'
    if damage == 'manifest_array':
        (bundle/'manifest.json').write_text('[]')
    elif damage == 'replaced':
        with pymupdf.open() as pdf:
            pdf.new_page(width=50,height=50).get_pixmap().save(bundle/'assets/figure.png')
    else:
        manifest = json.loads((bundle/'manifest.json').read_text())
        if damage == 'unregistered':
            manifest['assets'] = []
        else:
            manifest['document_id'] = 'other-doc'
        (bundle/'manifest.json').write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(service.ollama,'_post_chat', lambda payload: calls.append(payload))
    contexts = service._search_source_chunks('legend','p1',['d1'],question_vector=[1,0])
    result = service._draft_answer('In Figure 1 what color is the legend?',None,contexts)
    assert not calls and not result.citations


def test_mixed_table_figure_limit_keeps_three_visual_ranks(evidence):
    _, service, _ = evidence
    figures = [RetrievedContext(Citation(document_id='d1',parse_version='v5',excerpt='Figure '+str(i),score=6-i), 'Figure '+str(i),6-i,
        evidence_kind='figure', visual_rank=i) for i in range(3)]
    tables = [RetrievedContext(Citation(document_id='d1',parse_version='v5',table_id='table-'+str(i),
        block_type='table',excerpt='Table '+str(i)+'\n|a|b|\n|1|2|',score=60+i),
        'Table '+str(i)+'\n|a|b|\n|1|2|',60+i,evidence_kind='table') for i in range(1,4)]
    result = service._finalize_contexts(figures+tables,question='Compare the plot with Table 1 and Table 2 and Table 3')
    assert [c.visual_rank for c in result[:3]] == [0,1,2]


def test_long_figure_caption_budget_keeps_minimal_image_identity(evidence):
    _, service, _ = evidence
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 30
    figures = [RetrievedContext(Citation(excerpt='Figure '+str(i)+' '+('caption '*100),block_type='figure',score=6-i),
        'Figure '+str(i)+' '+('caption '*100),6-i,evidence_kind='figure',visual_rank=i) for i in range(3)]
    result = service._fit_contexts_to_token_budget(figures,question='Compare the plot panels')
    assert len(result) == 3
    assert sum(service._retrieval_token_counter(c.prompt_text) for c in result) <= 30


def test_partial_missing_image_is_disclosed_to_model_and_user(evidence,monkeypatch):
    db, service, root = evidence
    db.add(DocumentChunk(id='missing-fig',document_id='d1',parse_version='v5',chunk_role='child',
        block_type='figure',ordinal=1,text='Figure 2: ![figure](assets/missing.png)',embedding=[0.9,0.1]))
    db.commit()
    calls = []
    def post(payload):
        calls.append(payload)
        return {'message':{'content':'{"answer_markdown":"One picture visible.","citations":[0]}'}}
    monkeypatch.setattr(service.ollama,'_post_chat',post)
    contexts = service._search_source_chunks('figure','p1',['d1'],limit=5,question_vector=[1,0])
    result = service._draft_answer('Compare Figure 1 and Figure 2',None,contexts)
    prompt = calls[0]['messages'][1]['content']
    assert 'UNAVAILABLE IMAGE' in prompt
    assert "Do NOT say the figure is 'not included'" not in prompt
    assert '未发送' in result.answer_markdown


def test_image_limit_records_unsent_context(evidence,monkeypatch):
    from app.services import visual_evidence
    _,service,root = evidence
    monkeypatch.setattr(visual_evidence,'MAX_IMAGES',1)
    contexts=service._search_source_chunks('legend','p1',['d1'],question_vector=[1,0])
    _,_,skipped=visual_evidence.resolve_context_images(service.db,contexts*2,root,service.parse_version_map)
    assert skipped and skipped[0]['reason'] == 'image_limit'


@pytest.mark.parametrize('reordered', [False, True])
def test_visual_terminal_keeps_sent_pixels_despite_narrative_only_model_citation(evidence, monkeypatch, reordered):
    from app.schemas.agent import EvidencePack
    from app.services.search import PreparedEvidence
    db, service, root = evidence
    db.get(DocumentChunk, 'v5-fig').source_spans = [{'metadata': {'figure_id':'figure1', 'asset_id':'asset1'}}]
    db.flush()
    (root/'d1/v5/figures.json').write_text(json.dumps([{'figure_id':'figure1', 'asset_id':'asset1', 'asset_path':'assets/figure.png'}]))
    figure = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1, 0])[0]
    figure.citation.page_slug = 'same-page'
    texts = [RetrievedContext(Citation(document_id='d1', parse_version='v5', chunk_id='text-'+str(i),
        page_slug='same-page', excerpt='narrative '+str(i), score=90+i), 'narrative '+str(i), 90+i)
        for i in range(3)]
    # A real pixel resolver/answer finalizer; only the external model is offline.
    narrative_index = 0 if reordered else 1
    monkeypatch.setattr(service.ollama, '_post_chat', lambda payload: {'message': {'content': json.dumps({
        'answer_markdown':f'The color is purple [{narrative_index}].',
        'citations':[0,1,2] if reordered else [1,2,3]})}})
    prepared = PreparedEvidence('p1', 'pilot', 'Figure 1 color', 'd1', {'d1':'v5'},
        [*texts, figure] if reordered else [figure, *texts], EvidencePack(status='ok', items=[]))
    response = service.answer_from_evidence('pilot', 'In Figure 1 what color is the legend?', prepared)
    pixel_indexes = [i for i, c in enumerate(response.citations) if c.chunk_id == 'v5-fig']
    assert len(pixel_indexes) == 1
    sent = response.metadata['visual_evidence']['sent'][0]
    assert sent['citation_index'] == pixel_indexes[0]
    assert sent['document_id'] == 'd1' and sent['parse_version'] == 'v5'
    assert sent['figure_id'] == 'figure1' and sent['asset_id'] == 'asset1'
    assert response.citations[pixel_indexes[0]].figure_id == 'figure1'
    assert response.citations[pixel_indexes[0]].asset_id == 'asset1'
    assert f'[{pixel_indexes[0]}]' in response.answer_markdown


def test_generation_queue_cancels_before_ordinary_visual_model_transport(evidence, monkeypatch):
    import threading
    from app.schemas.agent import AgentConstraints
    from app.services.execution_budget import ExecutionBudget, BudgetExceeded, execution_budget_scope
    from app.services.model_runtime import ModelRuntime, model_runtime_scope
    _, service, _ = evidence
    contexts = service._search_source_chunks('legend', 'p1', ['d1'], question_vector=[1,0])
    runtime = ModelRuntime({'generation':1})
    entered, cancelled, calls = threading.Event(), threading.Event(), []
    monkeypatch.setattr(service.ollama, '_post_chat', lambda payload: calls.append(payload))
    # Cancellation occurs while the real runtime sees a queued answer ticket.
    def cancel_waiter():
        import time
        end = time.monotonic() + 2
        while not runtime.snapshot()['generation']['queued'] and time.monotonic() < end:
            entered.wait(0.01)
        cancelled.set()
    budget = ExecutionBudget(AgentConstraints(timeout_seconds=10), cancel_event=cancelled)
    with runtime.acquire('generation'):
        thread = threading.Thread(target=cancel_waiter)
        thread.start()
        try:
            with model_runtime_scope(runtime), execution_budget_scope(budget), pytest.raises(BudgetExceeded, match='cancelled'):
                service._draft_answer('Figure 1 color', None, contexts)
        finally:
            thread.join(3)
        assert runtime.snapshot()['generation']['queued'] == 0
    assert calls == [] and runtime.snapshot()['generation']['active'] == 0
