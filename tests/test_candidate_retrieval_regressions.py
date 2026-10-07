"""Real candidate failures: failed active status and bilingual numbered tables."""
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from app.db.session import Base
from app.models.records import Document, DocumentChunk, DocumentParseVersion, Project
from app.services.search import QueryService


@pytest.fixture
def db():
    engine=create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id='p',slug='pilot',name='Pilot'))
        session.add(Document(id='d',project_id='p',title='Corrective Retrieval Augmented Generation',
            file_name='CRAG.pdf',sha256='source',raw_path='raw/CRAG.pdf',status='activation_failed',active_parse_version='old'))
        session.add(DocumentParseVersion(id='v',document_id='d',version_key='staged',
            artifact_dir='parsed/d/staged',status='ready_to_activate'))
        session.add(DocumentChunk(id='text',document_id='d',ordinal=0,parse_version='staged',
            chunk_role='child',block_type='narrative',text='CRAG retrieval confidence chooses Correct Incorrect Ambiguous actions.',
            source_spans=[{'page_label':'2','page_index':1}]))
        session.commit()
        yield session


def test_ready_staged_version_of_failed_document_routes_and_retrieves_without_activation(db):
    service=QueryService(db,parse_version_map={'d':'staged'})
    papers=service._route_papers('How does CRAG use retrieval confidence?', 'p',document_id='d')
    assert [p.document.id for p in papers]==['d']
    contexts=service._search_source_chunks('CRAG retrieval confidence', 'p',['d'],question_vector=[],limit=5)
    assert [c.citation.chunk_id for c in contexts]==['text']
    assert contexts[0].citation.page_title
    assert db.get(Document,'d').active_parse_version=='old'
    assert db.get(Document,'d').status=='activation_failed'


@pytest.mark.parametrize('status',['queued','embedding_failed','indexing'])
def test_unready_staged_version_cannot_rescue_failed_document(db,status):
    db.get(DocumentParseVersion,'v').status=status
    db.commit()
    service=QueryService(db,parse_version_map={'d':'staged'})
    assert service._route_papers('CRAG retrieval confidence','p',document_id='d')==[]
    assert service._search_source_chunks('CRAG retrieval confidence','p',['d'],question_vector=[])==[]


def test_failed_active_document_remains_hidden_without_candidate_selection(db):
    assert QueryService(db)._route_papers('CRAG retrieval confidence','p',document_id='d')==[]
    assert QueryService(db,parse_version_map={})._route_papers('CRAG','p',document_id='d')==[]


@pytest.mark.parametrize('question,caption,row,value',[
    ('读取 ReAct 表1中 Act 和 ReAct 在 HotpotQA 与 Fever 上各自的成绩。',
     'Table 1: Prompting results','Act','25.7'),
    ('读取 MuRAG 表1中 CC 和 LAION 的数据规模与数据格式。',
     'Table 1: Pre-training datasets','CC','15M'),
])
def test_chinese_numbered_table_selects_canonical_target_not_similar_appendix(db,question,caption,row,value):
    document=db.get(Document,'d')
    document.status='ready'
    document.active_parse_version='staged'
    db.add_all([
        DocumentChunk(id='target',document_id='d',ordinal=1,parse_version='staged',chunk_role='child',block_type='table',
            text=f'{caption}\n| Method | Size | Format |\n| --- | --- | --- |\n| {row} | {value} | Image/Caption |',
            source_spans=[{'page_label':'3','metadata':{'table_id':'target-table'}}]),
        DocumentChunk(id='wrong',document_id='d',ordinal=2,parse_version='staged',chunk_role='child',block_type='table',
            text='Table 10: ReAct HotpotQA and Fever results MuRAG datasets\n| Method | Size | Format |\n| --- | --- | --- |\n| Act ReAct CC LAION | 999 | incorrect |',
            source_spans=[{'page_label':'12','metadata':{'table_id':'wrong-table'}}]),
    ])
    db.commit()
    service=QueryService(db)
    contexts=service._search_source_chunks(question,'p',['d'],question_vector=[],limit=1)
    assert contexts and contexts[0].citation.chunk_id=='target'
    assert contexts[0].citation.table_id=='target-table'
    assert any(f.value==value for f in contexts[0].table_facts)
    assert all(c.citation.table_id!='wrong-table' for c in contexts)


@pytest.mark.parametrize('question',['读取表1中Act和ReAct成绩','Read Table 1 Act and ReAct scores'])
def test_explicit_table_number_cannot_be_replaced_by_entity_matched_table10(question):
    block='Table 10: results\n| Model | Score |\n| --- | --- |\n| Act | 99 |\n| ReAct | 98 |'
    assert not QueryService._table_block_matches_query(question,block)
    assert not QueryService._table_group_matches_query(question,block)


@pytest.mark.parametrize('question',[
    '读取 MuRAG 表1中 CC 和 LAION 的数据规模与数据格式。',
    '读取MuRAG表1中CC和LAION的数据规模与数据格式。',
])
def test_answer_row_selection_keeps_explicit_two_letter_dataset_alongside_longer_name(question):
    from app.services.table_evidence import CanonicalTableChunk,assemble_table_context,extract_table_facts
    table=assemble_table_context([CanonicalTableChunk('c','d','staged','t',0,
        'Table 1: Pre-training Dataset Statistics\n| Dataset | #Size | Format | Source |\n| --- | --- | --- | --- |\n'
        '| CC | 15M | Image, Caption | Crawled |\n| LAION | 200M | Image, Alt-Text | Crawled |\n'
        '| PAQ | 65M | Passage, QA | Generated |')])
    rows=QueryService._generic_table_fact_value_rows(question,tuple(extract_table_facts(question,table)),table.markdown)
    by_label={row['group']:row['values'] for row in rows}
    assert set(by_label)=={'CC','LAION'}
    assert '15M' in by_label['CC'] and '200M' in by_label['LAION']
    from app.schemas.common import Citation
    from app.services.search import RetrievedContext
    context=RetrievedContext(Citation(document_id='d',parse_version='staged',table_id='t',
        chunk_id='c',block_type='table',excerpt=table.markdown,score=1),table.markdown,1,
        evidence_kind='table',table_context=table,table_facts=tuple(extract_table_facts(question,table)))
    answer=QueryService(None)._deterministic_generic_table_answer(question,[context],[0],'normal')
    assert answer and '15M' in answer.answer_markdown and '200M' in answer.answer_markdown
    assert '65M' not in answer.answer_markdown


def test_short_dataset_row_does_not_match_inside_an_unrelated_word():
    assert QueryService._generic_table_row_relevance('读取表1的accuracy成绩','AC','')==0


@pytest.mark.parametrize('marker',['CORRECT','INCORRECT','AMBIGUOUS','CLS','SEP','ISREL'])
def test_citation_cleanup_preserves_literal_scientific_protocol_tokens(marker):
    answer=f'If [{marker}], refine the evidence. [2]'
    normalized=QueryService._renumber_answer_citations(answer,[2])
    finalized=QueryService._drop_unreturned_citation_markers(normalized,1)
    assert finalized==f'If [{marker}], refine the evidence. [0]'


def test_protocol_token_preservation_does_not_keep_unresolved_source_links():
    answer='Use [CLS] but not [[sources/private.md]] or [unresolved-label]. [2]'
    normalized=QueryService._renumber_answer_citations(answer,[2])
    assert '[CLS]' in normalized
    assert 'sources/private' not in normalized
    assert '[unresolved-label]' not in normalized
    assert '[0]' in normalized


def test_deterministic_table_answer_passes_real_repetition_verifier_with_distinct_rows():
    from app.schemas.common import Citation
    from app.services.search import RetrievedContext
    from app.services.answer_verifier import AnswerVerifier
    question='读取表1中 Standard、CoT、Act 和 ReAct 在 HotpotQA 与 Fever 上的成绩。'
    text=('Table 1: Results\n| Method | HotpotQA (EM) | Fever (Acc) |\n| --- | --- | --- |\n'
          '| Standard | 28.7 | 57.1 |\n| CoT | 29.4 | 56.3 |\n| Act | 25.7 | 58.9 |\n'
          '| ReAct | 27.4 | 60.9 |')
    context=RetrievedContext(Citation(document_id='d',chunk_id='c',table_id='t',
        parse_version='v',block_type='table',excerpt=text,score=1),text,1,evidence_kind='table')
    answer=QueryService(None)._deterministic_generic_table_answer(question,[context],[0],'normal')
    assert answer is not None
    for value in ('28.7','57.1','29.4','56.3','25.7','58.9','27.4','60.9'):
        assert value in answer.answer_markdown
    verdict=AnswerVerifier().verify(question=question,answer_markdown=answer.answer_markdown,
        citations=[context.citation.model_dump()],route_type='table_or_metric')
    assert verdict['retry_recommended'] is False, verdict['warnings']


def test_parametric_memory_answer_cannot_gain_absent_forcefield_evidence_terms():
    from app.schemas.common import Citation
    from app.services.search import RetrievedContext
    source='RAG combines a parametric seq2seq model with a non-parametric Wikipedia index.'
    context=RetrievedContext(Citation(document_id='d',chunk_id='c',parse_version='v',
        excerpt=source,score=1),source,1)
    answer=QueryService._append_missing_supported_question_terms(
        'RAG 的参数化记忆与非参数化记忆是什么？','分别为 seq2seq 模型和 Wikipedia 索引。',[context])
    for unsupported in ('RESP','HF/6-31G','M05-2X','MP2/cc-pVQZ','Leu CMAP','Ile','Val CMAP','5 milliseconds'):
        assert unsupported not in answer


def test_parametric_memory_answer_does_not_treat_word_fragments_as_scientific_identifiers():
    from app.schemas.common import Citation
    from app.services.search import RetrievedContext
    source = 'RAG responds while using publication evidence with a parametric generator.'
    context = RetrievedContext(Citation(document_id='d', chunk_id='c', parse_version='v',
        excerpt=source, score=1), source, 1)
    answer = QueryService._append_missing_supported_question_terms(
        'RAG 的参数化记忆与非参数化记忆是什么？', '分别为生成器和外部索引。', [context])
    assert 'RESP' not in answer
    assert 'Ile' not in answer
    assert 'cation' not in answer


def test_real_parameterization_identifiers_are_preserved_when_adjacent_to_chinese():
    from app.schemas.common import Citation
    from app.services.search import RetrievedContext
    source = '参数化使用RESP；Ile通过Val CMAP验证。'
    context = RetrievedContext(Citation(document_id='d', chunk_id='c', parse_version='v',
        excerpt=source, score=1), source, 1)
    answer = QueryService._append_missing_supported_question_terms(
        '该参数化方案是什么？', '按证据中的方案执行。', [context])
    assert 'RESP' in answer and 'Ile' in answer
