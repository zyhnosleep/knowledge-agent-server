"""Real boundary regressions from the private server acceptance/review."""
import pytest

from app.schemas.common import Citation
from app.services.search import QueryService, RetrievedContext


def react_table():
    text = 'Table 1: ReAct results\n| Method | HotpotQA (EM) | Fever (Acc) |\n|---|---|---|\n| Act | 25.7 | 58.9 |\n| ReAct | 27.4 | 60.9 |'
    return RetrievedContext(Citation(document_id='react', parse_version='v5', chunk_id='row',
        block_type='table', table_id='t1', excerpt=text, score=40), text, 40, evidence_kind='table')


@pytest.mark.parametrize('question', [
    'Read ReAct Table 99 ReAct-XL scores. If absent, do not substitute any other table.',
    '先找到 ReAct 表99，再用表99里的 ReAct-XL 在隐藏测试集上的精确分数计算相对表1 ReAct 的提升。没有该表不能用表1冒充。',
])
def test_missing_explicit_table_never_returns_another_table(question):
    service = QueryService(None)
    assert service._deterministic_table_answer_if_supported(question, [react_table()], 'normal') is None


def test_absent_hyphenated_method_is_not_replaced_by_its_base_row():
    question = 'Read Table 1 ReAct-XL scores; do not substitute ReAct.'
    assert QueryService(None)._deterministic_table_answer_if_supported(question, [react_table()], 'normal') is None


@pytest.mark.parametrize('question', [
    'Calculate the difference between Act and ReAct in Table 1 for HotpotQA and Fever.',
    '读取表1 Act 与 ReAct 的两列成绩，计算提升并比较哪个任务提升更大。',
    '计算 Table 1 PAQ 和 VQA 的规模比例。',
])
def test_arithmetic_is_not_short_circuited_by_raw_value_template(question):
    assert QueryService(None)._deterministic_table_answer_if_supported(question, [react_table()], 'normal') is None


@pytest.mark.parametrize('answer', [
    'HotpotQA: 27.4 - 25.7 = 1.7; Fever: 60.9 - 58.9 = 2.0. [0]',
    'HotpotQA: 27.4−25.7=1.7；Fever: 60.9−58.9=2.0，Fever提升更大。 [0]',
])
def test_verified_source_bound_equations_survive_strict_numeric_validation(answer):
    assert QueryService._unsupported_answer_numbers(answer, [react_table()], [0], strict=True) == set()


@pytest.mark.parametrize('answer,want', [
    ('200M / 15M ≈ 13.33 [0]', set()),
    ('65M / 400K = 162.5 [0]', set()),
    ('65M / 400K = 9.9 [0]', {'9.9'}),
    ('65M / 0 = 162.5 [0]', {'162.5'}),
    ('200M / 15M = 13.33 [0]', {'13.33'}),
])
def test_scaled_ratio_requires_cited_operands_and_correct_exact_or_rounded_result(answer, want):
    text = 'Table 1\n| Dataset | Size |\n|---|---|\n| LAION | 200M |\n| CC | 15M |\n| PAQ | 65M |\n| VQA | 400K |'
    context = RetrievedContext(Citation(document_id='m', parse_version='v5', chunk_id='c', excerpt=text, score=1), text, 1)
    assert QueryService._unsupported_answer_numbers(answer, [context], [0], strict=True) == want


def test_comparison_delta_must_derive_from_already_verified_source_equations():
    answer = '27.4 - 25.7 = 1.7; 60.9 - 58.9 = 2.0; 2.0 - 1.7 = 0.3 [0]'
    assert QueryService._unsupported_answer_numbers(answer, [react_table()], [0], strict=True) == set()


@pytest.mark.parametrize('answer,unsupported', [
    ('HotpotQA: 27.4 - 25.7 = 9.8 [0]', '9.8'),
    ('HotpotQA: 29.9 - 25.7 = 4.2 [0]', '4.2'),
    ('提升 1.7；没有源操作数或算式。 [0]', '1.7'),
])
def test_arithmetic_does_not_admit_fabricated_or_unproven_results(answer, unsupported):
    assert unsupported in QueryService._unsupported_answer_numbers(answer, [react_table()], [0], strict=True)


@pytest.mark.parametrize('answer', [
    'LAION / CC = (200M ÷ 15M) ≈ 13.3 倍。 [0]',
    'PAQ / VQA = 65M ÷ 400K = 65,000,000 ÷ 400,000 = 162.5。 [0]',
    'PAQ / VQA = (65,000,000 / 400,000) = 162.5。 [0]',
])
def test_live_ratio_equation_notation_and_source_equivalent_expansion_are_verified(answer):
    text = 'Table 1\n| Dataset | Size |\n|---|---|\n| LAION | 200M |\n| CC | 15M |\n| PAQ | 65M |\n| VQA | 400K |'
    context = RetrievedContext(Citation(document_id='m',parse_version='v5',chunk_id='c',excerpt=text,score=1),text,1)
    assert QueryService._unsupported_answer_numbers(answer,[context],[0],strict=True) == set()


@pytest.mark.parametrize('answer', [
    '65M / 400K = 65,000,000 / 500,000 = 130 [0]',
    '(200M / 15M) = 13.3 [0]',
    '(65,000,000 / 400,000) = 999.9 [0]',
])
def test_equivalent_notation_does_not_relax_source_or_result_checks(answer):
    text = 'Table 1\n| Dataset | Size |\n|---|---|\n| LAION | 200M |\n| CC | 15M |\n| PAQ | 65M |\n| VQA | 400K |'
    context = RetrievedContext(Citation(document_id='m',parse_version='v5',chunk_id='c',excerpt=text,score=1),text,1)
    assert QueryService._unsupported_answer_numbers(answer,[context],[0],strict=True)


@pytest.mark.parametrize('question', ['检索表示是什么？', '比较两种表示方法。', '表达的含义是什么？', '什么时候发表论文？'])
def test_chinese_narrative_words_do_not_route_to_tables(question):
    assert not QueryService._is_table_query(question)


@pytest.mark.parametrize('question', ['表1是什么？', '表 99 中的分数', '这张表怎么读？', '表格里有哪些方法？', 'Table 1 results', 'tabular data'])
def test_real_table_references_keep_table_routing(question):
    assert QueryService._is_table_query(question)
