"""Regressions for source-balanced fixed retrieval and explicit visual targets."""
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.session import Base
from app.schemas.common import Citation
from app.services.search import QueryService, RetrievedContext


def context(doc, title, text, score=1, kind=None):
    return RetrievedContext(Citation(document_id=doc, page_title=title,
        excerpt=text, score=score, block_type=kind), text, score, evidence_kind=kind)


def test_comparison_context_window_keeps_both_named_sources():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        service = QueryService(db)
        contexts = [context('a', 'Alpha Paper', f'Alpha evidence {i}', 100-i)
                    for i in range(12)]
        contexts.append(context('b', 'Beta Paper', 'Beta independent evidence', 1))
        result = service._finalize_contexts(contexts, question='Compare Alpha and Beta methods.')
        assert {c.citation.document_id for c in result} == {'a', 'b'}
        assert len(result) <= 8


def test_prompt_separates_source_identity_from_evidence():
    c = context('private-id', 'Beta Paper', 'We use token-level interaction.')
    text = QueryService._prompt_context_text('Compare Alpha and Beta.', c)
    assert 'Beta Paper' in text
    assert 'We use token-level interaction.' in text
    assert 'private-id' not in text


def test_explicit_figure_selects_only_its_named_paper():
    contexts = [context('a', 'Alpha Paper', 'Figure 1: workflow', kind='figure'),
                context('b', 'Beta Paper', 'Figure 1: accuracy', kind='figure'),
                context('b', 'Beta Paper', 'Figure 2: examples', kind='figure')]
    selected = QueryService._requested_figure_indexes('Explain Beta 图1.', contexts)
    assert selected == [1]


def test_explicit_figure_number_does_not_match_a_longer_number():
    contexts = [context('b', 'Beta Paper', 'Figure 10: examples', kind='figure')]
    assert QueryService._requested_figure_indexes('Explain Beta 图1.', contexts) == []


def test_no_figure_number_keeps_candidates_for_visual_disambiguation():
    contexts = [context('a', 'Alpha Paper', 'Figure 1: workflow', kind='figure'),
                context('b', 'Beta Paper', 'Figure 2: accuracy', kind='figure')]
    assert QueryService._requested_figure_indexes('Compare these diagrams.', contexts) == [0, 1]


def test_figure_selector_uses_child_caption_not_parent_heading():
    c = context('b', 'Beta Paper', 'Figure 1: accuracy', kind='figure')
    c.prompt_text = 'Results section\nFigure 1: accuracy'
    assert QueryService._requested_figure_indexes('Explain Beta Figure 1.', [c]) == [0]


def test_figure_ownership_supports_possessive_and_postposed_source():
    contexts = [context('a', 'Alpha Paper', 'Figure 1: workflow', kind='figure'),
                context('b', 'Beta Paper', 'Figure 1: accuracy', kind='figure')]
    for question in ("Explain Beta's Figure 1.", 'Explain Figure 1 in Beta.'):
        assert QueryService._requested_figure_indexes(question, contexts) == [1]


def test_token_budget_does_not_starve_small_comparison_source():
    service = QueryService(None)
    service._retrieval_token_counter = lambda text: len(text.split())
    service.DRAFT_CONTEXT_TOKEN_BUDGET = 50
    contexts = [context('a', 'Alpha Paper', ' '.join(['alpha']*40), 10),
                context('b', 'Beta Paper', ' '.join(['beta']*9), 1)]
    result = service._fit_contexts_to_token_budget(contexts, question='Compare Alpha and Beta.')
    assert any(c.citation.document_id == 'b' for c in result)
    rendered = '\n\n'.join(f'[{i}] {service._prompt_context_text("Compare Alpha and Beta.", c)}'
                            for i, c in enumerate(result))
    assert len(rendered.split()) <= 50


def test_comparison_reservation_uses_trusted_matched_source_not_title_words():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        service = QueryService(db)
        contexts = [context('a', 'Token Interaction Architecture', f'Alpha evidence {i}', 100-i)
                    for i in range(12)]
        contexts.append(context('b', 'Span Conditioning Framework', 'Beta mechanism', 1))
        for c in contexts:
            c.comparison_source = True
        result = service._finalize_contexts(contexts, question='Compare Alpha and Beta.')
        assert {c.citation.document_id for c in result} == {'a', 'b'}
