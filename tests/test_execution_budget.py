"""Budget contracts use real accounting, a fake clock, and no network."""
import threading
import pytest
from app.schemas.agent import AgentConstraints


def budget_api():
    from app.services.execution_budget import ExecutionBudget, BudgetExceeded, execution_budget_scope, current_execution_budget
    return ExecutionBudget, BudgetExceeded, execution_budget_scope, current_execution_budget


def test_answer_and_empty_rescue_share_quota():
    Budget, _, _, _ = budget_api()
    budget = Budget(AgentConstraints(), max_answer_retries=1)
    assert budget.consume_answer_retry() is True
    assert budget.consume_answer_retry(empty_answer=True) is False
    zero = Budget(AgentConstraints())
    assert zero.consume_answer_retry() is False
    assert zero.consume_answer_retry(empty_answer=True) is True
    assert zero.consume_answer_retry(empty_answer=True) is False


def test_format_retry_is_request_wide():
    Budget, _, _, _ = budget_api()
    budget = Budget(AgentConstraints())
    assert budget.consume_format_retry() is True
    assert budget.consume_format_retry() is False


def test_token_admission_counts_whole_prompt():
    Budget, Exceeded, _, _ = budget_api()
    budget = Budget(AgentConstraints(budget_tokens=1))
    with pytest.raises(Exceeded, match='token'):
        budget.reserve_model_call(messages=[{'role':'system','content':'Use supplied evidence only.'},
            {'role':'user','content':'Evidence: Purple legend. Question: color?'}],
            model='text', max_output_tokens=128, context_length=4096)
    assert budget.usage_metadata()['model_requests'] == 0


def test_provider_usage_accumulates_without_double_settlement():
    Budget, _, _, _ = budget_api()
    budget = Budget(AgentConstraints())
    for prompt, completion in [(11,5),(13,7)]:
        reservation = budget.reserve_model_call(messages=[{'role':'user','content':'hello'}],
            model='text', max_output_tokens=128, context_length=4096)
        budget.finish_model_call(reservation, prompt_tokens=prompt, completion_tokens=completion)
        with pytest.raises(ValueError):
            budget.finish_model_call(reservation, prompt_tokens=prompt, completion_tokens=completion)
    usage = budget.usage_metadata()
    assert usage['prompt_tokens'] == 24
    assert usage['completion_tokens'] == 12
    assert usage['model_requests'] == 2
    assert usage['usage_source'] == 'provider'


def test_unknown_image_usage_reserves_context_without_refund():
    Budget, Exceeded, _, _ = budget_api()
    budget = Budget(AgentConstraints(budget_tokens=1000))
    reservation = budget.reserve_model_call(messages=[{'role':'user','content':'color?', 'images':['pixels']}],
        model='unknown-vision', max_output_tokens=100, context_length=900)
    budget.finish_model_call(reservation, prompt_tokens=None, completion_tokens=None)
    assert budget.usage_metadata()['usage_source'] == 'estimated'
    with pytest.raises(Exceeded):
        budget.reserve_model_call(messages=[{'role':'user','content':'color?', 'images':['pixels']}],
            model='unknown-vision', max_output_tokens=100, context_length=900)


def test_deadline_and_cancel_stop_admission():
    Budget, Exceeded, _, _ = budget_api()
    now = [10.0]
    budget = Budget(AgentConstraints(timeout_seconds=1), clock=lambda: now[0])
    assert budget.http_timeout(90) == 1
    now[0] = 11.0
    with pytest.raises(Exceeded, match='deadline'):
        budget.check_deadline()
    event = threading.Event()
    cancelled = Budget(AgentConstraints(), cancel_event=event)
    event.set()
    with pytest.raises(Exceeded, match='cancel'):
        cancelled.check_deadline()


def test_scope_resets_after_exception_and_is_thread_local():
    Budget, _, scope, current = budget_api()
    budget = Budget(AgentConstraints())
    with pytest.raises(RuntimeError):
        with scope(budget):
            assert current() is budget
            observed = []
            thread = threading.Thread(target=lambda: observed.append(current()))
            thread.start()
            thread.join()
            assert observed == [None]
            raise RuntimeError('test')
    assert current() is None


def test_invalid_provider_usage_is_not_credited():
    Budget, _, _, _ = budget_api()
    budget = Budget(AgentConstraints())
    reservation = budget.reserve_model_call(messages=[{'role':'user','content':'hello'}],
        model='text', max_output_tokens=128, context_length=4096)
    budget.finish_model_call(reservation, prompt_tokens=True, completion_tokens=-2)
    assert budget.usage_metadata()['usage_source'] == 'estimated'


def test_provider_overrun_blocks_further_generation():
    Budget, Exceeded, _, _ = budget_api()
    budget = Budget(AgentConstraints(budget_tokens=1000))
    reservation = budget.reserve_model_call(messages=[{'role':'user','content':'hi'}],
        model='text', max_output_tokens=20, context_length=512)
    budget.finish_model_call(reservation, prompt_tokens=900, completion_tokens=101)
    with pytest.raises(Exceeded, match='token'):
        budget.reserve_model_call(messages=[{'role':'user','content':'hi'}],
            model='text', max_output_tokens=20, context_length=512)
    assert budget.usage_metadata()['model_requests'] == 1
