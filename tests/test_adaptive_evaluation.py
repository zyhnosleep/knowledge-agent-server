import pytest

from scripts.evaluate_adaptive_agent import Case, _aggregate, collect_arm


def test_ungraded_structure_pass_is_not_accuracy():
    assert _aggregate([])["fact_accuracy"] is None
    result = _aggregate([{"verification_status": "passed"}, {"fact_correct": True}, {"fact_correct": False}])
    assert result["fact_accuracy"] == 0.5
    assert result["fact_judged_count"] == 2
    assert result["ungraded_count"] == 1
    assert result["citation_accuracy"] is None


@pytest.mark.parametrize("bad", [1, "true", [], {}])
def test_invalid_judgment_types_cannot_be_truthy_accuracy(bad):
    with pytest.raises(ValueError, match="judgment_type_invalid"):
        _aggregate([{"fact_correct": bad}])


def test_errors_are_reported_not_silently_removed():
    result = _aggregate([{"error": "request_failed"}, {"fact_correct": True, "citation_correct": False}])
    assert result["run_count"] == 2 and result["error_count"] == 1
    assert result["fact_accuracy"] == 1.0 and result["fact_judged_count"] == 1
    assert result["citation_accuracy"] == 0.0


class Response:
    def __init__(self, mode):
        self.mode = mode
    def raise_for_status(self):
        pass
    def json(self):
        return {"status": "completed", "answer_model": "qwen3-vl:4b", "final_answer": "answer",
            "metadata": {"execution_mode": self.mode}, "citations": [], "steps": [], "usage": {}}


class Client:
    def __init__(self, mode):
        self.mode, self.payloads = mode, []
    def post(self, url, json):
        self.payloads.append(json)
        return Response(self.mode)


def test_arms_have_identical_explicit_constraints_and_no_public_toggle():
    case = Case(id="one", question="先核对图再比较表", document_scope="d", reference_facts=["fact"],
        required_evidence=[], unanswerable=False)
    static, adaptive = Client("static"), Client("adaptive")
    a = collect_arm([case], client=static, base_url="http://127.0.0.1:18002", project="p", mode="static", context={})
    b = collect_arm([case], client=adaptive, base_url="http://127.0.0.1:18002", project="p", mode="adaptive", context={})
    assert static.payloads[0]["constraints"] == adaptive.payloads[0]["constraints"] == {
        "allow_external_network": False, "max_steps": 12, "max_tool_calls": 6,
        "budget_tokens": 60000, "timeout_seconds": 120}
    assert "adaptive_enabled" not in static.payloads[0]
    assert a[0]["fact_correct"] is None and b[0]["fact_correct"] is None


def test_requested_mode_cannot_mislabel_actual_arm():
    case = Case(id="one", question="q", document_scope=None, reference_facts=[], required_evidence=[], unanswerable=True)
    rows = collect_arm([case], client=Client("static"), base_url="http://127.0.0.1:18002",
        project="p", mode="adaptive", context={})
    assert rows[0]["error"] == "execution_mode_mismatch"


def test_changed_index_cannot_be_compared_as_same_experiment():
    from scripts.evaluate_adaptive_agent import compare_arms
    a = [{"id": "one", "case": {}, "constraints": {}, "context": {"index_sha256": "old"}}]
    b = [{"id": "one", "case": {}, "constraints": {}, "context": {"index_sha256": "new"}}]
    with pytest.raises(ValueError, match="paired_context_mismatch"):
        compare_arms(a, b)


def test_failed_arm_retains_actual_response_not_a_success_score():
    case = Case(id="one", question="q", document_scope=None, reference_facts=[], required_evidence=[], unanswerable=True)
    rows = collect_arm([case], client=Client("static"), base_url="http://127.0.0.1:18002",
        project="p", mode="adaptive", context={})
    assert rows[0]["error"] == "execution_mode_mismatch"
    assert rows[0]["response"]["metadata"]["execution_mode"] == "static"
    assert rows[0]["fact_correct"] is None


@pytest.mark.parametrize('label',['rag-direct','local-fallback','visual-evidence','no-generation'])
def test_non_model_answer_labels_require_real_base_inference_target(label):
    class TargetResponse(Response):
        def json(self):
            result=super().json()
            result['answer_model']=label
            result['steps']=[{'step_type':'route','metadata':{
                'inference_model':'qwen3-vl:4b','inference_profile':'generation'}}]
            return result
    class TargetClient(Client):
        def post(self,url,json):
            return TargetResponse(self.mode)
    case=Case(id='one',question='q',document_scope=None,reference_facts=[],required_evidence=[],unanswerable=False)
    rows=collect_arm([case],client=TargetClient('static'),base_url='http://127.0.0.1:18002',
        project='p',mode='static',context={'model':'qwen3-vl:4b'})
    assert rows[0]['error'] is None
    assert rows[0]['fact_correct'] is None


@pytest.mark.parametrize('label,target',[
    ('rag-direct',None),('local-fallback','foreign-model'),('foreign-model','qwen3-vl:4b'),
    ('qwen3-vl:4b','foreign-model'), ('visual-evidence',None),
    ('visual-evidence','foreign-model'), ('no-generation',None), ('no-generation','foreign-model')])
def test_answer_labels_cannot_hide_missing_or_foreign_model_targets(label,target):
    class TargetResponse(Response):
        def json(self):
            result=super().json()
            result['answer_model']=label
            result['steps']=([{'metadata':{'inference_model':target}}] if target else [])
            return result
    class TargetClient(Client):
        def post(self,url,json):
            return TargetResponse(self.mode)
    case=Case(id='one',question='q',document_scope=None,reference_facts=[],required_evidence=[],unanswerable=False)
    rows=collect_arm([case],client=TargetClient('static'),base_url='http://127.0.0.1:18002',
        project='p',mode='static',context={'model':'qwen3-vl:4b'})
    assert rows[0]['error']=='generation_model_mismatch'
