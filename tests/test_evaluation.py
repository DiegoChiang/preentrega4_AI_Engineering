import pytest

from incident_intelligence.rag.evaluation import evaluate_retrieval, load_golden_set
from incident_intelligence.rag.schemas import GoldenCase, RetrievedDocument


def _result(document_id):
    return RetrievedDocument(
        chunk_id=f"chunk-{document_id}",
        document_id=document_id,
        content="contenido",
        source="source.md",
        page=1,
        category="runbook",
    )


class FakeSystem:
    async def retrieve(self, query):
        return [_result(query), _result("distractor-1"), _result("distractor-2")]


@pytest.mark.asyncio
async def test_evaluation_computes_precision_and_recall_at_five():
    cases = [
        GoldenCase(question=f"doc-{index}", relevant_document_ids=[f"doc-{index}"])
        for index in range(5)
    ]

    report = await evaluate_retrieval(FakeSystem(), cases)

    assert report.k == 5
    assert report.queries == 5
    assert report.mean_precision_at_k == pytest.approx(0.2)
    assert report.mean_recall_at_k == pytest.approx(1.0)
    assert all(item.hits == 1 for item in report.results)


def test_repository_golden_set_contains_five_cases():
    cases = load_golden_set(__import__("pathlib").Path("evaluation/golden_set.json"))

    assert len(cases) == 5
    assert len({case.question for case in cases}) == 5
