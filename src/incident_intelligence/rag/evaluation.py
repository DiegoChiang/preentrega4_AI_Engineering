"""Métricas Precision@k y Recall@k para recuperación."""

import json
from pathlib import Path
from typing import Protocol

from incident_intelligence.rag.schemas import (
    EvaluationReport,
    GoldenCase,
    QueryMetrics,
    RetrievedDocument,
)


class RetrievalSystem(Protocol):
    async def retrieve(self, query: str) -> list[RetrievedDocument]: ...


def load_golden_set(path: Path) -> list[GoldenCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("El Golden Set debe ser una lista JSON")
    cases = [GoldenCase.model_validate(item) for item in raw]
    if len(cases) < 5:
        raise ValueError("El Golden Set debe contener al menos cinco preguntas")
    return cases


async def evaluate_retrieval(
    system: RetrievalSystem,
    cases: list[GoldenCase],
    *,
    k: int = 5,
) -> EvaluationReport:
    if k != 5:
        raise ValueError("La evaluación requerida usa k=5")
    if not cases:
        raise ValueError("Se requiere al menos un caso de evaluación")

    results: list[QueryMetrics] = []
    for case in cases:
        retrieved = (await system.retrieve(case.question))[:k]
        retrieved_ids = [document.document_id for document in retrieved]
        relevant = set(case.relevant_document_ids)
        hits = len(set(retrieved_ids) & relevant)
        results.append(
            QueryMetrics(
                question=case.question,
                retrieved_document_ids=retrieved_ids,
                relevant_document_ids=case.relevant_document_ids,
                hits=hits,
                precision_at_k=hits / k,
                recall_at_k=hits / len(relevant),
            )
        )

    return EvaluationReport(
        k=k,
        queries=len(results),
        mean_precision_at_k=sum(item.precision_at_k for item in results) / len(results),
        mean_recall_at_k=sum(item.recall_at_k for item in results) / len(results),
        results=results,
    )
