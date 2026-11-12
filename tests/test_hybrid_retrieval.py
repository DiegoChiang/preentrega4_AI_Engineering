import pytest
from langchain_core.documents import Document
from pinecone import Document as PineconeDocument

from incident_intelligence.rag.retrieval import RAGSystem, _matches_to_documents


class FakeEnsemble:
    async def ainvoke(self, query):
        assert query == "diagnóstico"
        return [
            Document(page_content="A1", metadata={"document_id": "a", "chunk_id": "a-1"}),
            Document(page_content="A2", metadata={"document_id": "a", "chunk_id": "a-2"}),
            Document(page_content="B", metadata={"document_id": "b", "chunk_id": "b-1"}),
            Document(page_content="C", metadata={"document_id": "c", "chunk_id": "c-1"}),
            Document(page_content="D", metadata={"document_id": "d", "chunk_id": "d-1"}),
            Document(page_content="E", metadata={"document_id": "e", "chunk_id": "e-1"}),
            Document(page_content="F", metadata={"document_id": "f", "chunk_id": "f-1"}),
        ]


@pytest.mark.asyncio
async def test_rag_system_returns_five_unique_documents():
    system = RAGSystem(FakeEnsemble())

    results = await system.retrieve("diagnóstico")

    assert [item.document_id for item in results] == ["a", "b", "c", "d", "e"]
    assert len(results) == 5


@pytest.mark.asyncio
async def test_rag_system_rejects_empty_query():
    system = RAGSystem(FakeEnsemble())

    with pytest.raises(ValueError, match="vacía"):
        await system.retrieve("  ")


def test_pinecone_document_keeps_identifier_score_and_metadata():
    match = PineconeDocument(
        {
            "_id": "chunk-1",
            "_score": 0.87,
            "text": "Diagnóstico",
            "document_id": "runbook",
            "source": "runbook.md",
            "page": 2,
            "category": "operaciones",
            "tags": ["api"],
            "chunk_index": 0,
        }
    )

    document = _matches_to_documents([match])[0]

    assert document.metadata["chunk_id"] == "chunk-1"
    assert document.metadata["semantic_score"] == pytest.approx(0.87)
    assert document.metadata["source"] == "runbook.md"
