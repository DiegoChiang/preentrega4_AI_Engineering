from types import SimpleNamespace

import pytest

from incident_intelligence.config import Settings
from incident_intelligence.rag.ingestion import PineconeIngestor
from incident_intelligence.rag.pinecone import ensure_serverless_index
from incident_intelligence.rag.schemas import CloudChunk


class FakeIndexes:
    def __init__(self, *, dimension=1536):
        self.dimension = dimension
        self.names = []
        self.create_calls = []

    def list(self):
        return [SimpleNamespace(name=name) for name in self.names]

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        self.names.append(kwargs["name"])

    def describe(self, name):
        assert name in self.names
        field = SimpleNamespace(dimension=self.dimension, metric="cosine")
        return SimpleNamespace(schema=SimpleNamespace(fields={"embedding": field}))


class FakeDocuments:
    def __init__(self):
        self.calls = []

    def upsert(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(upserted_count=len(kwargs["documents"]))


class FakeIndex:
    def __init__(self):
        self.documents = FakeDocuments()


class FakeClient:
    def __init__(self, *, dimension=1536):
        self.indexes = FakeIndexes(dimension=dimension)
        self.value = FakeIndex()

    def index(self, name):
        assert name == "incident-knowledge-cloud"
        return self.value


class FakeEmbeddings:
    async def aembed_documents(self, texts):
        return [[float(index)] * 1536 for index, _ in enumerate(texts)]


def test_ensure_index_creates_serverless_dense_vector_schema():
    settings = Settings(_env_file=None)
    client = FakeClient()

    index = ensure_serverless_index(settings, client)

    assert index is client.value
    call = client.indexes.create_calls[0]
    vector = call["schema"]["fields"]["embedding"]
    assert vector == {"type": "dense_vector", "dimension": 1536, "metric": "cosine"}
    assert call["deployment"] == {
        "deployment_type": "managed",
        "cloud": "aws",
        "region": "us-east-1",
    }


def test_ensure_index_rejects_dimension_mismatch():
    settings = Settings(_env_file=None)
    client = FakeClient(dimension=768)
    client.indexes.names.append(settings.pinecone_index_name)

    with pytest.raises(ValueError, match="dimensión 768"):
        ensure_serverless_index(settings, client)


@pytest.mark.asyncio
async def test_ingestion_preserves_text_and_advanced_metadata():
    index = FakeIndex()
    chunk = CloudChunk(
        id="chunk-1",
        document_id="runbook",
        content="Diagnóstico técnico",
        source="runbook.md",
        page=2,
        category="operaciones",
        tags=["pagos", "api"],
        chunk_index=0,
    )
    ingestor = PineconeIngestor(
        index,
        FakeEmbeddings(),
        namespace="technical-docs",
        embedding_dimension=1536,
    )

    summary = await ingestor.ingest([chunk], document_count=1)

    assert summary.upserted == 1
    call = index.documents.calls[0]
    assert call["namespace"] == "technical-docs"
    stored = call["documents"][0]
    assert stored["text"] == "Diagnóstico técnico"
    assert stored["source"] == "runbook.md"
    assert stored["page"] == 2
    assert stored["category"] == "operaciones"
    assert stored["tags"] == ["pagos", "api"]
    assert len(stored["embedding"]) == 1536
