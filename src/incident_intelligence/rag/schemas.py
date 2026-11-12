"""Contratos de ingesta, recuperación y evaluación."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int = Field(default=1, ge=1)
    category: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip().casefold() for value in values if value.strip()))


class CloudChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    category: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    chunk_index: int = Field(ge=0)

    def pinecone_document(self, embedding: list[float]) -> dict[str, Any]:
        return {
            "_id": self.id,
            "embedding": embedding,
            "text": self.content,
            "document_id": self.document_id,
            "source": self.source,
            "page": self.page,
            "category": self.category,
            "tags": self.tags,
            "chunk_index": self.chunk_index,
        }


class IngestionSummary(BaseModel):
    documents: int = Field(ge=0)
    chunks: int = Field(ge=0)
    upserted: int = Field(ge=0)
    namespace: str


class RetrievedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    content: str
    source: str
    page: int = Field(ge=1)
    category: str
    tags: list[str] = Field(default_factory=list)
    semantic_score: float | None = None


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    relevant_document_ids: list[str] = Field(min_length=1)


class QueryMetrics(BaseModel):
    question: str
    retrieved_document_ids: list[str]
    relevant_document_ids: list[str]
    hits: int = Field(ge=0)
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)


class EvaluationReport(BaseModel):
    k: int = Field(ge=1)
    queries: int = Field(ge=1)
    mean_precision_at_k: float = Field(ge=0, le=1)
    mean_recall_at_k: float = Field(ge=0, le=1)
    results: list[QueryMetrics]
