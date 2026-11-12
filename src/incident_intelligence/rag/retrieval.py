"""Recuperación híbrida con Pinecone semántico y BM25 local."""

import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from incident_intelligence.rag.pinecone import VECTOR_FIELD
from incident_intelligence.rag.schemas import CloudChunk, RetrievedDocument


def _match_value(match: Any, key: str, default: Any = None) -> Any:
    try:
        value = getattr(match, key)
    except AttributeError:
        value = None
    if value is not None:
        return value
    getter = getattr(match, "get", None)
    if callable(getter):
        value = getter(key, default)
        if value is not None:
            return value
    return default


def _matches_to_documents(matches: list[Any]) -> list[Document]:
    documents: list[Document] = []
    for match in matches:
        chunk_id = str(_match_value(match, "id", _match_value(match, "_id", "")))
        documents.append(
            Document(
                page_content=str(_match_value(match, "text", "")),
                metadata={
                    "chunk_id": chunk_id,
                    "document_id": str(_match_value(match, "document_id", "")),
                    "source": str(_match_value(match, "source", "desconocido")),
                    "page": int(_match_value(match, "page", 1)),
                    "category": str(_match_value(match, "category", "document")),
                    "tags": list(_match_value(match, "tags", [])),
                    "chunk_index": int(_match_value(match, "chunk_index", 0)),
                    "semantic_score": float(_match_value(match, "score", 0.0)),
                },
            )
        )
    return documents


class PineconeSemanticRetriever(BaseRetriever):
    """Adaptador de Pinecone compatible con EnsembleRetriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: Any
    embeddings: Any
    namespace: str
    k: int = Field(default=15, ge=1)

    def _search(self, query_vector: list[float]) -> list[Document]:
        try:
            from pinecone import DenseVectorQuery
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Instala la dependencia 'pinecone'") from exc
        response = self.index.documents.search(
            namespace=self.namespace,
            top_k=self.k,
            score_by=[DenseVectorQuery(field=VECTOR_FIELD, values=query_vector)],
            include_fields=[
                "text",
                "document_id",
                "source",
                "page",
                "category",
                "tags",
                "chunk_index",
            ],
        )
        return _matches_to_documents(list(response.matches))

    def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Document]:
        return self._search(self.embeddings.embed_query(query))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any,
    ) -> list[Document]:
        query_vector = await self.embeddings.aembed_query(query)
        return await asyncio.to_thread(self._search, query_vector)


class RAGSystem:
    """Recuperador Top-5 que encapsula la fusión semántica y léxica."""

    def __init__(self, ensemble_retriever: Any, *, top_k: int = 5) -> None:
        if top_k != 5:
            raise ValueError("top_k debe ser 5")
        self.ensemble_retriever = ensemble_retriever
        self.top_k = top_k

    @classmethod
    def build(
        cls,
        *,
        index: Any,
        embeddings: Any,
        chunks: list[CloudChunk],
        namespace: str,
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.4,
        top_k: int = 5,
    ) -> "RAGSystem":
        if abs((semantic_weight + lexical_weight) - 1.0) > 1e-9:
            raise ValueError("Los pesos del recuperador deben sumar 1")
        if not chunks:
            raise ValueError("Se requiere al menos un fragmento para BM25")
        try:
            from langchain_classic.retrievers.ensemble import EnsembleRetriever
            from langchain_community.retrievers import BM25Retriever
        except ImportError as exc:  # pragma: no cover
            message = "Instala langchain-classic, langchain-community y rank-bm25"
            raise RuntimeError(message) from exc

        candidate_k = max(15, top_k * 3)
        corpus = [
            Document(
                page_content=chunk.content,
                metadata={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "category": chunk.category,
                    "tags": chunk.tags,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk in chunks
        ]
        semantic = PineconeSemanticRetriever(
            index=index,
            embeddings=embeddings,
            namespace=namespace,
            k=candidate_k,
        )
        lexical = BM25Retriever.from_documents(corpus, k=candidate_k)
        ensemble = EnsembleRetriever(
            retrievers=[semantic, lexical],
            weights=[semantic_weight, lexical_weight],
            c=60,
        )
        return cls(ensemble, top_k=top_k)

    async def retrieve(self, query: str) -> list[RetrievedDocument]:
        clean = query.strip()
        if not clean:
            raise ValueError("La consulta no puede estar vacía")
        documents = await self.ensemble_retriever.ainvoke(clean)
        results: list[RetrievedDocument] = []
        seen: set[str] = set()
        for document in documents:
            metadata = document.metadata
            document_id = str(metadata.get("document_id", ""))
            if not document_id or document_id in seen:
                continue
            seen.add(document_id)
            results.append(
                RetrievedDocument(
                    chunk_id=str(metadata.get("chunk_id", "")),
                    document_id=document_id,
                    content=document.page_content,
                    source=str(metadata.get("source", "desconocido")),
                    page=int(metadata.get("page", 1)),
                    category=str(metadata.get("category", "document")),
                    tags=list(metadata.get("tags", [])),
                    semantic_score=metadata.get("semantic_score"),
                )
            )
            if len(results) == self.top_k:
                break
        return results
