"""Pipeline asíncrono de ingesta hacia Pinecone."""

import asyncio
from typing import Any, Protocol

from incident_intelligence.rag.schemas import CloudChunk, IngestionSummary


class AsyncEmbeddings(Protocol):
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...


class PineconeIngestor:
    def __init__(
        self,
        index: Any,
        embeddings: AsyncEmbeddings,
        *,
        namespace: str,
        embedding_dimension: int,
        batch_size: int = 100,
    ) -> None:
        if not namespace.strip():
            raise ValueError("namespace no puede estar vacío")
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        self.index = index
        self.embeddings = embeddings
        self.namespace = namespace
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size

    async def ingest(
        self,
        chunks: list[CloudChunk],
        *,
        document_count: int,
    ) -> IngestionSummary:
        if not chunks:
            return IngestionSummary(
                documents=document_count,
                chunks=0,
                upserted=0,
                namespace=self.namespace,
            )

        vectors = await self.embeddings.aembed_documents([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("La cantidad de embeddings no coincide con los fragmentos")
        if any(len(vector) != self.embedding_dimension for vector in vectors):
            raise ValueError("La dimensión de embeddings no coincide con el índice")

        upserted = 0
        documents = [
            chunk.pinecone_document(embedding)
            for chunk, embedding in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            response = await asyncio.to_thread(
                self.index.documents.upsert,
                namespace=self.namespace,
                documents=batch,
            )
            upserted += int(getattr(response, "upserted_count", len(batch)))

        return IngestionSummary(
            documents=document_count,
            chunks=len(chunks),
            upserted=upserted,
            namespace=self.namespace,
        )
