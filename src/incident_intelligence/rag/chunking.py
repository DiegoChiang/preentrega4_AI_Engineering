"""Fragmentación semántica con identificadores reproducibles."""

import hashlib
import re

from incident_intelligence.rag.schemas import CloudChunk, KnowledgeDocument


def count_lexical_tokens(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def split_documents(
    documents: list[KnowledgeDocument],
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[CloudChunk]:
    if not 500 <= chunk_size <= 800:
        raise ValueError("chunk_size debe estar entre 500 y 800")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe estar entre 0 y chunk_size")

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Instala la dependencia 'langchain-text-splitters'") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_lexical_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[CloudChunk] = []
    for document in documents:
        for index, content in enumerate(splitter.split_text(document.content)):
            digest = hashlib.sha256(
                f"{document.document_id}:{document.page}:{index}:{content}".encode()
            ).hexdigest()
            chunks.append(
                CloudChunk(
                    id=digest,
                    document_id=document.document_id,
                    content=content,
                    source=document.source,
                    page=document.page,
                    category=document.category,
                    tags=document.tags,
                    chunk_index=index,
                )
            )
    return chunks
