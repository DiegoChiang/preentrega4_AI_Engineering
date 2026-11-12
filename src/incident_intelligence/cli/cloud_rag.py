"""Interfaz de línea de comandos para el RAG híbrido en Pinecone."""

import argparse
import asyncio
import json
from typing import Any

from incident_intelligence.config import Settings, get_settings
from incident_intelligence.integrations.embeddings import build_embeddings
from incident_intelligence.rag.chunking import split_documents
from incident_intelligence.rag.documents import load_knowledge_documents
from incident_intelligence.rag.evaluation import evaluate_retrieval, load_golden_set
from incident_intelligence.rag.ingestion import PineconeIngestor
from incident_intelligence.rag.pinecone import ensure_serverless_index
from incident_intelligence.rag.retrieval import RAGSystem


def build_corpus(settings: Settings) -> tuple[list[Any], list[Any]]:
    documents = load_knowledge_documents(settings.data_dir)
    chunks = split_documents(
        documents,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    return documents, chunks


async def open_cloud_resources(settings: Settings) -> tuple[Any, Any]:
    index = await asyncio.to_thread(ensure_serverless_index, settings)
    embeddings = build_embeddings(settings)
    return index, embeddings


async def setup_index(settings: Settings) -> dict[str, object]:
    await asyncio.to_thread(ensure_serverless_index, settings)
    return {
        "index": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
        "dimension": settings.embedding_dimension,
        "metric": "cosine",
    }


async def ingest_documents(settings: Settings) -> dict[str, object]:
    documents, chunks = build_corpus(settings)
    index, embeddings = await open_cloud_resources(settings)
    ingestor = PineconeIngestor(
        index,
        embeddings,
        namespace=settings.pinecone_namespace,
        embedding_dimension=settings.embedding_dimension,
    )
    summary = await ingestor.ingest(chunks, document_count=len(documents))
    return summary.model_dump()


async def build_rag_system(settings: Settings) -> RAGSystem:
    _, chunks = build_corpus(settings)
    index, embeddings = await open_cloud_resources(settings)
    return RAGSystem.build(
        index=index,
        embeddings=embeddings,
        chunks=chunks,
        namespace=settings.pinecone_namespace,
        semantic_weight=settings.semantic_weight,
        lexical_weight=settings.lexical_weight,
        top_k=settings.rag_top_k,
    )


async def query_documents(settings: Settings, question: str) -> list[dict[str, object]]:
    system = await build_rag_system(settings)
    results = await system.retrieve(question)
    return [result.model_dump() for result in results]


async def evaluate_documents(settings: Settings) -> dict[str, object]:
    system = await build_rag_system(settings)
    cases = load_golden_set(settings.golden_set_path)
    report = await evaluate_retrieval(system, cases, k=settings.rag_top_k)
    return report.model_dump()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG híbrido con Pinecone Serverless y BM25")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("setup", help="Crear o validar el índice serverless")
    subcommands.add_parser("ingest", help="Procesar el corpus y cargarlo en Pinecone")
    query = subcommands.add_parser("query", help="Recuperar cinco documentos")
    query.add_argument("question", help="Pregunta técnica")
    subcommands.add_parser("evaluate", help="Calcular Precision@5 y Recall@5")
    return parser


async def _dispatch(args: argparse.Namespace, settings: Settings) -> object:
    if args.command == "setup":
        return await setup_index(settings)
    if args.command == "ingest":
        return await ingest_documents(settings)
    if args.command == "query":
        return await query_documents(settings, args.question)
    return await evaluate_documents(settings)


def main() -> None:
    args = _parser().parse_args()
    result = asyncio.run(_dispatch(args, get_settings()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
