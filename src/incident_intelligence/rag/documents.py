"""Carga documentos técnicos desde Markdown, texto, JSON y PDF."""

import json
from pathlib import Path
from typing import Any

from incident_intelligence.rag.schemas import KnowledgeDocument

SUPPORTED_SUFFIXES = {".md", ".txt", ".json", ".pdf"}


def _default_metadata(path: Path) -> tuple[str, list[str]]:
    tags = [part.casefold() for part in path.stem.replace("-", "_").split("_") if part]
    return path.suffix.lstrip(".").casefold() or "document", tags


def _relative_source(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _load_text(path: Path, root: Path) -> list[KnowledgeDocument]:
    category, tags = _default_metadata(path)
    return [
        KnowledgeDocument(
            document_id=path.stem,
            content=path.read_text(encoding="utf-8").strip(),
            source=_relative_source(path, root),
            page=1,
            category=category,
            tags=tags,
        )
    ]


def _load_json(path: Path, root: Path) -> list[KnowledgeDocument]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("documents", []) if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError(f"El JSON debe contener una lista de documentos: {path}")

    documents: list[KnowledgeDocument] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Documento JSON inválido en posición {index}: {path}")
        documents.append(
            KnowledgeDocument(
                document_id=str(
                    record.get("document_id")
                    or record.get("id")
                    or f"{path.stem}-{index}"
                ),
                content=str(record.get("content", "")).strip(),
                source=_relative_source(path, root),
                page=int(record.get("page", 1)),
                category=str(record.get("category", "json")),
                tags=[str(tag) for tag in record.get("tags", [])],
            )
        )
    return documents


def _load_pdf(path: Path, root: Path) -> list[KnowledgeDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Instala la dependencia 'pypdf'") from exc

    category, tags = _default_metadata(path)
    reader = PdfReader(str(path))
    documents: list[KnowledgeDocument] = []
    for page_number, page in enumerate(reader.pages, start=1):
        content = (page.extract_text() or "").strip()
        if content:
            documents.append(
                KnowledgeDocument(
                    document_id=path.stem,
                    content=content,
                    source=_relative_source(path, root),
                    page=page_number,
                    category=category,
                    tags=tags,
                )
            )
    return documents


def load_knowledge_documents(data_dir: Path) -> list[KnowledgeDocument]:
    if not data_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {data_dir}")

    documents: list[KnowledgeDocument] = []
    candidates = (
        item for item in data_dir.rglob("*") if item.suffix.casefold() in SUPPORTED_SUFFIXES
    )
    for path in sorted(candidates):
        suffix = path.suffix.casefold()
        if suffix in {".md", ".txt"}:
            documents.extend(_load_text(path, data_dir))
        elif suffix == ".json":
            documents.extend(_load_json(path, data_dir))
        else:
            documents.extend(_load_pdf(path, data_dir))
    return documents
