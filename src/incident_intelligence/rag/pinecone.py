"""Configuración del índice Pinecone Serverless."""

from typing import Any

from incident_intelligence.config import Settings

VECTOR_FIELD = "embedding"


def build_pinecone_client(settings: Settings) -> Any:
    try:
        from pinecone import Pinecone
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Instala la dependencia 'pinecone'") from exc
    return Pinecone(api_key=settings.require_pinecone_key())


def _index_names(indexes: Any) -> set[str]:
    names: set[str] = set()
    for item in indexes:
        name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        if name:
            names.add(str(name))
    return names


def ensure_serverless_index(settings: Settings, client: Any | None = None) -> Any:
    """Crea el índice administrado y valida su dimensión antes de usarlo."""

    active_client = client or build_pinecone_client(settings)
    indexes = active_client.indexes
    if settings.pinecone_index_name not in _index_names(indexes.list()):
        indexes.create(
            name=settings.pinecone_index_name,
            schema={
                "fields": {
                    VECTOR_FIELD: {
                        "type": "dense_vector",
                        "dimension": settings.embedding_dimension,
                        "metric": "cosine",
                    }
                }
            },
            deployment={
                "deployment_type": "managed",
                "cloud": settings.pinecone_cloud,
                "region": settings.pinecone_region,
            },
            tags={"project": "ai-engineering", "component": "hybrid-rag"},
        )

    description = indexes.describe(settings.pinecone_index_name)
    field = description.schema.fields[VECTOR_FIELD]
    dimension = int(field.dimension)
    metric = str(field.metric)
    if dimension != settings.embedding_dimension:
        raise ValueError(
            f"El índice usa dimensión {dimension}; se esperaba {settings.embedding_dimension}"
        )
    if metric != "cosine":
        raise ValueError(f"El índice usa métrica {metric}; se esperaba cosine")
    return active_client.index(settings.pinecone_index_name)
