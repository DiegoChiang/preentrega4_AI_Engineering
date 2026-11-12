"""Construcción del modelo de embeddings compartido por ingesta y consulta."""

from typing import Any

from incident_intelligence.certificates import configure_system_certificate_store
from incident_intelligence.config import Settings


def build_embeddings(settings: Settings) -> Any:
    configure_system_certificate_store()
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Instala la dependencia 'langchain-openai'") from exc
    return OpenAIEmbeddings(
        api_key=settings.require_openai_key(),
        model=settings.openai_embedding_model,
    )
