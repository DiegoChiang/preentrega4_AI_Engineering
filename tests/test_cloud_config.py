import pytest
from pydantic import ValidationError

from incident_intelligence.config import Settings


def test_cloud_defaults_match_embedding_and_hybrid_retrieval():
    settings = Settings(_env_file=None)

    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimension == 1536
    assert settings.rag_top_k == 5
    assert settings.rag_chunk_size == 600
    assert settings.semantic_weight + settings.lexical_weight == 1


def test_hybrid_weights_must_sum_one():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, semantic_weight=0.7, lexical_weight=0.4)


def test_missing_cloud_keys_are_reported_without_exposing_secrets():
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.require_openai_key()
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        settings.require_pinecone_key()
