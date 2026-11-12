"""Configuración central cargada desde variables de entorno."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros de ingesta y recuperación híbrida en Pinecone."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=1536, ge=1)

    pinecone_index_name: str = Field(default="incident-knowledge-cloud", min_length=1)
    pinecone_namespace: str = Field(default="technical-docs", min_length=1)
    pinecone_cloud: str = Field(default="aws", min_length=1)
    pinecone_region: str = Field(default="us-east-1", min_length=1)

    data_dir: Path = Path("data")
    golden_set_path: Path = Path("evaluation/golden_set.json")
    rag_top_k: int = Field(default=5, ge=5, le=5)
    rag_chunk_size: int = Field(default=600, ge=500, le=800)
    rag_chunk_overlap: int = Field(default=100, ge=0, le=200)
    semantic_weight: float = Field(default=0.6, ge=0, le=1)
    lexical_weight: float = Field(default=0.4, ge=0, le=1)

    @field_validator("rag_chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {})
        chunk_size = data.get("rag_chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("rag_chunk_overlap debe ser menor que rag_chunk_size")
        return value

    @field_validator("lexical_weight")
    @classmethod
    def validate_weights(cls, value: float, info: object) -> float:
        semantic = getattr(info, "data", {}).get("semantic_weight")
        if semantic is not None and abs((semantic + value) - 1.0) > 1e-9:
            raise ValueError("semantic_weight y lexical_weight deben sumar 1")
        return value

    def require_openai_key(self) -> str:
        secret = self.openai_api_key
        if secret is None or not secret.get_secret_value().strip():
            raise ValueError("Falta la variable de entorno OPENAI_API_KEY")
        return secret.get_secret_value()

    def require_pinecone_key(self) -> str:
        secret = self.pinecone_api_key
        if secret is None or not secret.get_secret_value().strip():
            raise ValueError("Falta la variable de entorno PINECONE_API_KEY")
        return secret.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

