# app/settings.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(ENV_PATH)

_ALLOWED_STRATEGIES = {"recursive", "semantic", "overlap"} 

class Settings(BaseSettings):
    # Requeridas
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    QDRANT_URL: str = Field(..., env="QDRANT_URL")
    QDRANT_API_KEY: str = Field(..., env="QDRANT_API_KEY")

    # Opcionales
    QDRANT_COLLECTION: str = "document_embeddings"
    TOP_K: int = 5

    EMBED_MODEL: str = "text-embedding-3-small"  # 1536 dim
    EMBED_DIM: int = 1536
    GEN_MODEL: str = "gpt-4.1-mini"

    MIN_SCORE: float = 0.35
    MAX_CONTEXT_CHARS: int = 3500

    # (solo para trazabilidad del indexado; la CLI de index pasa los reales)
    CHUNK_STRATEGY: str = "recursive"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("CHUNK_STRATEGY")
    @classmethod
    def _validate_strategy(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in _ALLOWED_STRATEGIES:
            raise ValueError(f"CHUNK_STRATEGY inválida: '{v}'. Valores permitidos: {sorted(_ALLOWED_STRATEGIES)}")
        return v

    @field_validator("CHUNK_SIZE", "CHUNK_OVERLAP")
    @classmethod
    def _validate_positive(cls, v: int, info):
        if v < 0:
            raise ValueError(f"{info.field_name} debe ser >= 0")
        return v

    @field_validator("EMBED_DIM")
    @classmethod
    def _warn_embed_dim(cls, v: int):
        if v <= 0:
            raise ValueError("EMBED_DIM debe ser > 0")
        return v

settings = Settings()
