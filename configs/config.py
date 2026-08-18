from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    provider: str
    params: dict[str, Any] = Field(default_factory=dict)


class VectorStoreConfig(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)

class RetrievalDenseConfig(BaseModel):
    source: dict[str, Any] = Field(default_factory=dict)


class RetrievalLexicalConfig(BaseModel):
    type: str
    params: dict[str, Any] = Field(default_factory=dict)

class RerankerConfig(BaseModel):
    type: str = "identity"

    top_k: int = 5

    params: dict[str, Any] = Field(
        default_factory=dict
    )

class RetrievalConfig(BaseModel):
    type: str
    candidate_k: int = 20

    params: dict[str, Any] = Field(
        default_factory=dict
    )

    dense: RetrievalDenseConfig | None = None
    lexical: RetrievalLexicalConfig | None = None

class ContextConfig(BaseModel):
    type: str = "simple"

    params: dict[str, Any] = Field(
        default_factory=dict
    )


class GenerationConfig(BaseModel):
    provider: str = "ollama"

    params: dict[str, Any] = Field(
        default_factory=dict
    )

class Settings(BaseModel):
    input_dir: Path
    output_dir: Path

    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig

    retrieval: RetrievalConfig
    reranker: RerankerConfig

    context: ContextConfig
    generation: GenerationConfig

    def resolve_paths(self) -> None:
        if not self.input_dir.is_absolute():
            self.input_dir = Path.cwd() / self.input_dir

        if not self.output_dir.is_absolute():
            self.output_dir = Path.cwd() / self.output_dir


def get_settings(config_path: str | Path = "configs/config.yaml") -> Settings:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a YAML mapping")

    settings = Settings.model_validate(data)
    settings.resolve_paths()

    return settings
