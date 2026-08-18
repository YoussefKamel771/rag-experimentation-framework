from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from src.ingestion.models import Chunk

@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    retriever: str
    metadata: dict[str, Any] = field(default_factory=dict)

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError
