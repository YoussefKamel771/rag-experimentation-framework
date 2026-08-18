from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.retrieval.base import RetrievalResult


@dataclass
class ContextItem:
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltContext:
    text: str
    items: list[ContextItem]
    total_characters: int
    total_chunks: int


class ContextBuilder(ABC):

    @abstractmethod
    def build(
        self,
        results: list[RetrievalResult],
    ) -> BuiltContext:
        raise NotImplementedError