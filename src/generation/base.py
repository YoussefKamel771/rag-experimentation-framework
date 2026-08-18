from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt: str

    sources: list[dict[str, Any]] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class Generator(ABC):

    @abstractmethod
    def generate(
        self,
        query: str,
        context: str,
        sources: list[dict[str, Any]] | None = None,
    ) -> GenerationResult:
        raise NotImplementedError