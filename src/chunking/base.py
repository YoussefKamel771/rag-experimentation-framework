from abc import ABC, abstractmethod

from src.ingestion.models import Chunk, Document


class Chunker(ABC):
    """Interface implemented by every chunking strategy."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError
