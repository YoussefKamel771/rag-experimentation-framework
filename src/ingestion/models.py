from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A source document before chunking."""

    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk that will be embedded and indexed."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
