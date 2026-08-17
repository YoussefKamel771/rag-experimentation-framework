from __future__ import annotations
import hashlib, re
from src.ingestion.models import Chunk, Document

def make_chunk(document: Document, text: str, index: int, strategy: str, extra_metadata=None):
    text=text.strip()
    key=f"{document.document_id}:{strategy}:{index}:{hashlib.sha1(text.encode()).hexdigest()[:12]}"
    metadata={**document.metadata,"document_id":document.document_id,"chunk_index":index,"chunking_strategy":strategy}
    if extra_metadata: metadata.update(extra_metadata)
    return Chunk(hashlib.sha1(key.encode()).hexdigest()[:16], text, metadata)

def sentence_split(text: str):
    return [p.strip() for p in re.split(r"(?<=[.!?؟。！])\s+|\n+", text) if p.strip()]
