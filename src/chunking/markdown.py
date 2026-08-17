from __future__ import annotations

import re

from src.ingestion.models import Document
from src.plugins.registry import chunker_registry

from .base import Chunker
from .utils import make_chunk


@chunker_registry.register("markdown")
class MarkdownChunker(Chunker):

    HEADING_PATTERN = re.compile(
        r"^#{1,6}\s+.+$"
    )

    def __init__(
        self,
        max_chunk_size: int = 1200,
        overlap: int = 100,
    ):
        if max_chunk_size <= 0:
            raise ValueError(
                "max_chunk_size must be greater than 0"
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk(self, document: Document):

        sections = self._split_sections(
            document.text
        )

        chunks = []

        for section in sections:

            if len(section) <= self.max_chunk_size:
                chunks.append(section)
                continue

            chunks.extend(
                self._split_large_section(section)
            )

        return [
            make_chunk(
                document,
                text,
                index,
                "markdown",
                {
                    "max_chunk_size": self.max_chunk_size,
                    "overlap": self.overlap,
                },
            )
            for index, text in enumerate(chunks)
        ]

    def _split_sections(self, text: str):

        sections = []
        current = []

        for line in text.splitlines():

            is_heading = bool(
                self.HEADING_PATTERN.match(line)
            )

            if is_heading and current:
                sections.append(
                    "\n".join(current).strip()
                )
                current = [line]

            else:
                current.append(line)

        if current:
            sections.append(
                "\n".join(current).strip()
            )

        return [
            section
            for section in sections
            if section
        ]

    def _split_large_section(self, section: str):

        paragraphs = [
            paragraph.strip()
            for paragraph in section.split("\n\n")
            if paragraph.strip()
        ]

        chunks = []
        current = ""

        for paragraph in paragraphs:

            candidate = (
                paragraph
                if not current
                else f"{current}\n\n{paragraph}"
            )

            if (
                current
                and len(candidate) > self.max_chunk_size
            ):
                chunks.append(current)

                overlap_text = current[-self.overlap:]

                current = (
                    f"{overlap_text}\n\n{paragraph}"
                ).strip()

            else:
                current = candidate

        if current:
            chunks.append(current)

        return chunks