from __future__ import annotations

from src.plugins.registry import context_builder_registry
from src.retrieval.base import RetrievalResult

from .base import (
    BuiltContext,
    ContextBuilder,
    ContextItem,
)


@context_builder_registry.register("simple")
class SimpleContextBuilder(ContextBuilder):

    def __init__(
        self,
        max_characters: int = 12000,
        separator: str = "\n\n---\n\n",
        include_metadata: bool = False,
    ):
        if max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than 0"
            )

        self.max_characters = max_characters
        self.separator = separator
        self.include_metadata = include_metadata

    def build(
        self,
        results: list[RetrievalResult],
    ) -> BuiltContext:

        if not results:
            return BuiltContext(
                text="",
                items=[],
                total_characters=0,
                total_chunks=0,
            )

        blocks: list[str] = []
        items: list[ContextItem] = []

        for result in results:

            source_number = len(items) + 1

            header = (
                f"[Source {source_number} "
                f"| chunk={result.chunk.chunk_id}]"
            )

            if (
                self.include_metadata
                and result.chunk.metadata
            ):
                header += (
                    f" metadata={result.chunk.metadata}"
                )

            chunk_text = result.chunk.text.strip()

            block = (
                f"{header}\n"
                f"{chunk_text}"
            )

            separator_length = (
                len(self.separator)
                if blocks
                else 0
            )

            new_total = (
                sum(len(x) for x in blocks)
                + separator_length
                + len(block)
            )

            # Stop if adding this chunk exceeds the limit.
            if blocks and new_total > self.max_characters:
                break

            # If the first chunk itself is too large,
            # truncate it.
            if not blocks and len(block) > self.max_characters:
                block = block[:self.max_characters]

            blocks.append(block)

            items.append(
                ContextItem(
                    chunk_id=result.chunk.chunk_id,
                    text=result.chunk.text,
                    score=result.score,
                    rank=result.rank,
                    metadata=result.chunk.metadata,
                )
            )

        context_text = self.separator.join(blocks)

        return BuiltContext(
            text=context_text,
            items=items,
            total_characters=len(context_text),
            total_chunks=len(items),
        )