from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from src.ingestion.models import Chunk
from src.plugins.registry import retriever_registry

from .base import RetrievalResult, Retriever


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """
    Convert text into lowercase word tokens.

    Example:
        "What is LiDAR?"
        -> ["what", "is", "lidar"]
    """
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------

@retriever_registry.register("bm25")
class BM25Retriever(Retriever):

    def __init__(
        self,
        chunks_path: str,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b

        # Load chunks
        self.chunks = self._load_chunks(
            chunks_path
        )

        if not self.chunks:
            raise ValueError(
                "No chunks found for BM25 retrieval"
            )

        # Tokenize every chunk
        self.tokens = [
            tokenize(chunk.text)
            for chunk in self.chunks
        ]

        # Document lengths
        self.document_lengths = [
            len(tokens)
            for tokens in self.tokens
        ]

        # Average document length
        self.average_document_length = (
            sum(self.document_lengths)
            / len(self.document_lengths)
        )

        # Term frequency:
        #
        # tf[document_index][term] = number of
        # times the term appears in that document
        self.term_frequencies = [
            Counter(tokens)
            for tokens in self.tokens
        ]

        # Inverse document frequency
        self.idf = self._calculate_idf()

    # -----------------------------------------------------
    # Loading
    # -----------------------------------------------------

    @staticmethod
    def _load_chunks(
        chunks_path: str,
    ) -> list[Chunk]:

        path = Path(chunks_path)

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return [
            Chunk(**item)
            for item in data
        ]

    # -----------------------------------------------------
    # IDF
    # -----------------------------------------------------

    def _calculate_idf(self) -> dict[str, float]:

        # Number of documents containing each term
        document_frequency = Counter()

        for tokens in self.tokens:

            # set() is important:
            # a word appearing 10 times in one document
            # should count as appearing in ONE document
            document_frequency.update(
                set(tokens)
            )

        number_of_documents = len(
            self.chunks
        )

        idf = {}

        for term, document_count in (
            document_frequency.items()
        ):

            idf[term] = math.log(
                1
                + (
                    number_of_documents
                    - document_count
                    + 0.5
                )
                / (
                    document_count
                    + 0.5
                )
            )

        return idf

    # -----------------------------------------------------
    # Score one document
    # -----------------------------------------------------

    def _score_document(
        self,
        query_tokens: list[str],
        document_index: int,
    ) -> float:

        score = 0.0

        document_length = (
            self.document_lengths[
                document_index
            ]
        )

        term_frequency = (
            self.term_frequencies[
                document_index
            ]
        )

        for term in query_tokens:

            frequency = term_frequency.get(
                term,
                0,
            )

            # Query term does not exist in document
            if frequency == 0:
                continue

            idf = self.idf.get(
                term,
                0.0,
            )

            length_normalization = (
                1
                - self.b
                + self.b
                * document_length
                / max(
                    self.average_document_length,
                    1e-12,
                )
            )

            denominator = (
                frequency
                + self.k1
                * length_normalization
            )

            term_score = (
                idf
                * frequency
                * (self.k1 + 1)
                / denominator
            )

            score += term_score

        return score

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:

        if not query.strip():
            return []

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        top_k = max(
            1,
            min(
                top_k,
                len(self.chunks),
            ),
        )

        # Calculate score for every document
        scored_documents = []

        for document_index in range(
            len(self.chunks)
        ):

            score = self._score_document(
                query_tokens,
                document_index,
            )

            scored_documents.append(
                (
                    document_index,
                    score,
                )
            )

        # Highest score first
        scored_documents.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # Convert to RetrievalResult
        results = []

        for rank, (
            document_index,
            score,
        ) in enumerate(
            scored_documents[:top_k],
            start=1,
        ):

            results.append(
                RetrievalResult(
                    chunk=self.chunks[
                        document_index
                    ],
                    score=float(score),
                    rank=rank,
                    retriever="bm25",
                    metadata={
                        "document_index":
                            document_index,
                    },
                )
            )

        return results