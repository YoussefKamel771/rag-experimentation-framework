from __future__ import annotations

from src.retrieval.base import RetrievalResult

DEFAULT_DOC_ID_FIELD = "document_id"


def dedup_to_document_ranking(
    results: list[RetrievalResult],
    doc_id_field: str = DEFAULT_DOC_ID_FIELD,
) -> list[str]:
    """
    Collapse a chunk-level ranked result list down to a document-level
    ranked list, for scoring against document-level qrels.

    Strategy: results are assumed to already be rank-ordered
    (best-first). The first occurrence of a given document_id is kept
    and later chunks from the same document are dropped -- i.e.
    "best-scoring chunk per document" wins, since it appears first in
    the ranking. This is a deliberate, single, documented dedup
    strategy so results stay comparable across experiments; if a
    different rule (e.g. average chunk score per doc) is needed later,
    swap this function rather than special-casing callers.

    Chunks missing `doc_id_field` in metadata are skipped rather than
    raising, since not every chunker/config path is guaranteed to
    stamp it (defensive against upstream config drift).
    """
    seen: set[str] = set()
    doc_ids: list[str] = []

    for result in results:
        doc_id = result.chunk.metadata.get(doc_id_field)

        if doc_id is None or doc_id in seen:
            continue

        seen.add(doc_id)
        doc_ids.append(str(doc_id))

    return doc_ids


def doc_ranking_to_run(doc_ids: list[str]) -> dict[str, float]:
    """
    Convert an ordered document-id list into the {doc_id: score} shape
    pytrec_eval expects for a single query's "run".

    Scores are synthetic and strictly decreasing by position rather
    than reusing the underlying retrieval/reranker scores, because:
    (a) dense/BM25/cross-encoder scores live on different scales and
    aren't comparable across stages, and (b) ties in the raw score
    would make ranking order ambiguous to pytrec_eval. Position is the
    only thing that actually matters for these metrics.
    """
    n = len(doc_ids)
    return {doc_id: float(n - index) for index, doc_id in enumerate(doc_ids)}