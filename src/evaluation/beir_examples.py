from __future__ import annotations

from .models import EvalExample


def beir_to_eval_examples(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> list[EvalExample]:
    """
    Build EvalExamples from BEIR's (queries, qrels) split output.

    Iterates over `qrels` rather than `queries`: a query without any
    qrels entries can't be scored (there's nothing to compare
    against), so it's silently excluded rather than producing an
    example with empty ground truth that would only pollute averages.
    """
    examples: list[EvalExample] = []

    for query_id, relevant_doc_ids in qrels.items():
        query_text = queries.get(query_id)

        if not query_text:
            continue

        examples.append(
            EvalExample(
                query_id=query_id,
                query_text=query_text,
                relevant_doc_ids=relevant_doc_ids,
            )
        )

    return examples