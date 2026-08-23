# src/ingestion/beir_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from src.ingestion.models import Document


class BEIRDatasetLoader:
    """
    Downloads (if needed) and loads a BEIR benchmark dataset,
    exposing the corpus as framework-native Document objects,
    plus the raw queries/qrels needed for retrieval evaluation.
    """

    BASE_URL = (
        "https://public.ukp.informatik.tu-darmstadt.de/"
        "thakur/BEIR/datasets/{}.zip"
    )

    def __init__(
        self,
        dataset: str,
        data_dir: str | Path = "data/beir",
        split: str = "test",
    ):
        self.dataset = dataset
        self.data_dir = Path(data_dir)
        self.split = split

        self.corpus: dict[str, dict[str, Any]] = {}
        self.queries: dict[str, str] = {}
        self.qrels: dict[str, dict[str, int]] = {}

    def download(self) -> Path:
        """Download + unzip the dataset if not already present."""
        url = self.BASE_URL.format(self.dataset)
        data_path = util.download_and_unzip(url, str(self.data_dir))
        return Path(data_path)

    def load(self) -> "BEIRDatasetLoader":
        """Download (if needed) and load corpus/queries/qrels."""
        data_path = self.download()

        self.corpus, self.queries, self.qrels = (
            GenericDataLoader(data_folder=str(data_path)).load(
                split=self.split
            )
        )

        return self

    def to_documents(self) -> list[Document]:
        """
        Convert the BEIR corpus into framework-native Documents.

        BEIR entries look like:
            {"_id": "31715818", "title": "...", "text": "...", "metadata": {}}

        Title is folded into the text (standard BEIR baseline practice)
        but also kept separately in metadata for inspection/debugging.
        """
        if not self.corpus:
            raise RuntimeError("Corpus is empty. Call .load() first.")

        documents = []

        for doc_id, fields in self.corpus.items():
            title = (fields.get("title") or "").strip()
            body = (fields.get("text") or "").strip()

            text = f"{title}\n\n{body}" if title else body
            if not text:
                continue

            documents.append(
                Document(
                    document_id=doc_id,
                    text=text,
                    metadata={
                        "source": "beir",
                        "dataset": self.dataset,
                        "title": title,
                        **(fields.get("metadata") or {}),
                    },
                )
            )

        return documents

    def get_queries(self) -> dict[str, str]:
        """query_id -> query text"""
        return self.queries

    def get_qrels(self) -> dict[str, dict[str, int]]:
        """query_id -> {corpus_id: relevance_score}"""
        return self.qrels