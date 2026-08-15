from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from src.ingestion.models import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def make_document_id(path: Path) -> str:
    """Create a stable, human-readable ID containing the original filename and hash."""
    resolved_path = path.resolve()
    # Get primary filename before multi-part extensions (e.g., 'S08_set1_a10' from 'S08_set1_a10.txt.clean')
    file_stem = path.name.split(".")[0]
    
    hash_digest = hashlib.sha1(str(resolved_path).encode("utf-8")).hexdigest()[:8]
    return f"{file_stem}_{hash_digest}"


def clean_text(text: str) -> str:
    """Conservative text normalization; avoid destroying document structure."""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue

        documents.append(
            Document(
                document_id=make_document_id(path),
                text=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "file_type": "pdf",
                    "page_number": page_number,
                },
            )
        )

    return documents


def load_text_file(path: Path) -> list[Document]:
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        return []

    return [
        Document(
            document_id=make_document_id(path),
            text=text,
            metadata={
                "source": str(path),
                "filename": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "page_number": None,
            },
        )
    ]

def load_csv_manifest(
    manifest_path: Path, 
    target_dir: Path | None = None,
    filter_suffix: str = ".clean"
) -> list[Document]:
    """Reads a CSV manifest and loads files listed in the 'file' column that match the filter suffix."""
    documents: list[Document] = []
    base_dir = target_dir or manifest_path.parent

    with manifest_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("file", "").strip()
            
            # Filter specifically for files ending with .clean (or your choice)
            if not filename or not filename.endswith(filter_suffix):
                continue

            file_path = base_dir / filename
            
            # Fallback search if the file is in a subdirectory relative to base_dir
            if not file_path.exists():
                matches = list(base_dir.rglob(filename))
                if matches:
                    file_path = matches[0]

            if not file_path.exists():
                print(f"      [Warning] File listed in manifest not found: {file_path}")
                continue

            # Load the text file and attach word count metadata if available from CSV
            loaded_docs = load_text_file(file_path)
            for doc in loaded_docs:
                if "words" in row and row["words"].isdigit():
                    doc.metadata["target_word_count"] = int(row["words"])
                documents.extend(loaded_docs)

    return documents

def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    print(f"      Loading {path} ({suffix})")

    if suffix == ".pdf":
        return load_pdf(path)

    if suffix in {".txt", ".md"} or path.name.endswith(".clean"):
        return load_text_file(path)

    if suffix == ".csv":
        return load_csv_manifest(path)

    raise ValueError(f"Unsupported file type: {path}")


def load_directory(input_dir: str | Path) -> list[Document]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    documents: list[Document] = []

    for path in sorted(input_path.rglob("*")):
        print(f"    Checking {path} ({path.suffix.lower()})")
        # Matches supported extensions OR filenames ending in .clean
        if path.is_file() and (path.suffix.lower() in SUPPORTED_EXTENSIONS or path.name.endswith(".clean")):
            documents.extend(load_file(path))

    if not documents:
        raise ValueError(
            f"No supported documents found in {input_path}. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return documents