"""vega — a general-purpose PDF + image ingestion library.

Parse and chunk PDFs and standalone images (born-digital or scanned) into
general-purpose ``{chunk_id, text, metadata}`` records, with pluggable,
GPU-capable OCR and first-class Indic-script recovery.

Quick start::

    from vega import ingest_file, ingest_directory, parse

    chunks = ingest_file("report.pdf")                 # English, auto OCR
    chunks = ingest_file("go.pdf", languages=["te"])   # Telugu (legacy-font aware)
    doc    = parse("scan.png", ocr_mode="tesseract")   # DocumentModel, no chunking

Lower-level building blocks (``DocumentModel``, ``IngestionPipeline``,
``select_backend`` …) are importable from their submodules.
"""

from vega.config import IngestConfig
from vega.model import DocumentModel, Element, ElementType, TableData
from vega.pipeline import (
    IngestionPipeline,
    IngestStats,
    ingest_directory,
    ingest_file,
    parse,
)
from vega.records import ChunkRecord, stable_chunk_id
from vega.writer import write_json, write_jsonl

__version__ = "0.1.0"

__all__ = [
    "ingest_file",
    "ingest_directory",
    "parse",
    "IngestionPipeline",
    "IngestConfig",
    "IngestStats",
    "ChunkRecord",
    "stable_chunk_id",
    "DocumentModel",
    "Element",
    "ElementType",
    "TableData",
    "write_jsonl",
    "write_json",
    "__version__",
]
