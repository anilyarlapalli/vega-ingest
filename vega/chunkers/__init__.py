"""Chunkers: ``DocumentModel`` → ``list[ChunkRecord]``.

``Chunker`` is the extension point: anything with ``chunk(model) -> records``
can be passed to ``IngestionPipeline(chunker=...)`` to replace the default
:class:`StructureChunker`. Chunking is deliberately decoupled from parsing —
a custom strategy sees the full structured model (headings, tables, pages,
reading order) and never touches parsing/OCR.

Limitation: a custom chunker applies **in-process only**. Multi-file runs with
``workers > 1`` rebuild their pipelines from the picklable config inside each
worker process, which would silently fall back to the default — so the
pipeline refuses that combination instead (see ``_iter_parallel``).
"""

from typing import List, Protocol, runtime_checkable

from vega.chunkers.structure import StructureChunker
from vega.model import DocumentModel
from vega.records import ChunkRecord


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, model: DocumentModel) -> List[ChunkRecord]:
        """Turn one parsed document into ordered chunk records."""
        ...


__all__ = ["Chunker", "StructureChunker"]
