"""Chunk record — the single retrieval unit + the single stable id.

There is exactly **one** id per chunk, minted once at ingestion from the
document source + structural position, so a rebuild doesn't churn every id.
``ChunkRecord.as_dict()`` emits the general-purpose ``{chunk_id, text, metadata}``
shape that any downstream embedder / store / KG can consume.

Adapted from the AgenticAI_Manufacturing ``doc_pipeline.ingestion.records`` module.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict


def stable_chunk_id(source: str, section_path: str, ordinal: int) -> str:
    """Deterministic, collision-resistant, content-addressable-ish id.

    Stable across re-ingests of the same file (same source + structural
    position ⇒ same id). 16 hex chars of SHA-1 is ample for single corpora.
    """
    basis = f"{source}\x1f{section_path}\x1f{ordinal}"
    return "c_" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


@dataclass
class ChunkRecord:
    """One retrieval unit. ``chunk_id`` is the only id used everywhere."""

    chunk_id: str
    text: str
    source: str = ""
    doc_type: str = ""
    strategy: str = ""                       # which chunker produced it
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """The general-purpose ``{chunk_id, text, metadata}`` contract."""
        meta = dict(self.metadata)
        meta.setdefault("source", self.source)
        meta.setdefault("doc_type", self.doc_type)
        meta.setdefault("strategy", self.strategy)
        return {"chunk_id": self.chunk_id, "text": self.text, "metadata": meta}
