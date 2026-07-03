"""Output writers — general-purpose JSON / JSONL, no app-specific schema.

  · ``write_jsonl`` — one chunk per line: ``{chunk_id, text, metadata}``. The
    portable retrieval-ingest format (stream it into any vector store / KG).
  · ``write_json``  — a single document: the ``DocumentModel`` structure (typed
    element tree) for callers that want the parse result rather than chunks.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from vega.model import DocumentModel, Element


def write_jsonl(records: Iterable[Union[Dict[str, Any], Any]], path: str | Path) -> int:
    """Write chunk dicts (or ``ChunkRecord`` objects) as JSONL. Returns count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            if hasattr(rec, "as_dict"):
                rec = rec.as_dict()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def _element_to_dict(el: Element) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "type": el.type.value,
        "text": el.text,
        "level": el.level,
        "page": el.page,
        "meta": el.meta,
    }
    if el.table is not None:
        d["table"] = {
            "headers": el.table.headers,
            "rows": el.table.rows,
            "caption": el.table.caption,
        }
    return d


def document_to_dict(model: DocumentModel) -> Dict[str, Any]:
    return {
        "source": model.source,
        "doc_type": model.doc_type,
        "metadata": model.metadata,
        "elements": [_element_to_dict(e) for e in model.elements],
    }


def write_json(obj: Union[DocumentModel, List[Dict[str, Any]], Dict[str, Any]],
               path: str | Path) -> None:
    """Write a ``DocumentModel`` (as its element tree) or any JSON-able object."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, DocumentModel):
        payload: Any = document_to_dict(obj)
    elif dataclasses.is_dataclass(obj):
        payload = dataclasses.asdict(obj)
    else:
        payload = obj
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
