"""Structure-aware + table-aware chunking.

Walks the ``DocumentModel`` in reading order while maintaining a **heading
breadcrumb**, and emits token-sized chunks that:

  · prepend their section path (``Doc › Section › Subsection``) so the embedding
    and the LLM both see where the chunk sits — one of the biggest retrieval
    wins available, and impossible in a flat-string design;
  · never split a table mid-row — tables become their own chunks, and a large
    table is split into row groups with the header repeated in each;
  · break cleanly at heading/table boundaries, with sentence-aware overlap only
    *within* a long section.

Sizing is by embedding tokens, not characters.

Adapted from the AgenticAI_Manufacturing
``doc_pipeline.ingestion.chunkers.structure`` module.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from vega.model import DocumentModel, Element, ElementType, TableData
from vega.records import ChunkRecord, stable_chunk_id
from vega.tokenization import (
    count_tokens, DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS,
)

_SENT = re.compile(r"(?<=[.!?])\s+")


class StructureChunker:
    def __init__(
        self,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        min_tokens: int = 48,
    ):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    def chunk(self, model: DocumentModel) -> List[ChunkRecord]:
        doc_title = _doc_title(model)
        breadcrumb: List[Tuple[int, str]] = []   # (level, text)
        records: List[ChunkRecord] = []
        buf: List[str] = []
        buf_tokens = 0
        buf_page: Optional[int] = None
        ordinal = 0

        def section_path() -> List[str]:
            path = [doc_title] if doc_title else []
            path += [t for _, t in breadcrumb]
            # drop consecutive duplicates (doc title == first heading is common)
            deduped: List[str] = []
            for p in path:
                if p and (not deduped or deduped[-1] != p):
                    deduped.append(p)
            return deduped

        def flush(overlap: bool = False):
            nonlocal buf, buf_tokens, buf_page, ordinal
            if not buf:
                return
            body = "\n".join(buf).strip()
            if body:
                records.append(_make_record(
                    model, section_path(), body, buf_page, ordinal,
                    strategy="structure",
                ))
                ordinal += 1
            carry: List[str] = []
            if overlap and self.overlap_tokens > 0:
                carry = _tail_sentences(body, self.overlap_tokens)
            buf = list(carry)
            buf_tokens = count_tokens(" ".join(buf)) if buf else 0
            buf_page = None

        for el in model.elements:
            if el.is_heading():
                flush()                      # clean break at section boundary
                _push_breadcrumb(breadcrumb, el)
                continue

            if el.type == ElementType.TABLE and el.table is not None:
                flush()                      # tables never share a chunk with prose
                table_recs = _table_records(
                    model, section_path(), el, ordinal, self.chunk_tokens,
                )
                records.extend(table_recs)
                if table_recs:
                    ordinal = table_recs[-1].metadata["ordinal"] + 1
                continue

            text = el.text.strip()
            if not text:
                continue
            if el.type == ElementType.FIGURE:
                text = f"[Figure] {text}"
            t = count_tokens(text)
            if buf and buf_tokens + t > self.chunk_tokens:
                flush(overlap=True)
            if buf_page is None:
                buf_page = el.page
            buf.append(text)
            buf_tokens += t

        flush()
        return _merge_small(records, self.min_tokens, self.chunk_tokens)


# ── helpers ─────────────────────────────────────────────────────────────────


def _doc_title(model: DocumentModel) -> str:
    for el in model.elements:
        if el.type == ElementType.TITLE and el.text.strip():
            return el.text.strip()
    # first H1 as a fallback title
    for el in model.elements:
        if el.type == ElementType.HEADING and el.level == 1 and el.text.strip():
            return el.text.strip()
    return model.metadata.get("filename", "")


def _push_breadcrumb(breadcrumb: List[Tuple[int, str]], el: Element) -> None:
    level = el.level or 1
    while breadcrumb and breadcrumb[-1][0] >= level:
        breadcrumb.pop()
    breadcrumb.append((level, el.text.strip()))


def _prefix(section_path: List[str]) -> str:
    return " › ".join(p for p in section_path if p)


def _make_record(model, section_path, body, page, ordinal, strategy) -> ChunkRecord:
    crumb = _prefix(section_path)
    text = f"{crumb}\n\n{body}" if crumb else body
    sp = " / ".join(section_path)
    return ChunkRecord(
        chunk_id=stable_chunk_id(model.source, sp, ordinal),
        text=text,
        source=model.source,
        doc_type=model.doc_type,
        strategy=strategy,
        metadata={
            "section_path": section_path,
            "heading": section_path[-1] if section_path else "",
            "page": page,
            "ordinal": ordinal,
            "filename": model.metadata.get("filename", ""),
        },
    )


def _table_records(model, section_path, el: Element, start_ordinal: int,
                   chunk_tokens: int) -> List[ChunkRecord]:
    """Emit a table as standalone chunk(s); split big tables by row group,
    repeating the header so each chunk is self-describing."""
    td: TableData = el.table
    full_md = td.to_markdown()
    records: List[ChunkRecord] = []
    ordinal = start_ordinal
    if count_tokens(full_md) <= chunk_tokens or td.n_rows <= 1:
        records.append(_table_chunk(model, section_path, td, ordinal, el.page,
                                    part=None))
        return records

    # Row-group split: size each group to the token budget, header repeated.
    header_tokens = count_tokens(td.to_markdown(max_rows=0)) or 1
    per_row = max(1, (count_tokens(full_md) - header_tokens) // max(1, td.n_rows))
    rows_per_group = max(1, (chunk_tokens - header_tokens) // per_row)
    part = 1
    for start in range(0, td.n_rows, rows_per_group):
        slice_td = TableData(headers=td.headers,
                             rows=td.rows[start:start + rows_per_group],
                             caption=td.caption)
        records.append(_table_chunk(model, section_path, slice_td, ordinal,
                                    el.page, part=part))
        ordinal += 1
        part += 1
    return records


def _table_chunk(model, section_path, td: TableData, ordinal, page, part) -> ChunkRecord:
    crumb = _prefix(section_path)
    label = f"Table ({td.n_rows}×{td.n_cols})"
    if part:
        label += f" — part {part}"
    head = f"{crumb}\n\n{label}\n" if crumb else f"{label}\n"
    sp = " / ".join(section_path) + f"::table#{ordinal}"
    return ChunkRecord(
        chunk_id=stable_chunk_id(model.source, sp, ordinal),
        text=head + td.to_markdown(),
        source=model.source,
        doc_type=model.doc_type,
        strategy="table",
        metadata={
            "section_path": section_path,
            "heading": section_path[-1] if section_path else "",
            "page": page,
            "ordinal": ordinal,
            "is_table": True,
            "table_shape": [td.n_rows, td.n_cols],
            "filename": model.metadata.get("filename", ""),
        },
    )


def _tail_sentences(text: str, overlap_tokens: int) -> List[str]:
    sents = [s for s in _SENT.split(text) if s.strip()]
    out: List[str] = []
    total = 0
    for s in reversed(sents):
        t = count_tokens(s)
        if total + t > overlap_tokens and out:
            break
        out.insert(0, s.strip())
        total += t
    return out


def _merge_small(records: List[ChunkRecord], min_tokens: int,
                 chunk_tokens: int) -> List[ChunkRecord]:
    """Fold sub-``min_tokens`` prose chunks into the previous prose chunk when
    they share a source and the merge stays within ~1.5× budget. Tables are
    never merged (kept self-contained)."""
    out: List[ChunkRecord] = []
    for rec in records:
        if (
            out
            and not rec.metadata.get("is_table")
            and not out[-1].metadata.get("is_table")
            and out[-1].source == rec.source
            # only merge within the same section — avoids stitching two
            # different breadcrumbs into one chunk
            and out[-1].metadata.get("section_path") == rec.metadata.get("section_path")
            and count_tokens(rec.text) < min_tokens
            and count_tokens(out[-1].text) + count_tokens(rec.text) <= int(chunk_tokens * 1.5)
        ):
            # bodies share a section ⇒ append rec's body without its breadcrumb
            body = rec.text.split("\n\n", 1)[-1] if "\n\n" in rec.text else rec.text
            out[-1].text += "\n" + body
            continue
        out.append(rec)
    return out
