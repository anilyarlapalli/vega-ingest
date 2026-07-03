"""Structured document model — the parse-stage output.

Parsers never collapse a file into one ``content`` string. They emit a
``DocumentModel`` — an *ordered tree of typed elements* that preserves headings,
tables, figures, and reading order. Every downstream improvement (heading-
breadcrumb chunks, table-aware chunking, OCR captions) is only possible because
structure survives the parse stage.

A parser's only job: bytes → ``DocumentModel``. A chunker's only job:
``DocumentModel`` → ``list[ChunkRecord]`` (see ``records.py``).

Adapted from the AgenticAI_Manufacturing ``doc_pipeline.ingestion.model`` module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class ElementType(str, Enum):
    TITLE = "title"          # document title (one per doc, ideally)
    HEADING = "heading"      # section heading; carries ``level`` 1..6
    PARAGRAPH = "paragraph"  # flowing prose
    LIST_ITEM = "list_item"  # one bullet / numbered item
    TABLE = "table"          # structured table (see ``TableData``)
    FIGURE = "figure"        # an image / diagram (may carry OCR text)
    CAPTION = "caption"      # figure/table caption
    CODE = "code"            # code / preformatted block


@dataclass
class TableData:
    """A parsed table kept structured rather than flattened to prose."""

    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    caption: str = ""

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.headers) or (len(self.rows[0]) if self.rows else 0)

    def to_markdown(self, max_rows: Optional[int] = None) -> str:
        """GitHub-flavoured Markdown table — the form LLMs read most reliably.

        ``max_rows`` truncates the body (with an ellipsis row) so a giant
        table can be summarised in a header card without dumping every row.
        """
        cols = self.n_cols
        if cols == 0:
            return ""
        headers = self.headers or [f"col_{i+1}" for i in range(cols)]
        headers = [(_clean_cell(h) or f"col_{i+1}") for i, h in enumerate(headers)]

        def _row(cells: List[str]) -> str:
            padded = [_clean_cell(c) for c in cells] + [""] * (cols - len(cells))
            return "| " + " | ".join(padded[:cols]) + " |"

        lines = [_row(headers), "| " + " | ".join(["---"] * cols) + " |"]
        body = self.rows if max_rows is None else self.rows[:max_rows]
        for r in body:
            lines.append(_row(r))
        if max_rows is not None and self.n_rows > max_rows:
            lines.append("| " + " | ".join(["…"] * cols) + " |")
        return "\n".join(lines)


def _clean_cell(value) -> str:
    """Normalise a cell to a single-line Markdown-safe string."""
    if value is None:
        return ""
    s = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return " ".join(s.split()).strip()


@dataclass
class Element:
    """One node of the document tree."""

    type: ElementType
    text: str = ""                       # for text-bearing elements
    table: Optional[TableData] = None    # set when type == TABLE
    level: int = 0                       # heading depth (1..6); 0 otherwise
    page: Optional[int] = None           # 1-based source page, when known
    meta: dict = field(default_factory=dict)

    def is_heading(self) -> bool:
        return self.type in (ElementType.TITLE, ElementType.HEADING)


@dataclass
class DocumentModel:
    """Parser output: ordered typed elements + document-level metadata."""

    source: str = ""                     # absolute path or uri
    doc_type: str = ""                   # "pdf" | "image" | "txt"
    elements: List[Element] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)   # filename, total_pages, …

    def add(self, element: Element) -> None:
        self.elements.append(element)

    def text_elements(self) -> List[Element]:
        return [e for e in self.elements if e.type != ElementType.TABLE]

    def tables(self) -> List[Element]:
        return [e for e in self.elements if e.type == ElementType.TABLE]

    def summary(self) -> dict:
        counts: dict = {}
        for e in self.elements:
            counts[e.type.value] = counts.get(e.type.value, 0) + 1
        return {
            "source": self.source,
            "doc_type": self.doc_type,
            "n_elements": len(self.elements),
            "by_type": counts,
        }
