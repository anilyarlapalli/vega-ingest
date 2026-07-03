"""Plain-text parser — recovers whatever light structure ``.txt`` carries.

Recognises a ``=== Section ===`` convention as headings and blank-line-delimited
paragraphs. Everything else is prose. Encoding-safe (``errors='replace'``) so a
stray byte can't abort the batch. Kept as a lightweight convenience path; vega's
focus is PDF + images.

Adapted from the AgenticAI_Manufacturing ``doc_pipeline.ingestion.parsers.text``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from vega.model import DocumentModel, Element, ElementType
from vega.records import normalize_source


class TextParser:
    def parse(self, path: Path) -> DocumentModel:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        model = DocumentModel(source=normalize_source(str(path)), doc_type="txt",
                              metadata={"filename": path.name})
        para: List[str] = []

        def flush():
            if para:
                joined = " ".join(" ".join(para).split())
                if joined:
                    model.add(Element(type=ElementType.PARAGRAPH, text=joined))
                para.clear()

        for line in text.splitlines():
            s = line.strip()
            if s.startswith("===") and s.endswith("===") and len(s) > 6:
                flush()
                model.add(Element(type=ElementType.HEADING,
                                  text=s.strip("= ").strip(), level=1))
            elif not s:
                flush()
            else:
                para.append(s)
        flush()
        return model
