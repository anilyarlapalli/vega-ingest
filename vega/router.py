"""Format router — file extension → parser.

vega parses **PDF** and **standalone image** files (plus ``.txt`` as a light
convenience). The ``Parser`` protocol leaves room for content sniffing or a
layout-model parser to drop in behind the same interface without touching
callers.

Adapted from the AgenticAI_Manufacturing ``doc_pipeline.ingestion.router`` module
(scoped to PDF + images; parsers are built per-call carrying the OCR backend and
language routing rather than cached, since those vary per run).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from vega.parsers.base import Parser
from vega.parsers.image import ImageParser
from vega.parsers.pdf import PDFParser
from vega.parsers.text import TextParser

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp")
# The advertised core: vega's scope is PDF + images.
CORE_EXTENSIONS = (".pdf",) + IMAGE_EXTENSIONS
# ``.txt`` is an explicit *convenience extra*, not part of the core PDF+image
# scope — handy for mixing plain-text notes into a corpus. Documented as such in
# the README; kept in the supported set so directory ingestion picks it up.
TEXT_EXTENSIONS = (".txt",)
SUPPORTED_EXTENSIONS = CORE_EXTENSIONS + TEXT_EXTENSIONS


def is_supported(path: Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def get_parser(
    path: Path,
    *,
    ocr_backend=None,
    recovery_script: Optional[str] = None,
    candidate_langs: Optional[list] = None,
    figure_ocr: bool = False,
    dpi: int = 300,
    scanned_dpi: int = 200,
    page_workers: int = 1,
    columns: bool = True,
    batch_ocr: bool = True,
) -> Optional[Parser]:
    """Build the parser for ``path``, wiring the OCR backend + language routing.

    PDFs and images carry the OCR backend, the primary declared ``recovery_script``
    (Tesseract code) and the full ``candidate_langs`` (ISO codes) so per-page
    recovery routes to the right pack. ``.txt`` needs none of it.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return PDFParser(
            ocr_backend=ocr_backend, recovery_script=recovery_script,
            candidate_langs=candidate_langs, figure_ocr=figure_ocr,
            dpi=dpi, scanned_dpi=scanned_dpi, page_workers=page_workers,
            columns=columns, batch_ocr=batch_ocr,
        )
    if ext in IMAGE_EXTENSIONS:
        return ImageParser(
            ocr_backend=ocr_backend, recovery_script=recovery_script,
            candidate_langs=candidate_langs,
        )
    if ext == ".txt":
        return TextParser()
    return None
