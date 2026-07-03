"""Standalone image parser — OCR a ``.png/.jpg/.tiff/.bmp/.webp`` into structure.

An image file *is* a scanned page: it always needs OCR (there is no text layer
to skip to). Language routing mirrors the PDF scanned path — declared language →
Tesseract OSD among the declared candidates → every candidate pack — falling
back to plain English OCR when no non-English language is declared.

Text is split on blank lines into paragraphs so the structure chunker has
something to size; there is no heading/table structure to recover from a raw
image, so everything is prose on page 1.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from vega import text_recovery
from vega.model import DocumentModel, Element, ElementType
from vega.records import normalize_source

logger = logging.getLogger("vega.parsers.image")


def _to_png_bytes(path: Path) -> bytes:
    """Load any supported image and re-encode as PNG (the OCR backend contract).
    Normalises to RGB so TIFF/BMP/WebP/CMYK inputs all OCR consistently."""
    from PIL import Image  # noqa: PLC0415
    with Image.open(path) as im:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


class ImageParser:
    def __init__(self, ocr_backend=None, recovery_script: Optional[str] = None,
                 candidate_langs: Optional[list] = None):
        self._backend = ocr_backend
        self._recovery_script = recovery_script
        self._candidate_langs = candidate_langs or []

    def _ocr(self, png: bytes) -> Tuple[str, Optional[str], bool]:
        """Return (text, script, ocr_used). Routes Indic scripts via
        text_recovery; falls back to English OCR."""
        if self._backend is None:
            return ("", None, False)
        if self._candidate_langs or self._recovery_script:
            rec = text_recovery.ocr_scanned(
                render_png=lambda: png,
                backend=self._backend,
                candidate_langs=self._candidate_langs,
                declared_script=self._recovery_script,
            )
            if rec.was_recovered:
                return (rec.text, rec.script, True)
        try:
            text = self._backend.image_to_text(png, "eng")
        except Exception as e:
            logger.debug("image OCR failed: %r", e)
            text = ""
        return (text, "eng" if text else None, bool(text))

    def parse(self, path: Path) -> DocumentModel:
        path = Path(path)
        model = DocumentModel(
            source=normalize_source(str(path)), doc_type="image",
            metadata={"filename": path.name, "total_pages": 1},
        )
        try:
            png = _to_png_bytes(path)
        except Exception as e:
            logger.warning("could not read image %s: %r", path.name, e)
            model.metadata["ocr_pages"] = []
            model.metadata["ocr_backend"] = getattr(self._backend, "name", None)
            return model

        text, script, used = self._ocr(png)
        paragraphs: List[str] = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs and text.strip():
            paragraphs = [text.strip()]
        for para in paragraphs:
            model.add(Element(type=ElementType.PARAGRAPH,
                              text=" ".join(para.split()), page=1))

        model.metadata["ocr_pages"] = [1] if used else []
        model.metadata["ocr_backend"] = getattr(self._backend, "name", None)
        model.metadata["ocr_script"] = script
        logger.info("parsed image %s: %d paragraph(s), script=%s",
                    path.name, len(paragraphs), script)
        return model
