"""Shared test fixtures — stub OCR backends + a generated born-digital PDF.

Nothing here touches a GPU, the network, or real Tesseract language packs: the
OCR engines are replaced by in-memory stubs so the pluggable seam is exercised
deterministically.
"""

from __future__ import annotations

from typing import List, Set

import pytest

from vega.ocr.base import BaseOCRBackend


class StubOCRBackend(BaseOCRBackend):
    """A fake OCR engine. Returns queued text and records every call, so tests
    can assert *which* script was routed without a real Tesseract install."""

    def __init__(self, name: str = "stub", scripts=None, output: str = ""):
        self.name = name
        self._scripts: Set[str] = set(scripts or {"eng"})
        self._output = output
        self.calls: List[tuple] = []          # (script, n_bytes)

    def available_scripts(self) -> Set[str]:
        return set(self._scripts)

    def image_to_text(self, image_png: bytes, script: str) -> str:
        self.calls.append((script, len(image_png)))
        return self._output


@pytest.fixture
def stub_backend():
    return StubOCRBackend(name="stub", scripts={"eng", "tel", "hin"},
                          output="recovered text")


@pytest.fixture
def make_ocr_stub():
    """Factory for parameterised stub backends (scripts + canned output)."""
    def _make(scripts=("eng",), output="", name="stub"):
        return StubOCRBackend(name=name, scripts=set(scripts), output=output)
    return _make


@pytest.fixture
def scanned_pdf(tmp_path):
    """A one-page PDF with **no text layer** — just an embedded image. Exercises
    the per-page "needs OCR" (scanned) branch: raw text < threshold, no tables."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from PIL import Image

    img = Image.new("RGB", (800, 400), "white")
    pdf_path = tmp_path / "scanned.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    c.drawImage(ImageReader(img), 72, 400, width=400, height=200)
    c.showPage()
    c.save()
    return pdf_path


@pytest.fixture
def image_file(tmp_path):
    """A standalone .png — an image file *is* a scanned page (always OCR'd)."""
    from PIL import Image

    p = tmp_path / "scan.png"
    Image.new("RGB", (400, 200), "white").save(p)
    return p


@pytest.fixture
def multipage_pdf(tmp_path):
    """A born-digital PDF with several sections across several pages — used to
    check page-level parallelism produces byte-identical results to serial."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    p = tmp_path / "multi.pdf"
    c = canvas.Canvas(str(p), pagesize=LETTER)
    w, h = LETTER
    for i in range(6):
        c.setFont("Helvetica-Bold", 15)
        c.drawString(72, h - 90, f"Section {i}")
        c.setFont("Helvetica", 11)
        c.drawString(72, h - 120,
                     f"Body text for page {i} with enough words here to size "
                     f"into a real token-bounded chunk with a breadcrumb.")
        c.showPage()
    c.save()
    return p


@pytest.fixture
def born_digital_pdf(tmp_path):
    """A tiny multi-section born-digital PDF (real text layer, no images).
    Generated with reportlab so the no-OCR parse+chunk path runs end to end."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=LETTER)
    width, height = LETTER

    # Page 1 — a title, a heading, and body prose.
    c.setFont("Helvetica-Bold", 22)
    c.drawString(72, height - 90, "Vega Test Document")
    c.setFont("Helvetica-Bold", 15)
    c.drawString(72, height - 140, "Introduction")
    c.setFont("Helvetica", 11)
    body = (
        "Vega parses born-digital PDFs without any OCR because the text layer "
        "is already present. This paragraph exists so the structure chunker has "
        "real prose to size into a token-bounded chunk with a heading breadcrumb."
    )
    y = height - 170
    for line in _wrap(body, 90):
        c.drawString(72, y, line)
        y -= 16
    c.showPage()

    # Page 2 — another section.
    c.setFont("Helvetica-Bold", 15)
    c.drawString(72, height - 90, "Details")
    c.setFont("Helvetica", 11)
    body2 = (
        "The second section lives on a second page so page numbers propagate "
        "into chunk metadata. Reading order is top to bottom, left to right."
    )
    y = height - 120
    for line in _wrap(body2, 90):
        c.drawString(72, y, line)
        y -= 16
    c.showPage()
    c.save()
    return pdf_path


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines
