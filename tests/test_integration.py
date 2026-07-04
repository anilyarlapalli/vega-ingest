"""Integration tests — real Tesseract, real threads/processes, real corrupt input.

All are marked ``integration`` and skip themselves when their dependency is
absent, so the DEFAULT suite never needs a GPU, network, or real language pack.
Run everything (including these) with, e.g., ``pytest -m 'integration or not
integration'`` on a host that has Tesseract installed.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _tesseract_langs():
    try:
        import pytesseract
        return set(pytesseract.get_languages())
    except Exception:
        return set()


_HAS_TESS = shutil.which("tesseract") is not None
_LANGS = _tesseract_langs()
requires_tess = pytest.mark.skipif(not _HAS_TESS, reason="tesseract binary not installed")


# ── real OCR on a synthesized English scan ───────────────────────────────────

@requires_tess
@pytest.mark.skipif("eng" not in _LANGS, reason="tesseract 'eng' pack missing")
def test_real_tesseract_ocrs_english_image(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    from vega import ingest_file

    img = Image.new("RGB", (900, 160), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    except Exception:
        font = ImageFont.load_default()
    d.text((25, 55), "Integration OCR smoke test", fill="black", font=font)
    p = tmp_path / "scan.png"
    img.save(p)

    chunks = ingest_file(p, ocr_mode="tesseract")
    assert chunks
    text = " ".join(c["text"] for c in chunks).lower()
    assert "integration" in text and "test" in text
    assert chunks[0]["metadata"]["ocr_used"] is True


@requires_tess
@pytest.mark.skipif("tel" not in _LANGS, reason="tesseract 'tel' (Telugu) pack missing")
def test_real_tesseract_indic_routing(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    from vega import ingest_file

    telugu = "పరిపాలన పరిషత్తు నుండి ఉత్తర్వు"
    font_path = "/usr/share/fonts/truetype/teluguvijayam/Suravaram.ttf"
    try:
        font = ImageFont.truetype(font_path, 44)
    except Exception:
        pytest.skip("no Telugu-capable font available")
    img = Image.new("RGB", (1000, 140), "white")
    ImageDraw.Draw(img).text((25, 45), telugu, fill="black", font=font)
    p = tmp_path / "te.png"
    img.save(p)

    chunks = ingest_file(p, languages=["te"], ocr_mode="tesseract")
    assert chunks
    joined = " ".join(c["text"] for c in chunks)
    # A good chunk of the output should be Telugu-script characters.
    telu = sum(1 for ch in joined if 0x0C00 <= ord(ch) <= 0x0C7F)
    assert telu >= 10


# ── the reported bug: born-digital Tamil in a legacy ASCII glyph font ────────
#
# tamil.pdf is an untracked fixture in the repo root (VANAVIL-Avvaiyar/SunTommy,
# WinAnsiEncoding, no /ToUnicode). Under a DEFAULT run (no --lang) it used to
# extract as ASCII mojibake ("jkpo;ehL muR"), tagged 'en', ocr_used=false. The
# fix must detect the glyph mojibake generically, OCR via all-supported OSD, and
# emit clean Tamil Unicode tagged 'ta'. Skips if the file or 'tam' pack is absent.
_TAMIL_PDF = Path("/home/anil-y/app_ideas/manufacture/vega/tamil.pdf")


@requires_tess
@pytest.mark.skipif(not _TAMIL_PDF.exists(), reason="tamil.pdf fixture not present")
@pytest.mark.skipif("tam" not in _LANGS, reason="tesseract 'tam' (Tamil) pack missing")
def test_tamil_glyph_font_recovered_under_default_lang():
    from vega import ingest_file

    # Default config: no --lang declared (default 'en'), real Tesseract.
    chunks = ingest_file(str(_TAMIL_PDF), ocr_mode="tesseract")
    assert chunks, "tamil.pdf should yield chunks"
    joined = " ".join(c["text"] for c in chunks)

    # 1) Clean Tamil Unicode recovered — the government header must be present.
    assert "தமிழ்நாடு அரசு" in joined
    # 2) No glyph mojibake remains anywhere in the output.
    assert "jkpo;ehL" not in joined
    tamil_chars = sum(1 for ch in joined if 0x0B80 <= ord(ch) <= 0x0BFF)
    assert tamil_chars > 100

    # 3) A recovered chunk is tagged 'ta' and marked ocr_used.
    recovered = [c for c in chunks if "தமிழ்நாடு அரசு" in c["text"]]
    assert recovered
    assert recovered[0]["metadata"]["language"] == "ta"
    assert any(c["metadata"]["ocr_used"] for c in recovered)


# ── cache under concurrent access (no real pack needed) ──────────────────────

def test_cache_race_is_consistent(tmp_path):
    """Many threads OCR the same key at once through one shared cache dir; the
    atomic write must yield the same value with no corruption / exception."""
    from vega.ocr.base import BaseOCRBackend
    from vega.ocr.cache import CachingOCRBackend

    class SlowStub(BaseOCRBackend):
        name = "slow"
        def available_scripts(self):
            return {"eng"}
        def image_to_text(self, png, script):
            return "consistent-" + png.decode()

    cache = CachingOCRBackend(SlowStub(), tmp_path / "c")
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda _: cache.image_to_text(b"same", "eng"),
                              range(64)))
    assert set(results) == {"consistent-same"}
    assert list((tmp_path / "c").glob("*.tmp")) == []


# ── real process-pool parallelism across files ───────────────────────────────

def test_parallel_workers_match_serial(tmp_path):
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    from vega import ingest_directory

    for i in range(6):
        c = canvas.Canvas(str(tmp_path / f"doc{i}.pdf"), pagesize=LETTER)
        c.drawString(72, 700, f"Document {i} body text with enough words to chunk.")
        c.showPage(); c.save()

    serial = ingest_directory(tmp_path, ocr_mode="none", workers=1)
    parallel = ingest_directory(tmp_path, ocr_mode="none", workers=4)
    assert {c["chunk_id"] for c in serial} == {c["chunk_id"] for c in parallel}
    assert len(serial) == len(parallel) > 0


# ── corrupt input is isolated, never fatal ───────────────────────────────────

def test_corrupt_pdf_is_isolated(tmp_path):
    from vega.config import IngestConfig
    from vega.pipeline import IngestionPipeline

    good = tmp_path / "ok.pdf"
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(good), pagesize=LETTER)
    c.drawString(72, 700, "A perfectly fine document with enough words here.")
    c.showPage(); c.save()
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.5 \n garbage not a pdf \n")

    pipe = IngestionPipeline(IngestConfig(ocr_mode="none"))
    recs = pipe.ingest_directory(tmp_path)
    assert pipe.stats.files_failed == 1
    assert pipe.stats.files_parsed == 1
    assert recs                                  # the good file still produced chunks


@requires_tess
def test_real_table_extraction(tmp_path):
    """A ruled table in a born-digital PDF is captured as a structured table
    chunk (needs no OCR, but exercises the real find_tables path)."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    from reportlab.lib import colors
    from vega import ingest_file

    p = tmp_path / "table.pdf"
    doc = SimpleDocTemplate(str(p), pagesize=LETTER)
    data = [["Parameter", "Value"], ["Voltage", "230 V"],
            ["Current", "5 A"], ["Power", "1150 W"]]
    t = Table(data)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    doc.build([t])

    chunks = ingest_file(p, ocr_mode="none")
    tables = [c for c in chunks if c["metadata"].get("is_table")]
    assert tables, "expected a structured table chunk"
    assert "230 V" in tables[0]["text"]
