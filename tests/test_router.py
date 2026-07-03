"""Format router — extension → parser, support gate, OCR wiring."""

from pathlib import Path

import pytest

from vega.parsers.image import ImageParser
from vega.parsers.pdf import PDFParser
from vega.parsers.text import TextParser
from vega.router import (
    CORE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    TEXT_EXTENSIONS,
    get_parser,
    is_supported,
)


def test_txt_is_an_explicit_extra_not_core():
    # Finding 21: .txt is a documented convenience extra, outside the core
    # PDF+image scope, but still handled.
    assert ".txt" not in CORE_EXTENSIONS
    assert ".txt" in TEXT_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert isinstance(get_parser(Path("notes.txt")), TextParser)


def test_all_contract_image_extensions_supported():
    # The acceptance contract's image set, exactly.
    want = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
    assert want <= set(IMAGE_EXTENSIONS)
    assert ".pdf" in SUPPORTED_EXTENSIONS
    for ext in want | {".pdf"}:
        assert is_supported(Path(f"x{ext}"))


def test_is_supported_is_case_insensitive():
    assert is_supported(Path("REPORT.PDF"))
    assert is_supported(Path("scan.JPG"))
    assert not is_supported(Path("notes.docx"))
    assert not is_supported(Path("archive.zip"))


@pytest.mark.parametrize("name,cls", [
    ("a.pdf", PDFParser),
    ("a.png", ImageParser),
    ("a.jpeg", ImageParser),
    ("a.tiff", ImageParser),
    ("a.webp", ImageParser),
    ("a.txt", TextParser),
])
def test_get_parser_dispatches_by_extension(name, cls):
    assert isinstance(get_parser(Path(name)), cls)


def test_get_parser_unsupported_returns_none():
    assert get_parser(Path("thing.docx")) is None


def test_get_parser_threads_ocr_backend_and_language_routing(stub_backend):
    p = get_parser(
        Path("go.pdf"), ocr_backend=stub_backend,
        recovery_script="tel", candidate_langs=["te", "en"],
        figure_ocr=True, dpi=250, scanned_dpi=150,
    )
    assert isinstance(p, PDFParser)
    assert p._backend is stub_backend
    assert p._recovery_script == "tel"
    assert p._candidate_langs == ["te", "en"]
    assert p._figure_ocr is True
    assert p._dpi == 250 and p._scanned_dpi == 150


def test_get_parser_image_carries_backend_and_script(stub_backend):
    p = get_parser(Path("scan.png"), ocr_backend=stub_backend,
                   recovery_script="hin", candidate_langs=["hi"])
    assert isinstance(p, ImageParser)
    assert p._backend is stub_backend
    assert p._recovery_script == "hin"
