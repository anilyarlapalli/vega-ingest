"""Pluggable OCR backend selection, GPU auto-detection, fallback + cache.

Nothing here imports torch/easyocr or touches a real pack — the GPU path is
exercised by monkeypatching the CUDA probe and the importability check.
"""

from vega.ocr import selection
from vega.ocr.cache import CachingOCRBackend
from vega.ocr.selection import (
    FallbackOCRBackend,
    gpu_available,
    select_backend,
)
from vega.ocr.tesseract import TesseractBackend


def test_mode_none_disables_ocr():
    assert select_backend("none") is None


def test_mode_tesseract_builds_cpu_backend():
    b = select_backend("tesseract")
    assert isinstance(b, TesseractBackend)
    assert b.name == "tesseract"


def test_auto_without_gpu_picks_tesseract(monkeypatch):
    monkeypatch.setattr(selection, "gpu_available", lambda: False)
    b = select_backend("auto")
    assert isinstance(b, TesseractBackend)


def test_auto_with_gpu_and_easyocr_composes_fallback(monkeypatch):
    monkeypatch.setattr(selection, "gpu_available", lambda: True)
    monkeypatch.setattr(selection, "_easyocr_importable", lambda: True)
    b = select_backend("auto")
    assert isinstance(b, FallbackOCRBackend)
    assert b.name == "fallback"
    # Union of EasyOCR's neural scripts and Tesseract's host packs.
    scripts = b.available_scripts()
    assert "tel" in scripts               # both backends know Telugu


def test_explicit_easyocr_falls_back_when_absent(monkeypatch):
    monkeypatch.setattr(selection, "_easyocr_importable", lambda: False)
    b = select_backend("easyocr")
    assert isinstance(b, TesseractBackend)   # graceful degrade, no raise


def test_auto_gpu_flag_but_no_easyocr_degrades(monkeypatch):
    monkeypatch.setattr(selection, "_easyocr_importable", lambda: False)
    b = select_backend("auto", gpu=True)     # forced GPU, but no neural backend
    assert isinstance(b, TesseractBackend)


def test_cache_dir_wraps_backend(tmp_path):
    b = select_backend("tesseract", cache_dir=str(tmp_path / "ocr"))
    assert isinstance(b, CachingOCRBackend)
    assert b.name == "tesseract"             # name delegates to inner backend


def test_gpu_available_is_false_without_torch():
    # No torch / no CUDA in the test env → probe must answer False, not raise.
    assert gpu_available() is False


def test_fallback_routes_each_script_to_a_capable_backend(make_ocr_stub):
    neural = make_ocr_stub(scripts=("eng", "tel"), output="neural", name="neural")
    tess = make_ocr_stub(scripts=("eng", "mal", "guj"), output="tess", name="tess")
    fb = FallbackOCRBackend([neural, tess])

    assert fb.available_scripts() == {"eng", "tel", "mal", "guj"}
    assert fb.image_to_text(b"png", "tel") == "neural"   # only neural has tel
    assert fb.image_to_text(b"png", "mal") == "tess"     # only tess has mal
    # An unknown script routes to *some* backend rather than crashing.
    assert fb.image_to_text(b"png", "xyz") in ("neural", "tess", "")


def test_caching_backend_hits_disk_on_repeat(make_ocr_stub, tmp_path):
    inner = make_ocr_stub(scripts=("eng",), output="cached-value")
    cache = CachingOCRBackend(inner, tmp_path / "cache")

    first = cache.image_to_text(b"IDENTICAL-BYTES", "eng")
    second = cache.image_to_text(b"IDENTICAL-BYTES", "eng")
    assert first == second == "cached-value"
    assert len(inner.calls) == 1             # second call served from disk
