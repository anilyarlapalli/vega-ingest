"""Pluggable OCR backend selection, GPU auto-detection, fallback + cache.

Nothing here imports torch/easyocr or touches a real pack — the GPU path is
exercised by monkeypatching the CUDA probe and the importability check.
"""

import pytest

from vega.ocr import selection
from vega.ocr.base import BaseOCRBackend
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


def test_unknown_ocr_mode_raises(monkeypatch):
    # Finding 20: an unknown mode must fail loudly, not silently become "auto".
    with pytest.raises(ValueError):
        select_backend("banana")


def test_fallback_tries_next_backend_when_first_is_empty(make_ocr_stub):
    # Finding 6: a capable backend returning "" must not strand the page.
    empty = make_ocr_stub(scripts=("eng", "tel"), output="", name="neural")
    tess = make_ocr_stub(scripts=("eng", "tel"), output="from-tesseract", name="tess")
    fb = FallbackOCRBackend([empty, tess])
    assert fb.image_to_text(b"png", "tel") == "from-tesseract"


def test_fallback_routes_multi_nonlatin_combo_away_from_easyocr(make_ocr_stub):
    # Finding 7: EasyOCR can't do two non-Latin scripts in one call; the router
    # must send tel+hin to a backend that can_handle it (Tesseract).
    from vega.ocr.easyocr_backend import EasyOCRBackend
    easy = EasyOCRBackend()                       # can_handle("tel+hin") is False
    tess = make_ocr_stub(scripts=("eng", "tel", "hin"), output="combo", name="tess")
    fb = FallbackOCRBackend([easy, tess])
    assert fb.image_to_text(b"png", "tel+hin") == "combo"
    assert easy.can_handle("tel+hin") is False
    assert easy.can_handle("tel+eng") is True


# ── cache correctness (findings 1, 17, 18, 19) ───────────────────────────────

def test_cache_write_is_atomic_no_tmp_left(make_ocr_stub, tmp_path):
    inner = make_ocr_stub(scripts=("eng",), output="value")
    cache = CachingOCRBackend(inner, tmp_path / "c")
    cache.image_to_text(b"BYTES", "eng")
    leftovers = list((tmp_path / "c").glob("*.tmp")) + list((tmp_path / "c").glob(".*tmp*"))
    assert leftovers == []                        # temp file was atomically renamed


def test_cache_filenames_are_sanitized(make_ocr_stub, tmp_path):
    inner = make_ocr_stub(scripts=("eng",), output="v", name="weird/name")
    cache = CachingOCRBackend(inner, tmp_path / "c")
    # A script string with path/relative characters must not escape the dir:
    # the key is a single sanitized filename component under the cache dir.
    p = cache._path(b"x", "tel+eng/../../etc")
    import os
    assert os.sep not in p.name
    assert p.resolve().parent == (tmp_path / "c").resolve()   # no traversal


def test_cache_version_bump_invalidates(make_ocr_stub, tmp_path):
    # Finding 18: an engine/pack upgrade must not serve stale cached text.
    d = tmp_path / "c"
    a = make_ocr_stub(scripts=("eng",), output="old")
    a.cache_version = lambda: "engine:1"          # type: ignore[assignment]
    ca = CachingOCRBackend(a, d)
    assert ca.image_to_text(b"IMG", "eng") == "old"

    b = make_ocr_stub(scripts=("eng",), output="new")
    b.cache_version = lambda: "engine:2"          # type: ignore[assignment]
    cb = CachingOCRBackend(b, d)
    assert cb.image_to_text(b"IMG", "eng") == "new"   # different key → recompute


def test_cache_batches_all_misses_in_one_call(tmp_path):
    # Finding 19: the cache wrapper must not serialize batch OCR.
    class BatchStub(BaseOCRBackend):
        name = "batch"
        def __init__(self):
            self.batch_calls = 0
        def available_scripts(self):
            return {"eng"}
        def image_to_text(self, png, script):
            return png.decode()
        def image_to_text_batch(self, images, script):
            self.batch_calls += 1
            return [im.decode() for im in images]

    inner = BatchStub()
    cache = CachingOCRBackend(inner, tmp_path / "c")
    out = cache.image_to_text_batch([b"a", b"b", b"c"], "eng")
    assert out == ["a", "b", "c"]
    assert inner.batch_calls == 1                 # all three misses → one batch
    # Repeat: fully cached, backend not touched again.
    assert cache.image_to_text_batch([b"a", b"b", b"c"], "eng") == ["a", "b", "c"]
    assert inner.batch_calls == 1
