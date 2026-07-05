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
    monkeypatch.setattr(selection, "_surya_importable", lambda: False)
    b = select_backend("auto")
    assert isinstance(b, FallbackOCRBackend)
    assert b.name == "fallback"
    # Union of EasyOCR's neural scripts and Tesseract's host packs.
    scripts = b.available_scripts()
    assert "tel" in scripts               # both backends know Telugu


def test_auto_with_gpu_prefers_surya_first(monkeypatch):
    from vega.ocr.easyocr_backend import EasyOCRBackend
    from vega.ocr.surya_backend import SuryaBackend
    monkeypatch.setattr(selection, "gpu_available", lambda: True)
    monkeypatch.setattr(selection, "_easyocr_importable", lambda: True)
    monkeypatch.setattr(selection, "_surya_importable", lambda: True)
    b = select_backend("auto")
    assert isinstance(b, FallbackOCRBackend)
    names = [type(inner) for inner in b._backends]
    assert names == [SuryaBackend, EasyOCRBackend, TesseractBackend]


def test_auto_order_is_driven_by_neural_preference_tuple(monkeypatch):
    # Reordering NEURAL_PREFERENCE must be the ONLY change needed to flip the
    # auto-mode engine priority.
    from vega.ocr.easyocr_backend import EasyOCRBackend
    from vega.ocr.surya_backend import SuryaBackend
    monkeypatch.setattr(selection, "gpu_available", lambda: True)
    monkeypatch.setattr(selection, "_easyocr_importable", lambda: True)
    monkeypatch.setattr(selection, "_surya_importable", lambda: True)
    monkeypatch.setattr(selection, "NEURAL_PREFERENCE", ("easyocr", "surya"))
    b = select_backend("auto")
    assert [type(inner) for inner in b._backends] == [
        EasyOCRBackend, SuryaBackend, TesseractBackend]


def test_auto_with_gpu_and_only_surya(monkeypatch):
    from vega.ocr.surya_backend import SuryaBackend
    monkeypatch.setattr(selection, "gpu_available", lambda: True)
    monkeypatch.setattr(selection, "_easyocr_importable", lambda: False)
    monkeypatch.setattr(selection, "_surya_importable", lambda: True)
    b = select_backend("auto")
    assert isinstance(b, FallbackOCRBackend)
    assert isinstance(b._backends[0], SuryaBackend)


def test_explicit_easyocr_falls_back_when_absent(monkeypatch):
    monkeypatch.setattr(selection, "_easyocr_importable", lambda: False)
    b = select_backend("easyocr")
    assert isinstance(b, TesseractBackend)   # graceful degrade, no raise


def test_explicit_surya_falls_back_when_absent(monkeypatch):
    monkeypatch.setattr(selection, "_surya_importable", lambda: False)
    b = select_backend("surya")
    assert isinstance(b, TesseractBackend)   # graceful degrade, no raise


def test_mode_surya_builds_backend_when_importable(monkeypatch):
    from vega.ocr.surya_backend import SuryaBackend
    monkeypatch.setattr(selection, "_surya_importable", lambda: True)
    b = select_backend("surya")              # construction is lazy — no import
    assert isinstance(b, SuryaBackend)
    assert b.name == "surya"


def test_auto_gpu_flag_but_no_neural_backend_degrades(monkeypatch):
    monkeypatch.setattr(selection, "_easyocr_importable", lambda: False)
    monkeypatch.setattr(selection, "_surya_importable", lambda: False)
    b = select_backend("auto", gpu=True)     # forced GPU, but no neural backend
    assert isinstance(b, TesseractBackend)


def test_cache_dir_wraps_backend(tmp_path):
    b = select_backend("tesseract", cache_dir=str(tmp_path / "ocr"))
    assert isinstance(b, CachingOCRBackend)
    assert b.name == "tesseract"             # name delegates to inner backend


def test_gpu_available_is_false_without_torch(monkeypatch):
    # With torch absent the probe must answer False, not raise. (Simulated —
    # the dev machine may well have a CUDA torch installed.)
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)   # import torch → ImportError
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


def test_easyocr_concurrent_reader_construction_is_single(monkeypatch):
    # Page workers racing _reader must not build the same Reader N times
    # (N× VRAM + thread-unsafe torch model loading).
    import sys
    import threading
    import time
    import types
    from vega.ocr.easyocr_backend import EasyOCRBackend

    built = {"n": 0}
    fake = types.ModuleType("easyocr")

    def _reader_ctor(langs, gpu=False):
        built["n"] += 1
        time.sleep(0.02)                 # widen the race window
        return object()
    fake.Reader = _reader_ctor
    monkeypatch.setitem(sys.modules, "easyocr", fake)

    b = EasyOCRBackend(gpu=False)
    threads = [threading.Thread(target=b._reader, args=(["kn", "en"],))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert built["n"] == 1
    assert ("kn", "en") in b._readers


# ── per-page engine attribution + composite cache identity ──────────────────

def test_fallback_attributes_winning_engine(make_ocr_stub):
    empty = make_ocr_stub(scripts=("eng", "tel"), output="", name="neural")
    tess = make_ocr_stub(scripts=("eng", "tel"), output="text", name="tess")
    fb = FallbackOCRBackend([empty, tess])
    out, engine = fb.image_to_text_attributed(b"png", "tel")
    assert (out, engine) == ("text", "tess")     # winner's name, not "fallback"
    out2, engine2 = fb.image_to_text_attributed(b"png", "xyz")
    if not out2:
        assert engine2 is None                   # empty result → no attribution


def test_fallback_cache_version_folds_members_in_order(make_ocr_stub):
    a = make_ocr_stub(scripts=("eng",), name="a")
    b = make_ocr_stub(scripts=("eng",), name="b")
    a.cache_version = lambda: "a:1"              # type: ignore[assignment]
    b.cache_version = lambda: "b:1"              # type: ignore[assignment]
    v_ab = FallbackOCRBackend([a, b]).cache_version()
    v_ba = FallbackOCRBackend([b, a]).cache_version()
    assert "a:1" in v_ab and "b:1" in v_ab
    # Reordering engines changes the fingerprint → old cached text is not
    # served to a composite whose winner may differ.
    assert v_ab != v_ba


def test_cache_preserves_engine_attribution(make_ocr_stub, tmp_path):
    inner = make_ocr_stub(scripts=("eng",), output="hello", name="stub")
    cache = CachingOCRBackend(inner, tmp_path / "c")
    out, engine = cache.image_to_text_attributed(b"IMG", "eng")
    assert (out, engine) == ("hello", "stub")    # miss: attributed + persisted
    out, engine = cache.image_to_text_attributed(b"IMG", "eng")
    assert (out, engine) == ("hello", "stub")    # hit: engine from sidecar
    assert len(inner.calls) == 1
    # A pre-attribution cache entry (no sidecar) still hits, attributing None.
    for sidecar in (tmp_path / "c").rglob("*.engine"):
        sidecar.unlink()
    out, engine = cache.image_to_text_attributed(b"IMG", "eng")
    assert (out, engine) == ("hello", None)
    assert len(inner.calls) == 1                 # still served from disk


# ── cache correctness (findings 1, 17, 18, 19) ───────────────────────────────

def test_cache_never_persists_empty_results(make_ocr_stub, tmp_path):
    # Phase 0.1: "" usually means a transient failure — caching it would
    # poison the page until manual deletion (observed on a real kannada run).
    inner = make_ocr_stub(scripts=("eng",), output="")
    cache = CachingOCRBackend(inner, tmp_path / "c")
    assert cache.image_to_text(b"IMG", "eng") == ""
    assert cache.image_to_text(b"IMG", "eng") == ""
    assert len(inner.calls) == 2                  # re-tried, not served from disk
    assert list((tmp_path / "c").rglob("*.txt")) == []

    # Once the backend recovers, the non-empty result IS cached.
    inner._output = "recovered now"
    assert cache.image_to_text(b"IMG", "eng") == "recovered now"
    assert cache.image_to_text(b"IMG", "eng") == "recovered now"
    assert len(inner.calls) == 3                  # last call was a disk hit


def test_cache_batch_never_persists_empty_results(make_ocr_stub, tmp_path):
    inner = make_ocr_stub(scripts=("eng",), output="")
    cache = CachingOCRBackend(inner, tmp_path / "c")
    assert cache.image_to_text_batch([b"a", b"b"], "eng") == ["", ""]
    assert list((tmp_path / "c").rglob("*.txt")) == []
    assert cache.image_to_text_batch([b"a", b"b"], "eng") == ["", ""]
    assert len(inner.calls) == 4                  # both misses re-tried


def test_cache_write_is_atomic_no_tmp_left(make_ocr_stub, tmp_path):
    inner = make_ocr_stub(scripts=("eng",), output="value")
    cache = CachingOCRBackend(inner, tmp_path / "c")
    cache.image_to_text(b"BYTES", "eng")
    leftovers = list((tmp_path / "c").rglob("*.tmp")) + list((tmp_path / "c").rglob(".*tmp*"))
    assert leftovers == []                        # temp file was atomically renamed


def test_cache_filenames_are_sanitized(make_ocr_stub, tmp_path):
    inner = make_ocr_stub(scripts=("eng",), output="v", name="weird/name")
    cache = CachingOCRBackend(inner, tmp_path / "c")
    # A script string with path/relative characters must not escape the dir:
    # the key is a single sanitized filename component under a two-hex shard
    # subdirectory of the cache dir.
    p = cache._path(b"x", "tel+eng/../../etc")
    import os
    import re
    assert os.sep not in p.name
    assert re.fullmatch(r"[0-9a-f]{2}", p.parent.name)        # shard component
    assert p.resolve().parent.parent == (tmp_path / "c").resolve()  # no traversal


def test_cache_entries_shard_into_hash_subdirs(make_ocr_stub, tmp_path):
    # Phase 4: ~10^5+ entries must not pile into one flat directory.
    inner = make_ocr_stub(scripts=("eng",), output="text")
    cache = CachingOCRBackend(inner, tmp_path / "c")
    for i in range(24):
        cache.image_to_text(b"IMG-%d" % i, "eng")
    shards = [d for d in (tmp_path / "c").iterdir() if d.is_dir()]
    assert len(shards) > 1                                    # keys spread
    assert all(len(d.name) == 2 for d in shards)
    assert len(list((tmp_path / "c").rglob("*.txt"))) == 24
    # And the round trip still hits.
    assert cache.image_to_text(b"IMG-0", "eng") == "text"
    assert len(inner.calls) == 24


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
