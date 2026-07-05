"""Phase 1 batch OCR — planner/executor units + golden equality vs the
single-page path. All engines are stubs; nothing here needs a GPU or packs.

The golden tests are the contract from docs/DESIGN-scale-ocr.md: batch mode
(`batch_ocr=True`, the default) must produce byte-identical DocumentModels to
the single-page path (`batch_ocr=False`) — same text, same metadata, same
suspect flags — for recovery successes AND failures.
"""

from typing import List

import pytest

from vega import text_recovery as tr
from vega.parsers.pdf import PDFParser

TELUGU = "పరిపాలన పరిషత్తు నుండి ఉత్తర్వు జారీ చేయబడింది"
MOJIBAKE = "Bçñüéàæþ§«ßÿîïçœ ëêÀÁÂÃ æþðøµ¶·"


# ── planner units ─────────────────────────────────────────────────────────────

def test_plan_recover_matches_recover_noop_conditions(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    # Clean text → recover() no-ops without OCR → no plan.
    assert tr.plan_recover("Perfectly clean English text here.", [],
                           b"PNG", backend=backend) is None
    # Garbled with a font hint → a plan, resolved like recover() would.
    plan = tr.plan_recover(MOJIBAKE, ["SHREE-TEL"], b"PNG", backend=backend,
                           candidate_langs=["te"])
    assert plan is not None
    assert plan.kind == "recover"
    assert plan.attempts == ["tel"]
    assert plan.verify == ["tel"]
    assert plan.original_text == MOJIBAKE
    # No backend → no plan (recover() would no-op too).
    assert tr.plan_recover(MOJIBAKE, ["SHREE-TEL"], b"PNG", backend=None) is None


def test_plan_scanned_builds_attempts_like_ocr_scanned(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel", "hin"), output=TELUGU)
    plan = tr.plan_scanned(b"PNG", backend=backend,
                           candidate_langs=["te", "hi"], declared_script=None)
    assert plan is not None and plan.kind == "scanned"
    # OSD fails on the fake PNG → attempts fall back to the multi-pack join.
    assert plan.attempts == ["tel+hin"]
    assert plan.verify == ["tel", "hin"]
    # Nothing to try (English-only backend, no declared langs) → None,
    # mirroring ocr_scanned's pre-OCR no-op.
    assert tr.plan_scanned(b"PNG", backend=make_ocr_stub(scripts=("eng",)),
                           candidate_langs=[]) is None


# ── executor units ────────────────────────────────────────────────────────────

class _BatchCountingStub:
    """Stub recording each batch call as (script, n_images)."""
    name = "stub"

    def __init__(self, scripts, output_for=None, output=""):
        self._scripts = set(scripts)
        self._output_for = output_for or {}
        self._output = output
        self.batch_calls: List[tuple] = []

    def available_scripts(self):
        return set(self._scripts)

    def can_handle(self, script):
        return all(p in self._scripts for p in script.split("+") if p)

    def cache_version(self):
        return "stub:1"

    def image_to_text(self, png, script):
        return self.image_to_text_batch([png], script)[0]

    def image_to_text_batch(self, images, script):
        self.batch_calls.append((script, len(images)))
        out = self._output_for.get(script, self._output)
        return [out for _ in images]


def test_execute_plans_groups_one_script_into_one_batch():
    backend = _BatchCountingStub(("eng", "tel"), output=TELUGU)
    plans = [tr.OCRPlan(png=b"P%d" % i, kind="recover", attempts=["tel"],
                        verify=["tel"], original_text=MOJIBAKE)
             for i in range(5)]
    recs = tr.execute_plans(plans, backend)
    assert all(r.was_recovered and r.text == TELUGU and r.script == "tel"
               for r in recs)
    assert all(r.engine == "stub" for r in recs)
    # One batched call for all five pages (bilingual co-load applied).
    assert backend.batch_calls == [("tel+eng", 5)]


def test_execute_plans_windows_large_groups(monkeypatch):
    monkeypatch.setenv("VEGA_OCR_WINDOW", "2")
    backend = _BatchCountingStub(("eng", "tel"), output=TELUGU)
    plans = [tr.OCRPlan(png=b"x", kind="recover", attempts=["tel"],
                        verify=["tel"], original_text=MOJIBAKE)
             for _ in range(5)]
    recs = tr.execute_plans(plans, backend)
    assert all(r.was_recovered for r in recs)
    assert backend.batch_calls == [("tel+eng", 2), ("tel+eng", 2), ("tel+eng", 1)]


def test_execute_plans_low_confidence_noops_to_original():
    backend = _BatchCountingStub(("eng", "tel"), output="qwerty latin junk")
    plans = [tr.OCRPlan(png=b"x", kind="recover", attempts=["tel"],
                        verify=["tel"], original_text=MOJIBAKE)]
    rec = tr.execute_plans(plans, backend)[0]
    assert rec.was_recovered is False
    assert rec.text == MOJIBAKE            # original preserved, like recover()


def test_execute_plans_scanned_retries_next_attempt():
    # First attempt (tel) yields junk; the multi-pack retry (tel+hin) succeeds.
    backend = _BatchCountingStub(
        ("eng", "tel", "hin"),
        output_for={"tel+eng": "zzz junk", "tel+hin+eng": TELUGU})
    plans = [tr.OCRPlan(png=b"x", kind="scanned",
                        attempts=["tel", "tel+hin"], verify=["tel", "hin"])]
    rec = tr.execute_plans(plans, backend)[0]
    assert rec.was_recovered is True
    assert rec.text == TELUGU
    assert [c[0] for c in backend.batch_calls] == ["tel+eng", "tel+hin+eng"]


def test_execute_plans_scanned_all_low_confidence_noops():
    backend = _BatchCountingStub(("eng", "tel"), output="zzz junk")
    plans = [tr.OCRPlan(png=b"x", kind="scanned", attempts=["tel"],
                        verify=["tel"])]
    rec = tr.execute_plans(plans, backend)[0]
    assert rec.was_recovered is False and rec.text == ""


# ── golden: batch parse == single-page parse ─────────────────────────────────

@pytest.fixture
def mojibake_pdf(tmp_path):
    """Three born-digital pages whose text layer is Latin-1 glyph mojibake —
    every page trips is_garbled() via the accented-density signal."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    p = tmp_path / "mojibake.pdf"
    c = canvas.Canvas(str(p), pagesize=LETTER)
    for i in range(3):
        y = 700
        for _ in range(6):
            c.drawString(72, y, MOJIBAKE + f" page {i}")
            y -= 24
        c.showPage()
    c.save()
    return p


def _model_fingerprint(model):
    return (
        [(e.type, e.text, e.page) for e in model.elements],
        {k: v for k, v in model.metadata.items()},
    )


def _parse_both_ways(pdf_path, make_backend):
    m_batch = PDFParser(ocr_backend=make_backend(), candidate_langs=["te"],
                        batch_ocr=True).parse(pdf_path)
    m_single = PDFParser(ocr_backend=make_backend(), candidate_langs=["te"],
                         batch_ocr=False).parse(pdf_path)
    return m_batch, m_single


def test_golden_batch_equals_single_on_recovery_success(mojibake_pdf, make_ocr_stub):
    m_batch, m_single = _parse_both_ways(
        mojibake_pdf, lambda: make_ocr_stub(scripts=("eng", "tel"), output=TELUGU))
    assert _model_fingerprint(m_batch) == _model_fingerprint(m_single)
    assert m_batch.metadata["ocr_pages"] == [1, 2, 3]
    assert m_batch.metadata["ocr_page_engines"] == {1: "stub", 2: "stub", 3: "stub"}
    assert all(TELUGU in e.text for e in m_batch.elements)


def test_golden_batch_equals_single_on_recovery_failure(mojibake_pdf, make_ocr_stub):
    # OCR yields junk → both modes must keep the mojibake + set suspect pages.
    m_batch, m_single = _parse_both_ways(
        mojibake_pdf,
        lambda: make_ocr_stub(scripts=("eng", "tel"), output="zzz junk"))
    assert _model_fingerprint(m_batch) == _model_fingerprint(m_single)
    assert m_batch.metadata["ocr_pages"] == []
    assert m_batch.metadata["garble_suspect_pages"] == [1, 2, 3]


def test_golden_batch_equals_single_on_scanned_pages(scanned_pdf, make_ocr_stub):
    # eng-only stub: plan_scanned yields None → inline English fallback in both.
    m_batch, m_single = _parse_both_ways(
        scanned_pdf, lambda: make_ocr_stub(scripts=("eng",), output="scan text"))
    assert _model_fingerprint(m_batch) == _model_fingerprint(m_single)
    assert m_batch.metadata["ocr_pages"] == [1]


def test_no_batch_ocr_flag_reaches_parser(monkeypatch):
    from vega.config import IngestConfig
    from vega.pipeline import IngestionPipeline
    pipe = IngestionPipeline(IngestConfig(ocr_mode="none", batch_ocr=False))
    parser = pipe._parser_for(__import__("pathlib").Path("x.pdf"))
    assert parser._batch_ocr is False


# ── tesseract CPU batch threading ─────────────────────────────────────────────
# The deferred window is serial by default (BaseOCRBackend); TesseractBackend
# overrides it with a thread pool — every page is its own subprocess, so
# threads parallelize the window across cores (~5× measured on real pages).

def _thread_recording_ocr(record):
    import threading
    import time

    def fake(self, image_png, script):
        time.sleep(0.02)              # widen the overlap window
        record.append(threading.get_ident())
        return f"text:{image_png.decode()}"
    return fake


def test_tesseract_batch_preserves_order_and_parallelizes(monkeypatch):
    from vega.ocr.tesseract import TesseractBackend
    threads: List[int] = []
    monkeypatch.setattr(TesseractBackend, "image_to_text",
                        _thread_recording_ocr(threads))
    imgs = [f"p{i}".encode() for i in range(6)]
    out = TesseractBackend().image_to_text_batch(imgs, "tam")
    # Positional contract: output i belongs to image i, exactly as serial.
    assert out == [f"text:p{i}" for i in range(6)]
    assert len(set(threads)) > 1


def test_tesseract_batch_env_forces_serial(monkeypatch):
    from vega.ocr.tesseract import TesseractBackend
    monkeypatch.setenv("VEGA_CPU_OCR_THREADS", "1")
    threads: List[int] = []
    monkeypatch.setattr(TesseractBackend, "image_to_text",
                        _thread_recording_ocr(threads))
    out = TesseractBackend().image_to_text_batch(
        [b"a", b"b", b"c"], "tam")
    assert out == ["text:a", "text:b", "text:c"]
    assert len(set(threads)) == 1     # the caller's thread, no pool


def test_cpu_ocr_threads_precedence(monkeypatch):
    # config.resolve_* owns the rule: explicit > env > auto default.
    from vega import config as c
    monkeypatch.setenv("VEGA_CPU_OCR_THREADS", "3")
    assert c.resolve_cpu_ocr_threads() == 3
    assert c.resolve_cpu_ocr_threads(explicit=5) == 5      # explicit wins
    monkeypatch.setenv("VEGA_CPU_OCR_THREADS", "not-a-number")
    assert 1 <= c.resolve_cpu_ocr_threads() <= c.MAX_CPU_OCR_THREADS
    monkeypatch.delenv("VEGA_CPU_OCR_THREADS")
    assert 1 <= c.resolve_cpu_ocr_threads() <= c.MAX_CPU_OCR_THREADS


def test_ocr_window_precedence(monkeypatch):
    from vega import config as c
    monkeypatch.delenv("VEGA_OCR_WINDOW", raising=False)
    assert c.resolve_ocr_window() == c.DEFAULT_OCR_WINDOW
    monkeypatch.setenv("VEGA_OCR_WINDOW", "4")
    assert c.resolve_ocr_window() == 4
    assert c.resolve_ocr_window(explicit=9) == 9           # explicit wins


def test_tesseract_batch_constructor_knob_wins_env(monkeypatch):
    from vega.ocr.tesseract import TesseractBackend
    import threading
    monkeypatch.setenv("VEGA_CPU_OCR_THREADS", "8")
    threads: List[int] = []

    def fake(self, image_png, script):
        threads.append(threading.get_ident())
        return "x"
    monkeypatch.setattr(TesseractBackend, "image_to_text", fake)
    TesseractBackend(batch_threads=1).image_to_text_batch(
        [b"a", b"b", b"c"], "tam")
    assert len(set(threads)) == 1     # explicit 1 beat the env var


def test_tesseract_batch_caps_omp_and_restores(monkeypatch):
    import os
    from vega.ocr.tesseract import TesseractBackend
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)
    seen: List[str] = []

    def fake(self, image_png, script):
        seen.append(os.environ.get("OMP_THREAD_LIMIT"))
        return "x"
    monkeypatch.setattr(TesseractBackend, "image_to_text", fake)
    TesseractBackend().image_to_text_batch([b"a", b"b", b"c", b"d"], "tam")
    assert set(seen) == {"1"}                       # capped during the pool
    assert "OMP_THREAD_LIMIT" not in os.environ     # restored after

    # An explicit user setting wins and survives.
    monkeypatch.setenv("OMP_THREAD_LIMIT", "4")
    seen.clear()
    TesseractBackend().image_to_text_batch([b"a", b"b"], "tam")
    assert set(seen) == {"4"}
    assert os.environ["OMP_THREAD_LIMIT"] == "4"


def test_resolvers_clamp_uniformly(monkeypatch):
    # Every knob clamps to >=1 the same way, whether explicit or env-sourced;
    # the GPU pair still passes None through (backend auto-sizes from VRAM).
    from vega import config as c
    for name in ("VEGA_OCR_WINDOW", "VEGA_CPU_OCR_THREADS",
                  "VEGA_GPU_BATCH", "VEGA_GPU_DET_BATCH"):
        monkeypatch.delenv(name, raising=False)
    assert c.resolve_ocr_window(explicit=-5) == 1
    assert c.resolve_cpu_ocr_threads(explicit=0) == 1
    assert c.resolve_gpu_batch(explicit=-5) == 1
    assert c.resolve_gpu_det_batch(explicit=0) == 1
    assert c.resolve_gpu_batch() is None
    assert c.resolve_gpu_det_batch() is None
    monkeypatch.setenv("VEGA_GPU_BATCH", "-3")
    assert c.resolve_gpu_batch() == 1
