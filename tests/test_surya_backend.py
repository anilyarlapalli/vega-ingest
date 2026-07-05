"""SuryaBackend unit tests — no real surya/torch import, no model download.

The backend is exercised through fakes: predictor construction failure is
simulated via sys.modules stubs, inference via hand-built predictor doubles.
"""

import io
import sys
import types
from types import SimpleNamespace

from vega.ocr.surya_backend import SuryaBackend


def _tiny_png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buf, format="PNG")
    return buf.getvalue()


def _result(*line_texts):
    return SimpleNamespace(
        text_lines=[SimpleNamespace(text=t) for t in line_texts])


def test_advertises_every_vega_script():
    scripts = SuryaBackend().available_scripts()
    # Surya covers the packs EasyOCR lacks — that is its selling point here.
    for code in ("eng", "hin", "mar", "tam", "tel", "kan",
                 "mal", "ben", "asm", "guj", "pan", "ori"):
        assert code in scripts


def test_can_handle_multi_nonlatin_combo():
    # Language-agnostic model: tel+hin in one pass (EasyOCR must decline this).
    b = SuryaBackend()
    assert b.can_handle("tel+hin") is True
    assert b.can_handle("tam+eng") is True
    assert b.can_handle("xyz") is False


def test_result_text_strips_formatting_tags():
    r = _result("<b>தமிழ்</b> text", "line<br/>break", "<math>x^2</math>", "")
    assert SuryaBackend._result_text(r) == "தமிழ் text\nline\nbreak\nx^2"


def test_construction_failure_is_negative_cached(monkeypatch):
    calls = {"n": 0}

    class BoomModule(types.ModuleType):
        def __getattr__(self, name):
            calls["n"] += 1
            raise RuntimeError("weights unavailable")

    monkeypatch.setitem(sys.modules, "surya", types.ModuleType("surya"))
    for sub in ("surya.detection", "surya.foundation", "surya.recognition"):
        monkeypatch.setitem(sys.modules, sub, BoomModule(sub))

    b = SuryaBackend()
    assert b.image_to_text(_tiny_png(), "tam") == ""
    first = calls["n"]
    assert first > 0
    # Second call must short-circuit — no re-attempted model construction.
    assert b.image_to_text(_tiny_png(), "tam") == ""
    assert calls["n"] == first


def test_batch_uses_one_inference_call_and_keeps_length():
    seen = {"batches": 0}

    def fake_recognition(pils, **kwargs):
        seen["batches"] += 1
        assert kwargs.get("det_predictor") is fake_detection
        return [_result(f"page-{i}") for i in range(len(pils))]

    fake_detection = object()
    b = SuryaBackend()
    b._predictors = (fake_recognition, fake_detection)

    out = b.image_to_text_batch([_tiny_png()] * 3, "tel")
    assert out == ["page-0", "page-1", "page-2"]
    assert seen["batches"] == 1


def test_inference_error_yields_empty_strings_not_raise():
    def broken_recognition(pils, **kwargs):
        raise RuntimeError("CUDA OOM")

    b = SuryaBackend()
    b._predictors = (broken_recognition, object())
    assert b.image_to_text_batch([_tiny_png()] * 2, "kan") == ["", ""]


def test_concurrent_build_constructs_predictors_exactly_once(monkeypatch):
    # --page-workers races _build from several threads; model construction must
    # be serialized (transformers meta-device init is process-global) and run
    # exactly once. This is the real-world "Cannot copy out of meta tensor"
    # failure mode observed with 3 page workers.
    import threading
    import time

    built = {"n": 0}

    class FakePredictorModule(types.ModuleType):
        def __getattr__(self, name):
            def _ctor(*a, **kw):
                if name == "FoundationPredictor":
                    built["n"] += 1
                    time.sleep(0.02)      # widen the race window
                return object()
            return _ctor

    monkeypatch.setitem(sys.modules, "surya", types.ModuleType("surya"))
    for sub in ("surya.detection", "surya.foundation", "surya.recognition"):
        monkeypatch.setitem(sys.modules, sub, FakePredictorModule(sub))

    b = SuryaBackend()
    threads = [threading.Thread(target=b._build) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert built["n"] == 1
    assert b._predictors is not None


def test_forced_cpu_omits_gpu_batch_size(monkeypatch):
    monkeypatch.setenv("VEGA_GPU_BATCH", "32")
    captured = {}

    def fake_recognition(pils, **kwargs):
        captured.update(kwargs)
        return [_result("x") for _ in pils]

    b = SuryaBackend(gpu=False)
    b._predictors = (fake_recognition, object())
    b.image_to_text(_tiny_png(), "hin")
    assert "recognition_batch_size" not in captured

    b2 = SuryaBackend()
    b2._predictors = (fake_recognition, object())
    b2.image_to_text(_tiny_png(), "hin")
    assert captured.get("recognition_batch_size") == 32


# ── Phase 2: VRAM-aware / configurable recognition batch size ────────────────

def test_batch_size_env_override_wins(monkeypatch):
    from vega.ocr import surya_backend as sb
    monkeypatch.setenv("VEGA_GPU_BATCH", "512")
    assert sb._resolve_recognition_batch() == 512
    monkeypatch.setenv("VEGA_GPU_BATCH", "not-a-number")   # ignored, no raise
    assert sb._resolve_recognition_batch() in (32, None)


def test_batch_size_auto_probes_vram(monkeypatch):
    from vega.ocr import surya_backend as sb
    monkeypatch.delenv("VEGA_GPU_BATCH", raising=False)

    class FakeProps:
        def __init__(self, total):
            self.total_memory = total

    class FakeCuda:
        def __init__(self, total):
            self._t = total
        def is_available(self):
            return True
        def get_device_properties(self, i):
            return FakeProps(self._t)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = FakeCuda(4 * 1024 ** 3)              # 4 GB card
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert sb._resolve_recognition_batch() == 32           # guardrail

    fake_torch.cuda = FakeCuda(96 * 1024 ** 3)             # 96 GB card
    assert sb._resolve_recognition_batch() is None         # surya's default


# ── Phase 0.2: one poison page must not blank a batch window ────────────────

def test_batch_failure_retries_pages_individually():
    calls = {"n": 0}

    def fake_recognition(pils, **kwargs):
        calls["n"] += 1
        if len(pils) > 1:
            raise RuntimeError("poison image in batch")
        if calls["n"] == 3:                  # second single-page call fails
            raise RuntimeError("this page really is broken")
        return [_result("ok") for _ in pils]

    b = SuryaBackend(gpu=False)
    b._predictors = (fake_recognition, object())
    out = b.image_to_text_batch([_tiny_png()] * 3, "tel")
    assert out == ["ok", "", "ok"]           # one page lost, not the window
    assert calls["n"] == 4                   # 1 batch attempt + 3 singles


def test_single_image_failure_is_not_retried():
    calls = {"n": 0}

    def broken_recognition(pils, **kwargs):
        calls["n"] += 1
        raise RuntimeError("boom")

    b = SuryaBackend(gpu=False)
    b._predictors = (broken_recognition, object())
    assert b.image_to_text(_tiny_png(), "kan") == ""
    assert calls["n"] == 1                   # no pointless double inference
