"""Surya backend — GPU-capable neural OCR (surya-ocr 0.17.x, in-process torch).

Surya's foundation model is **language-agnostic**: one recognition pass reads
any supported script (90+ languages), so unlike EasyOCR there is no
one-non-Latin-script-per-reader limit — ``tel+hin`` style combos are fine.
The predictors are heavy to construct (detection + foundation nets) and are
built lazily and cached — importing this module never imports ``surya`` or
``torch``, so it stays import-safe on hosts without them.

Construction failures are negative-cached: if the predictors cannot be built
(missing weights, OOM, broken install) we warn **once** and short-circuit every
later call instead of re-paying the failed model load per page (the lesson from
EasyOCR's broken Tamil checkpoint, where each page silently re-loaded the
detector before failing).
"""

from __future__ import annotations

import io
import logging
import re
import threading
from typing import List, Optional, Set

from vega.config import resolve_gpu_batch, resolve_gpu_det_batch
from vega.ocr.base import BaseOCRBackend

logger = logging.getLogger("vega.ocr.surya_backend")

# Tesseract script codes Surya's multilingual model covers (all of vega's
# languages — Surya reads Malayalam/Gujarati/Gurmukhi/Odia, which EasyOCR lacks).
_SCRIPTS = frozenset({
    "eng", "hin", "mar", "tam", "tel", "kan", "mal",
    "ben", "asm", "guj", "pan", "ori",
})

# Surya emits light formatting tags inside line text (<b>, <i>, <math>…). The
# OCRBackend contract is plain text, so they are stripped; <br> becomes \n.
_TAG_RE = re.compile(r"</?(?:b|i|u|del|mark|sub|sup|small|math)[^>]*>")
_BR_RE = re.compile(r"<br\s*/?>")

# Recognition batch sizing (Phase 2 of docs/DESIGN-scale-ocr.md). Precedence
# (vega.config owns the rule): constructor arg > env VEGA_GPU_BATCH > auto:
#   · < 8 GB VRAM — 32, the guardrail a 4 GB card needs to not OOM;
#   · ≥ 8 GB VRAM — None: let Surya use its own tuned default (hundreds),
#                   which is what large cards want for throughput.
_SMALL_GPU_BATCH = 32
_SMALL_GPU_BYTES = 8 * 1024 ** 3
_UNSET = object()                  # lazy-resolution sentinel


def _small_gpu() -> Optional[bool]:
    """True on a <8GB CUDA card, False on a big one, None when unknowable."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            return total < _SMALL_GPU_BYTES
    except Exception:
        pass
    return None


def _resolve_recognition_batch(explicit: Optional[int] = None) -> Optional[int]:
    v = resolve_gpu_batch(explicit)          # clamped ≥1 by vega.config
    if v is not None:
        return v
    small = _small_gpu()
    return None if small is False else _SMALL_GPU_BATCH


def _resolve_detection_batch(explicit: Optional[int] = None) -> Optional[int]:
    """Detection allocates per-*page* tensors, so a multi-page batch OOMs a
    4 GB card even when recognition is capped (observed: 792 MB for 3 pages at
    300 dpi). Small cards detect one page at a time — same peak memory as the
    single-page path; big cards keep Surya's default."""
    v = resolve_gpu_det_batch(explicit)      # clamped ≥1 by vega.config
    if v is not None:
        return v
    small = _small_gpu()
    return None if small is False else 1


class SuryaBackend(BaseOCRBackend):
    name = "surya"

    def __init__(self, gpu: Optional[bool] = None,
                 recognition_batch: Optional[int] = None,
                 detection_batch: Optional[int] = None):
        # ``gpu``: False forces CPU; True/None let Surya auto-place on CUDA.
        # Batch sizes: explicit value > VEGA_GPU_BATCH / VEGA_GPU_DET_BATCH >
        # VRAM-aware auto (vega.config owns the precedence rule).
        self._gpu = gpu
        self._explicit_rec_batch = recognition_batch
        self._explicit_det_batch = detection_batch
        self._predictors = None          # (recognition, detection) once built
        self._init_error: Optional[str] = None
        # Serialize inference: one predictor on one device — concurrent page
        # workers calling it in parallel fail/empty out on small GPUs, silently
        # demoting pages to the next backend in a fallback composite.
        self._infer_lock = threading.Lock()
        self._rec_batch = _UNSET         # resolved lazily, once per instance
        self._det_batch = _UNSET

    def available_scripts(self) -> Set[str]:
        return set(_SCRIPTS)

    # BaseOCRBackend.can_handle is correct here: any ``+``-combo of advertised
    # scripts is served in a single language-agnostic pass.

    def cache_version(self) -> str:
        ver = "unknown"
        try:
            from importlib.metadata import version  # noqa: PLC0415
            ver = version("surya-ocr")
        except Exception:  # pragma: no cover - metadata absent
            pass
        return f"surya:{ver}"

    def _build(self):
        """Build (and cache) the predictors; negative-cache any failure.

        Construction is serialized under the cross-backend ``MODEL_INIT_LOCK``
        (double-checked): concurrent page workers must not race model loading —
        transformers' meta-device init is process-global and two interleaved
        builds corrupt each other ("Cannot copy out of meta tensor")."""
        if self._init_error is not None:
            return None
        if self._predictors is not None:
            return self._predictors
        from vega.ocr.base import MODEL_INIT_LOCK  # noqa: PLC0415
        with MODEL_INIT_LOCK:
            if self._init_error is not None:        # re-check under the lock
                return None
            if self._predictors is not None:
                return self._predictors
            try:
                from surya.detection import DetectionPredictor  # noqa: PLC0415
                from surya.foundation import FoundationPredictor  # noqa: PLC0415
                from surya.recognition import RecognitionPredictor  # noqa: PLC0415

                device = "cpu" if self._gpu is False else None  # None → auto
                logger.info("constructing Surya predictors (device=%s)",
                            device or "auto")
                foundation = (FoundationPredictor(device=device) if device
                              else FoundationPredictor())
                detection = (DetectionPredictor(device=device) if device
                             else DetectionPredictor())
                self._predictors = (RecognitionPredictor(foundation), detection)
            except Exception as e:  # noqa: BLE001
                self._init_error = repr(e)
                logger.warning(
                    "Surya predictors failed to build (%s) — surya OCR "
                    "disabled for this run, falling back where possible", e)
                return None
        return self._predictors

    def _to_pil(self, image_png: bytes):
        from PIL import Image  # noqa: PLC0415
        return Image.open(io.BytesIO(image_png)).convert("RGB")

    @staticmethod
    def _result_text(result) -> str:
        lines = []
        for line in getattr(result, "text_lines", None) or []:
            text = _TAG_RE.sub("", _BR_RE.sub("\n", line.text or "")).strip()
            if text:
                lines.append(text)
        return "\n".join(lines).strip()

    def _recognition_batch(self) -> Optional[int]:
        if self._rec_batch is _UNSET:
            self._rec_batch = _resolve_recognition_batch(
                self._explicit_rec_batch)
        return self._rec_batch

    def _detection_batch(self) -> Optional[int]:
        if self._det_batch is _UNSET:
            self._det_batch = _resolve_detection_batch(
                self._explicit_det_batch)
        return self._det_batch

    @staticmethod
    def _purge_cuda() -> None:
        """Release cached CUDA memory after an OOM so a retry starts from a
        clean pool instead of inheriting the failed call's fragmentation."""
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - best effort
            pass

    def _infer(self, images: List[bytes], script: str) -> List[str]:
        """One recognition call for ``images``; raises on failure (callers
        handle isolation). Inference is serialized (one predictor, one device)."""
        recognition, detection = self._predictors
        pils = [self._to_pil(im) for im in images]
        kwargs = {"det_predictor": detection, "sort_lines": True,
                  "math_mode": False}
        if self._gpu is not False:
            batch = self._recognition_batch()
            if batch is not None:
                kwargs["recognition_batch_size"] = batch
            det = self._detection_batch()
            if det is not None:
                kwargs["detection_batch_size"] = det
        with self._infer_lock:
            results = recognition(pils, **kwargs)
        out = [self._result_text(r) for r in results]
        # Length contract even if surya drops a page.
        out += ["" for _ in range(len(images) - len(out))]
        return out[:len(images)]

    def image_to_text(self, image_png: bytes, script: str) -> str:
        return self.image_to_text_batch([image_png], script)[0]

    def image_to_text_batch(self, images: List[bytes], script: str) -> List[str]:
        built = self._build()
        if built is None:
            return ["" for _ in images]
        try:
            return self._infer(images, script)
        except Exception as e:  # noqa: BLE001
            if len(images) <= 1:
                logger.debug("surya OCR failed (script=%s): %r", script, e)
                return ["" for _ in images]
            # One poison page must not blank the whole window — retry each
            # page individually so a failure costs exactly one page (and keeps
            # engine failover page-granular for deterministic composites).
            logger.debug("surya batch OCR failed (script=%s): %r — retrying "
                         "%d pages individually", script, e, len(images))
            self._purge_cuda()
            out: List[str] = []
            for im in images:
                try:
                    out.extend(self._infer([im], script))
                except Exception as e2:  # noqa: BLE001
                    logger.debug("surya OCR failed on a page: %r", e2)
                    self._purge_cuda()
                    out.append("")
            return out
