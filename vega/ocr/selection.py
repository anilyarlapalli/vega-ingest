"""OCR backend selection — config/CLI flag **and** GPU auto-detection.

Policy:
  · ``mode="none"``       → no OCR (born-digital only).
  · ``mode="tesseract"``  → Tesseract (CPU).
  · ``mode="easyocr"``    → EasyOCR (neural; CUDA if present).
  · ``mode="surya"``      → Surya (neural, language-agnostic; CUDA if present).
  · ``mode="auto"``       → prefer GPU backends when a CUDA GPU is present
                            (Surya first — best Indic fidelity — then EasyOCR),
                            else Tesseract. In the GPU case the engines are
                            composed so any script/page one backend cannot serve
                            falls through to the next, ending at Tesseract.

Everything degrades gracefully: if torch/easyocr are absent, ``auto`` and even
an explicit ``easyocr`` request fall back to Tesseract rather than raising.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

from vega.ocr.base import BaseOCRBackend, OCRBackend
from vega.ocr.cache import CachingOCRBackend
from vega.ocr.tesseract import TesseractBackend

logger = logging.getLogger("vega.ocr.selection")


def gpu_available() -> bool:
    """True iff a CUDA device is visible to torch. False when torch is absent."""
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _easyocr_importable() -> bool:
    import importlib.util  # noqa: PLC0415
    return importlib.util.find_spec("easyocr") is not None


def _surya_importable() -> bool:
    import importlib.util  # noqa: PLC0415
    return importlib.util.find_spec("surya") is not None


# Auto-mode GPU engine priority — reorder this tuple to change which neural
# backend ``auto`` tries first; everything else (import checks, construction,
# composition with Tesseract) follows from it. Surya leads: best measured Indic
# fidelity, full script coverage, working Tamil model.
NEURAL_PREFERENCE = ("surya", "easyocr")


def _build_neural(name: str, gpu: bool,
                  gpu_batch: Optional[int] = None,
                  gpu_det_batch: Optional[int] = None) -> Optional[OCRBackend]:
    """Construct one neural backend by name, or None when not installed."""
    if name == "surya" and _surya_importable():
        from vega.ocr.surya_backend import SuryaBackend  # noqa: PLC0415
        return SuryaBackend(gpu=gpu, recognition_batch=gpu_batch,
                            detection_batch=gpu_det_batch)
    if name == "easyocr" and _easyocr_importable():
        from vega.ocr.easyocr_backend import EasyOCRBackend  # noqa: PLC0415
        return EasyOCRBackend(gpu=gpu)
    return None


class FallbackOCRBackend(BaseOCRBackend):
    """Routes each script to the first wrapped backend that supports it.

    Used for ``auto`` on a GPU host: the neural backends (Surya, then EasyOCR)
    handle the scripts they know; Tesseract covers the rest. ``available_scripts``
    is the union, so ``text_recovery``'s availability check sees full coverage.
    """

    name = "fallback"

    def __init__(self, backends: List[OCRBackend]):
        self._backends = [b for b in backends if b is not None]

    def available_scripts(self) -> Set[str]:
        out: Set[str] = set()
        for b in self._backends:
            out |= b.available_scripts()
        return out

    def _ranked(self, script: str) -> List[OCRBackend]:
        """Backends ordered best-first for ``script``: those that can serve the
        *whole* combo in one call (``can_handle``) first, then by how many parts
        they cover. Used so we can try the next backend if the first yields
        nothing."""
        parts = [p for p in script.split("+") if p]

        def key(b: OCRBackend):
            av = b.available_scripts()
            covers = sum(1 for p in parts if p in av)
            whole = 1 if b.can_handle(script) else 0
            return (whole, covers)

        ranked = sorted(self._backends, key=key, reverse=True)
        # Drop backends that cover nothing at all.
        return [b for b in ranked if key(b)[1] > 0] or ranked

    def cache_version(self) -> str:
        # Fold every member's fingerprint in, **in order** — adding, upgrading,
        # or reordering engines must invalidate cached text, because a different
        # engine may now win the same page (a bare "fallback:v1" served
        # EasyOCR-era text to a Surya-first composite).
        parts = []
        for b in self._backends:
            try:
                parts.append(b.cache_version())
            except Exception:  # pragma: no cover - defensive
                parts.append(getattr(b, "name", "backend"))
        return f"fallback({','.join(parts)})"

    def image_to_text(self, image_png: bytes, script: str) -> str:
        return self.image_to_text_attributed(image_png, script)[0]

    def image_to_text_attributed(self, image_png: bytes, script: str):
        # Try capable backends in order; a backend that returns empty (or raises)
        # shouldn't strand the page — fall through to the next one. The winner's
        # own name is reported (surya/easyocr/tesseract), not "fallback".
        for b in self._ranked(script):
            try:
                out = b.image_to_text(image_png, script)
            except Exception:  # noqa: BLE001
                out = ""
            if out:
                return out, getattr(b, "name", None)
        return "", None

    def image_to_text_batch(self, images, script: str):
        return self.image_to_text_batch_attributed(images, script)[0]

    def image_to_text_batch_attributed(self, images, script: str):
        """Batched routing with per-item attribution: the first ranked backend
        gets the whole batch; any empties are re-batched through the next
        backend(s), and each item records the engine that actually filled it."""
        out: List[str] = ["" for _ in images]
        engines: List[Optional[str]] = [None for _ in images]
        for b in self._ranked(script):
            missing = [i for i, t in enumerate(out) if not t]
            if not missing:
                break
            try:
                res = b.image_to_text_batch([images[i] for i in missing], script)
            except Exception:  # noqa: BLE001
                res = ["" for _ in missing]
            for i, txt in zip(missing, res):
                if txt:
                    out[i] = txt
                    engines[i] = getattr(b, "name", None)
        return out, engines


def select_backend(
    mode: str = "auto",
    *,
    gpu: Optional[bool] = None,
    tessdata_dir: Optional[str] = None,
    cache_dir=None,
    gpu_batch: Optional[int] = None,
    gpu_det_batch: Optional[int] = None,
    cpu_ocr_threads: Optional[int] = None,
) -> Optional[OCRBackend]:
    """Build the OCR backend for ``mode``. Returns ``None`` for ``mode="none"``.

    When ``cache_dir`` is given the chosen backend is wrapped in a disk cache.
    The batch/thread knobs are pass-throughs to the backend constructors;
    ``None`` defers to the env-var/auto defaults (vega.config owns the rule).
    """
    mode = (mode or "auto").lower()
    valid = ("auto", "tesseract", "easyocr", "surya", "none")
    if mode not in valid:
        raise ValueError(
            f"unknown OCR mode {mode!r}; expected one of {', '.join(valid)}")
    if mode == "none":
        return None

    def _tesseract() -> TesseractBackend:
        return TesseractBackend(tessdata_dir, batch_threads=cpu_ocr_threads)

    backend: Optional[OCRBackend]
    if mode == "tesseract":
        backend = _tesseract()
    elif mode == "easyocr":
        if _easyocr_importable():
            from vega.ocr.easyocr_backend import EasyOCRBackend  # noqa: PLC0415
            backend = EasyOCRBackend(gpu=gpu)
        else:
            logger.warning("easyocr requested but not installed — using Tesseract")
            backend = _tesseract()
    elif mode == "surya":
        if _surya_importable():
            from vega.ocr.surya_backend import SuryaBackend  # noqa: PLC0415
            backend = SuryaBackend(gpu=gpu, recognition_batch=gpu_batch,
                                   detection_batch=gpu_det_batch)
        else:
            logger.warning("surya requested but not installed — using Tesseract")
            backend = _tesseract()
    else:  # auto
        use_gpu = gpu if gpu is not None else gpu_available()
        neural: List[OCRBackend] = []
        if use_gpu:
            neural = [b for b in (_build_neural(n, gpu=True,
                                                gpu_batch=gpu_batch,
                                                gpu_det_batch=gpu_det_batch)
                                  for n in NEURAL_PREFERENCE) if b is not None]
        if neural:
            logger.info("auto: CUDA GPU present → %s (Tesseract fallback)",
                        " → ".join(b.name for b in neural))
            backend = FallbackOCRBackend([*neural, _tesseract()])
        else:
            logger.info("auto: no CUDA GPU → Tesseract (CPU)")
            backend = _tesseract()

    if cache_dir is not None and backend is not None:
        backend = CachingOCRBackend(backend, cache_dir)
    return backend
