"""OCR backend selection — config/CLI flag **and** GPU auto-detection.

Policy:
  · ``mode="none"``       → no OCR (born-digital only).
  · ``mode="tesseract"``  → Tesseract (CPU).
  · ``mode="easyocr"``    → EasyOCR (neural; CUDA if present).
  · ``mode="auto"``       → prefer the GPU backend when a CUDA GPU is present
                            (EasyOCR), else Tesseract. In the GPU case the two
                            are composed so scripts EasyOCR lacks (Malayalam,
                            Gujarati, Gurmukhi, Odia) fall back to Tesseract.

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


class FallbackOCRBackend(BaseOCRBackend):
    """Routes each script to the first wrapped backend that supports it.

    Used for ``auto`` on a GPU host: EasyOCR (fast, batched) handles the scripts
    it knows; Tesseract covers the rest. ``available_scripts`` is the union, so
    ``text_recovery``'s availability check sees full coverage.
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

    def image_to_text(self, image_png: bytes, script: str) -> str:
        # Try capable backends in order; a backend that returns empty (or raises)
        # shouldn't strand the page — fall through to the next one.
        for b in self._ranked(script):
            try:
                out = b.image_to_text(image_png, script)
            except Exception:  # noqa: BLE001
                out = ""
            if out:
                return out
        return ""

    def image_to_text_batch(self, images, script: str):
        ranked = self._ranked(script)
        if not ranked:
            return ["" for _ in images]
        out = ranked[0].image_to_text_batch(images, script)
        # Fill any empties from the next capable backend(s).
        for b in ranked[1:]:
            if all(out):
                break
            for i, txt in enumerate(out):
                if not txt:
                    try:
                        out[i] = b.image_to_text(images[i], script)
                    except Exception:  # noqa: BLE001
                        pass
        return out


def select_backend(
    mode: str = "auto",
    *,
    gpu: Optional[bool] = None,
    tessdata_dir: Optional[str] = None,
    cache_dir=None,
) -> Optional[OCRBackend]:
    """Build the OCR backend for ``mode``. Returns ``None`` for ``mode="none"``.

    When ``cache_dir`` is given the chosen backend is wrapped in a disk cache.
    """
    mode = (mode or "auto").lower()
    valid = ("auto", "tesseract", "easyocr", "none")
    if mode not in valid:
        raise ValueError(
            f"unknown OCR mode {mode!r}; expected one of {', '.join(valid)}")
    if mode == "none":
        return None

    backend: Optional[OCRBackend]
    if mode == "tesseract":
        backend = TesseractBackend(tessdata_dir)
    elif mode == "easyocr":
        if _easyocr_importable():
            from vega.ocr.easyocr_backend import EasyOCRBackend  # noqa: PLC0415
            backend = EasyOCRBackend(gpu=gpu)
        else:
            logger.warning("easyocr requested but not installed — using Tesseract")
            backend = TesseractBackend(tessdata_dir)
    else:  # auto
        use_gpu = gpu if gpu is not None else gpu_available()
        if use_gpu and _easyocr_importable():
            from vega.ocr.easyocr_backend import EasyOCRBackend  # noqa: PLC0415
            logger.info("auto: CUDA GPU present → EasyOCR (Tesseract fallback)")
            backend = FallbackOCRBackend([
                EasyOCRBackend(gpu=True),
                TesseractBackend(tessdata_dir),
            ])
        else:
            logger.info("auto: no CUDA GPU → Tesseract (CPU)")
            backend = TesseractBackend(tessdata_dir)

    if cache_dir is not None and backend is not None:
        backend = CachingOCRBackend(backend, cache_dir)
    return backend
