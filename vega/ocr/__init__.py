"""Pluggable OCR backends.

Public surface:
  · ``OCRBackend``           — the protocol every engine implements.
  · ``select_backend``       — flag + GPU-auto-detection factory.
  · ``TesseractBackend``     — CPU default.
  · ``EasyOCRBackend``       — GPU-capable neural backend (lazy import).
  · ``CachingOCRBackend``    — transparent disk cache decorator.
  · ``FallbackOCRBackend``   — per-script routing across backends.
  · ``detect_osd_script``    — engine-agnostic OSD script detection.
"""

from vega.ocr.base import BaseOCRBackend, OCRBackend
from vega.ocr.cache import CachingOCRBackend
from vega.ocr.selection import (
    FallbackOCRBackend,
    gpu_available,
    select_backend,
)
from vega.ocr.tesseract import TesseractBackend, detect_osd_script

__all__ = [
    "OCRBackend",
    "BaseOCRBackend",
    "TesseractBackend",
    "CachingOCRBackend",
    "FallbackOCRBackend",
    "select_backend",
    "gpu_available",
    "detect_osd_script",
    "EasyOCRBackend",
]


def __getattr__(name):
    # Lazy so importing vega.ocr never imports easyocr/torch.
    if name == "EasyOCRBackend":
        from vega.ocr.easyocr_backend import EasyOCRBackend
        return EasyOCRBackend
    raise AttributeError(name)
