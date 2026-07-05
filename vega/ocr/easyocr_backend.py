"""EasyOCR backend — GPU-capable neural OCR.

EasyOCR runs on CPU and **auto-uses CUDA when available**, so it is vega's GPU
path. The ``Reader`` is heavy to construct (loads detection + recognition nets)
and is therefore cached per language set and built lazily — importing this module
never imports ``easyocr`` or ``torch``, so it stays import-safe on hosts without
them (``available_scripts`` still answers from a static map).

EasyOCR's Indic coverage is narrower than Tesseract's: it ships Devanagari
(hi/mr), Tamil, Telugu, Kannada and Bengali but not Malayalam, Gujarati,
Gurmukhi or Odia. Those simply aren't advertised in :meth:`available_scripts`,
so the fallback backend routes them to Tesseract automatically.
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional, Set

from vega.ocr.base import BaseOCRBackend

logger = logging.getLogger("vega.ocr.easyocr_backend")

# Tesseract script code → EasyOCR language code (only the ones EasyOCR supports).
_TESS_TO_EASY = {
    "eng": "en",
    "hin": "hi",
    "mar": "mr",
    "tam": "ta",
    "tel": "te",
    "kan": "kn",
    "ben": "bn",
    "asm": "as",
}


class EasyOCRBackend(BaseOCRBackend):
    name = "easyocr"

    def __init__(self, gpu: Optional[bool] = None):
        # ``gpu``: True/False forces device; None auto-detects torch.cuda at use.
        self._gpu = gpu
        self._readers: dict = {}

    # Non-Latin script codes (English co-loads with any single one of these).
    _NON_LATIN = frozenset(_TESS_TO_EASY) - {"eng"}

    def available_scripts(self) -> Set[str]:
        return set(_TESS_TO_EASY)

    def can_handle(self, script: str) -> bool:
        """EasyOCR loads at most **one** non-Latin script per Reader (plus Latin).
        A request naming two non-Latin scripts (e.g. ``tel+hin``) cannot be served
        faithfully, so we decline it — the fallback router then sends it to
        Tesseract, which does combine packs. Every part must also be known."""
        parts = [p for p in script.split("+") if p]
        if not parts or any(p not in _TESS_TO_EASY for p in parts):
            return False
        return sum(1 for p in parts if p in self._NON_LATIN) <= 1

    def cache_version(self) -> str:
        ver = "unknown"
        try:
            import easyocr  # noqa: PLC0415
            ver = getattr(easyocr, "__version__", "unknown")
        except Exception:  # pragma: no cover - easyocr absent
            pass
        return f"easyocr:{ver}"

    def _resolve_gpu(self) -> bool:
        if self._gpu is not None:
            return self._gpu
        try:
            import torch  # noqa: PLC0415
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _easy_langs(self, script: str) -> List[str]:
        """Map a ``+``-joined Tesseract script string to EasyOCR codes.

        EasyOCR only allows one non-Latin script per Reader (English co-loads
        with any), so we keep the first supported non-English script + English.
        """
        primary: Optional[str] = None
        want_en = False
        for part in script.split("+"):
            easy = _TESS_TO_EASY.get(part)
            if easy is None:
                continue
            if easy == "en":
                want_en = True
            elif primary is None:
                primary = easy
        langs = [primary] if primary else []
        if want_en or not langs:
            langs.append("en")
        return langs

    def _reader(self, langs: List[str]):
        # Construction is serialized under the cross-backend MODEL_INIT_LOCK:
        # concurrent page workers racing here built the same Reader N times
        # (N× VRAM) and interleaved torch model loading, which is not
        # thread-safe (meta-device init is process-global).
        key = tuple(langs)
        if key in self._readers:
            return self._readers[key]
        from vega.ocr.base import MODEL_INIT_LOCK  # noqa: PLC0415
        with MODEL_INIT_LOCK:
            if key not in self._readers:            # re-check under the lock
                import easyocr  # noqa: PLC0415
                logger.info("constructing EasyOCR Reader%s gpu=%s",
                            list(key), self._resolve_gpu())
                self._readers[key] = easyocr.Reader(list(key),
                                                    gpu=self._resolve_gpu())
        return self._readers[key]

    def _to_array(self, image_png: bytes):
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        return np.array(Image.open(io.BytesIO(image_png)).convert("RGB"))

    def image_to_text(self, image_png: bytes, script: str) -> str:
        try:
            reader = self._reader(self._easy_langs(script))
            lines = reader.readtext(self._to_array(image_png), detail=0,
                                    paragraph=True)
            return "\n".join(str(l) for l in lines).strip()
        except Exception as e:
            logger.debug("easyocr OCR failed (script=%s): %r", script, e)
            return ""

    def image_to_text_batch(self, images: List[bytes], script: str) -> List[str]:
        # EasyOCR batches internally per-image; a shared Reader across the batch
        # keeps the (expensive) model resident and CUDA-warm.
        reader = None
        out: List[str] = []
        for im in images:
            try:
                if reader is None:
                    reader = self._reader(self._easy_langs(script))
                lines = reader.readtext(self._to_array(im), detail=0, paragraph=True)
                out.append("\n".join(str(l) for l in lines).strip())
            except Exception as e:
                logger.debug("easyocr batch item failed: %r", e)
                out.append("")
        return out
