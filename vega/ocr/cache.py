"""Disk-backed OCR cache — a transparent decorator over any ``OCRBackend``.

OCR is the expensive step; re-running vega over the same corpus (or the same
scanned page under a different chunking config) should not re-OCR. The cache key
is a content hash of the *rendered page/image bytes* plus the backend name and
script, so it is correct across files and safe to share on disk.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Set

from vega.ocr.base import BaseOCRBackend, OCRBackend

logger = logging.getLogger("vega.ocr.cache")


class CachingOCRBackend(BaseOCRBackend):
    """Wraps a backend; persists ``image_to_text`` results as files on disk."""

    def __init__(self, backend: OCRBackend, cache_dir):
        self._backend = backend
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._backend.name

    def available_scripts(self) -> Set[str]:
        return self._backend.available_scripts()

    def _path(self, image_png: bytes, script: str) -> Path:
        h = hashlib.sha1(image_png).hexdigest()
        safe_script = script.replace("+", "-")
        return self._dir / f"{self._backend.name}__{safe_script}__{h}.txt"

    def image_to_text(self, image_png: bytes, script: str) -> str:
        p = self._path(image_png, script)
        if p.exists():
            try:
                return p.read_text(encoding="utf-8")
            except Exception:
                pass
        out = self._backend.image_to_text(image_png, script)
        try:
            p.write_text(out, encoding="utf-8")
        except Exception as e:  # cache is best-effort, never fatal
            logger.debug("OCR cache write failed: %r", e)
        return out

    def image_to_text_batch(self, images: List[bytes], script: str) -> List[str]:
        # Reuse the cached per-image path so partial hits are honoured.
        return [self.image_to_text(im, script) for im in images]
