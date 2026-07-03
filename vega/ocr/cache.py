"""Disk-backed OCR cache — a transparent decorator over any ``OCRBackend``.

OCR is the expensive step; re-running vega over the same corpus (or the same
scanned page under a different chunking config) should not re-OCR. The cache key
is a content hash of the *rendered page/image bytes* plus the backend name, its
**version fingerprint**, and the script — so it is correct across files, safe to
share on disk, and self-invalidates when the engine or its packs are upgraded.

Writes are atomic (temp file + ``os.replace``) so concurrent worker processes
sharing one cache dir never read a half-written file; OCR is deterministic, so
the occasional duplicate compute on a race is harmless.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import List, Set

from vega.ocr.base import BaseOCRBackend, OCRBackend

logger = logging.getLogger("vega.ocr.cache")

# Strict filename alphabet — never trust a backend name / script in a path.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize(value: str) -> str:
    return _SAFE.sub("-", value or "")


class CachingOCRBackend(BaseOCRBackend):
    """Wraps a backend; persists ``image_to_text`` results as files on disk."""

    def __init__(self, backend: OCRBackend, cache_dir):
        self._backend = backend
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Compute the version fingerprint once (querying it can shell out).
        try:
            self._version = _sanitize(backend.cache_version())
        except Exception:  # pragma: no cover - defensive
            self._version = _sanitize(getattr(backend, "name", "backend"))

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._backend.name

    def available_scripts(self) -> Set[str]:
        return self._backend.available_scripts()

    def can_handle(self, script: str) -> bool:
        return self._backend.can_handle(script)

    def cache_version(self) -> str:
        return self._version

    def _path(self, image_png: bytes, script: str) -> Path:
        # Version + script fold into the hash (correctness) *and* the readable
        # filename prefix is sanitized (safety); the hash disambiguates.
        digest = hashlib.sha1(
            image_png + b"\x1f" + self._version.encode("utf-8")
            + b"\x1f" + script.encode("utf-8")
        ).hexdigest()
        prefix = _sanitize(self._backend.name)
        safe_script = _sanitize(script)
        return self._dir / f"{prefix}__{safe_script}__{digest}.txt"

    def _read(self, p: Path):
        if not p.exists():
            return None
        try:
            return p.read_text(encoding="utf-8")
        except Exception:  # partial/locked read — treat as a miss
            return None

    def _write_atomic(self, p: Path, text: str) -> None:
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, p)          # atomic on POSIX + Windows
        except Exception as e:          # cache is best-effort, never fatal
            logger.debug("OCR cache write failed: %r", e)
            try:
                tmp.unlink()
            except OSError:
                pass

    def image_to_text(self, image_png: bytes, script: str) -> str:
        p = self._path(image_png, script)
        hit = self._read(p)
        if hit is not None:
            return hit
        out = self._backend.image_to_text(image_png, script)
        self._write_atomic(p, out)
        return out

    def image_to_text_batch(self, images: List[bytes], script: str) -> List[str]:
        """Serve hits from disk; batch **all misses** through the wrapped backend
        in one call so a GPU backend keeps its batching (findings: don't
        serialize batch OCR through the cache)."""
        paths = [self._path(im, script) for im in images]
        out: List[str] = [None] * len(images)   # type: ignore[list-item]
        miss_idx: List[int] = []
        for i, p in enumerate(paths):
            hit = self._read(p)
            if hit is not None:
                out[i] = hit
            else:
                miss_idx.append(i)
        if miss_idx:
            miss_imgs = [images[i] for i in miss_idx]
            results = self._backend.image_to_text_batch(miss_imgs, script)
            for i, res in zip(miss_idx, results):
                out[i] = res
                self._write_atomic(paths[i], res)
        return out  # type: ignore[return-value]
