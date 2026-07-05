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
        # Entries shard into 256 two-hex-digit subdirectories so a large corpus
        # (~10^5–10^6 pages) never piles every file into one flat directory
        # (Phase 4 of docs/DESIGN-scale-ocr.md). Pre-shard flat entries simply
        # miss and re-OCR — the cache is disposable by design.
        digest = hashlib.sha1(
            image_png + b"\x1f" + self._version.encode("utf-8")
            + b"\x1f" + script.encode("utf-8")
        ).hexdigest()
        prefix = _sanitize(self._backend.name)
        safe_script = _sanitize(script)
        return self._dir / digest[:2] / f"{prefix}__{safe_script}__{digest}.txt"

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
            p.parent.mkdir(parents=True, exist_ok=True)   # shard subdir
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, p)          # atomic on POSIX + Windows
        except Exception as e:          # cache is best-effort, never fatal
            logger.debug("OCR cache write failed: %r", e)
            try:
                tmp.unlink()
            except OSError:
                pass

    def image_to_text(self, image_png: bytes, script: str) -> str:
        return self.image_to_text_attributed(image_png, script)[0]

    def image_to_text_attributed(self, image_png: bytes, script: str):
        """Cached OCR that preserves **engine attribution**: the producing
        engine's name is stored in a tiny ``.engine`` sidecar next to the text
        entry. Pre-sidecar cache entries still hit — they just attribute None."""
        p = self._path(image_png, script)
        hit = self._read(p)
        if hit is not None:
            engine = self._read(p.with_suffix(p.suffix + ".engine"))
            return hit, (engine.strip() or None) if engine else None
        out, engine = self._attributed(self._backend, image_png, script)
        # Never persist an empty result: "" usually means a *transient* failure
        # (GPU contention, model-load race), and caching it would poison the
        # page until someone hand-deletes the entry. Genuinely blank pages
        # re-OCR each run — rare and cheap next to that failure mode.
        if out:
            self._write_atomic(p, out)
            if engine:
                self._write_atomic(p.with_suffix(p.suffix + ".engine"), engine)
        return out, engine

    @staticmethod
    def _attributed(backend, image_png: bytes, script: str):
        fn = getattr(backend, "image_to_text_attributed", None)
        if fn is not None:
            return fn(image_png, script)
        out = backend.image_to_text(image_png, script)
        return out, (getattr(backend, "name", None) if out else None)

    def image_to_text_batch(self, images: List[bytes], script: str) -> List[str]:
        return self.image_to_text_batch_attributed(images, script)[0]

    def image_to_text_batch_attributed(self, images: List[bytes], script: str):
        """Serve hits from disk (with engine sidecars); batch **all misses**
        through the wrapped backend in one attributed call so a GPU backend
        keeps its batching (findings: don't serialize batch OCR through the
        cache). Empty results are never persisted (transient failures)."""
        paths = [self._path(im, script) for im in images]
        out: List[str] = [None] * len(images)   # type: ignore[list-item]
        engines: List = [None] * len(images)
        miss_idx: List[int] = []
        for i, p in enumerate(paths):
            hit = self._read(p)
            if hit is not None:
                out[i] = hit
                eng = self._read(p.with_suffix(p.suffix + ".engine"))
                engines[i] = (eng.strip() or None) if eng else None
            else:
                miss_idx.append(i)
        if miss_idx:
            miss_imgs = [images[i] for i in miss_idx]
            results, res_engines = self._batch_attributed(
                self._backend, miss_imgs, script)
            for i, res, eng in zip(miss_idx, results, res_engines):
                out[i] = res
                engines[i] = eng
                if res:                       # empty = transient; don't poison
                    self._write_atomic(paths[i], res)
                    if eng:
                        self._write_atomic(
                            paths[i].with_suffix(paths[i].suffix + ".engine"), eng)
        return out, engines  # type: ignore[return-value]

    @staticmethod
    def _batch_attributed(backend, images: List[bytes], script: str):
        fn = getattr(backend, "image_to_text_batch_attributed", None)
        if fn is not None:
            return fn(images, script)
        texts = backend.image_to_text_batch(images, script)
        name = getattr(backend, "name", None)
        return texts, [(name if t else None) for t in texts]
