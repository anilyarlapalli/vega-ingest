"""Tesseract OCR backend (CPU default) + engine-independent OSD script detection.

Tesseract is the default because it has broad, high-quality Indic coverage via
its language packs and runs anywhere without a GPU. Language packs are found via
``tessdata_dir`` (set as ``TESSDATA_PREFIX`` for the process) or the ambient
Tesseract install.

``detect_osd_script`` lives here because Orientation-and-Script-Detection needs
only ``osd.traineddata`` (no language pack) and is genuinely engine-agnostic —
even when the *text* OCR runs on a neural backend, script detection can lean on
Tesseract OSD.
"""

from __future__ import annotations

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Set

from vega.config import resolve_cpu_ocr_threads
from vega.ocr.base import BaseOCRBackend

logger = logging.getLogger("vega.ocr.tesseract")


class TesseractBackend(BaseOCRBackend):
    name = "tesseract"

    def __init__(self, tessdata_dir: Optional[str] = None,
                 batch_threads: Optional[int] = None):
        # ``batch_threads``: thread-pool width for a batch window (the CPU
        # analogue of Surya's recognition batch). Every image_to_text call is
        # its own tesseract subprocess — GIL-free — so a pool scales the
        # deferred window across cores; measured on a real 29-page 300-dpi
        # Tamil PDF (16-core dev box): whole file 52s → 19s, byte-identical.
        # None ⇒ VEGA_CPU_OCR_THREADS, then min(8, cores) (vega.config).
        self._tessdata_dir = tessdata_dir
        self._batch_threads = batch_threads
        if tessdata_dir:
            # Tesseract 4/5 resolve packs from TESSDATA_PREFIX; set it for this
            # process so workers pointed at a user-local pack dir just work.
            os.environ["TESSDATA_PREFIX"] = tessdata_dir
        self._cached_langs: Optional[Set[str]] = None
        self._version: Optional[str] = None

    def _config(self) -> str:
        return f'--tessdata-dir "{self._tessdata_dir}"' if self._tessdata_dir else ""

    def cache_version(self) -> str:
        """tesseract engine version + tessdata location — both change OCR output,
        so both belong in the cache key. Queried once and memoised."""
        if getattr(self, "_version", None) is None:
            ver = "unknown"
            try:
                import pytesseract  # noqa: PLC0415
                ver = str(pytesseract.get_tesseract_version())
            except Exception:  # pragma: no cover - tesseract absent
                pass
            self._version = f"tesseract:{ver}:{self._tessdata_dir or 'ambient'}"
        return self._version

    def available_scripts(self) -> Set[str]:
        if self._cached_langs is None:
            try:
                import pytesseract  # noqa: PLC0415
                self._cached_langs = set(
                    pytesseract.get_languages(config=self._config())
                )
            except Exception as e:  # pragma: no cover - tesseract absent
                logger.debug("pytesseract.get_languages failed: %r", e)
                self._cached_langs = set()
        return self._cached_langs

    def image_to_text(self, image_png: bytes, script: str) -> str:
        try:
            import pytesseract  # noqa: PLC0415
            from PIL import Image  # noqa: PLC0415
            img = Image.open(io.BytesIO(image_png))
            return (pytesseract.image_to_string(
                img, lang=script, config=self._config()) or "").strip()
        except Exception as e:
            logger.debug("tesseract OCR failed (lang=%s): %r", script, e)
            return ""

    def image_to_text_batch(self, images: List[bytes], script: str) -> List[str]:
        # Each page is an independent subprocess, so a thread pool turns the
        # deferred batch window (docs/DESIGN-scale-ocr.md Phase 1) from serial
        # into parallel on CPU. Order is preserved; each item still degrades
        # to "" on failure exactly like the single-page path.
        workers = min(len(images), resolve_cpu_ocr_threads(self._batch_threads))
        if workers <= 1:
            return [self.image_to_text(im, script) for im in images]
        # Tesseract's own OpenMP threads busy-spin at barriers; W pool workers
        # × 4 OMP threads each oversubscribes the box and burns ~5× the CPU
        # for the same pages (measured: 48s → 19s wall on a 29-page file just
        # by capping this). The pool provides the parallelism, so each
        # subprocess gets one OMP thread. An explicit user setting wins; the
        # deferred pass is single-threaded per file, so the temporary global
        # is restored before anyone else reads it.
        preset = os.environ.get("OMP_THREAD_LIMIT")
        if preset is None:
            os.environ["OMP_THREAD_LIMIT"] = "1"
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(lambda im: self.image_to_text(im, script),
                                   images))
        finally:
            if preset is None:
                os.environ.pop("OMP_THREAD_LIMIT", None)


def detect_osd_script(image_png: bytes, candidate_iso: List[str]) -> Optional[str]:
    """Detect a page's script via Tesseract OSD and map it to a Tesseract pack
    code among ``candidate_iso`` (ISO-639-1 codes). None on failure / no match.

    Used when a scanned page has no font hint — OSD reads the pixels and names
    the script; :func:`vega.languages.iso_for_osd_script` disambiguates shared
    scripts (Devanagari → hi vs mr) by which languages the caller declared.
    """
    if not candidate_iso:
        return None
    try:
        import pytesseract  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from vega.languages import iso_for_osd_script, to_tesseract  # noqa: PLC0415
        osd = pytesseract.image_to_osd(Image.open(io.BytesIO(image_png)))
        name = ""
        for line in osd.splitlines():
            if line.startswith("Script:"):
                name = line.split(":", 1)[1].strip()
                break
        iso = iso_for_osd_script(name, candidate_iso)
        pack = to_tesseract(iso) if iso else None
        if pack:
            logger.info("OSD detected script=%s → pack=%s", name, pack)
        return pack
    except Exception as e:  # OSD fails on sparse/short pages ("too few chars")
        logger.debug("OSD failed: %r", e)
        return None
