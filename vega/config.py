"""Runtime configuration — one picklable settings object for the pipeline.

``IngestConfig`` is deliberately a plain dataclass with only primitive fields so
it can cross a ``ProcessPoolExecutor`` boundary. Each worker rebuilds its own OCR
backend from this config (backends — Tesseract sessions, EasyOCR readers — are
not picklable, so they are never sent between processes).

Environment overrides (all optional):
  · ``VEGA_TESSDATA_DIR``   — directory of ``*.traineddata`` packs (TESSDATA_PREFIX).
  · ``VEGA_OCR_CACHE_DIR``  — where OCR results are cached (default under ~/.cache).
  · ``VEGA_EMBEDDING_MODEL``— tokenizer used for token-aware chunk sizing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from vega.tokenization import DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS

# OCR backend selection modes.
OCR_MODES = ("auto", "tesseract", "easyocr", "none")


def default_cache_dir() -> Path:
    env = os.environ.get("VEGA_OCR_CACHE_DIR")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "vega" / "ocr"


def default_tessdata_dir() -> Optional[str]:
    return os.environ.get("VEGA_TESSDATA_DIR") or os.environ.get("TESSDATA_PREFIX")


@dataclass
class IngestConfig:
    """Everything a pipeline (and its worker processes) needs to run."""

    # Language handling ------------------------------------------------------
    # ISO-639-1 codes the caller declares for the corpus (e.g. ["te", "en"]).
    # Drives per-page OCR routing and per-chunk language tagging. Empty / ["en"]
    # ⇒ English path; legacy-font detection + OSD still apply as a fallback.
    languages: List[str] = field(default_factory=lambda: ["en"])

    # OCR --------------------------------------------------------------------
    ocr_mode: str = "auto"                 # auto | tesseract | easyocr | none
    gpu: Optional[bool] = None             # None ⇒ auto-detect (torch.cuda)
    figure_ocr: bool = False               # OCR embedded figures (expensive)
    dpi: int = 300                         # render DPI for recovery / figure OCR
    scanned_dpi: int = 200                 # render DPI for plain scanned pages
    cache_dir: Optional[str] = None        # None ⇒ default_cache_dir()
    tessdata_dir: Optional[str] = None     # None ⇒ default_tessdata_dir()
    ocr_cache: bool = True                 # wrap backend in a disk cache

    # Chunking ---------------------------------------------------------------
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    min_tokens: int = 48

    # Throughput -------------------------------------------------------------
    workers: int = 1                       # process-pool size for multi-file runs

    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir else default_cache_dir()

    def resolved_tessdata_dir(self) -> Optional[str]:
        return self.tessdata_dir or default_tessdata_dir()
