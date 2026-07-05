"""Runtime configuration — one picklable settings object for the pipeline.

``IngestConfig`` is deliberately a plain dataclass with only primitive fields so
it can cross a ``ProcessPoolExecutor`` boundary. Each worker rebuilds its own OCR
backend from this config (backends — Tesseract sessions, EasyOCR readers — are
not picklable, so they are never sent between processes).

Environment overrides (all optional):
  · ``VEGA_TESSDATA_DIR``    — directory of ``*.traineddata`` packs (TESSDATA_PREFIX).
  · ``VEGA_OCR_CACHE_DIR``   — where OCR results are cached (default under ~/.cache).
  · ``VEGA_EMBEDDING_MODEL`` — tokenizer used for token-aware chunk sizing.
  · ``VEGA_OCR_WINDOW``      — pages per batched-OCR window (default 16).
  · ``VEGA_GPU_BATCH``       — Surya recognition batch size (default: VRAM-aware —
                               32 under 8 GB, Surya's own default above).
  · ``VEGA_GPU_DET_BATCH``   — Surya detection batch size (default: VRAM-aware —
                               1 under 8 GB, Surya's own default above).
  · ``VEGA_CPU_OCR_THREADS`` — thread-pool width for a Tesseract batch window
                               (default min(8, cores)).

This module is the **single point of truth for tuning knobs**: one precedence
rule — explicit config/constructor value > ``VEGA_*`` env var > auto default —
implemented once in the ``resolve_*`` helpers below. Components take the value
through their constructor/parameter and fall back to the resolver; nothing
outside this module reads the environment for tuning.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from vega.tokenization import DEFAULT_CHUNK_TOKENS, DEFAULT_OVERLAP_TOKENS

logger = logging.getLogger("vega.config")

# OCR backend selection modes.
OCR_MODES = ("auto", "tesseract", "easyocr", "surya", "none")

# ── tuning-knob resolution (single point of truth) ───────────────────────────

# Pages per batched-OCR window. A host-RAM knob, not VRAM — the backend's own
# batch sizes cap per-forward memory regardless (docs/DESIGN-scale-ocr.md C3).
DEFAULT_OCR_WINDOW = 16

# Ceiling for the Tesseract batch-window thread pool: gains flatten past 8
# while host-RAM and load spikes grow (measured on a 16-core dev box).
MAX_CPU_OCR_THREADS = 8


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("ignoring non-integer %s=%r", name, raw)
        return None


def resolve_ocr_window(explicit: Optional[int] = None) -> int:
    v = explicit if explicit is not None else _env_int("VEGA_OCR_WINDOW")
    return max(1, v) if v is not None else DEFAULT_OCR_WINDOW


def resolve_cpu_ocr_threads(explicit: Optional[int] = None) -> int:
    v = explicit if explicit is not None else _env_int("VEGA_CPU_OCR_THREADS")
    if v is not None:
        return max(1, v)
    return min(MAX_CPU_OCR_THREADS, os.cpu_count() or 1)


def resolve_gpu_batch(explicit: Optional[int] = None) -> Optional[int]:
    """Surya recognition batch size; None ⇒ the backend auto-sizes from VRAM."""
    v = explicit if explicit is not None else _env_int("VEGA_GPU_BATCH")
    return max(1, v) if v is not None else None


def resolve_gpu_det_batch(explicit: Optional[int] = None) -> Optional[int]:
    """Surya detection batch size; None ⇒ the backend auto-sizes from VRAM."""
    v = explicit if explicit is not None else _env_int("VEGA_GPU_DET_BATCH")
    return max(1, v) if v is not None else None


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
    batch_ocr: bool = True                 # batch a file's page-OCR into
                                           # script-grouped GPU windows
                                           # (--no-batch-ocr forces per-page)

    # Layout -----------------------------------------------------------------
    # Detect multi-column PDF pages and read column-by-column. A no-op on
    # single-column pages; disable if a document's layout confuses the detector.
    columns: bool = True

    # Chunking ---------------------------------------------------------------
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    min_tokens: int = 48

    # Throughput -------------------------------------------------------------
    # None on the Optional fields ⇒ resolve via env var, then auto default
    # (the resolve_* helpers above — the single precedence rule).
    workers: int = 1                       # process-pool size for multi-file runs
    page_workers: int = 1                  # thread-pool size for pages of ONE PDF;
                                           # a single-file run also borrows ``workers``
                                           # so a large PDF uses all the cores.
    ocr_window: Optional[int] = None       # pages per batched-OCR window
    gpu_batch: Optional[int] = None        # Surya recognition batch size
    gpu_det_batch: Optional[int] = None    # Surya detection batch size
    cpu_ocr_threads: Optional[int] = None  # Tesseract batch-window thread pool

    # Directory discovery ----------------------------------------------------
    # ``_``-prefixed paths are ingested by default (general-purpose contract).
    # Opt in to skip them (e.g. an original-binary audit archive).
    skip_underscored: bool = False

    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir) if self.cache_dir else default_cache_dir()

    def resolved_tessdata_dir(self) -> Optional[str]:
        return self.tessdata_dir or default_tessdata_dir()
