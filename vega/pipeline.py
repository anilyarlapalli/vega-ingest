"""Ingestion orchestrator — parse → chunk → ``ChunkRecord``, fault-tolerant.

Public surface (also re-exported from ``vega``):

  · ``IngestionPipeline(config)`` — stateful driver holding the OCR backend + stats.
  · ``ingest_file(path, ...)``    → list[dict]   (module-level convenience)
  · ``ingest_directory(dir, ...)``→ list[dict]   (recursive, sorted, parallel)
  · ``parse(path, ...)``          → DocumentModel (lower-level, single file)

Per-file errors are isolated: one corrupt/locked/garbled file is logged and
skipped, never aborts the batch. Multi-file runs parallelise across a process
pool (``config.workers``); each worker builds its own OCR backend from the
picklable ``IngestConfig`` (engines don't cross process boundaries).

Adapted from the AgenticAI_Manufacturing ``doc_pipeline.ingestion.pipeline``
module — repo-specific ``core.document_acl`` coupling is dropped, OCR backend
construction + GPU auto-selection is wired in, and parallelism is added.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from vega.chunkers.structure import StructureChunker
from vega.config import IngestConfig
from vega.languages import language_of_text, normalize_languages
from vega.model import DocumentModel
from vega.records import ChunkRecord
from vega.router import get_parser, is_supported
from vega.text_recovery import script_for_language

logger = logging.getLogger("vega.pipeline")


@dataclass
class IngestStats:
    files_seen: int = 0
    files_parsed: int = 0
    files_failed: int = 0
    chunks: int = 0
    errors: List[str] = field(default_factory=list)
    by_doctype: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "files_seen": self.files_seen,
            "files_parsed": self.files_parsed,
            "files_failed": self.files_failed,
            "chunks": self.chunks,
            "by_doctype": dict(self.by_doctype),
            "errors": list(self.errors),
        }


class IngestionPipeline:
    def __init__(self, config: Optional[IngestConfig] = None):
        self.config = config or IngestConfig()
        self.stats = IngestStats()
        self.chunker = StructureChunker(
            chunk_tokens=self.config.chunk_tokens,
            overlap_tokens=self.config.overlap_tokens,
            min_tokens=self.config.min_tokens,
        )
        # Declared languages (ISO). Non-English candidates drive OCR routing; the
        # primary one is the back-compat single ``recovery_script``.
        self._languages = normalize_languages(self.config.languages) or ["en"]
        non_en = [l for l in self._languages if l != "en"]
        self._candidate_langs = non_en
        self._recovery_script = script_for_language(non_en[0]) if non_en else None
        self._backend = None
        self._backend_built = False

    # ── OCR backend (lazy; built once per process) ──────────────────────────

    @property
    def backend(self):
        if not self._backend_built:
            from vega.ocr import select_backend  # noqa: PLC0415
            cache_dir = (
                str(self.config.resolved_cache_dir())
                if self.config.ocr_cache else None
            )
            self._backend = select_backend(
                self.config.ocr_mode,
                gpu=self.config.gpu,
                tessdata_dir=self.config.resolved_tessdata_dir(),
                cache_dir=cache_dir,
            )
            self._backend_built = True
            logger.info("OCR backend: %s",
                        getattr(self._backend, "name", None) or "disabled")
        return self._backend

    # ── enrichment / tagging ────────────────────────────────────────────────

    def _enrich(self, records: List[ChunkRecord], model: DocumentModel) -> None:
        """Attach source / source_file / doc_type / ocr provenance to every
        record's metadata."""
        ocr_pages = set(model.metadata.get("ocr_pages") or [])
        backend_name = model.metadata.get("ocr_backend")
        src_name = Path(model.source).name if model.source else ""
        for r in records:
            r.metadata.setdefault("source", model.source)
            r.metadata.setdefault("source_file", src_name)
            r.metadata.setdefault("doc_type", model.doc_type)
            page = r.metadata.get("page")
            r.metadata["ocr_used"] = bool(ocr_pages) and page in ocr_pages
            if backend_name:
                r.metadata["backend"] = backend_name

    def _tag_languages(self, records: List[ChunkRecord]) -> None:
        """Write each chunk's detected language (ISO) to ``metadata['language']``.

        Detected from the chunk's Unicode text bounded by the declared candidate
        set; 'en'/primary fallback for Latin text. Single-language corpora are
        tagged with their one language too (cheap, and useful for downstream)."""
        primary = self._languages[0]
        mult_i = len(self._languages) > 1
        for r in records:
            if mult_i:
                detected = language_of_text(r.text or "", self._languages)
                r.metadata["language"] = detected or (
                    "en" if "en" in self._languages else primary)
            else:
                r.metadata.setdefault("language", primary)

    # ── single file ─────────────────────────────────────────────────────────

    def parse(self, path: str | Path) -> DocumentModel:
        """Lower-level: parse one file into a ``DocumentModel`` (no chunking)."""
        path = Path(path)
        parser = get_parser(
            path, ocr_backend=self.backend,
            recovery_script=self._recovery_script,
            candidate_langs=self._candidate_langs,
            figure_ocr=self.config.figure_ocr,
            dpi=self.config.dpi, scanned_dpi=self.config.scanned_dpi,
        )
        if parser is None:
            raise ValueError(f"unsupported type {path.suffix} for {path.name}")
        return parser.parse(path)

    def ingest_file(self, path: str | Path) -> List[ChunkRecord]:
        path = Path(path)
        self.stats.files_seen += 1
        parser = get_parser(
            path, ocr_backend=self.backend,
            recovery_script=self._recovery_script,
            candidate_langs=self._candidate_langs,
            figure_ocr=self.config.figure_ocr,
            dpi=self.config.dpi, scanned_dpi=self.config.scanned_dpi,
        )
        if parser is None:
            msg = f"unsupported type {path.suffix} for {path.name}"
            logger.warning(msg)
            self.stats.files_failed += 1
            self.stats.errors.append(msg)
            return []
        try:
            model = parser.parse(path)
            records = self.chunker.chunk(model)
            self._enrich(records, model)
            self._tag_languages(records)
            self.stats.files_parsed += 1
            self.stats.chunks += len(records)
            self.stats.by_doctype[model.doc_type] = (
                self.stats.by_doctype.get(model.doc_type, 0) + len(records))
            logger.info("ingested %s → %d chunks", path.name, len(records))
            return records
        except Exception as e:                  # fault isolation — never abort batch
            msg = f"{path.name}: {type(e).__name__}: {e}"
            logger.exception("failed to ingest %s", path.name)
            self.stats.files_failed += 1
            self.stats.errors.append(msg)
            return []

    # ── batches ─────────────────────────────────────────────────────────────

    def ingest_paths(self, paths: Iterable[str | Path]) -> List[ChunkRecord]:
        files = [Path(p) for p in paths]
        workers = max(1, int(self.config.workers))
        if workers == 1 or len(files) <= 1:
            out: List[ChunkRecord] = []
            for p in files:
                out.extend(self.ingest_file(p))
            return out
        return self._ingest_parallel(files, workers)

    def _ingest_parallel(self, files: List[Path], workers: int) -> List[ChunkRecord]:
        """Fan out across a process pool; each worker builds its own backend.
        Order of results follows ``files`` for deterministic ids/output."""
        out: List[ChunkRecord] = []
        logger.info("parallel ingest: %d files across %d workers", len(files), workers)
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker,
            initargs=(self.config,),
        ) as ex:
            for result in ex.map(_work_one, [str(f) for f in files]):
                self.stats.files_seen += 1
                if result["error"]:
                    self.stats.files_failed += 1
                    self.stats.errors.append(result["error"])
                    continue
                self.stats.files_parsed += 1
                recs = [ChunkRecord(**r) for r in result["records"]]
                self.stats.chunks += len(recs)
                dt = result["doc_type"]
                self.stats.by_doctype[dt] = self.stats.by_doctype.get(dt, 0) + len(recs)
                out.extend(recs)
        return out

    def ingest_directory(self, dir_path: str | Path) -> List[ChunkRecord]:
        directory = Path(dir_path)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        def _kept(p: Path) -> bool:
            # ``_``-prefixed paths (e.g. an original-binary archive) are never
            # ingested — they exist for audit / re-extraction only.
            return p.is_file() and not any(
                part.startswith("_") for part in p.relative_to(directory).parts)

        files = sorted(p for p in directory.rglob("*") if _kept(p) and is_supported(p))
        records = self.ingest_paths(files)
        logger.info(
            "directory %s: %d/%d files ok, %d chunks (%d failed)",
            directory, self.stats.files_parsed, self.stats.files_seen,
            self.stats.chunks, self.stats.files_failed)
        return records


# ── process-pool worker (module-level so it is picklable) ─────────────────────

_WORKER: Dict[str, Any] = {}


def _init_worker(config: IngestConfig) -> None:
    _WORKER["pipe"] = IngestionPipeline(config)


def _work_one(path: str) -> Dict[str, Any]:
    pipe: IngestionPipeline = _WORKER["pipe"]
    p = Path(path)
    try:
        model = pipe.parse(p)
        records = pipe.chunker.chunk(model)
        pipe._enrich(records, model)
        pipe._tag_languages(records)
        return {
            "error": None,
            "doc_type": model.doc_type,
            "records": [
                {"chunk_id": r.chunk_id, "text": r.text, "source": r.source,
                 "doc_type": r.doc_type, "strategy": r.strategy,
                 "metadata": r.metadata}
                for r in records
            ],
        }
    except Exception as e:
        return {"error": f"{p.name}: {type(e).__name__}: {e}",
                "doc_type": "", "records": []}


# ── module-level convenience API ──────────────────────────────────────────────


def ingest_file(path: str | Path, config: Optional[IngestConfig] = None,
                **kwargs) -> List[Dict[str, Any]]:
    """Ingest one file → list of ``{chunk_id, text, metadata}`` dicts.
    ``kwargs`` override fields on a default (or supplied) ``IngestConfig``."""
    cfg = _merge_config(config, kwargs)
    pipe = IngestionPipeline(cfg)
    return [r.as_dict() for r in pipe.ingest_file(path)]


def ingest_directory(path: str | Path, config: Optional[IngestConfig] = None,
                     **kwargs) -> List[Dict[str, Any]]:
    """Ingest a directory (recursive) → list of ``{chunk_id, text, metadata}``."""
    cfg = _merge_config(config, kwargs)
    pipe = IngestionPipeline(cfg)
    return [r.as_dict() for r in pipe.ingest_directory(path)]


def parse(path: str | Path, config: Optional[IngestConfig] = None,
          **kwargs) -> DocumentModel:
    """Parse one file → ``DocumentModel`` (structure only, no chunking)."""
    cfg = _merge_config(config, kwargs)
    return IngestionPipeline(cfg).parse(path)


def _merge_config(config: Optional[IngestConfig], kwargs: Dict[str, Any]) -> IngestConfig:
    import dataclasses  # noqa: PLC0415
    cfg = config or IngestConfig()
    if kwargs:
        cfg = dataclasses.replace(cfg, **kwargs)
    return cfg
