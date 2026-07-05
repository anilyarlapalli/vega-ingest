"""``vega`` command-line interface.

    vega ingest <path-or-dir> [--lang te,hi] [--ocr auto|tesseract|easyocr|surya|none]
                              [--workers N] [--out out.jsonl] [--json doc.json]
    vega info                 # OCR backend / GPU / language support

Exposed as the ``vega`` console script (see pyproject.toml).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from vega import __version__
from vega.config import IngestConfig, OCR_MODES
from vega.languages import language_name, normalize_languages, supported_languages
from vega.pipeline import IngestionPipeline
from vega.writer import document_to_dict, write_json, write_jsonl


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vega",
        description="Parse + chunk PDFs and images (GPU-capable, Indic-aware OCR).",
    )
    p.add_argument("--version", action="version", version=f"vega {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="parse + chunk a file or directory")
    ing.add_argument("path", help="file or directory to ingest")
    ing.add_argument("--lang", "--langs", dest="lang", default="en",
                     help="declared language(s), e.g. 'te,hi,en' (default: en). "
                          f"supported: {','.join(supported_languages())} — "
                          "names work too ('Telugu'); see also 'vega info'")
    ing.add_argument("--ocr", choices=OCR_MODES, default="auto",
                     help="OCR backend selection (default: auto)")
    ing.add_argument("--workers", type=int, default=1,
                     help="process-pool size for multi-file runs; a single-file "
                          "run spends these on the PDF's pages (default: 1)")
    ing.add_argument("--page-workers", type=int, default=1,
                     help="thread-pool size for the pages of one PDF (default: 1)")
    ing.add_argument("--no-columns", dest="columns", action="store_false",
                     default=True, help="disable multi-column reading-order detection")
    ing.add_argument("--skip-underscored", action="store_true",
                     help="skip '_'-prefixed paths during directory ingestion")
    ing.add_argument("--out", default=None,
                     help="write chunks as JSONL to this path (default: stdout)")
    ing.add_argument("--json", dest="json_out", default=None,
                     help="also write the DocumentModel(s) as JSON to this path")
    ing.add_argument("--dpi", type=int, default=300, help="OCR render DPI (default: 300)")
    ing.add_argument("--figure-ocr", action="store_true",
                     help="OCR embedded figures too (slower)")
    ing.add_argument("--no-cache", action="store_true", help="disable the OCR disk cache")
    ing.add_argument("--no-batch-ocr", dest="batch_ocr", action="store_false",
                     default=True,
                     help="run page OCR one page at a time instead of batched "
                          "GPU windows (debug / conservative fallback)")
    ing.add_argument("--tessdata-dir", default=None,
                     help="directory of Tesseract *.traineddata packs")
    ing.add_argument("--chunk-tokens", type=int, default=None, help="max tokens per chunk")
    ing.add_argument("--gpu", dest="gpu", action="store_true", default=None,
                     help="force GPU for the neural backend")
    ing.add_argument("--no-gpu", dest="gpu", action="store_false",
                     help="force CPU even if a GPU is present")
    ing.add_argument("--stats", action="store_true", help="print ingest stats to stderr")
    ing.add_argument("-v", "--verbose", action="store_true", help="verbose logging")

    sub.add_parser("info", help="show OCR backend, GPU and language support")
    return p


def _cmd_ingest(args) -> int:
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    langs = normalize_languages(args.lang) or ["en"]
    cfg = IngestConfig(
        languages=langs,
        ocr_mode=args.ocr,
        gpu=args.gpu,
        figure_ocr=args.figure_ocr,
        dpi=args.dpi,
        cache_dir=None,
        tessdata_dir=args.tessdata_dir,
        ocr_cache=not args.no_cache,
        batch_ocr=args.batch_ocr,
        workers=args.workers,
        page_workers=args.page_workers,
        columns=args.columns,
        skip_underscored=args.skip_underscored,
    )
    if args.chunk_tokens:
        cfg.chunk_tokens = args.chunk_tokens

    # A --json dump reuses the DocumentModels parsed during ingest rather than
    # re-parsing (which would repeat OCR and dodge fault isolation). Retaining
    # models needs the in-process path, so force single-worker when dumping.
    want_models = bool(args.json_out)
    if want_models:
        cfg.workers = 1
    pipe = IngestionPipeline(cfg, keep_models=want_models)
    target = Path(args.path)
    if not target.exists():
        print(f"vega: no such path: {target}", file=sys.stderr)
        return 2

    started_at = time.perf_counter()
    # Chunks stream to the sink as each file completes — RAM stays O(one file)
    # for a 10k-PDF directory run instead of O(corpus).
    from itertools import chain  # noqa: PLC0415
    record_lists = (
        pipe.iter_ingest_directory(target) if target.is_dir()
        else iter([pipe.ingest_file(target)])
    )
    records = chain.from_iterable(record_lists)

    if args.out:
        n = write_jsonl(records, args.out)
        print(f"wrote {n} chunks → {args.out}", file=sys.stderr)
    else:
        for r in records:
            print(json.dumps(r.as_dict(), ensure_ascii=False))

    if args.json_out:
        docs = [document_to_dict(m) for m in pipe.documents]
        payload = docs[0] if len(docs) == 1 else docs
        write_json(payload, args.json_out)
        print(f"wrote DocumentModel JSON → {args.json_out}", file=sys.stderr)

    elapsed_seconds = time.perf_counter() - started_at
    if args.stats:
        print(json.dumps(pipe.stats.as_dict(), indent=2), file=sys.stderr)
    if pipe.stats.files_parsed > 0:
        print(f"processed {target} in {elapsed_seconds:.3f}s", file=sys.stderr)
    return 0


def _cmd_info(_args) -> int:
    from vega.ocr import gpu_available, select_backend  # noqa: PLC0415
    from vega.config import default_tessdata_dir  # noqa: PLC0415

    print(f"vega {__version__}")
    print(f"CUDA GPU available: {gpu_available()}")
    tess = default_tessdata_dir()
    print(f"tessdata dir: {tess or '(ambient tesseract default)'}")
    backend = select_backend("auto", tessdata_dir=tess)
    name = getattr(backend, "name", "disabled")
    print(f"auto-selected OCR backend: {name}")
    try:
        scripts = sorted(backend.available_scripts()) if backend else []
        print(f"available OCR packs ({len(scripts)}): {' '.join(scripts)}")
    except Exception as e:
        print(f"available OCR packs: (could not query: {e})")
    print("supported languages:")
    for iso in supported_languages():
        print(f"  {iso}  {language_name(iso)}")
    return 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "info":
        return _cmd_info(args)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
