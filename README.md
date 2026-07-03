# vega

**General-purpose PDF + image ingestion — parse and chunk, no embedding.**

vega turns PDFs and standalone images (born-digital *or* scanned) into portable
`{chunk_id, text, metadata}` records ready for any embedder, vector store, or
knowledge graph. It preserves document **structure** (headings, tables, reading
order) through the parse stage, decides **per page** whether OCR is even needed,
and routes Indic scripts — including **legacy non-Unicode fonts** — to the right
OCR pack. The OCR engine is **pluggable**: CPU Tesseract by default, GPU-capable
EasyOCR auto-selected when a CUDA device is present.

- **Formats:** PDF + images (`.png .jpg .jpeg .tiff .tif .bmp .webp`), plus a
  light `.txt` convenience path.
- **Languages:** English + all ten Indic scripts vega OCRs — Telugu, Hindi,
  Marathi, Tamil, Kannada, Malayalam, Bengali, Gujarati, Punjabi, Odia.
- **Output:** JSONL (one chunk per line) or a whole-document JSON dump of the
  structured element tree.
- **Scope:** parse + chunk **only**. No embeddings, no retrieval — that is the
  next stage's job, and deliberately out of scope.

---

## Install

vega's core (Tesseract CPU path) needs only its base dependencies:

```bash
pip install -e .            # from a checkout
# extras:
pip install -e '.[test]'    # + pytest, reportlab (test suite)
pip install -e '.[easyocr]' # + easyocr, torch (GPU-capable neural OCR)
pip install -e '.[tokenizer]'  # + transformers (token-exact chunk sizing)
```

### System Tesseract + Indic language packs

The Tesseract binary and language packs are **system** packages, installed with
`apt`, not pip. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr            # the engine
sudo apt-get install -y tesseract-ocr-osd        # orientation/script detection

# Indic language packs (install the ones your corpus needs):
sudo apt-get install -y \
  tesseract-ocr-tel tesseract-ocr-hin tesseract-ocr-mar tesseract-ocr-tam \
  tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-ben tesseract-ocr-guj \
  tesseract-ocr-pan tesseract-ocr-ori
```

`tesseract --list-langs` shows what is installed. If you keep packs in a custom
directory, point vega at it with `--tessdata-dir /path/to/tessdata` or the
`VEGA_TESSDATA_DIR` environment variable (either sets `TESSDATA_PREFIX` for the
run). A declared language whose pack is missing degrades gracefully: the page is
left as-is rather than crashing the batch.

### GPU / EasyOCR (optional)

```bash
pip install -e '.[easyocr]'    # pulls easyocr + torch
```

EasyOCR **auto-uses CUDA when `torch.cuda.is_available()`**. Nothing else is
required — vega detects the GPU and selects EasyOCR automatically (see
[GPU enablement](#gpu-enablement)). With no GPU (or no torch/easyocr installed),
vega silently falls back to Tesseract.

---

## Command-line usage

```bash
# Inspect the environment: auto-selected backend, GPU, installed packs, languages
vega info

# Born-digital PDF → JSONL on stdout (one chunk per line). No OCR is run.
vega ingest report.pdf --ocr none

# A directory, recursively, in parallel, written to a file
vega ingest ./corpus --workers 8 --out chunks.jsonl

# One large PDF, pages parallelised across threads
vega ingest big.pdf --page-workers 8 --out big.jsonl
# (a single-file run also spends --workers on the PDF's pages automatically)

# Telugu (+ English) documents — legacy-font recovery + scanned-page OCR
vega ingest go.pdf --lang te,en --out go.jsonl

# Force a backend / device; also dump the structured DocumentModel as JSON
vega ingest scan.png --ocr tesseract --json doc.json --out chunks.jsonl
vega ingest big.pdf  --ocr easyocr --gpu          # force neural GPU backend
vega ingest big.pdf  --no-gpu                     # force CPU even if a GPU exists
```

Key flags: `--lang` (declared languages, e.g. `te,hi,en`), `--ocr`
(`auto|tesseract|easyocr|none`), `--workers` (files, in parallel; a single-file
run spends them on pages), `--page-workers` (pages of one PDF, in parallel),
`--no-columns` (disable multi-column reading-order detection), `--skip-underscored`
(skip `_`-prefixed paths in directory ingestion), `--out` (JSONL), `--json`
(DocumentModel dump — reuses the models parsed during ingest, no re-parse),
`--dpi`, `--figure-ocr`, `--no-cache`, `--tessdata-dir`, `--chunk-tokens`,
`--gpu/--no-gpu`, `--stats`, `-v`. Stdout is **pure JSONL** — progress and stats
go to stderr, so `vega ingest … | jq` just works.

> **`.txt` is a convenience extra**, not part of the core PDF+image scope. Plain
> `.txt` files are still parsed (a light `=== Section ===` convention is honoured)
> and are picked up by directory ingestion, but the focus — and all the OCR /
> layout / recovery machinery — is PDF + images.

## Python API

```python
from vega import ingest_file, ingest_directory, parse, write_jsonl

# One file → list of {chunk_id, text, metadata} dicts
chunks = ingest_file("report.pdf", languages=["en"], ocr_mode="none")

# A directory, in parallel
chunks = ingest_directory("corpus/", languages=["te", "en"], workers=8)
write_jsonl(chunks, "corpus.jsonl")

# Lower-level: just the structured parse (no chunking)
doc = parse("scan.png", ocr_mode="tesseract")   # -> DocumentModel
print(doc.summary())

# Full control via a config object
from vega import IngestConfig, IngestionPipeline
cfg = IngestConfig(languages=["te", "en"], ocr_mode="auto", workers=4,
                   chunk_tokens=400)
pipe = IngestionPipeline(cfg)
chunks = pipe.ingest_directory("corpus/")
print(pipe.stats.as_dict())
```

### Output record shape

```json
{
  "chunk_id": "c_dfd66f5cf3e6d0cb",
  "text": "Vega Ingestion Report › Overview\n\nVega parses born-digital PDFs …",
  "metadata": {
    "source": "/abs/report.pdf", "source_file": "report.pdf",
    "doc_type": "pdf", "page": 1, "pages": [1],
    "section_path": ["Vega Ingestion Report", "Overview"],
    "heading": "Overview", "language": "en", "ocr_used": false,
    "backend": "tesseract", "ordinal": 0
  }
}
```

`chunk_id` is a stable content-addressed id (normalised source path + structural
position), so re-ingesting the same file — even by a different relative/absolute
path — does not churn ids. `page` is the first contributing page; `pages` lists
**every** page a chunk draws from (a chunk can span a page break), and
`ocr_used` is true if *any* of those pages was OCR'd. `language` is the chunk's
**dominant** language (Latin counts as English), so a mostly-English chunk with
a stray Indic word is tagged `en`.

---

## OCR backend plugin model

Every OCR engine implements one small protocol, `vega.ocr.OCRBackend`:

```python
class OCRBackend(Protocol):
    name: str
    def available_scripts(self) -> Set[str]: ...          # e.g. {"eng", "tel"}
    def image_to_text(self, image_png: bytes, script: str) -> str: ...
    def image_to_text_batch(self, images, script) -> list[str]: ...
```

Parsers and the text-recovery cascade only ever talk to this interface — they
never import an engine. A new backend (PaddleOCR, Surya, a cloud OCR) drops in by
implementing the protocol and returning it from `select_backend`. Provided
backends:

| Backend | `name` | Notes |
|---|---|---|
| `TesseractBackend` | `tesseract` | CPU default; broad Indic pack coverage. |
| `EasyOCRBackend` | `easyocr` | GPU-capable neural OCR; lazy import (safe without torch). |
| `FallbackOCRBackend` | `fallback` | Routes each script to the first wrapped backend that supports it. |
| `CachingOCRBackend` | *(delegates)* | Transparent disk cache keyed on page-bytes + script. |

`image_to_text` must never raise — it returns `""` on failure so the pipeline's
per-file fault isolation holds.

### Throughput

- **Per-page OCR skip** — born-digital pages with a real text layer are never
  rendered or OCR'd; only text-empty (scanned) pages are.
- **File parallelism** — `--workers N` fans files out across a process pool; each
  worker builds its own backend (engines aren't picklable).
- **Page parallelism** — `--page-workers N` fans the pages of one PDF across a
  thread pool (each thread opens its own document handle, since PyMuPDF is not
  safe to share across threads). A single-file run automatically spends
  `--workers` on pages, so a large PDF uses all the cores. Output order is
  deterministic, so chunk ids are identical to a serial run.
- **Disk cache** — OCR results are cached by a content hash of the rendered
  page bytes + the backend's **version fingerprint** + script. The version
  fingerprint (Tesseract version + tessdata location, or EasyOCR version) means
  an engine/pack upgrade transparently invalidates stale entries. Writes are
  atomic (temp file + `os.replace`) so parallel workers can share one cache dir
  safely (`--no-cache` to disable; `VEGA_OCR_CACHE_DIR` to relocate).

### Reading order & multi-column pages

Reading order is reconstructed per page. A conservative **column detector**
(left-edge clustering with a gutter test) reads clearly two-column pages
column-by-column; single-column pages are unaffected. It is on by default and a
no-op when no clear columns are present — disable with `--no-columns` (or
`IngestConfig(columns=False)`) if a particular layout confuses it. The detector
targets the common 2-column case; dense mixed / newspaper layouts may still
interleave and are a known limitation.

---

## GPU enablement

Selection policy (`--ocr auto`, the default):

1. `torch.cuda.is_available()` is true → **EasyOCR** (neural, batched), composed
   with Tesseract so scripts EasyOCR lacks (Malayalam, Gujarati, Gurmukhi, Odia)
   fall back automatically.
2. otherwise → **Tesseract** (CPU).

Override with `--ocr tesseract|easyocr|none`, and force the device with
`--gpu` / `--no-gpu`. Everything degrades gracefully: if `torch`/`easyocr` are
not installed, even an explicit `--ocr easyocr` falls back to Tesseract rather
than raising. Confirm what your machine will do with `vega info`.

---

## Language table

| ISO | Language | Tesseract pack | Script (Unicode block) | EasyOCR |
|-----|----------|----------------|------------------------|:-------:|
| en  | English   | eng | Latin              | ✓ |
| te  | Telugu    | tel | Telugu  (U+0C00)   | ✓ |
| hi  | Hindi     | hin | Devanagari (U+0900)| ✓ |
| mr  | Marathi   | mar | Devanagari (U+0900)| ✓ |
| ta  | Tamil     | tam | Tamil   (U+0B80)   | ✓ |
| kn  | Kannada   | kan | Kannada (U+0C80)   | ✓ |
| ml  | Malayalam | mal | Malayalam (U+0D00) | — (Tesseract) |
| bn  | Bengali   | ben | Bengali (U+0980)   | ✓ |
| gu  | Gujarati  | guj | Gujarati (U+0A80)  | — (Tesseract) |
| pa  | Punjabi   | pan | Gurmukhi (U+0A00)  | — (Tesseract) |
| or  | Odia      | ori | Odia    (U+0B00)   | — (Tesseract) |

Language declaration is forgiving — `"Telugu"`, `"te"`, and `"tel"` all
normalise to `te`; a comma/slash list like `"te,hi,en"` is accepted everywhere.

### Legacy-font recovery

Indic documents are routinely typeset in legacy **non-Unicode** fonts (Shree-Tel,
Anu, Kruti Dev …) whose glyphs sit on Latin-1 codepoints — a Telugu page then
extracts as mojibake, not text. vega detects this (by font name and glyph-density
fingerprint), re-renders the page, and OCRs it with the correct script pack,
replacing the garbage with clean Unicode. The English path is completely
untouched — a clean page is never rendered.

---

## Development

```bash
pip install -e '.[test]'
pytest                          # default suite: stubs only, no real pack needed
pytest -m integration           # real Tesseract / threads / process pool / corrupt input
vega info                       # sanity-check the local OCR environment
```

The **default** suite uses in-memory stub OCR engines — no GPU, network, or real
language pack is required to run it green. Real-OCR, concurrency, and
corrupt-input tests are marked `integration` and **skip themselves** when their
dependency (e.g. a Tesseract pack) is absent, so they never make the default
suite flaky. See [docs/DEMO.md](docs/DEMO.md) for a recorded end-to-end run.
