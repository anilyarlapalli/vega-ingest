# vega

**General-purpose PDF + image ingestion — parse and chunk, no embedding.**

vega turns PDFs and standalone images (born-digital *or* scanned) into portable
`{chunk_id, text, metadata}` records ready for any embedder, vector store, or
knowledge graph. It preserves document **structure** (headings, tables, reading
order) through the parse stage, decides **per page** whether OCR is even needed,
and routes Indic scripts — including **legacy non-Unicode fonts** — to the right
OCR pack. The OCR engine is **pluggable**: CPU Tesseract by default; on a CUDA
machine the installed neural backends (**Surya**, then **EasyOCR**) are
auto-selected best-first, with Tesseract as the final fallback.

- **Formats:** PDF + images (`.png .jpg .jpeg .tiff .tif .bmp .webp`), plus a
  light `.txt` convenience path.
- **Languages:** English + the eleven Indic languages vega OCRs — Telugu, Hindi,
  Marathi, Tamil, Kannada, Malayalam, Bengali, Assamese, Gujarati, Punjabi, Odia.
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
pip install -e '.[surya]'   # + surya-ocr (<0.18), torch (GPU neural OCR, best Indic fidelity)
pip install -e '.[tokenizer]'  # + transformers (token-exact chunk sizing)
```

For reproducible installs add `-c constraints.txt` — the known-good lockfile
(pins versions without adding packages; both Dockerfiles use it). Unpinned
resolution has shipped a silently-broken Surya before (see
`docs/TEST-vast.md`, F5/F6).

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
  tesseract-ocr-kan tesseract-ocr-mal tesseract-ocr-ben tesseract-ocr-asm \
  tesseract-ocr-guj tesseract-ocr-pan tesseract-ocr-ori
```

`tesseract --list-langs` shows what is installed. If you keep packs in a custom
directory, point vega at it with `--tessdata-dir /path/to/tessdata` or the
`VEGA_TESSDATA_DIR` environment variable (either sets `TESSDATA_PREFIX` for the
run). A declared language whose pack is missing degrades gracefully: the page is
left as-is rather than crashing the batch.

### GPU / neural OCR (optional)

```bash
pip install -e '.[surya]'      # surya-ocr (<0.18) + torch — preferred in auto mode
pip install -e '.[easyocr]'    # easyocr + torch
```

Both neural engines **auto-use CUDA when `torch.cuda.is_available()`**. Nothing
else is required — vega detects the GPU and composes the installed engines
automatically, Surya first (see [GPU enablement](#gpu-enablement)). With no GPU
(or neither engine installed), vega silently falls back to Tesseract.

Notes for GPU installs:

- Both engines **download their models on first use** (Surya from Hugging Face,
  EasyOCR per-language checkpoints — a few hundred MB each); the first ingest is
  correspondingly slow, later runs are warm.
- Surya is pinned `<0.18` because 0.20+ ("Surya 2") switched to a VLM served by
  external runtimes (a vllm Docker container or a spawned `llama-server`) and no
  longer runs in-process; 0.17.x is the last embeddable release.
- On small cards (≈4 GB) only **one** neural engine fits at a time. That is fine
  inside one vega process (engines load lazily, per page, as needed), but two
  concurrent vega processes both using the GPU can starve each other — the
  loser logs a one-time construction warning and produces empty OCR output.
- Surya's recognition batch size is **VRAM-aware**: cards under 8 GB get a
  conservative cap of 32 (a 4 GB card OOMs on Surya's default), larger cards
  use Surya's own tuned default. Override with `IngestConfig(gpu_batch=N)` or
  `VEGA_GPU_BATCH=N` (see the tuning-knobs table under Throughput).

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
vega ingest big.pdf  --ocr surya                  # one specific neural engine, alone
vega ingest big.pdf  --ocr easyocr --gpu          # force neural GPU backend
vega ingest big.pdf  --no-gpu                     # force CPU even if a GPU exists
```

Key flags: `--lang` (declared languages, e.g. `te,hi,en`), `--ocr`
(`auto|tesseract|easyocr|surya|none`), `--workers` (files, in parallel; a single-file
run spends them on pages), `--page-workers` (pages of one PDF, in parallel),
`--no-columns` (disable multi-column reading-order detection), `--skip-underscored`
(skip `_`-prefixed paths in directory ingestion), `--out` (JSONL), `--json`
(DocumentModel dump — reuses the models parsed during ingest, no re-parse),
`--dpi`, `--figure-ocr`, `--force-ocr` (re-OCR pages that already have a text
layer — for bad/legacy layers the garble detector misses; the original text is
kept when OCR can't beat it), `--no-cache`, `--no-batch-ocr` (per-page OCR
instead of batched GPU windows), `--tessdata-dir`, `--chunk-tokens`,
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

### Custom chunking

Chunking is decoupled from parsing: parsers emit a structured `DocumentModel`
(headings, tables, pages, reading order) and the chunker consumes it. Any
object satisfying the `vega.chunkers.Chunker` protocol —
`chunk(model) -> list[ChunkRecord]` — can replace the default
`StructureChunker`:

```python
class WholePageChunker:
    def chunk(self, model):  # sees full structure; never touches parsing/OCR
        ...

pipe = IngestionPipeline(cfg, chunker=WholePageChunker())
```

One limitation, enforced loudly: a custom chunker is **in-process only**.
Multi-file runs with `workers > 1` rebuild pipelines from the picklable
config inside each worker process, which would silently fall back to the
default chunker — so the pipeline raises instead; use `workers=1` with a
custom chunker (page-level and OCR parallelism are unaffected).

### Output record shape

```json
{
  "chunk_id": "c_dfd66f5cf3e6d0cb",
  "text": "Vega Ingestion Report › Overview\n\nVega parses born-digital PDFs …",
  "metadata": {
    "source": "/abs/report.pdf", "source_file": "report.pdf",
    "doc_type": "pdf", "page": 1, "pages": [1],
    "section_path": ["Vega Ingestion Report", "Overview"],
    "heading": "Overview", "language": "en", "ocr_used": true,
    "ocr_engine": "surya", "garble_suspected": false,
    "backend": "fallback", "ordinal": 0
  }
}
```

`chunk_id` is a stable content-addressed id (normalised source path + structural
position), so re-ingesting the same file — even by a different relative/absolute
path — does not churn ids. `page` is the first contributing page; `pages` lists
**every** page a chunk draws from (a chunk can span a page break), and
`ocr_used` is true if *any* of those pages was OCR'd. `language` is the chunk's
**dominant** language (Latin counts as English), so a mostly-English chunk with
a stray Indic word is tagged `en`. `garble_suspected` is true if any
contributing page ships with text that looks like unrecovered mojibake (see
[Legacy-font recovery](#legacy-font-recovery)) — filter or reprocess those
chunks downstream rather than embedding them blind. `backend` is the name of
the backend object that served the file — under `--ocr auto` on a GPU host that
is the composite `"fallback"`. `ocr_engine` (present only on OCR'd chunks) is
the engine that actually produced the text — `"surya"`, or `"surya+tesseract"`
when a chunk spans pages won by different engines; per-page attribution also
appears in the `-v` recovery logs (`engine=surya`) and survives the OCR disk
cache. Chunks OCR'd before attribution existed (old cache entries) simply omit
the key.

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
never import an engine. A new backend (PaddleOCR, a cloud OCR …) drops in by
implementing the protocol and returning it from `select_backend`. Provided
backends:

| Backend | `name` | Notes |
|---|---|---|
| `TesseractBackend` | `tesseract` | CPU default; broad Indic pack coverage. |
| `EasyOCRBackend` | `easyocr` | GPU-capable neural OCR; lazy import (safe without torch). |
| `SuryaBackend` | `surya` | GPU-capable multilingual neural OCR (surya-ocr, pinned `<0.18` — later releases require external VLM runtimes); one language-agnostic pass covers every vega script, so multi-non-Latin combos like `tel+hin` are fine. Lazy import. |
| `FallbackOCRBackend` | `fallback` | Routes each script to the first wrapped backend that supports it. |
| `CachingOCRBackend` | *(delegates)* | Transparent disk cache keyed on page-bytes + script. |

`image_to_text` must never raise — it returns `""` on failure so the pipeline's
per-file fault isolation holds.

### Throughput

- **Per-page OCR skip** — born-digital pages with a real text layer are never
  rendered or OCR'd; only text-empty (scanned) pages are.
- **Batched page OCR** — within a file, pages that need OCR (recovery or
  scanned) are grouped by resolved script and sent to the backend in windowed
  batches (`ocr_window`, default 16) instead of one call per page — this
  is what lets a large GPU actually stay busy. Semantics are identical to the
  per-page path (same confidence gates; failed pages re-run through it);
  `--no-batch-ocr` opts out. Note: neural engines are not bit-deterministic
  across batch compositions, so a rerun in the other mode can differ by a few
  characters on low-confidence decorative text.
- **CPU batch OCR** — Tesseract runs a batch window across a thread pool
  (`cpu_ocr_threads`, default `min(8, cores)`; each subprocess capped to
  one OpenMP thread while the window runs). Measured on a 29-page 300-dpi
  Tamil PDF: whole-file wall 52s → 19s vs the per-page path, byte-identical
  output — batching speeds up CPU-only runs too, not just GPU ones.
- **File parallelism** — `--workers N` fans files out across a process pool; each
  worker builds its own backend (engines aren't picklable).
- **Page parallelism** — `--page-workers N` fans the pages of one PDF across a
  thread pool (each thread opens its own document handle, since PyMuPDF is not
  safe to share across threads). A single-file run automatically spends
  `--workers` on pages. Output order is deterministic, so chunk ids are
  identical to a serial run. **Caveat:** with batch OCR on (the default), page
  threads only cover parsing and page rendering — both GIL-bound in PyMuPDF —
  so they add little or nothing (measured: a 29-page file got slightly
  *slower* at `--page-workers 8`). The flag earns its keep in `--no-batch-ocr`
  mode, where OCR runs inline in those threads. For throughput across a
  corpus, `--workers` (a process pool, one file per worker) is the knob that
  scales — each process parses, renders and OCRs independently, so all cores
  stay busy regardless of the GIL.
- **Disk cache** — OCR results are cached by a content hash of the rendered
  page bytes + the backend's **version fingerprint** + script. The version
  fingerprint (Tesseract version + tessdata location, or the easyocr/surya-ocr
  package version) means an engine/pack upgrade transparently invalidates stale
  entries. Writes are
  atomic (temp file + `os.replace`) so parallel workers can share one cache dir
  safely (`--no-cache` to disable; `VEGA_OCR_CACHE_DIR` to relocate). Entries
  shard into 256 hash-prefix subdirectories so corpus-scale runs (10⁵+ pages)
  never pile into one flat directory; empty OCR results are never persisted,
  so transient failures self-heal on the next run.
- **Streaming output** — the CLI writes each file's chunks as the file
  completes (`IngestionPipeline.iter_ingest` / `iter_ingest_directory` in the
  API), so a directory run holds one file's chunks in memory, not the corpus'.

### Tuning knobs — single point of truth

Every performance knob lives on `IngestConfig` and resolves with **one
precedence rule**, implemented once in `vega/config.py`:
**explicit config value > `VEGA_*` env var > auto default.** Nothing outside
that module reads the environment for tuning.

| `IngestConfig` field | CLI flag | Env override | Auto default | Controls |
|---|---|---|---|---|
| `workers` | `--workers` | — | 1 | process pool across files |
| `page_workers` | `--page-workers` | — | 1 | thread pool across one PDF's pages |
| `batch_ocr` | `--no-batch-ocr` | — | on | batched vs per-page OCR |
| `force_ocr` | `--force-ocr` | — | off | re-OCR pages that have a text layer |
| `ocr_window` | — | `VEGA_OCR_WINDOW` | 16 | pages per batched-OCR window (host-RAM knob) |
| `gpu_batch` | — | `VEGA_GPU_BATCH` | VRAM-aware (32 under 8 GB, else Surya's own) | Surya recognition batch size |
| `gpu_det_batch` | — | `VEGA_GPU_DET_BATCH` | VRAM-aware (1 under 8 GB, else Surya's own) | Surya detection batch size |
| `cpu_ocr_threads` | — | `VEGA_CPU_OCR_THREADS` | `min(8, cores)` | Tesseract batch-window thread pool |
| `dpi` / `scanned_dpi` | `--dpi` | — | 300 / 200 | OCR render resolution |
| `cache_dir` | — | `VEGA_OCR_CACHE_DIR` | `~/.cache/vega/ocr` | OCR disk cache location |
| `tessdata_dir` | `--tessdata-dir` | `VEGA_TESSDATA_DIR` | ambient install | Tesseract language packs |

```python
# programmatic control — no env vars needed
cfg = IngestConfig(ocr_mode="auto", workers=8, gpu_batch=256, ocr_window=32)
```

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

1. `torch.cuda.is_available()` is true → the installed neural backends composed
   best-first: **Surya** (best measured Indic fidelity, full script coverage),
   then **EasyOCR**, ending at Tesseract — any page or script one engine cannot
   serve falls through to the next.
2. otherwise → **Tesseract** (CPU).

The neural priority is a single tuple — reordering it is the only change needed
to flip which engine `auto` tries first:

```python
# vega/ocr/selection.py
NEURAL_PREFERENCE = ("surya", "easyocr")
```

Override with `--ocr tesseract|easyocr|surya|none` (an explicit engine is used
*alone*, with no fallback chain — deliberate, so benchmarks and debugging stay
honest), and force the device with `--gpu` / `--no-gpu`. Everything degrades
gracefully: if `torch`/`easyocr`/`surya` are not installed, even an explicit
`--ocr easyocr` or `--ocr surya` falls back to Tesseract rather than raising.
Confirm what your machine will do with `vega info`.

### Indicative engine comparison

One real scanned page per script, 300 dpi, warm engines, RTX 3050 Mobile 4 GB —
indicative only, not a rigorous benchmark. *Ratio* is the share of non-space
characters landing in the expected Unicode block (it cannot see
wrong-letter-right-block errors, where manual inspection favoured Surya —
e.g. EasyOCR systematically reads Kannada ಮ as ವು).

| Page | Tesseract (CPU) | EasyOCR (GPU) | Surya (GPU) |
|---|---|---|---|
| Telugu (assembly record) | 3.2 s / 0.912 | 3.0 s / 0.929 | 8.0 s / 0.917 |
| Kannada (book page)      | 2.5 s / 0.943 | 2.9 s / 0.959 | 5.8 s / 0.949 |
| Tamil (exam material)    | 1.2 s / 0.914 | — (broken model) | 4.1 s / 0.968 |

Rule of thumb: **Surya** for fidelity (and the only neural Tamil), **EasyOCR**
for speed on the scripts it serves well, **Tesseract** when there is no GPU —
which is exactly the `auto` composition order.

---

## Language table

| ISO | Language | Tesseract pack | Script (Unicode block) | EasyOCR | Surya |
|-----|----------|----------------|------------------------|:-------:|:-----:|
| en  | English   | eng | Latin              | ✓ | ✓ |
| te  | Telugu    | tel | Telugu  (U+0C00)   | ✓ | ✓ |
| hi  | Hindi     | hin | Devanagari (U+0900)| ✓ | ✓ |
| mr  | Marathi   | mar | Devanagari (U+0900)| ✓ | ✓ |
| ta  | Tamil     | tam | Tamil   (U+0B80)   | ✓* | ✓ |
| kn  | Kannada   | kan | Kannada (U+0C80)   | ✓ | ✓ |
| ml  | Malayalam | mal | Malayalam (U+0D00) | — (Tesseract) | ✓ |
| bn  | Bengali   | ben | Bengali (U+0980)   | ✓ | ✓ |
| as  | Assamese  | asm | Bengali (U+0980)   | ✓ | ✓ |
| gu  | Gujarati  | guj | Gujarati (U+0A80)  | — (Tesseract) | ✓ |
| pa  | Punjabi   | pan | Gurmukhi (U+0A00)  | — (Tesseract) | ✓ |
| or  | Odia      | ori | Odia    (U+0B00)   | — (Tesseract) | ✓ |

\* EasyOCR 1.7.2's released Tamil checkpoint cannot load (upstream charset
mismatch), so Tamil requests to EasyOCR return empty and fall through to the
next backend in `auto` mode.

Language declaration is forgiving — `"Telugu"`, `"te"`, and `"tel"` all
normalise to `te`; a comma/slash list like `"te,hi,en"` is accepted everywhere.
`--lang` is optional even for scanned Indic input: with no declared language,
both the mojibake-recovery path and the scanned-page path resolve the script
per page via Tesseract OSD over every supported language (a Latin/`eng`
detection keeps the plain-English path). Declaring languages is still
recommended — it bounds detection for shared-script corpora and rescues sparse
pages where OSD has too few characters to answer.
Languages sharing a script (Hindi/Marathi on Devanagari, Bengali/Assamese on
the Bengali block) OCR with their own pack when declared, but block-histogram
language *tagging* cannot tell them apart — undeclared shared-script text tags
as the script's canonical owner (`hi`, `bn`).

### Legacy-font recovery

Indic documents are routinely typeset in legacy **non-Unicode** fonts (Shree-Tel,
Anu, Kruti Dev …) whose glyphs sit on Latin-1 codepoints — a Telugu page then
extracts as mojibake, not text. vega detects this (by font name and glyph-density
fingerprint), re-renders the page, and OCRs it with the correct script pack,
replacing the garbage with clean Unicode. The English path is completely
untouched — a clean page is never rendered.

Three disjoint mojibake families are detected, each by its own signal:
**Latin-1 glyph fonts** (Shree/Kruti Dev — accented-glyph density),
**ASCII glyph fonts** (TAB/TSCII/Bamini/Nudi — implausible Latin word shapes),
and **broken ToUnicode CMaps** (real script letters polluted with wrong-block
codepoints, Private-Use-Area glyphs, or U+FFFD — a script-independent Unicode
sanity check on NFC-normalised text). Detection sees table cells too, so a
garbled page that happens to trip the table detector cannot smuggle mojibake
through as structured data.

Recovery is two-tier: heavy corruption OCR-replaces the whole page (still
verified by a Unicode-block confidence gate, so a false fire self-heals to a
no-op); *partial* corruption — say one garbled title line on an otherwise-clean
page, where replacing 95% clean born-digital text with OCR output would be a
net loss — keeps the text and sets `garble_suspected` on the affected chunks
instead, so downstream can filter or reprocess. Known limitation: corruption
that stays **coherent within one script** (wrong-letters-right-block CMaps,
visual-order matras) passes the sanity check undetected.

---

## Deployment (VM / Docker)

Nothing in the core changes for deployment — these are packaging wrappers
around the same `pip install .` + apt steps documented under Install.

### Docker (CPU — parse + chunk with Tesseract)

```bash
docker build -t vega .
docker run --rm \
  -v ./corpus:/data -v ./out:/out -v vega-cache:/cache \
  vega ingest /data --lang ta --out /out/chunks.jsonl
```

~350 MB image: python-slim + Tesseract + all 11 Indic packs + vega.
Volume layout: `/data` read-only input, `/out` output, `/cache` the OCR disk
cache — keep the named volume and repeat runs skip already-OCR'd pages
(writes are atomic, so parallel containers can share it).

> **Bind-mount permissions.** The container runs as a non-root user
> (uid 1000). A bind-mounted output dir (`-v ./out:/out`) is writable only if
> your host uid is also 1000 — otherwise the run dies with
> `PermissionError` on the first write. Two fixes, pick one:
>
> ```bash
> # a) run as your host uid/gid (works with any bind mounts)
> docker run --rm --user "$(id -u):$(id -g)" \
>   -v ./corpus:/data -v ./out:/out -v ./cache:/cache vega ingest /data --out /out/chunks.jsonl
> # b) keep the default user; make the host dir writable by uid 1000
> chown 1000:1000 ./out    # or chmod 777 for a throwaway dir
> ```
>
> With option (a), bind-mount the cache too (as shown) — the `vega-cache`
> *named* volume is initialized owned by uid 1000 and would be unwritable
> for a different `--user`. Named volumes and the default user always work
> together; mixing named volumes with `--user` does not.

### Docker (GPU — Surya + EasyOCR + Tesseract auto mode)

```bash
docker build -f Dockerfile.gpu -t vega-gpu .
docker run --rm --gpus all \
  -v ./corpus:/data -v ./out:/out -v vega-cache:/cache \
  vega-gpu ingest /data --lang ta --out /out/chunks.jsonl
```

Host prerequisites (not containerizable): the NVIDIA driver + 
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Neural models are **not** baked into the image — Surya/EasyOCR download on
first use into `/cache` (`HF_HOME`, `EASYOCR_MODULE_PATH`), so the first run
is slow and every later container start is warm. Tuning knobs pass through as
env vars, e.g. `-e VEGA_GPU_BATCH=256 -e VEGA_OCR_WINDOW=32` on a large card
(see the tuning-knobs table).

### Bare VM (no Docker)

```bash
git clone git@github.com:anilyarlapalli/vega-ingest.git && cd vega-ingest
scripts/setup_vm.sh                # CPU: apt packs + venv + vega, ends with `vega info`
scripts/setup_vm.sh --gpu --warm   # + torch/Surya/EasyOCR, pre-download models
.venv/bin/vega ingest corpus/ --lang ta --out chunks.jsonl
```

The script is idempotent (safe to re-run) and self-verifies. For GPU the only
host prerequisite is a working NVIDIA driver (`nvidia-smi`); torch's pip wheel
bundles its own CUDA runtime.

Note on the OCR cache: entries are keyed by engine **version**, so a cache
warmed on one machine won't pre-warm an environment with a different
Tesseract/Surya version — by design (different engine builds produce
different text).

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
