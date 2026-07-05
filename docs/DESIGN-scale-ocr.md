# Design: scaling vega OCR to large corpora (10k PDFs / ~300k OCR pages)

**Status:** Phases 0, 1 and 2 implemented 2026-07-05. Phase 0: empty results
are no longer cached; Surya batch failures retry per page (with a CUDA cache
purge between retries — an immediate retry after OOM otherwise inherits the
failed call's fragmentation and OOMs again). Phase 2: recognition batch size
is VRAM-aware (`VEGA_GPU_BATCH` override), and — found during Phase 1
verification — Surya's *detection* batch needed the same treatment
(`VEGA_GPU_DET_BATCH`; small cards detect one page at a time, since detection
allocates per-page tensors and a 3-page window OOM'd the 4 GB card). Phase 1:
`plan_recover`/`plan_scanned`/`execute_plans` in `text_recovery`, deferred
mode in the PDF parser (`--no-batch-ocr` opts out), failed pages re-parse
through the single-page path. Golden tests (stub engines) assert page-for-page
equality between modes; on the real GPU, note that neural engines are not
bit-deterministic across batch compositions — a real 6-page kannada run
matched single-page mode on 5/6 chunks byte-for-byte, with one ~0.1% diff on
decorative cover text (predicted by critique C7). Phase 4 core implemented
2026-07-05: cache entries shard into 256 two-hex subdirectories (pre-shard
flat entries miss and re-OCR — disposable by design), and the CLI streams
chunks per-file via `IngestionPipeline.iter_ingest`/`iter_ingest_directory`
(RAM is O(one file); `ingest_paths`/`ingest_directory` keep their list-return
API on top of the iterator). Deferred from Phase 4: LRU eviction and the
resume manifest (both optional; revisit when a real large run wants them).
Phase 3 remains gated on measurement. Written 2026-07-05; assumes the target
deployment gains a large-VRAM GPU (~96 GB class) while dev remains a 4 GB
laptop.

**Goal:** cut wall-clock for a 10k-PDF Indic corpus from *days-to-weeks* to
*about a day or less*, without changing CLI semantics, chunk IDs, the recovery
correctness gates, cache correctness, or per-page engine attribution.

**Non-goals:** multi-GPU, multi-machine distribution, embedding/retrieval.
10k PDFs does not justify distributed complexity on a 96 GB card.

---

## 1. Baseline (what limits throughput today)

| # | Bottleneck | Where |
|---|---|---|
| B1 | OCR runs **one page per call** — recovery and scanned paths call `image_to_text` per page | `text_recovery.recover` / `ocr_scanned`, called from `pdf.py._collect_page` |
| B2 | Surya inference is serialized per process (`_infer_lock`) — correct for one small GPU, but combined with B1 the GPU idles between pages | `surya_backend.py` |
| B3 | `_GPU_RECOGNITION_BATCH = 32` hardcoded for the 4 GB dev card | `surya_backend.py` |
| B4 | `--workers N` = N processes × N **full model copies** (backends aren't picklable) | `pipeline.py` / `config.py` design |
| B5 | Flat OCR cache dir → ~600k files (text + `.engine` sidecars) in one directory at target scale; no eviction | `ocr/cache.py` |
| B6 | Directory ingest accumulates **all chunks in RAM** before writing (~500k chunks ≈ GBs) | `pipeline.py` / CLI writer |
| B7 | Empty OCR results are cached forever — a transient GPU failure permanently poisons a page (observed in the 2026-07-05 kannada run: two `ratio=0.00` pages served from cache) | `ocr/cache.py` |

Napkin throughput: this corpus profile is ~100 % OCR pages (mojibake or
scanned). 300k pages × ~10 s/page/process = ~35 process-days. The 96 GB card
changes nothing by itself — B1–B3 keep it idle.

---

## 2. Design, phased by risk/benefit

### Phase 0 — prerequisites (do first, tiny)

* **P0.1 Don't cache empty OCR results.** `CachingOCRBackend` skips the write
  when `out == ""` (both single and batch paths). Transient failures then
  self-heal on the next run. *This is a correctness fix that matters more the
  bigger the run.*
* **P0.2 Batch failure isolation in `SuryaBackend.image_to_text_batch`.**
  Today one poison image inside a batch throws and the whole window returns
  `""`×N. On exception, retry pages **individually** so one bad render costs
  one page, not a window.

### Phase 1 — batch OCR *within a file* (the big win, no operational change)

Restructure the per-file flow from "detect→render→OCR→verify per page inside
`_collect_page`" into three passes:

1. **Pass A (CPU, page-worker threads, unchanged parallelism):** parse each
   page; run garble detection / scanned detection; resolve the script per page
   (font hint → declared → OSD → candidates — existing cascade, unchanged);
   render PNGs for pages that need OCR. **Defer** the OCR call itself.
2. **Pass B (GPU):** group deferred pages by resolved script string
   (`"kan+eng"`), and per group call `image_to_text_batch(images, script)` in
   **windows of W pages** (W ≈ 16–32; see critique C3). One new plumbing piece:
   `image_to_text_batch_attributed` on the composite + cache, so per-page
   engine attribution survives batching.
3. **Pass C (CPU):** run the existing per-page verification gate
   (`_best_ratio` ≥ `MIN_SCRIPT_CONF`) on each page's text; apply
   replacements; low-confidence pages keep original text + `garble_suspected`
   exactly as today. Pages whose batch output failed verification re-enter a
   **retry batch** with the multi-pack script (mirroring `ocr_scanned`'s
   retry), then give up per current semantics.

Implementation stance: **do not modify `recover()`/`ocr_scanned()`** — they
stay as the single-page path (images, `.txt`, non-batch callers, and the
fallback when batching is disabled). Add a separate batch orchestrator in
`text_recovery` that reuses the same primitives (`is_garbled`, script
resolution, `_ocr_lang`, `_best_ratio`). Golden tests assert batch output ==
single-page output, page for page, on the real sample corpus (tamil, kannada,
malayalam, ap_assembly excerpts).

Why this wins: Surya batches *text-line crops* across input images internally;
giving it 16–32 pages per call yields thousands of line crops per forward pass
instead of ~60, which is exactly what a 96 GB card wants.

### Phase 2 — VRAM-aware batch sizing (hours of work)

* `IngestConfig.gpu_batch` + env `VEGA_GPU_BATCH` (recognition batch size), and
  a window-size knob for Phase 1.
* Default `auto`: probe `torch.cuda.get_device_properties(0).total_memory` —
  < 8 GB → 32 (today's guardrail), ≥ 8 GB → Surya's own default (≈ hundreds).
* The `_infer_lock` **stays** — with batching it no longer costs throughput,
  and it keeps small-GPU deployments safe.

### Phase 3 — process architecture (decide *after* measuring 1+2)

Options, in increasing complexity:

* **(a) Status quo:** N worker processes, each its own model copy, each
  batching its own file. On 96 GB: ~2 GB models + ~0.5 GB CUDA context per
  worker ⇒ 16 workers ≈ 40 GB, fits. CUDA timeslices kernels, so aggregate
  speedup is sublinear (~3–6×, not 16×) — but combined with Phases 1–2 this
  may already hit the target. **Zero new code.**
* **(b) GPU-owning OCR service:** parser workers become CPU-only and submit
  PNG windows to a single GPU process over a queue; cross-*file* batching,
  maximum GPU utilization, but adds daemon lifecycle, backpressure and IPC to
  what is today a pure CLI.
* **(c) Surya 2 via vllm (upstream's own answer to (b)):** on a 96 GB card,
  `surya-ocr` 0.20+ served by vllm *is* the batch OCR service — built,
  maintained, and continuously improved upstream. vega would add a thin
  `surya2` backend speaking the OpenAI-compatible protocol to a configured
  endpoint (the 0.17 in-process backend stays for laptop/dev). See critique
  C1 — this likely beats building (b) ourselves.

**Recommendation:** measure after Phase 1+2 under (a); if GPU utilization is
still low, go (c), not (b).

### Phase 4 — hygiene at scale (independent, hours each)

* **Sharded cache dir** (`ab/cd/<key>`) + optional max-size LRU eviction by
  mtime. Migration: read misses on old flat paths just re-OCR (cache is
  disposable by design); or a one-shot `vega cache migrate`.
* **Streaming JSONL writer:** write each file's chunks as the file completes
  (per-file ordering preserved; `--stats` unchanged). Bounds RAM at O(one
  file), not O(corpus).
* **Resumable manifest** (nice-to-have): a progress log of completed source
  paths so a re-run skips parsed-and-written files entirely rather than
  re-parsing into cache hits.

---

## 3. Self-critique

* **C1 — The 0.17 pin points the wrong way for the 96 GB target.** We pinned
  `surya-ocr<0.18` because 0.20 needs external runtimes — the right call for a
  4 GB laptop, the wrong ceiling for a server GPU. Upstream's vllm-served
  Surya 2 is a better batch-OCR service than anything Phase 3(b) would
  hand-roll, with better models. The design therefore treats Phase 3(b) as a
  last resort and 3(c) as the intended endgame; the honest cost is maintaining
  two Surya integration modes (in-process 0.17 for dev, remote 0.20 for
  deployment).
* **C2 — Phase 1 touches the crown jewel.** `recover()`'s cascade is the most
  hard-won logic in the repo (three mojibake families, two-tier floors,
  OSD symmetry). Mitigated by *not editing it* — the batch orchestrator is
  additive and golden-tested for page-for-page equality against the
  single-page path, and a config flag can force the old path in production.
  Residual risk: drift if future cascade changes forget the orchestrator;
  the golden tests are the guard.
* **C3 — Window size W is a RAM knob, not a VRAM knob.** Rendered 300-dpi
  PNGs are 2–6 MB each; W=32 across 8 page-worker threads can hold ~1.5 GB of
  pixels in host RAM. Surya's `recognition_batch_size` caps the per-forward
  VRAM regardless of W, so W beyond ~32 adds RAM pressure with little GPU
  benefit. Default W=16, cap deferred-render memory, document it.
* **C4 — Grouping by script may fragment.** Batch groups form per (file,
  script). Mixed-script files degrade to small groups — i.e., to today's
  behavior, never worse. Single-language files (the dominant case in this
  corpus) form one big group. Acceptable.
* **C5 — Phase 3(a)'s sublinearity is unproven on this hardware.** The 3–6×
  estimate for multi-process GPU sharing is folklore, not measurement. Hence
  the explicit "measure before choosing" gate — the design refuses to commit
  to (b)/(c) on guesswork.
* **C6 — Cache write amplification and sidecars double file counts.** Phase 4
  sharding handles lookup, but 600k small files is still backup/rsync-hostile.
  Considered and rejected for now: SQLite cache backend (single file,
  concurrent-writer complexity across worker *processes*). Revisit only if
  the sharded dir actually hurts operations.
* **C7 — Determinism must survive batching.** Chunk IDs are content-addressed;
  identical text ⇒ identical IDs, and batch OCR of the same pixels with the
  same engine is deterministic. But *engine mixing* isn't: if a batch window
  fails over to a different engine than the single-page path would have, text
  differs across runs. Mitigation: P0.2's per-page retry keeps failover at
  page granularity (same as today), and attribution makes any mixing visible
  in metadata rather than silent.
* **C8 — What this design deliberately ignores:** Tesseract-path batching
  (subprocess-per-page; parallelizes fine across CPU workers already), EasyOCR
  batch tuning (secondary engine now), multi-GPU (out of scope until a second
  card exists), and the `548 > 512` tokenizer warning (cosmetic, unrelated).

## 4. Suggested execution order

| Step | Content | Effort | Risk |
|---|---|---|---|
| 1 | P0.1 empty-result caching + P0.2 batch failure isolation | hours | low |
| 2 | Phase 2 batch-size config (useful even before Phase 1) | hours | low |
| 3 | Phase 1 batch orchestrator + golden tests | 1–2 days | medium (C2) |
| 4 | Phase 4 sharded cache + streaming writer | hours each | low |
| 5 | Measure on real corpus under Phase 3(a) | half day | — |
| 6 | Phase 3(c) `surya2` remote backend — only if step 5 says so | 1–2 days | medium |
