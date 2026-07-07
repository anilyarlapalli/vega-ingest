# Testing the vega GPU image on Vast.ai

Plan for validating `Dockerfile.gpu` (`vega-gpu`) on a rented Vast.ai GPU
instance. Reference: https://docs.vast.ai/guides/instances/connect/overview

Two Vast.ai facts shape the whole plan:

- **Vast pulls images from a registry.** Instances are themselves containers,
  so you cannot build an image on one — push `vega-gpu` to Docker Hub (or
  GHCR) first.
- **SSH launch mode overrides the image ENTRYPOINT** and drops you in a shell,
  which is ideal for interactive testing. Entrypoint launch mode runs the
  image as-is (`ENTRYPOINT ["vega"]` + your args) — that's the production
  shape, tested separately in Phase 3.

## Phase 0 — Local prep (one-time)

1. Push the image to a registry:

   ```bash
   docker login
   docker tag vega-gpu:latest <dockerhub-user>/vega-gpu:0.1
   docker push <dockerhub-user>/vega-gpu:0.1     # ~5.85 GB, takes a while
   ```

2. Add your SSH public key in the Vast portal: Account → Keys → SSH Keys
   (`cat ~/.ssh/id_ed25519.pub`).

### Public vs private image

Nothing secret is baked into the image (base + apt/pip packages + `vega/`
source — no keys, no data), so public isn't dangerous per se. But a public
image exposes the source code: anyone can `docker pull` it and read the
`vega` package out of the layers.

**Decision (2026-07-07): keep the repo public for the test window and delete
it from Docker Hub afterwards.** This skips two setup steps that a private
repo would need — Docker Hub's free tier allows only 1 private repo, and
Vast would need pull credentials (a read-only access token in the template's
**Docker Repository Authentication** section). Revisit if the image ever
hosts long-lived deployments.

## Phase 1 — Rent the instance

3. Create a **template** in the portal:
   - Image: `<dockerhub-user>/vega-gpu:0.1`
   - Launch mode: **SSH**
   - Disk: **≥ 40 GB** (6 GB image + ~2 GB neural models on first run +
     corpus + outputs)

4. On the Search page, filter offers:
   - Any modest GPU works (RTX 3060/3090 is plenty — dev target was an
     RTX 3050).
   - **CUDA ≥ 13.0** — the image bundles `torch 2.12.1+cu130` (verified on
     an instance 2026-07-07: a CUDA 12.6 host gives `cuda: False`, "driver
     too old"). The host driver is the one thing the container can't fix
     (see the `Dockerfile.gpu` header comment). NB: the dev venv runs cu126
     torch, but the unpinned `pip install '.[gpu]'` in the image resolved
     cu130 — check `torch.version.cuda` inside the image whenever it's
     rebuilt, and filter Vast offers accordingly. (This is the deferred
     lockfile gap from the production critique biting in practice.)
   - Rent **on-demand** (not interruptible) for a test session.

## Phase 2 — Smoke test over SSH

5. Connect with the command shown on the Instances page:
   `ssh -p <port> root@<ip>`. You land as root with the entrypoint bypassed;
   `vega` is on PATH.

6. Environment sanity checks:

   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   vega --help
   tesseract --list-langs   # should show tel, kan, tam, hin, ...
   ```

7. Upload a test PDF from your machine:

   ```bash
   scp -P <port> kannada.pdf root@<ip>:/data/
   ```

   Kannada is the right choice — we have a local GPU baseline
   (`kannada_gpu.jsonl`) to diff against. Telugu is also verified-good.
   **Avoid Tamil**: the EasyOCR 1.7.2 Tamil model is broken upstream and
   silently falls back to Tesseract, so it wouldn't exercise the GPU path.

8. Cold run (first run downloads ~2 GB of Surya/EasyOCR models into
   `/cache`):

   ```bash
   time vega ingest /data/kannada.pdf --lang kn --out /out/kannada_vast.jsonl
   ```

   In a second SSH session, run `nvidia-smi -l 1` during OCR and confirm the
   python process actually sits on the GPU with real utilization.

9. Warm run: clear the result cache first so you time model-warm OCR rather
   than a cache hit, then rerun:

   ```bash
   rm -rf /cache/ocr
   time vega ingest /data/kannada.pdf --lang kn --out /out/kannada_vast2.jsonl
   ```

   This gives the number that matters for cost estimates: **pages/minute at
   $/hr**.

10. Verify output: `scp` the JSONL back and diff chunk counts and text
    against the local `kannada_gpu.jsonl`. Same model versions should give
    essentially identical text.

## Phase 3 — Entrypoint mode (optional, production shape)

The SSH test validates the software; **Entrypoint launch mode** validates the
image as designed — Vast runs the container as-is, so `ENTRYPOINT ["vega"]`
fires with the args you pass in the template (e.g.
`ingest /data --lang te --out /out/chunks.jsonl`). The catch: no interactive
access, so data must arrive via Vast's cloud-sync (S3/Drive) or a pre-loaded
volume. Worth doing once before any real batch job; skippable for a smoke
test.

## Phase 4 — Teardown

11. **Destroy** (not just stop) the instance from the portal — stopped
    instances still bill for disk. Record total spend and the warm pages/min
    number for the cost model.

---

# Findings log — 2026-07-07 test session

Everything below is the evidence base for the next round of changes.
Findings are numbered **F1–F16**; the action list at the end references them,
and future commits/PRs should cite the finding IDs they address.

## Session record

- **Images pushed** (Docker Hub `yak2373`, public for the test window,
  to be deleted after — see Phase 0 decision):
  - `yak2373/vega-gpu:0.1` — production image, from local `vega-gpu:latest`
    (image id `510504cada96`, built 2026-07-05), digest `sha256:82591bcf…`.
  - `yak2373/vega-gpu:0.1-vast` — SSH-test variant (see caveat section),
    digest `sha256:2cb45417…`.
- **Instances rented** (3 total; first two were teaching moments):
  1. CUDA 12.6 host, production `0.1` image — unreachable (F2).
  2. Same host class, first `0.1-vast` build (openssh only) — unreachable
     (F3).
  3. **The one that worked**: RTX 3060 12 GB, driver 590.48.01 / CUDA 13.1,
    128 cores, 503 GB RAM, `23.227.184.228` — all benchmarks below.
- **Test corpus**: `kannada.pdf` (46 pages, children's storybook, large
  print) → 46 chunks on every run, all backends.

## Benchmark numbers (RTX 3060, ~$0.12/hr)

| Run | Backend | Time | Pages/min |
|---|---|---|---|
| Cold (incl. ~100 MB EasyOCR download) | EasyOCR (Surya broken — F5) | 212.6 s | 13.0 |
| Warm | EasyOCR | 208.7 s | 13.2 |
| Surya, first run (incl. 1.5 GB model download) | Surya-first auto | 278.4 s | 9.9 |
| Surya, warm | Surya-first auto | 279.2 s | 9.9 |
| 2 files × 46 pp, `--workers 2 --no-cache` | Surya-first auto | 370.7 s / 92 pp | **14.9** |

Cost at these rates: **~$0.02 per 100 pages**, ≈ 7–8k pages per dollar —
roughly constant across card classes (F12).

## Findings — Vast.ai platform (deployment how-to)

- **F1** Vast pulls images from a registry only; instances are themselves
  containers, so no on-instance builds. Public Hub repo avoids template
  auth; delete the repo after the test window (proprietary source in
  layers).
- **F2** Vast **SSH launch mode cannot reach the production image**: it
  needs `openssh-server` in the image and a root user; `python:3.12-slim`
  ships neither and `Dockerfile.gpu` ends with `USER vega`. Symptom: the
  instance card shows "Connect" (blue) but the SSH port refuses
  connections.
- **F3** Vast's provisioning appends an **auto-tmux block to
  `/root/.bashrc`**. With tmux missing from the image, *every* shell —
  interactive and `ssh host 'cmd'` alike — dies during bashrc. There is no
  rescue path from outside: Vast's sshd config omits the SFTP subsystem,
  and piping commands into `ssh -tt` loses the race with bashrc. The fix
  must be baked into the image: install `tmux` **and** pre-create
  `/root/.no_auto_tmux` (Vast's official opt-out marker, keeps
  non-interactive commands working). Full recipe in the caveat section
  below.
- **F4** Push logistics: a derived tag reuses all base layers (the
  `0.1-vast` push uploaded ~30 MB, not 5.85 GB). First Docker Hub push
  failed with `unauthorized: access token has insufficient scopes` — the
  login token was read-only; pushing needs a read-write PAT.

## Findings — image bugs

All four are the deferred lockfile gap (item 2 of the 2026-07-05 production
critique) materializing.

- **F5** **`requests` is not declared** but Surya's import chain needs it.
  In the image, Surya fails to build (`No module named 'requests'`) and
  auto mode **silently** degrades to EasyOCR — one WARNING line, exit code
  0, plausible-looking output. Dev venvs hide the bug via transitive
  installs. This is the nastiest failure mode: quality drops with no error.
- **F6** **Unpinned `transformers` resolved 5.13.0** in the image;
  surya-ocr 0.17.1 needs 4.x. Symptom once F5 was patched:
  `'SuryaDecoderConfig' object has no attribute 'pad_token_id'`, again a
  silent WARNING + EasyOCR fallback. Known-good pairing (local venv):
  surya-ocr 0.17.1 + transformers **4.57.6**. On-instance fix that made
  Surya work: `pip install 'transformers==4.57.6'`.
- **F7** **Unpinned torch resolved 2.12.1+cu130** (dev venv: cu126). cu130
  wheels need host driver ≥ 580 / CUDA 13.0; on a CUDA 12.6 host
  `torch.cuda.is_available()` is False ("driver too old") and vega falls
  back to CPU. Host-picking rule until pinned: **filter Vast offers to
  CUDA ≥ 13.0**, and re-check `torch.version.cuda` inside the image after
  every rebuild.
- **F8** **Surya's models bypass the `/cache` volume**: surya 0.17 caches
  to `~/.cache/datalab` (1.5 GB observed), controlled by `MODEL_CACHE_DIR`
  (pydantic BaseSettings, `surya/settings.py:24`) — not `HF_HOME`.
  `Dockerfile.gpu` redirects HF and EasyOCR but not this, so every fresh
  container re-downloads 1.5 GB.

## Findings — performance (why 6 s/page single-stream)

- **F9** **Single-stream is pipeline-bound, not GPU-bound.** Evidence on
  the 128-core box: GPU utilization bursty (5 s samples mostly 0%, spikes
  to 100%); CPU averaged ~2.4 cores (11m22s user / 4m40s real); only
  4–5.4 GB of 12 GB VRAM used. The pipeline alternates serially: rasterize
  a 16-page window (`ocr_window=16`) on ~1 GIL-bound core → GPU burst →
  repeat. Each side waits for the other.
- **F10** **Batch-size knobs are NOT the limiter on ≥ 8 GB cards**:
  `surya_backend._small_gpu()` already hands big cards Surya's own tuned
  defaults (`VEGA_GPU_BATCH`/`VEGA_GPU_DET_BATCH` resolve to None ≥ 8 GB).
  Don't spend effort there.
- **F11** **`--workers 2` (process per file) gave +50%** on the same card:
  9.9 → 14.9 pp/min, GPU pegged at 98–100% during overlap, VRAM
  11.8/12.3 GB. **~5.4 GB per Surya worker process ⇒ 2 workers is a 12 GB
  card's ceiling.** Even at 2 workers the sampler caught idle gaps (both
  workers in CPU phase simultaneously) — headroom remains.
- **F12** **Scaling economics**: pages/$ is roughly flat (~7–8k pages/$)
  across card classes; bigger cards buy wall-clock and ops simplicity, not
  cheaper pages. A 24 GB RTX 4090 (~$0.35–0.40/hr) fits ~4 workers + ~2×
  kernels ⇒ estimated **40–60 pp/min** (unvalidated — see A6).
- **F13** **The single-stream fix is code, not hardware**: prefetch/overlap
  the next window's rasterization while the GPU chews the current one
  (producer-consumer). This is DESIGN-scale-ocr Phase 3 territory and
  raises every deployment's numbers, including the laptop.
- **F14** **Benchmark-corpus representativeness**: the storybook has a few
  large-print lines per page, underfilling Surya's recognition batches.
  Dense Telugu/government documents will show different (likely better)
  GPU utilization per page. Re-benchmark on representative documents
  before hardware decisions.

## Findings — real Telugu corpus (2026-07-07 evening,
`NirvachanaRamayanamAyodhyaKandam.pdf`, 627 pp, 33 MB)

- **F17** **The flagship use case silently fails**: this real target
  document is a scanned book with a *legacy-font-corrupted* OCR text layer
  (`సుర|పభుఃబోలి`, `|ప` = legacy encoding of ప్ర; `(బోలి` = ఁబోలి). The
  text layer is "coherent within one script", so the sanity check accepts
  it: vega emits the corrupt text unchanged (39 `|` + 89 `(` artifacts in a
  10-page slice), never OCRs, exit 0. The known documented limitation, now
  confirmed on the actual corpus vega exists for. Legacy-font recovery did
  not trigger on this font.
  **FIXED same day (A9a)**: new detection signal 5 in `text_recovery.py` —
  legacy *symbol-glyph splicing* (ASCII symbols spliced inside Indic words;
  `_looks_like_symbol_glyphs`, own recover/suspect floors, wired into
  `is_garbled`/`garble_suspect`). Verified on the real slice: 8/10 pages
  auto-recover via Surya (conf 0.77–0.99), output 79% similar to the
  pure-OCR baseline vs 10% to the corrupt layer.
- **F18** **No override exists**: `--ocr surya` selects a backend but OCR
  still only runs on pages the text-layer gate rejects; there is no
  `--force-ocr`. Workaround used for testing: strip the text layer with
  PyMuPDF redactions (`apply_redactions(images=PDF_REDACT_IMAGE_NONE)`),
  which leaves the scan images → pages classify as scanned → OCR runs.
  **FIXED same day (A9b)**: `--force-ocr` flag (`force_ocr` config field)
  routes text-layer pages through the *recovery* path with the clean gate
  skipped — the verify gate still applies, so the original text is kept
  whenever OCR can't beat it. Mirrored in `recover()`/`plan_recover()`
  (inline + batch parity tested). 13 new tests; suite 189 → 202 green.
- **F19** Stripped 10-page slice on the local RTX 3050: OCR triggered
  (Surya, 367 text lines), 44.1 s ≈ **13.6 pp/min single-stream**, and
  output quality far above the corrupt text layer (artifacts 39 → 12;
  భూమినింబోలి బుద్ధి గురుంబోలి read correctly). Residual noise: occasional
  cross-script leakage (Burmese/Bengali glyphs) on this degraded old scan —
  quantify on the full corpus before judging.

## Findings — RTX 4090 validation run (2026-07-07, A6)

Host: RTX 4090 24 GB, driver 590.48.01, 128 cores. Corpus: the stripped
Ramayanam (627 pp as 4 parts). Image deps fixed on-instance (`requests` +
`transformers==4.57.6`) before running.

- **F20** **4 Surya workers do NOT fit on 24 GB.** Each worker takes ~6 GB
  on the 4090 (more than the 3060's 5.4 GB — big-card batch defaults cost
  VRAM too); at `--workers 4` the card sat at 24.0/24.6 GB and a worker
  died mid-run with a Rust-side `pyo3_runtime.PanicException` (the
  `tokenizers` layer under Surya, under memory pressure). Practical
  ceiling: **3 workers**, or 4 with `VEGA_GPU_BATCH` capped.
- **F21** **Worker crashes destroy the whole run** (robustness bug, needs a
  fix): the panic was unpicklable, so `ProcessPoolExecutor.result()` raised
  `PicklingError` in the coordinator and the entire directory ingest
  aborted — 10m42s of compute, **zero bytes written** (single-file baseline
  from the same session was unaffected). Directory ingestion needs
  per-file error containment (catch per future, record in `stats.errors`,
  keep going) and ideally incremental flushing of completed files' chunks.
- Single-stream baseline (part1, 157 pp, incl. model download):
  **413.5 s = 22.8 pp/min** — 2.3× the 3060 single-stream number.
- **The accepted run — `--workers 3`, 627 pp: 968.4 s = 38.9 pp/min
  makespan, 0 failures, 303 chunks.** The raw number is skewed by load
  imbalance (4 files over 3 workers: one worker ran two parts serially, the
  others idled at the end). Steady-state estimate for the fully-loaded
  phase: phase 2 (part4 solo) ≈ the warm single-stream time ≈ 400 s, so
  phase 1 ≈ 570 s for 471 pp ≈ **~50 pp/min** with 3 busy workers — quote
  "~50, floor 39". **A6 acceptance (≥ 40 pp/min): PASS** on steady-state.
  A balanced run (3 or 6 equal files) would pin the exact number. VRAM at
  3 workers: ~21.9 GB (≈ 7 GB/worker on the 4090 — budget 7, not 5.4).
- Chunks flush **per completed file** (228 chunks were on disk mid-run) —
  the w4 crash zeroed output only because no ~157-page file had finished
  yet. A10's containment is still needed; the flushing half exists.
- **F22** Surya misreads this old book's printed ర-vattu stroke as `|`
  (~1.2/page: `|పవేశించెన్` for ప్రవేశించెన్). Not corruption leakage —
  the text layer was stripped — a genuine recognizer misread on degraded
  print. Candidate fix (A12): a small post-OCR Telugu normalizer mapping
  interior `|C` → `C‌్ర` patterns; needs a labeled sample to validate.

## Findings — quality & methodology

- **F15** **Quality verified by eye, not by diff**: Vast Surya output reads
  the known-hard glyphs correctly (ಮಕ್ಕಳ with correct ಮ where EasyOCR reads
  ವು; ಕನ್ನಡ ಅಭಿವೃದ್ಧಿ ಪ್ರಾಧಿಕಾರ clean). Cross-machine similarity scores were
  low (0.28–0.31 mean) but **meaningless**: local baselines were produced
  by different backend mixes/settings (`kannada_gpu_surya.jsonl` even
  contains font-CMap mojibake from a different pipeline mode). Regenerate
  local baselines after the lockfile lands, and only then treat text diffs
  as regression signal.
- **F16** Methodology notes for the next session:
  - `nvidia-smi` sampling at 30 s misses EasyOCR/Surya bursts entirely
    (looked like 0% GPU); 5 s resolves them.
  - Duplicate-file throughput tests need `--no-cache` — the OCR cache keys
    on rendered-page bytes, so copies of one PDF would be cache hits.
  - Remote `time vega …` output over `ssh | grep` pipes is block-buffered;
    nothing appears until the command ends. Check progress via file sizes
    and `nvidia-smi` on the box instead.
  - Cold ≈ warm on datacenter bandwidth; don't burn time separating them.

## Action list (priority order — cite these IDs in commits)

- **A1 (F5, F6, F7)** Declare and pin GPU-path deps in `pyproject.toml`:
  add `requests`, pin `transformers>=4.57,<5` next to the surya pin,
  decide a torch CUDA-flavor policy — then the full lockfile/constraints
  work deferred from the critique. Rebuild the image and verify Surya
  builds in the *bare image* (not a dev venv).
- **A2 (F8)** One-line Dockerfile.gpu fix: add
  `MODEL_CACHE_DIR=/cache/datalab` to the ENV block.
- **A3 (F5, F6)** Make backend degradation loud: auto mode currently
  drops Surya on a single WARNING. Consider a `--require-ocr surya`
  fail-fast flag and/or making `vega info` report per-backend
  availability so a container smoke test catches this class of bug. CI
  (critique item 1) running `vega info` inside the built GPU image would
  have caught both F5 and F6 before any rental.
- **A4 (F2, F3)** Decide where the Vast SSH variant lives: commit a
  `Dockerfile.vast` (or a documented build-arg) rather than leaving the
  recipe only in this doc and a scratchpad.
- **A5 (F13)** Design + implement OCR-window prefetch (DESIGN-scale-ocr
  Phase 3): overlap rasterization with GPU inference in the single-stream
  path.
- **A6 (F11, F12, F14)** Validation rental before production choice: a
  24 GB RTX 4090, representative dense-document corpus, `--workers 4`,
  ~30 min ≈ $0.20. Accept the production recipe if it lands ≥ 40 pp/min.
- **A7 (F15)** After A1: regenerate local baselines (`kannada_gpu*.jsonl`
  etc.) with locked versions and record the backend/settings provenance in
  the filenames or a manifest, so future diffs are regression signal.
- **A8 (F1)** Teardown discipline: destroy the instance (not stop), delete
  the public `yak2373/vega-gpu` Hub repo, and revisit private-registry
  auth (read-only PAT in the Vast template) before any long-lived
  deployment.
- **A9 (F17, F18)** ✅ **DONE 2026-07-07** — both shapes implemented: (a)
  detection signal 5 (symbol-glyph splicing) auto-rejects legacy pages into
  the OCR path; (b) `--force-ocr` as the manual override, verify-gated.
  See the FIXED notes under F17/F18. Uncommitted at time of writing.
- **A10 (F21)** ✅ **DONE 2026-07-07** — two-layer containment: `_work_one`
  now catches `BaseException` (pyo3 panics are not `Exception` subclasses;
  KeyboardInterrupt/SystemExit re-raised), and `_iter_parallel` switched
  from `ex.map` to per-future `submit`/`result()` with a try/except per
  file, so an exception that still crosses the boundary raw (unpicklable,
  `BrokenProcessPool` after an OOM kill) fails only its own file. Chunk
  flushing per completed file already existed. Suite 202 → 204.
- **A11 (F20)** Document/enforce the workers-per-VRAM rule: ~7 GB per Surya
  worker on big cards ⇒ `min(vram_gb // 7, cpu_workers)`; a startup warning
  (or auto-clamp) when the budget exceeds detected VRAM would have
  prevented the crashed run.
- **A12 (F22)** Post-OCR Telugu normalizer for the `|`-as-ర-vattu misread
  (~1.2/page on old print). Validate against a labeled sample first.

## Known caveat — SSH into a slim image

Vast's SSH mode expects to run an sshd inside the container, and
`python:3.12-slim` doesn't ship `openssh-server`; the production image also
ends with `USER vega`, so Vast's bootstrap can't install one either.
**Confirmed 2026-07-07**: the instance reaches "Connect" status but the SSH
port refuses connections. Fix: a test-only tag layered on the production
image —

```dockerfile
FROM yak2373/vega-gpu:0.1
USER root
RUN apt-get update && apt-get install -y --no-install-recommends openssh-server tmux \
    && rm -rf /var/lib/apt/lists/* \
    && touch /root/.no_auto_tmux
```

pushed as `yak2373/vega-gpu:0.1-vast` (~30 MB extra layer; the rest is
shared). Use this tag in the SSH-mode template. Entrypoint mode (Phase 3)
should keep using the clean `0.1` production tag.

Why tmux and the marker file: Vast's provisioning appends an auto-tmux block
to `/root/.bashrc`. With tmux missing, **every** shell — interactive logins
and non-interactive `ssh host 'cmd'` alike — dies during bashrc, so the
instance is unreachable even though sshd answers (confirmed 2026-07-07; SFTP
is no escape hatch either, Vast's sshd config omits the subsystem).
Installing tmux fixes interactive logins; `~/.no_auto_tmux` is Vast's
official opt-out marker and keeps non-interactive commands (scp, remote
benchmarking) working.
