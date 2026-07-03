# vega Demo

A recorded end-to-end run of the `vega` CLI on the host it was finished on
(4 Tesseract packs present: `eng hin osd tel`; **no** CUDA GPU). Every command
below was executed as shown.

---

## 1. Environment — `vega info`

```
$ vega info
vega 0.1.0
CUDA GPU available: False
tessdata dir: (ambient tesseract default)
auto-selected OCR backend: tesseract
available OCR packs (4): eng hin osd tel
supported languages:
  en  English
  te  Telugu
  hi  Hindi
  mr  Marathi
  ta  Tamil
  kn  Kannada
  ml  Malayalam
  bn  Bengali
  gu  Gujarati
  pa  Punjabi
  or  Odia
```

No GPU → `auto` selects the CPU Tesseract backend.

---

## 2. Born-digital PDF — parse + chunk, **no OCR**

A two-section born-digital PDF (`report.pdf`, generated with reportlab). Because
the text layer is present, **no OCR runs** — the per-page skip fires on every
page.

```
$ vega ingest report.pdf --ocr none --stats
```

Chunks (JSONL on stdout, one per line — shown here field-by-field):

```
--- chunk 0 ---
chunk_id     : c_dfd66f5cf3e6d0cb
doc_type     : pdf | page: 1 | ocr_used: False | language: en
section_path : ['Vega Ingestion Report', 'Overview']
text[:120]   : Vega Ingestion Report › Overview /  / Vega parses born-digital PDFs with no OCR because the text layer is present. It preser
--- chunk 1 ---
chunk_id     : c_38049f38791214e0
doc_type     : pdf | page: 1 | ocr_used: False | language: en
section_path : ['Vega Ingestion Report', 'Method']
text[:120]   : Vega Ingestion Report › Method /  / The OCR engine is pluggable behind a backend interface. Tesseract runs on CPU; EasyOCR i
```

Stats (stderr):

```json
{
  "files_seen": 1,
  "files_parsed": 1,
  "files_failed": 0,
  "chunks": 2,
  "by_doctype": { "pdf": 2 },
  "errors": []
}
```

Note `ocr_used: False` on every chunk, and the section breadcrumb
(`Report › Overview`) prepended to each chunk's text.

---

## 3. Scanned English image — **real Tesseract OCR**

A synthesized scanned image with no text layer (`scan_en.png`). An image file
*is* a scanned page, so it is always OCR'd — here by the real Tesseract engine.

```
$ vega ingest scan_en.png --ocr tesseract
```

```
chunk_id : c_1bd8d2afc37c715d
doc_type : image | ocr_used: True | backend: tesseract
text     : 'scan_en.png\n\nVega recovers text from scanned images\nusing a pluggable OCR backend.'
```

`ocr_used: True`, `backend: tesseract` — the text was recovered from pixels.

---

## 4. Scanned **Telugu** image — Indic OCR routing + recovery

A synthesized Telugu scanned image (`scan_te.png`, rendered from a Telugu font).
Declaring `--lang te` routes the page to the Telugu pack via the text-recovery
cascade (declared → OSD → pack), using the host's real `tel` traineddata.

```
$ vega ingest scan_te.png --lang te --ocr tesseract -v
```

```
INFO vega.text_recovery: text_recovery: OCR'd scanned page (pack=tel, detected=tel, conf=1.00, chars=46)

doc_type : image | ocr_used: True | backend: tesseract | language: te
text     : scan_te.png

           పరిపాలన పరిషత్తు నుండి ఉత్తర్వు జారీ చేయబడింది
```

Routed to `pack=tel`, script-verification `conf=1.00`, clean Telugu Unicode
recovered — the same cascade recovers legacy non-Unicode-font pages in real PDFs.

---

## 5. OCR backend seam + GPU auto-selection

Confirmed the pluggable seam and GPU auto-selection do not crash when no CUDA
device is present:

```
gpu_available()   : False            # torch present, no CUDA device
auto backend      : tesseract        # auto → CPU Tesseract when no GPU
easyocr scripts   : ['ben','eng','hin','kan','mar','tam','tel']   # static map, no torch import
EasyOCRBackend can be constructed and answers available_scripts() without importing torch/easyocr.
```

The absent-`torch`/`easyocr` degradation paths are covered deterministically in
`tests/test_ocr_selection.py` (the importability probe is monkeypatched).

---

## 6. Test suite

The full suite runs green with only stub OCR engines — no GPU, network, or real
language pack required:

```
$ pytest
............................................................             [100%]
60 passed in 1.54s
```
