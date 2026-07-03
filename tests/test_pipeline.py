"""End-to-end: born-digital PDF → parse → chunk → general-purpose records.

The whole no-OCR path runs against a real generated PDF; the one OCR assertion
uses an injected recording stub so no real pack is needed.
"""

import json

from vega import ingest_file, parse
from vega.config import IngestConfig
from vega.pipeline import IngestionPipeline
from vega.writer import write_jsonl


def test_ingest_born_digital_end_to_end(born_digital_pdf):
    chunks = ingest_file(born_digital_pdf, languages=["en"], ocr_mode="none")
    assert chunks, "born-digital PDF should yield at least one chunk"

    for ch in chunks:
        assert set(ch) == {"chunk_id", "text", "metadata"}   # the contract shape
        assert ch["chunk_id"].startswith("c_")
        assert ch["text"].strip()
        md = ch["metadata"]
        for key in ("source", "source_file", "doc_type", "page",
                    "section_path", "language", "ocr_used"):
            assert key in md, f"missing metadata key {key!r}"
        assert md["doc_type"] == "pdf"
        assert md["source_file"] == "sample.pdf"
        assert md["language"] == "en"
        assert md["ocr_used"] is False           # born-digital → no OCR


def test_chunk_ids_stable_across_ingests(born_digital_pdf):
    a = ingest_file(born_digital_pdf, ocr_mode="none")
    b = ingest_file(born_digital_pdf, ocr_mode="none")
    assert [c["chunk_id"] for c in a] == [c["chunk_id"] for c in b]


def test_born_digital_pipeline_never_calls_ocr(born_digital_pdf, make_ocr_stub):
    # Inject a recording backend into a live pipeline; born-digital ⇒ no calls.
    pipe = IngestionPipeline(IngestConfig(languages=["en"], ocr_mode="tesseract"))
    stub = make_ocr_stub(scripts=("eng", "tel"), output="X")
    pipe._backend, pipe._backend_built = stub, True

    recs = pipe.ingest_file(born_digital_pdf)
    assert recs
    assert stub.calls == []
    assert pipe.stats.files_parsed == 1
    assert pipe.stats.files_failed == 0


def test_parse_returns_document_model(born_digital_pdf):
    model = parse(born_digital_pdf, ocr_mode="none")
    assert model.doc_type == "pdf"
    assert model.metadata["total_pages"] == 2
    kinds = model.summary()["by_type"]
    assert kinds.get("heading", 0) >= 1          # structure survived the parse


def test_unsupported_file_is_isolated_not_fatal(tmp_path):
    bad = tmp_path / "note.docx"
    bad.write_text("not really a docx")
    pipe = IngestionPipeline(IngestConfig(ocr_mode="none"))
    recs = pipe.ingest_file(bad)
    assert recs == []
    assert pipe.stats.files_failed == 1          # skipped, batch survives


def test_writer_jsonl_roundtrip(born_digital_pdf, tmp_path):
    chunks = ingest_file(born_digital_pdf, ocr_mode="none")
    out = tmp_path / "chunks.jsonl"
    n = write_jsonl(chunks, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == len(lines) == len(chunks)
    first = json.loads(lines[0])
    assert set(first) == {"chunk_id", "text", "metadata"}
