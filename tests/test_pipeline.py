"""End-to-end: born-digital PDF → parse → chunk → general-purpose records.

The whole no-OCR path runs against a real generated PDF; the one OCR assertion
uses an injected recording stub so no real pack is needed.
"""

import json

from vega import ingest_directory, ingest_file, parse
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


def test_stable_ids_across_relative_and_absolute_paths(born_digital_pdf, monkeypatch):
    # Finding 11: relative vs absolute ingestion of the same file → same ids.
    abs_ids = [c["chunk_id"] for c in ingest_file(born_digital_pdf, ocr_mode="none")]
    monkeypatch.chdir(born_digital_pdf.parent)
    rel_ids = [c["chunk_id"] for c in ingest_file(born_digital_pdf.name, ocr_mode="none")]
    assert abs_ids == rel_ids


def test_ocr_used_is_or_across_merged_pages():
    # Finding 5: a chunk merging an OCR'd page + a born-digital page is ocr_used.
    from vega.model import DocumentModel
    from vega.chunkers.structure import StructureChunker
    pipe = IngestionPipeline(IngestConfig(languages=["en"], ocr_mode="none"))
    m = DocumentModel(source="/x/mix.pdf", doc_type="pdf",
                      metadata={"filename": "mix.pdf", "ocr_pages": [2]})
    from vega.model import Element, ElementType
    m.add(Element(type=ElementType.HEADING, text="S", level=1, page=1))
    m.add(Element(type=ElementType.PARAGRAPH, page=1, text="Born digital page one."))
    m.add(Element(type=ElementType.PARAGRAPH, page=2, text="Scanned page two text."))
    recs = StructureChunker(min_tokens=1).chunk(m)
    pipe._enrich(recs, m)
    spanning = [r for r in recs if set(r.metadata["pages"]) == {1, 2}]
    assert spanning and spanning[0].metadata["ocr_used"] is True


def test_directory_underscore_paths_included_by_default(tmp_path):
    # Finding 12: '_'-prefixed files are ingested unless explicitly skipped.
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    for name in ("visible.pdf", "_hidden.pdf"):
        c = canvas.Canvas(str(tmp_path / name), pagesize=LETTER)
        c.drawString(72, 700, f"Content of {name} with a few words to chunk here.")
        c.showPage(); c.save()

    default = ingest_directory(tmp_path, ocr_mode="none")
    files_default = {c["metadata"]["source_file"] for c in default}
    assert files_default == {"visible.pdf", "_hidden.pdf"}

    skipped = ingest_directory(tmp_path, ocr_mode="none", skip_underscored=True)
    files_skipped = {c["metadata"]["source_file"] for c in skipped}
    assert files_skipped == {"visible.pdf"}


def test_language_tag_uses_dominant_not_presence():
    # Finding 14: a mostly-English chunk with a stray Telugu word is tagged 'en'.
    from vega.model import DocumentModel, Element, ElementType
    from vega.chunkers.structure import StructureChunker
    pipe = IngestionPipeline(IngestConfig(languages=["te", "en"], ocr_mode="none"))
    m = DocumentModel(source="/x/mix.pdf", doc_type="pdf",
                      metadata={"filename": "mix.pdf"})
    m.add(Element(type=ElementType.PARAGRAPH, page=1, text=(
        "The government order was issued by the పరిషత్ office this morning.")))
    recs = StructureChunker(min_tokens=1).chunk(m)
    pipe._tag_languages(recs)
    assert recs[0].metadata["language"] == "en"


def test_tag_languages_detect_always_under_default_lang_en():
    # Detect-always: even with only 'en' declared, a recovered mostly-Tamil chunk
    # tags 'ta', while a clean-English chunk still tags 'en'. (This is what makes
    # the Tamil fix visible in metadata.language under the default --lang en.)
    from vega.records import ChunkRecord
    pipe = IngestionPipeline(IngestConfig(languages=["en"], ocr_mode="none"))
    ta_rec = ChunkRecord(chunk_id="c_ta", text=(
        "தமிழ்நாடு அரசு வேலைவாய்ப்பு மற்றும் பயிற்சித் துறை பொதுத் தமிழ் "
        "இலக்கணம் பகுதி இலக்கணக் குறிப்பறிதல் காப்புரிமை பணியாளர்"))
    en_rec = ChunkRecord(chunk_id="c_en", text=(
        "This clean English paragraph has enough ordinary words to be tagged as "
        "English by the dominant-language detector without any ambiguity here."))
    pipe._tag_languages([ta_rec, en_rec])
    assert ta_rec.metadata["language"] == "ta"    # recovered Tamil → 'ta'
    assert en_rec.metadata["language"] == "en"    # clean English still 'en'


def test_multilingual_pipeline_does_not_pin_first_language():
    # Finding 2 (pipeline half): several non-English languages ⇒ no pinned
    # recovery_script, so per-page OSD/candidate detection decides instead.
    multi = IngestionPipeline(IngestConfig(languages=["te", "hi", "en"]))
    assert multi._recovery_script is None
    single = IngestionPipeline(IngestConfig(languages=["te", "en"]))
    assert single._recovery_script == "tel"      # one non-English ⇒ unambiguous


def test_cli_json_reuses_parsed_models(tmp_path, monkeypatch):
    # Finding 13: --json must reuse ingest's parsed models, not re-parse. We
    # assert the pipeline retains exactly one model per parsed file and that a
    # corrupt file (skipped during ingest) is absent from the retained set.
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas
    good = tmp_path / "good.pdf"
    c = canvas.Canvas(str(good), pagesize=LETTER)
    c.drawString(72, 700, "A good document with enough words to chunk nicely.")
    c.showPage(); c.save()
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")

    pipe = IngestionPipeline(IngestConfig(ocr_mode="none"), keep_models=True)
    pipe.ingest_directory(tmp_path)
    assert pipe.stats.files_failed == 1          # corrupt file isolated
    assert [m.metadata["filename"] for m in pipe.documents] == ["good.pdf"]


def test_writer_jsonl_roundtrip(born_digital_pdf, tmp_path):
    chunks = ingest_file(born_digital_pdf, ocr_mode="none")
    out = tmp_path / "chunks.jsonl"
    n = write_jsonl(chunks, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert n == len(lines) == len(chunks)
    first = json.loads(lines[0])
    assert set(first) == {"chunk_id", "text", "metadata"}
