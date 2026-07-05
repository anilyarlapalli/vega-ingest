"""Per-page "needs OCR" decision — born-digital skips OCR, scanned triggers it.

Uses a call-recording stub backend so we can assert *whether* OCR ran (and with
which script) without any real Tesseract pack, GPU, or network.
"""

from vega.parsers.image import ImageParser
from vega.parsers.pdf import PDFParser


def test_born_digital_pdf_skips_ocr_entirely(born_digital_pdf, make_ocr_stub):
    stub = make_ocr_stub(scripts=("eng", "tel"), output="SHOULD-NOT-RUN")
    model = PDFParser(ocr_backend=stub).parse(born_digital_pdf)

    assert stub.calls == []                       # text layer present → no OCR
    assert model.metadata["ocr_pages"] == []
    assert model.doc_type == "pdf"
    assert any(e.text for e in model.elements)    # real prose was extracted


def test_scanned_pdf_page_triggers_ocr(scanned_pdf, make_ocr_stub):
    stub = make_ocr_stub(scripts=("eng",), output="text recovered from the scan")
    model = PDFParser(ocr_backend=stub).parse(scanned_pdf)

    assert stub.calls, "scanned page (no text layer) must be OCR'd"
    assert stub.calls[0][0] == "eng"              # English scanned path
    assert model.metadata["ocr_pages"] == [1]


def test_scanned_page_without_backend_degrades_gracefully(scanned_pdf):
    # OCR seam disabled (backend=None): no crash, simply no OCR text.
    model = PDFParser(ocr_backend=None).parse(scanned_pdf)
    assert model.metadata["ocr_pages"] == []
    assert model.metadata["ocr_backend"] is None


def test_standalone_image_is_always_ocred(image_file, make_ocr_stub):
    stub = make_ocr_stub(scripts=("eng",), output="label text on the image")
    model = ImageParser(ocr_backend=stub).parse(image_file)

    assert stub.calls and stub.calls[0][0] == "eng"
    assert model.doc_type == "image"
    assert model.metadata["ocr_pages"] == [1]
    assert model.metadata["total_pages"] == 1


def test_image_without_backend_is_noop(image_file):
    model = ImageParser(ocr_backend=None).parse(image_file)
    assert model.metadata["ocr_pages"] == []
    assert model.doc_type == "image"


def test_scanned_pdf_records_producing_engine(scanned_pdf, make_ocr_stub):
    stub = make_ocr_stub(scripts=("eng",), output="scan text", name="stub")
    model = PDFParser(ocr_backend=stub).parse(scanned_pdf)
    assert model.metadata["ocr_page_engines"] == {1: "stub"}


def test_image_records_producing_engine(image_file, make_ocr_stub):
    stub = make_ocr_stub(scripts=("eng",), output="label", name="stub")
    model = ImageParser(ocr_backend=stub).parse(image_file)
    assert model.metadata["ocr_page_engines"] == {1: "stub"}


def test_no_ocr_means_no_engine_map(born_digital_pdf, make_ocr_stub):
    stub = make_ocr_stub(scripts=("eng",), output="SHOULD-NOT-RUN")
    model = PDFParser(ocr_backend=stub).parse(born_digital_pdf)
    assert model.metadata["ocr_page_engines"] == {}
