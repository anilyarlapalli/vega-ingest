"""PDF parser internals — bold headings, column reading order, page parallelism."""

from vega.model import ElementType
from vega.parsers.pdf import PDFParser, _Item, _detect_columns, _order_items


# ── finding 9: bold short lines are treated as headings ──────────────────────

def test_bold_short_line_becomes_heading():
    p = PDFParser()
    it = _Item("text", 100.0, 72.0, text="Configuration", size=11.0, bold=True)
    el = p._text_element(it, page_no=1, body_size=11.0, heading_sizes=[16.0])
    assert el.type == ElementType.HEADING
    assert el.meta.get("bold") is True


def test_bold_sentence_is_not_a_heading():
    # A bold *sentence* (ends with a period, long) stays prose.
    p = PDFParser()
    it = _Item("text", 100.0, 72.0, size=11.0, bold=True,
               text="This is a full bold sentence that should remain a paragraph.")
    el = p._text_element(it, page_no=1, body_size=11.0, heading_sizes=[16.0])
    assert el.type == ElementType.PARAGRAPH


# ── finding 10: multi-column reading order ───────────────────────────────────

def test_two_column_reading_order():
    items = []
    for y in (100, 150, 200):
        items.append(_Item("text", y, 50.0, text=f"L{y}"))
    for y in (100, 150, 200):
        items.append(_Item("text", y, 350.0, text=f"R{y}"))
    _order_items(items, page_width=600.0, enable=True)
    order = [it.text for it in items]
    assert order == ["L100", "L150", "L200", "R100", "R150", "R200"]


def test_single_column_is_plain_top_to_bottom():
    assert _detect_columns([_Item("text", 0, 50.0)] * 3, 600.0) == [(0.0, 600.0)]
    items = [_Item("text", 200.0, 55.0, text="b"), _Item("text", 100.0, 50.0, text="a")]
    _order_items(items, page_width=600.0, enable=True)
    assert [it.text for it in items] == ["a", "b"]


def test_columns_flag_off_falls_back_to_scan_order():
    items = []
    for y in (100, 150):
        items.append(_Item("text", y, 50.0, text=f"L{y}"))
    for y in (100, 150):
        items.append(_Item("text", y, 350.0, text=f"R{y}"))
    _order_items(items, page_width=600.0, enable=False)
    # top-to-bottom across the whole page interleaves the columns by row
    assert [it.text for it in items] == ["L100", "R100", "L150", "R150"]


# ── finding 4: page-level parallelism is byte-identical to serial ────────────

def test_page_parallel_matches_serial(multipage_pdf):
    serial = PDFParser(page_workers=1).parse(multipage_pdf)
    parallel = PDFParser(page_workers=4).parse(multipage_pdf)
    to_tuples = lambda m: [(e.type.value, e.text, e.page) for e in m.elements]
    assert to_tuples(serial) == to_tuples(parallel)
    assert serial.metadata["total_pages"] == parallel.metadata["total_pages"] == 6


# ── broken-CMap suspect flag: page-level wiring ──────────────────────────────

def test_clean_pdf_has_no_garble_suspect_pages(born_digital_pdf):
    model = PDFParser().parse(born_digital_pdf)
    assert model.metadata["garble_suspect_pages"] == []


def test_suspect_pages_recorded_even_with_ocr_disabled(born_digital_pdf, monkeypatch):
    # Suspicion is a pure text-analysis verdict — it must be surfaced even when
    # no OCR backend exists (--ocr none), so downstream can still filter.
    from vega import text_recovery
    monkeypatch.setattr(text_recovery, "garble_suspect", lambda t: True)
    model = PDFParser(ocr_backend=None).parse(born_digital_pdf)
    assert model.metadata["garble_suspect_pages"] == [1, 2]


def test_garbled_but_unrecovered_page_is_suspect(born_digital_pdf, monkeypatch):
    # Detector fires but recovery cannot replace the page (here: no pack for the
    # script) → the page ships with its original text AND the suspect flag.
    from vega import text_recovery
    monkeypatch.setattr(text_recovery, "is_garbled", lambda t, f=None: True)

    class _NoPackBackend:
        name = "stub"
        def available_scripts(self):
            return set()
        def image_to_text(self, png, script):
            return ""

    model = PDFParser(ocr_backend=_NoPackBackend()).parse(born_digital_pdf)
    assert model.metadata["garble_suspect_pages"] == [1, 2]


def test_detection_text_includes_table_cells():
    # A garbled page frequently trips find_tables — the mojibake detector must
    # see table-cell text too, or the garbage ships as clean TableData.
    from vega.model import TableData
    from vega.parsers.pdf import _detection_text
    table = _Item("table", 0.0, 0.0,
                  table=TableData(headers=["കϓͩ", "ിളി"],
                                  rows=[["ഒരിടെͰാരϛ", "കാΎϙൽ"]]))
    out = _detection_text(["prose part"], [table])
    assert "prose part" in out
    assert "കϓͩ" in out and "കാΎϙൽ" in out
