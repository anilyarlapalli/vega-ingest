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
