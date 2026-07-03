"""Structure-aware chunker — breadcrumbs, section boundaries, tables, stable ids."""

from vega.chunkers.structure import StructureChunker
from vega.model import DocumentModel, Element, ElementType, TableData


def _doc():
    m = DocumentModel(source="/x/doc.pdf", doc_type="pdf",
                      metadata={"filename": "doc.pdf"})
    m.add(Element(type=ElementType.TITLE, text="My Doc"))
    m.add(Element(type=ElementType.HEADING, text="Introduction", level=1, page=1))
    m.add(Element(type=ElementType.PARAGRAPH, page=1, text=(
        "Vega keeps the document structure so that each chunk can be prefixed "
        "with its section breadcrumb. This paragraph is deliberately long enough "
        "to clear the minimum-token merge threshold and stand as its own chunk "
        "without being folded into a neighbour by the small-chunk merge pass.")))
    m.add(Element(type=ElementType.HEADING, text="Details", level=1, page=2))
    m.add(Element(type=ElementType.PARAGRAPH, page=2, text=(
        "The second section is on the second page. Because it lives under a "
        "different heading it must never be merged with the first section, and "
        "its page number should propagate into the chunk metadata for citation.")))
    return m


def test_breadcrumb_prefix_and_section_metadata():
    recs = StructureChunker().chunk(_doc())
    assert len(recs) >= 2
    intro = next(r for r in recs if "Introduction" in r.metadata["section_path"])
    details = next(r for r in recs if "Details" in r.metadata["section_path"])

    # Section path carries the doc title then the heading; text is prefixed.
    assert intro.metadata["section_path"] == ["My Doc", "Introduction"]
    assert intro.text.startswith("My Doc › Introduction")
    assert intro.metadata["page"] == 1
    assert details.metadata["page"] == 2
    assert details.metadata["heading"] == "Details"


def test_sections_do_not_merge_across_headings():
    recs = StructureChunker().chunk(_doc())
    paths = {tuple(r.metadata["section_path"]) for r in recs}
    assert ("My Doc", "Introduction") in paths
    assert ("My Doc", "Details") in paths


def test_chunk_ids_are_stable_across_runs():
    ids1 = [r.chunk_id for r in StructureChunker().chunk(_doc())]
    ids2 = [r.chunk_id for r in StructureChunker().chunk(_doc())]
    assert ids1 == ids2
    assert all(cid.startswith("c_") for cid in ids1)


def test_table_becomes_its_own_chunk():
    m = DocumentModel(source="/x/t.pdf", doc_type="pdf",
                      metadata={"filename": "t.pdf"})
    m.add(Element(type=ElementType.HEADING, text="Specs", level=1, page=1))
    m.add(Element(type=ElementType.TABLE, page=1, table=TableData(
        headers=["Param", "Value"],
        rows=[["Voltage", "230V"], ["Current", "5A"]])))
    recs = StructureChunker().chunk(m)
    tables = [r for r in recs if r.metadata.get("is_table")]
    assert len(tables) == 1
    t = tables[0]
    assert t.strategy == "table"
    assert t.metadata["table_shape"] == [2, 2]
    assert "| Param | Value |" in t.text          # rendered as markdown
    assert "230V" in t.text


def test_small_trailing_chunk_merges_within_section():
    m = DocumentModel(source="/x/s.pdf", doc_type="pdf",
                      metadata={"filename": "s.pdf"})
    m.add(Element(type=ElementType.HEADING, text="One", level=1, page=1))
    m.add(Element(type=ElementType.PARAGRAPH, page=1, text="Tiny."))
    m.add(Element(type=ElementType.PARAGRAPH, page=1, text="Also tiny."))
    recs = StructureChunker().chunk(m)
    # Two sub-min paragraphs in one section collapse to a single chunk.
    assert len(recs) == 1
    assert "Tiny." in recs[0].text and "Also tiny." in recs[0].text
