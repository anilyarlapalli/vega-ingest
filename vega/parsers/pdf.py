"""PDF parser — PyMuPDF (fitz) with structured tables + figure/OCR handling.

The high-value path. Rather than flattening a page to a string:

  · **Tables** are extracted as structured ``TableData`` via ``page.find_tables``
    and their page regions are *excluded* from the prose pass — so table text is
    captured once, structured, not twice as noise.
  · **Headings** are inferred from font size (doc-level body-size baseline →
    larger spans become headings, bucketed into levels) so the chunker can build
    a real section hierarchy.
  · **OCR is decided per page** — a born-digital page with a good text layer
    skips OCR entirely; a text-empty page is rendered + OCR'd as a scanned page;
    legacy non-Unicode Indic fonts are detected and re-OCR'd (``text_recovery``).
  · The actual OCR engine is a pluggable ``vega.ocr.OCRBackend`` (Tesseract CPU
    or a GPU neural backend), injected here — the parser never imports an engine.

Reading order is approximated by sorting blocks top-to-bottom, left-to-right —
good for single-column documents; a layout-model parser can drop in later behind
the same ``Parser`` protocol.

Adapted from the AgenticAI_Manufacturing ``doc_pipeline.ingestion.parsers.pdf``
module — the pytesseract-direct calls become OCR-backend calls, and OCR usage is
tracked per page for chunk metadata.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF

from vega import text_recovery
from vega.model import DocumentModel, Element, ElementType, TableData

logger = logging.getLogger("vega.parsers.pdf")

# PyMuPDF prints advisories to stdout by default. vega's CLI writes JSONL to
# stdout, so a stray advisory would corrupt the stream. Two independent muzzles,
# both best-effort so an older PyMuPDF without one API still works:
#  · ``find_tables`` prints a bare "use pymupdf_layout" hint via ``print()`` —
#    silenced by ``no_recommend_layout()`` / the documented env override;
#  · genuine PyMuPDF messages are routed to stderr via ``set_messages``.
os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")
try:  # pragma: no cover - depends on PyMuPDF build
    fitz.no_recommend_layout()
except Exception:  # noqa: BLE001
    pass
try:  # pragma: no cover - depends on PyMuPDF build
    fitz.set_messages(stream=sys.stderr)
except Exception:  # noqa: BLE001
    pass

_BBox = Tuple[float, float, float, float]


def _overlaps(a: _BBox, b: _BBox, iou_min: float = 0.3) -> bool:
    """True if rectangle ``a`` is mostly inside ``b`` (used to drop prose
    blocks that fall within a detected table region)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(1e-6, (ax1 - ax0) * (ay1 - ay0))
    return (inter / area_a) >= iou_min


def _looks_like_text(s: str) -> bool:
    """Quality gate for OCR output — reject pure garbage ('mi}', 'Oo').

    Requires several real alphabetic words and a reasonable alpha ratio, so
    diagram labels survive while OCR noise on textures/icons is dropped.
    """
    s = (s or "").strip()
    if len(s) < 12:
        return False
    words = [w for w in s.split() if sum(c.isalpha() for c in w) >= 3]
    if len(words) < 3:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in s)
    return (alpha / max(1, len(s))) >= 0.55


def _norm_boiler(text: str) -> str:
    """Normalise a margin line so page numbers collapse together
    ('Page 3' ~ 'Page 4') for repeated-header/footer detection."""
    import re  # noqa: PLC0415
    s = re.sub(r"\d+", "#", (text or "").lower())
    return " ".join(s.split())[:80]


def _in_margin(bbox: Optional[_BBox], page_h: float, frac: float = 0.10) -> bool:
    if not bbox or page_h <= 0:
        return False
    y0, y1 = bbox[1], bbox[3]
    return y1 <= page_h * frac or y0 >= page_h * (1.0 - frac)


def _detect_boilerplate(page_items, page_heights, min_pages: int = 3,
                        frac_pages: float = 0.4) -> set:
    """Normalised margin lines that recur across many pages = running
    headers/footers. Returned as a set of normalised strings to drop."""
    from collections import Counter  # noqa: PLC0415
    n_pages = len(page_items)
    if n_pages < min_pages:
        return set()
    counts: Counter = Counter()
    for items, page_h in zip(page_items, page_heights):
        seen = set()
        for it in items:
            if it.kind != "text" or not _in_margin(it.bbox, page_h):
                continue
            norm = _norm_boiler(it.text)
            if norm and len(norm) >= 3 and norm not in seen:
                counts[norm] += 1
                seen.add(norm)
    threshold = max(min_pages, int(n_pages * frac_pages))
    return {norm for norm, c in counts.items() if c >= threshold}


def _same_region(a: _BBox, b: _BBox) -> bool:
    """Symmetric overlap test for table dedup across detection strategies."""
    return _overlaps(a, b, iou_min=0.5) or _overlaps(b, a, iou_min=0.5)


class _Item:
    """Intermediate, positioned page item before reading-order assembly."""

    __slots__ = ("kind", "y0", "x0", "text", "size", "bbox", "table", "bold")

    def __init__(self, kind, y0, x0, text="", size=0.0, bbox=None,
                 table=None, bold=False):
        self.kind = kind          # "text" | "table" | "figure"
        self.y0 = y0
        self.x0 = x0
        self.text = text
        self.size = size
        self.bbox = bbox
        self.table = table
        self.bold = bold


class PDFParser:
    """Bytes → structured ``DocumentModel`` for PDF files."""

    def __init__(self, ocr_backend=None, recovery_script: Optional[str] = None,
                 candidate_langs: Optional[list] = None, figure_ocr: bool = False,
                 dpi: int = 300, scanned_dpi: int = 200):
        # ``ocr_backend``: a vega.ocr.OCRBackend, or None to disable OCR entirely
        # (born-digital only). ``recovery_script``: Tesseract code for the primary
        # declared language (legacy single-language path). ``candidate_langs``:
        # ISO codes of every language the corpus may contain (multilingual) —
        # bounds per-page OCR routing (font-name → declared → OSD → multi-pack).
        # None ⇒ font-name inference only; clean English PDFs never trigger
        # recovery either way.
        self._backend = ocr_backend
        self._recovery_script = recovery_script
        self._candidate_langs = candidate_langs or []
        self._figure_ocr = figure_ocr
        self._dpi = dpi
        self._scanned_dpi = scanned_dpi

    def _ocr_image(self, pix) -> str:
        """OCR a PyMuPDF pixmap (plain English path). '' when OCR disabled."""
        if self._backend is None:
            return ""
        try:
            return self._backend.image_to_text(pix.tobytes("png"), "eng")
        except Exception as e:
            logger.debug("OCR failed: %r", e)
            return ""

    def parse(self, path: Path) -> DocumentModel:
        path = Path(path)
        doc = fitz.open(str(path))
        ocr = self._backend is not None
        fig_ocr = ocr and self._figure_ocr
        self._fig_count = 0
        model = DocumentModel(
            source=str(path), doc_type="pdf",
            metadata={"filename": path.name, "total_pages": doc.page_count},
        )

        page_items: List[List[_Item]] = []
        page_heights: List[float] = []
        page_ocr: List[bool] = []
        size_hist: dict = {}

        for page in doc:
            items, used = self._collect_page(page, ocr, fig_ocr)
            page_items.append(items)
            page_ocr.append(used)
            page_heights.append(float(page.rect.height) or 1.0)
            for it in items:
                if it.kind == "text" and it.size:
                    key = round(it.size)
                    size_hist[key] = size_hist.get(key, 0) + len(it.text)

        body_size = _modal_body_size(size_hist)
        heading_sizes = _heading_levels(size_hist, body_size)
        boilerplate = _detect_boilerplate(page_items, page_heights)

        for page_no, items in enumerate(page_items, start=1):
            page_h = page_heights[page_no - 1]
            # Reading order: top→bottom, then left→right. Tolerance groups
            # near-equal y into the same visual row before sorting by x.
            items.sort(key=lambda it: (round(it.y0 / 4.0), it.x0))
            for it in items:
                # Drop running headers/footers: short margin text repeated
                # across many pages (page numbers, doc title).
                if (
                    it.kind == "text"
                    and _in_margin(it.bbox, page_h)
                    and _norm_boiler(it.text) in boilerplate
                ):
                    continue
                if it.kind == "table" and it.table is not None:
                    model.add(Element(
                        type=ElementType.TABLE, table=it.table, page=page_no,
                    ))
                elif it.kind == "figure":
                    model.add(Element(
                        type=ElementType.FIGURE, text=it.text, page=page_no,
                        meta={"ocr": bool(it.text)},
                    ))
                elif it.kind == "text" and it.text.strip():
                    model.add(self._text_element(
                        it, page_no, body_size, heading_sizes,
                    ))

        doc.close()
        model.metadata["figure_count_raw"] = self._fig_count
        model.metadata["ocr_pages"] = [i + 1 for i, u in enumerate(page_ocr) if u]
        model.metadata["ocr_backend"] = getattr(self._backend, "name", None)
        logger.info("parsed %s: %s (ocr pages=%s)", path.name,
                    model.summary()["by_type"], model.metadata["ocr_pages"])
        return model

    # ── per-page collection ────────────────────────────────────────────────

    def _collect_page(self, page, ocr: bool, fig_ocr: bool) -> Tuple[List[_Item], bool]:
        """Return (items, ocr_used) for one page. ``ocr_used`` is True when any
        OCR (recovery, scanned-page, or figure) produced text for this page."""
        items: List[_Item] = []
        table_bboxes: List[_BBox] = []
        ocr_used = False

        # 1) Tables first, so their regions can be excluded from prose. Neither
        #    find_tables strategy dominates: ``lines_strict`` catches ruled
        #    tables and rejects column-aligned prose; ``lines`` also catches
        #    borderless/whitespace tables but over-detects. We union both
        #    (strict first, so it wins on overlap) and let _is_real_table be the
        #    precision gate that rejects prose-masquerading-as-table.
        candidates: List[Tuple[_BBox, TableData]] = []
        for strat in ("lines_strict", "lines"):
            try:
                found = page.find_tables(strategy=strat)
            except Exception as e:
                logger.debug("find_tables(%s) failed: %r", strat, e)
                continue
            for tab in (found.tables if found else []):
                td = _to_table_data(tab)
                if td is not None and _is_real_table(td):
                    candidates.append((tuple(tab.bbox), td))
        for bbox, td in candidates:
            if any(_same_region(bbox, ab) for ab in table_bboxes):
                continue   # already captured by an earlier (stricter) strategy
            table_bboxes.append(bbox)
            items.append(_Item("table", bbox[1], bbox[0], bbox=bbox, table=td))

        # 2) Text blocks (excluding anything inside a table region).
        raw_text_len = 0
        page_text_parts: List[str] = []
        page_fonts: set = set()
        try:
            data = page.get_text("dict")
            for block in data.get("blocks", []):
                if block.get("type") != 0:          # 0 == text block
                    continue
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                if any(_overlaps(bbox, tb) for tb in table_bboxes):
                    continue
                text, size, bold, fonts = _block_text(block)
                if not text.strip():
                    continue
                raw_text_len += len(text)
                page_text_parts.append(text)
                page_fonts |= fonts
                items.append(_Item("text", bbox[1], bbox[0], text=text,
                                   size=size, bbox=bbox, bold=bold))
        except Exception as e:
            logger.debug("get_text failed on a page: %r", e)

        # 2b) Legacy non-Unicode font recovery (e.g. Shree-Tel Telugu). The page
        #     HAS extractable text, so the scanned-page path below never fires —
        #     but that text is glyph-garbage. text_recovery detects it (by font
        #     name / glyph density) and re-OCRs the page with the script's pack,
        #     replacing the mojibake with clean Unicode. Zero cost and a no-op on
        #     clean English pages: is_garbled() returns False and we never render.
        if ocr and raw_text_len >= 20:
            page_text = " ".join(page_text_parts)
            if text_recovery.is_garbled(page_text, page_fonts):
                rec = text_recovery.recover(
                    page_text, page_fonts,
                    render_png=lambda: page.get_pixmap(dpi=self._dpi).tobytes("png"),
                    backend=self._backend,
                    declared_script=self._recovery_script,
                    candidate_langs=self._candidate_langs,
                )
                if rec.was_recovered:
                    # Replace the page wholesale: tables/headings inferred from a
                    # broken encoding are equally unreliable, so drop everything
                    # collected so far and emit the recovered clean text.
                    return ([_Item("text", 0.0, 0.0, text=rec.text,
                                   size=0.0, bbox=tuple(page.rect))], True)

        # 3) Scanned page? Almost no extractable text but the page has area →
        #    render + OCR the whole page. For a non-English / multilingual
        #    corpus, route to the right pack (declared / OSD / multi-pack); fall
        #    back to plain English OCR otherwise. Born-digital pages never reach
        #    here — this is the per-page "needs OCR" decision.
        if ocr and raw_text_len < 20 and not table_bboxes:
            if self._candidate_langs or self._recovery_script:
                rec = text_recovery.ocr_scanned(
                    render_png=lambda: page.get_pixmap(dpi=self._dpi).tobytes("png"),
                    backend=self._backend,
                    candidate_langs=self._candidate_langs,
                    declared_script=self._recovery_script,
                )
                if rec.was_recovered:
                    return ([_Item("text", 0.0, 0.0, text=rec.text, size=0.0,
                                   bbox=tuple(page.rect))], True)
            text = self._ocr_image(page.get_pixmap(dpi=self._scanned_dpi))
            if text:
                items.append(_Item("text", 0.0, 0.0, text=text, size=0.0,
                                   bbox=tuple(page.rect)))
                return (items, True)  # scanned page: the page *is* one figure

        # 4) Embedded figures — record their presence; OCR to recover labels.
        try:
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 1:          # 1 == image block
                    continue
                bbox = tuple(block.get("bbox", (0, 0, 0, 0)))
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < 40 or h < 40:                 # skip icons / rules
                    continue
                ocr_text = ""
                if fig_ocr:
                    try:
                        ocr_text = self._ocr_image(
                            page.get_pixmap(clip=fitz.Rect(bbox), dpi=self._scanned_dpi))
                    except Exception:
                        ocr_text = ""
                # Only keep figures that yielded readable text — a figure with
                # no recoverable text adds noise, not retrieval signal.
                self._fig_count += 1
                if _looks_like_text(ocr_text):
                    items.append(_Item("figure", bbox[1], bbox[0], text=ocr_text, bbox=bbox))
                    ocr_used = True
        except Exception as e:
            logger.debug("image enumeration failed on a page: %r", e)

        return (items, ocr_used)

    # ── element classification ─────────────────────────────────────────────

    def _text_element(self, it: _Item, page_no: int, body_size: float,
                      heading_sizes: List[float]) -> Element:
        text = " ".join(it.text.split())
        # Heading: span notably larger than body, OR short+bold.
        level = _heading_level_for(it.size, heading_sizes)
        if level and len(text) <= 160:
            return Element(type=ElementType.HEADING, text=text, level=level,
                           page=page_no, meta={"size": it.size})
        if _looks_like_list_item(it.text):
            return Element(type=ElementType.LIST_ITEM, text=text, page=page_no)
        return Element(type=ElementType.PARAGRAPH, text=text, page=page_no)


# ── module helpers ──────────────────────────────────────────────────────────


def _to_table_data(tab) -> Optional[TableData]:
    try:
        rows = tab.extract()
    except Exception:
        return None
    rows = [[("" if c is None else str(c)) for c in r] for r in (rows or [])]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return None
    headers: List[str] = []
    try:
        if getattr(tab, "header", None) and tab.header.names:
            headers = [("" if h is None else str(h)) for h in tab.header.names]
    except Exception:
        headers = []
    if headers and rows and _row_equals(headers, rows[0]):
        body = rows[1:]
    elif headers:
        body = rows
    else:
        headers, body = rows[0], rows[1:]
    if not body:
        return None
    return TableData(headers=headers, rows=body)


def _is_real_table(td: TableData) -> bool:
    """Reject prose-masquerading-as-table — the dominant find_tables failure.

    A genuine table has ≥2 columns, ≥2 body rows, short cells, and isn't mostly
    empty. A column-aligned paragraph block trips find_tables but fails these.
    """
    if td.n_cols < 2 or td.n_rows < 2:
        return False
    cells = [c for r in td.rows for c in r]
    if not cells:
        return False
    # A single huge cell ⇒ a paragraph captured as a 1×N "table".
    if max((len(c) for c in cells), default=0) > 200:
        return False
    avg_len = sum(len(c) for c in cells) / len(cells)
    if avg_len > 80:
        return False
    empty_frac = sum(1 for c in cells if not c.strip()) / len(cells)
    if empty_frac > 0.6:
        return False
    return True


def _row_equals(a: List[str], b: List[str]) -> bool:
    norm = lambda xs: [" ".join(str(x).split()).lower() for x in xs]
    return norm(a) == norm(b)


def _block_text(block) -> Tuple[str, float, bool, set]:
    """Concatenate a block's spans; return (text, max_span_size, any_bold,
    font_names). Font names feed legacy-font detection in text_recovery."""
    parts: List[str] = []
    max_size = 0.0
    bold = False
    fonts: set = set()
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            t = span.get("text", "")
            if not t:
                continue
            parts.append(t)
            max_size = max(max_size, float(span.get("size", 0.0)))
            font = span.get("font")
            if font:
                fonts.add(font)
            # bit 4 (16) of flags == bold in PyMuPDF
            if int(span.get("flags", 0)) & 16:
                bold = True
        parts.append("\n")
    return ("".join(parts).strip(), max_size, bold, fonts)


def _modal_body_size(size_hist: dict) -> float:
    """Body font size = the size carrying the most characters."""
    if not size_hist:
        return 0.0
    return float(max(size_hist.items(), key=lambda kv: kv[1])[0])


def _heading_levels(size_hist: dict, body_size: float) -> List[float]:
    """Distinct sizes meaningfully larger than body, largest first → levels."""
    if body_size <= 0:
        return []
    bigger = sorted(
        {s for s in size_hist if s >= body_size * 1.15}, reverse=True
    )
    return [float(s) for s in bigger[:4]]   # cap at 4 heading levels


def _heading_level_for(size: float, heading_sizes: List[float]) -> int:
    for i, hs in enumerate(heading_sizes, start=1):
        if size >= hs - 0.5:
            return i
    return 0


_BULLETS = ("•", "-", "*", "·", "‣", "◦", "–")


def _looks_like_list_item(text: str) -> bool:
    s = text.lstrip()
    if s[:1] in _BULLETS:
        return True
    # "1. " / "1) " / "a) " numbered list markers
    head = s[:4]
    return bool(head) and (
        (head[:1].isdigit() and head[1:2] in ".)")
        or (head[:1].isalpha() and head[1:2] in ".)" and len(s) > 2)
    )
