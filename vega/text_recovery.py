"""Text recovery — rescue clean Unicode from broken/legacy-font extraction.

This module sits *below* the format parsers and owns the one concern they
shouldn't: that a span of "text" pulled off a page may not be real Unicode at
all. Indic documents are routinely typeset in **legacy non-Unicode fonts**
(Shree-Tel, Anu, APHTA …) whose glyphs are mapped onto Latin-1 codepoints.
PyMuPDF faithfully returns those codepoints, so a Telugu page extracts as
``B…{«§æþ`` — mojibake, not recoverable by any downstream embedder or LLM.

The module is **language-independent in logic, parameterised by script.** It
hardcodes no language: detection works for any script, the mechanism (detect →
render → OCR → verify) is identical everywhere, and only the *parameters* (which
Tesseract pack, which Unicode block to verify against) are script-specific data.
Script enters either from the caller's declared language, the legacy font name,
or Tesseract OSD on the rendered pixels.

The actual OCR call is delegated to a pluggable :class:`vega.ocr.OCRBackend`, so
recovery works identically whether the engine is CPU Tesseract or a GPU neural
backend. OSD script detection uses ``vega.ocr.detect_osd_script`` (engine-
agnostic: it needs only ``osd.traineddata``).

Adapted from the AgenticAI_Manufacturing ``doc_pipeline.ingestion.text_recovery``
module — the pytesseract-direct calls are replaced by the OCR backend seam.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("vega.text_recovery")


# ── Script reference data (the only language-specific part) ──────────────────
#
# Canonical script id == the Tesseract language pack code (``tel``, ``hin`` …),
# so one key drives both OCR and verification.

# Major Indic Unicode blocks, keyed by Tesseract code → (lo, hi) codepoints.
_SCRIPT_BLOCKS: Dict[str, Tuple[int, int]] = {
    "hin": (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, …)
    "mar": (0x0900, 0x097F),
    "ben": (0x0980, 0x09FF),  # Bengali
    "pan": (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    "guj": (0x0A80, 0x0AFF),  # Gujarati
    "ori": (0x0B00, 0x0B7F),  # Odia
    "tam": (0x0B80, 0x0BFF),  # Tamil
    "tel": (0x0C00, 0x0C7F),  # Telugu
    "kan": (0x0C80, 0x0CFF),  # Kannada
    "mal": (0x0D00, 0x0D7F),  # Malayalam
}

# ISO-639-1 (the caller's ``languages`` value) → Tesseract code.
_ISO_TO_TESS: Dict[str, str] = {
    "hi": "hin", "mr": "mar", "bn": "ben", "pa": "pan", "gu": "guj",
    "or": "ori", "ta": "tam", "te": "tel", "kn": "kan", "ml": "mal",
}

# Legacy non-Unicode font families → script. Matched as a case-insensitive
# substring of the PyMuPDF font name (after stripping the ``ABCDEE+`` subset
# prefix). Add rows as new legacy fonts are encountered.
_LEGACY_FONT_SCRIPTS: Dict[str, str] = {
    "shree-tel": "tel", "shreetel": "tel", "anu": "tel", "hemalatha": "tel",
    "shree-dev": "hin", "shree-deo": "hin", "kruti dev": "hin", "krutidev": "hin",
    "shree-tam": "tam", "shree-kan": "kan", "shree-mal": "mal",
    "shree-guj": "guj", "shree-ben": "ben", "shree-pun": "pan", "shree-ori": "ori",
}


def script_for_language(language: Optional[str]) -> Optional[str]:
    """Map a caller ``language`` (ISO-639-1, e.g. ``te``) to a Tesseract code
    (``tel``). Returns None for English / unknown — i.e. "no recovery needed"."""
    if not language:
        return None
    lang = language.strip().lower()
    if lang in ("en", "eng", ""):
        return None
    return _ISO_TO_TESS.get(lang) or (lang if lang in _SCRIPT_BLOCKS else None)


def _normalize_font(name: str) -> str:
    # PyMuPDF font names carry a subset tag like ``ABCDEE+SHREE-TEL-0900``.
    return (name or "").split("+", 1)[-1].strip().lower()


def script_from_fonts(font_names) -> Optional[str]:
    """Infer script from a legacy font name, e.g. ``SHREE-TEL-0900`` → ``tel``.
    Returns None when no known legacy font is present."""
    for raw in font_names or ():
        norm = _normalize_font(raw)
        for sig, script in _LEGACY_FONT_SCRIPTS.items():
            if sig in norm:
                return script
    return None


# ── Detection ────────────────────────────────────────────────────────────────

# Latin-1 Supplement accented letters — the dumping ground a legacy Indic font
# maps its glyphs into. A high density of these with no real words is the
# encoding fingerprint of mojibake.
def _garbage_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    accented = sum(1 for c in letters if 0x00C0 <= ord(c) <= 0x024F)
    return accented / len(letters)


def is_garbled(text: str, font_names=None) -> bool:
    """True if ``text`` looks like legacy-font mojibake rather than real text.

    Two independent signals, either sufficient:
      1. A known legacy Indic font is present (decisive, deterministic).
      2. Heuristic: a high density of accented-Latin glyphs — the byte pattern
         a non-Unicode Indic font produces — which clean English/ASCII text and
         normal European prose never reach.

    Deliberately conservative on the heuristic so clean documents never enter
    recovery; the font-name signal carries the real-world cases.
    """
    if script_from_fonts(font_names):
        return True
    return _garbage_ratio(text or "") >= 0.40


# ── Verification ─────────────────────────────────────────────────────────────

def script_ratio(text: str, script: str) -> float:
    """Fraction of letters that fall in ``script``'s Unicode block — the
    post-recovery quality check (clean Telugu ⇒ ~1.0; still-garbage ⇒ ~0.0)."""
    block = _SCRIPT_BLOCKS.get(script)
    if not block or not text:
        return 0.0
    lo, hi = block
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)


# ── Result + orchestration ───────────────────────────────────────────────────

# Minimum fraction of recovered letters that must fall in the target script's
# Unicode block for OCR output to be trusted (else it's garbage — discard).
MIN_SCRIPT_CONF = 0.30


@dataclass
class Recovery:
    text: str               # recovered Unicode, or the original if no-op
    was_recovered: bool     # True only when text was actually replaced
    script: Optional[str]   # Tesseract code used, if any
    method: str             # "none" | "ocr"
    confidence: float       # script_ratio of the result (0..1)


def _noop(text: str = "") -> Recovery:
    """A no-op result that carries the *original* text, so a caller can use
    ``rec.text`` unconditionally rather than special-casing ``was_recovered``."""
    return Recovery(text=text or "", was_recovered=False, script=None,
                    method="none", confidence=0.0)


def _ocr_available(script: str, backend) -> bool:
    # ``script`` may be a ``+``-joined multi-pack (e.g. "tel+hin") when the page
    # script is ambiguous in a multilingual corpus — every part must be present.
    if backend is None:
        return False
    installed = backend.available_scripts()
    return all(p in installed for p in script.split("+"))


def _ocr_lang(script: str, backend) -> str:
    """Tesseract-style ``lang`` string for the page. Indic government/legal docs
    are almost always bilingual (Telugu prose, English names/acronyms/numerals),
    so we co-load ``eng`` when the backend has it — Telugu-only OCR mangles the
    Latin runs (``MR.SPEAKER`` → garbage). Script comes first so it wins."""
    try:
        has_eng = "eng" in backend.available_scripts()
    except Exception:
        has_eng = False
    return f"{script}+eng" if has_eng and "eng" not in script.split("+") else script


def _ocr_png(png_bytes: bytes, script: str, backend) -> str:
    return backend.image_to_text(png_bytes, _ocr_lang(script, backend))


def _best_ratio(text: str, packs: List[str]) -> tuple:
    """Max ``script_ratio`` over a set of Tesseract packs → (ratio, winning pack).
    Used to verify recovered text in a multilingual corpus (we don't know which
    script the page turned out to be until after OCR)."""
    best, win = 0.0, None
    seen = set()
    for p in packs:
        for sp in str(p).split("+"):
            if not sp or sp in seen:
                continue
            seen.add(sp)
            r = script_ratio(text, sp)
            if r > best:
                best, win = r, sp
    return best, win


def recover(
    text: str,
    font_names,
    render_png: Callable[[], bytes],
    *,
    backend=None,
    declared_script: Optional[str] = None,
    candidate_langs: Optional[List[str]] = None,
) -> Recovery:
    """Rescue a page's text to clean Unicode, or return a no-op result.

    Script selection cascade (multilingual-aware): legacy **font name** →
    ``declared_script`` → **Tesseract OSD** on the rendered image (true scanned
    pages) → all ``candidate_langs`` packs joined (let the engine decide). The
    candidate set (the caller's ``languages``, as ISO codes) bounds detection so
    a mixed-language corpus routes each page to the right pack.

    ``was_recovered`` is False for every clean page — the English path is
    completely untouched, and no image is ever rendered for a clean page.
    """
    if not is_garbled(text, font_names):
        return _noop(text)
    if backend is None:
        return _noop(text)

    cand_packs = [p for p in (script_for_language(l) for l in (candidate_langs or [])) if p]

    # ── resolve the OCR pack ──
    script = None
    fs = script_from_fonts(font_names)
    if fs and (not cand_packs or fs in cand_packs):
        script = fs                                    # legacy-font hint wins
    elif declared_script and (not cand_packs or declared_script in cand_packs):
        script = declared_script
    if not script and cand_packs:
        script = _detect_script_osd(render_png, list(candidate_langs or []))  # OSD
    if not script and cand_packs:
        script = "+".join(dict.fromkeys(cand_packs))   # try every candidate pack
    if not script:
        script = fs or declared_script

    if not script:
        logger.warning("text_recovery: garbled text but no script resolved "
                       "(fonts=%s); leaving as-is", list(font_names or [])[:4])
        return _noop(text)
    if not _ocr_available(script, backend):
        logger.warning("text_recovery: script %r detected but OCR pack(s) "
                       "unavailable in backend %r — leaving text as-is (mojibake).",
                       script, getattr(backend, "name", "?"))
        return _noop(text)

    try:
        recovered = _ocr_png(render_png(), script, backend)
    except Exception as e:
        logger.warning("text_recovery: OCR failed for script %r: %r", script, e)
        return _noop(text)

    # Verify against whichever candidate script actually came out on top.
    verify = cand_packs or [script]
    conf, win = _best_ratio(recovered, verify)
    if not recovered or conf < MIN_SCRIPT_CONF:
        logger.warning("text_recovery: OCR produced low-confidence output "
                       "(packs=%s ratio=%.2f) — discarding", script, conf)
        return _noop(text)

    logger.info("text_recovery: recovered page via OCR (pack=%s, detected=%s, "
                "conf=%.2f, chars=%d)", script, win, conf, len(recovered))
    return Recovery(text=recovered, was_recovered=True, script=(win or script),
                    method="ocr", confidence=conf)


def _detect_script_osd(render_png: Callable[[], bytes],
                       candidate_langs: List[str]) -> Optional[str]:
    """OSD script detection on a rendered page, mapped to a candidate Tesseract
    pack. Engine-agnostic (Tesseract OSD needs only ``osd.traineddata``)."""
    if not candidate_langs:
        return None
    try:
        from vega.ocr.tesseract import detect_osd_script  # noqa: PLC0415
        return detect_osd_script(render_png(), list(candidate_langs))
    except Exception as e:
        logger.debug("text_recovery: OSD failed: %r", e)
        return None


def ocr_scanned(
    render_png: Callable[[], bytes],
    *,
    backend=None,
    candidate_langs: Optional[List[str]] = None,
    declared_script: Optional[str] = None,
) -> Recovery:
    """OCR a **scanned page** (no extractable text) in a non-English/multilingual
    corpus, deciding the script **per page** rather than blindly using the first
    declared language:

      1. a single ``declared_script`` (unambiguous single-language corpus), else
      2. Tesseract **OSD** on the rendered pixels, mapped to a declared language, else
      3. every candidate pack joined (let the engine choose).

    The result is gated by the same Unicode-block confidence check as legacy-font
    recovery: if the first script's output is low-confidence it retries with the
    full candidate-pack set, and if that is still garbage it returns a no-op so
    the caller can fall back to plain English OCR."""
    if backend is None:
        return _noop()
    cand_packs = [p for p in (script_for_language(l) for l in (candidate_langs or [])) if p]

    # Build an ordered list of scripts to try (per-page decision, best first).
    attempts: List[str] = []

    def _add(s: Optional[str]) -> None:
        if s and s not in attempts and _ocr_available(s, backend):
            attempts.append(s)

    if declared_script and (not cand_packs or declared_script in cand_packs):
        _add(declared_script)
    if cand_packs:
        _add(_detect_script_osd(render_png, list(candidate_langs or [])))
        _add("+".join(dict.fromkeys(cand_packs)))   # multi-pack fallback
    if not attempts:
        return _noop()

    verify = cand_packs or attempts
    best: Optional[Recovery] = None
    for script in attempts:
        try:
            recovered = _ocr_png(render_png(), script, backend)
        except Exception as e:
            logger.warning("text_recovery: scanned OCR failed (%s): %r", script, e)
            continue
        if not recovered:
            continue
        conf, win = _best_ratio(recovered, verify)
        cand = Recovery(text=recovered, was_recovered=True, script=(win or script),
                        method="ocr", confidence=conf)
        if best is None or conf > best.confidence:
            best = cand
        if conf >= MIN_SCRIPT_CONF:
            logger.info("text_recovery: OCR'd scanned page (pack=%s, detected=%s, "
                        "conf=%.2f, chars=%d)", script, win, conf, len(recovered))
            return cand

    # Every attempt was low-confidence → discard so the caller falls back to
    # plain English OCR rather than emitting mis-scripted garbage.
    if best is not None:
        logger.warning("text_recovery: scanned OCR low-confidence (best=%.2f) — "
                       "discarding, caller falls back to English",
                       best.confidence)
    return _noop()
