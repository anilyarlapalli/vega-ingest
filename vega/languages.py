"""Canonical language registry — the single source of truth for language codes.

Normalizes user-facing input (full names like ``"Telugu"``, ISO-639-1 ``"te"``,
or Tesseract/ISO-639-2 ``"tel"``) to a canonical **ISO-639-1** code, and resolves
per-language assets (display name, Tesseract OCR pack, Unicode script block,
OSD script name). One place so a caller's declared ``languages`` can be friendly
while everything internal stays ISO.

Supports English + the eleven Indic languages vega OCRs:
te hi mr ta kn ml bn gu pa or as (Assamese shares the Bengali Unicode block
but has its own Tesseract pack, ``asm``).

Adapted from the AgenticAI_Manufacturing ``doc_pipeline.languages`` module.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("vega.languages")

# ISO-639-1 → (English name, Tesseract code, Unicode block (lo, hi) | None,
#              Tesseract-OSD script name)
_LANGS: Dict[str, Tuple[str, str, Optional[Tuple[int, int]], Optional[str]]] = {
    "en": ("English",   "eng", None,             "Latin"),
    "te": ("Telugu",    "tel", (0x0C00, 0x0C7F), "Telugu"),
    "hi": ("Hindi",     "hin", (0x0900, 0x097F), "Devanagari"),
    "mr": ("Marathi",   "mar", (0x0900, 0x097F), "Devanagari"),
    "ta": ("Tamil",     "tam", (0x0B80, 0x0BFF), "Tamil"),
    "kn": ("Kannada",   "kan", (0x0C80, 0x0CFF), "Kannada"),
    "ml": ("Malayalam", "mal", (0x0D00, 0x0D7F), "Malayalam"),
    "bn": ("Bengali",   "ben", (0x0980, 0x09FF), "Bengali"),
    "gu": ("Gujarati",  "guj", (0x0A80, 0x0AFF), "Gujarati"),
    "pa": ("Punjabi",   "pan", (0x0A00, 0x0A7F), "Gurmukhi"),
    "or": ("Odia",      "ori", (0x0B00, 0x0B7F), "Oriya"),
    # Assamese shares the Bengali–Assamese block/OSD script with bn. It sits
    # AFTER bn so bn wins block-histogram ties (dominant_language) and is the
    # canonical owner for an undeclared OSD "Bengali" — same-script languages
    # can only be told apart by the caller's declaration (as with hi/mr).
    "as": ("Assamese",  "asm", (0x0980, 0x09FF), "Bengali"),
}

# Forgiving alias table → ISO-639-1. Built from the registry + variant spellings.
_ALIASES: Dict[str, str] = {}
for _iso, (_name, _tess, _blk, _osd) in _LANGS.items():
    _ALIASES[_iso] = _iso
    _ALIASES[_name.lower()] = _iso
    _ALIASES[_tess] = _iso
_ALIASES.update({
    "oriya": "or", "odiya": "or", "bangla": "bn", "panjabi": "pa",
    "asomiya": "as", "eng": "en",
})


def supported_languages() -> List[str]:
    """All ISO codes vega knows how to route / OCR."""
    return list(_LANGS)


def normalize_language(value) -> Optional[str]:
    """A single name/code → canonical ISO-639-1, or None if unrecognized."""
    if not value:
        return None
    return _ALIASES.get(str(value).strip().lower())


def normalize_languages(values) -> List[str]:
    """A name, comma/slash string, or list → ordered-unique ISO-639-1 list.

    Accepts e.g. ``"Telugu, Hindi, English"``, ``["te","hi"]``, ``"telugu"``.
    Unknown entries are logged and skipped (forgiving, never raises).
    """
    if values is None:
        return []
    if isinstance(values, str):
        parts = re.split(r"[,/]", values) if re.search(r"[,/]", values) else [values]
    else:
        parts = list(values)
    out: List[str] = []
    for p in parts:
        iso = normalize_language(p)
        if iso:
            if iso not in out:
                out.append(iso)
        elif str(p).strip():
            logger.warning("unknown language %r — skipped (supported: %s)",
                           p, ",".join(supported_languages()))
    return out


def language_name(iso: Optional[str]) -> Optional[str]:
    """English display name for an ISO code (e.g. ``te`` → ``Telugu``)."""
    iso = normalize_language(iso)
    return _LANGS[iso][0] if iso else None


def to_tesseract(iso: Optional[str]) -> Optional[str]:
    """Tesseract pack code for an ISO code; None for English/unknown (no OCR)."""
    iso = normalize_language(iso)
    if not iso or iso == "en":
        return None
    return _LANGS[iso][1]


def script_block(iso: Optional[str]) -> Optional[Tuple[int, int]]:
    """Unicode (lo, hi) block for an ISO code; None for Latin/English."""
    iso = normalize_language(iso)
    return _LANGS[iso][2] if iso else None


def language_of_text(text: str, candidates: Optional[List[str]] = None) -> Optional[str]:
    """Detect the dominant non-Latin language of ``text`` by Unicode-block
    histogram, restricted to ``candidates`` (ISO list) when given. Returns the
    ISO code with the most in-block characters, or None (e.g. pure Latin/empty).
    """
    if not text:
        return None
    pool = [c for c in (candidates or list(_LANGS)) if normalize_language(c)]
    pool = [normalize_language(c) for c in pool]
    best, best_n = None, 0
    for iso in pool:
        blk = _LANGS[iso][2]
        if not blk:
            continue
        lo, hi = blk
        n = sum(1 for ch in text if lo <= ord(ch) <= hi)
        if n > best_n:
            best, best_n = iso, n
    return best


def dominant_language(text: str, candidates: Optional[List[str]] = None) -> Optional[str]:
    """Dominant language of ``text`` by character count, **including Latin → 'en'**.

    Unlike :func:`language_of_text` (which only counts non-Latin blocks and so
    fires on a single Telugu char in an English sentence), this compares each
    candidate script's char count against Latin and returns the winner — robust
    for short / mixed text. ``None`` for empty / no-letter text.
    """
    if not text:
        return None
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    best: Optional[str] = "en" if latin else None
    best_n = latin
    pool = [normalize_language(c) for c in (candidates or list(_LANGS))
            if normalize_language(c)]
    for iso in pool:
        blk = _LANGS[iso][2]
        if not blk:
            continue
        lo, hi = blk
        n = sum(1 for ch in text if lo <= ord(ch) <= hi)
        if n > best_n:
            best, best_n = iso, n
    return best


def iso_for_osd_script(osd_script: str, candidates: List[str]) -> Optional[str]:
    """Map a Tesseract-OSD script name (e.g. ``Devanagari``) to an ISO code from
    ``candidates``. Disambiguates shared scripts (Devanagari → hi vs mr) by which
    candidate languages the caller declared."""
    if not osd_script:
        return None
    name = osd_script.strip().lower()
    cand = [normalize_language(c) for c in candidates if normalize_language(c)]
    matches = [iso for iso in cand
               if (_LANGS[iso][3] or "").lower() == name]
    if matches:
        return matches[0]  # first declared candidate using this script
    # Not in candidates but a known script → return the canonical owner anyway.
    for iso, (_n, _t, _b, osd) in _LANGS.items():
        if (osd or "").lower() == name:
            return iso
    return None
