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
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("vega.text_recovery")


# ── Script reference data (the only language-specific part) ──────────────────
#
# Canonical script id == the Tesseract language pack code (``tel``, ``hin`` …),
# so one key drives both OCR and verification.

# Major Indic Unicode blocks, keyed by Tesseract code → (lo, hi) codepoints.
# FOLLOW-UP: this table and ``vega.languages._LANGS`` duplicate the block data;
# derive one from the other in a later cleanup (out of scope for this fix).
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


# ── Generic ASCII glyph-mojibake detector (language/script independent) ───────
#
# A second, disjoint family of legacy fonts (TAB/TSCII/Bamini/VANAVIL/SunTommy …)
# maps its glyphs onto **plain ASCII** (a-z, ';', ':') rather than Latin-1
# accented codepoints, so ``_garbage_ratio`` reads 0.0 and misses them entirely.
# The tell-tale is that the ASCII "words" are not Latin words: they have almost
# no vowels and are peppered with mid-word ';'/':' (a virama/pulli glyph mapped
# onto punctuation). This detector is script-independent — it keys on the
# *failure of Latin plausibility*, not on any target language.
#
# Thresholds are pinned as a COMPOUND condition and were calibrated against the
# real ``jkpo;ehL``-style tamil.pdf page plus a negative corpus (English prose,
# Python source, URL lists, SKU/ID tables, base64). Measured values on that
# corpus (word-token count / ASCII-letter dominance / vowel ratio among ASCII
# letters / mid-word-punct-word fraction / no-vowel-word fraction /
# digit ratio / uppercase ratio):
#   jkpo;ehL mojibake : 40 / 1.00 / 0.10 / 0.62 / 0.60 / 0.00 / 0.18  → GARBLED
#   English prose     : 32 / 1.00 / 0.38 / 0.00 / 0.03 / 0.00 / 0.01  → clean
#   Python source     : 22 / 1.00 / 0.36 / 0.00 / 0.00 / 0.03 / 0.02  → clean
#   URL list          :  4 / 1.00 / 0.31 / 0.00 / 0.21 / 0.03 / 0.00  → clean
#   SKU / ID table    : 12 / 1.00 / 0.11 / 0.08 / 0.50 / 0.44 / 0.96  → clean
#   base64 blob       :  3 / 1.00 / 0.39 / 0.00 / 0.20 / 0.08 / 0.72  → clean
#
# No single signal is used alone — in particular the no-vowel-word signal is
# NEVER decisive by itself (it fires on ID/part-number tables); it only counts
# when the text also reads as lowercase running prose (few digits, little upper).
_MIN_WORD_TOKENS = 8       # guard: short strings (headers, IDs) are never judged
_MIN_ASCII_DOM = 0.85      # must be ASCII-letter dominated (Latin-1 → other path)
_MAX_VOWEL_RATIO = 0.35    # Latin prose sits ~0.38–0.40; mojibake is far below
_MIN_INWORD_PUNCT_FRAC = 0.15   # path A: ';'/':' *between letters* (not code colons)
_MIN_NOVOWEL_FRAC = 0.40        # path B: consonant-run words …
_MAX_DIGIT_RATIO = 0.10         # … but only when NOT a code/ID table (few digits …
_MAX_UPPER_RATIO = 0.30         # … and mostly lowercase, i.e. running prose)

_INWORD_PUNCT = re.compile(r"[A-Za-z][;:][A-Za-z]")
_VOWELS = frozenset("aeiouAEIOU")


def _looks_like_glyph_mojibake(text: str) -> bool:
    """True if ``text`` is ASCII-glyph legacy-font mojibake (script-independent).

    Compound test (see the block comment above for pinned thresholds): enough
    word tokens, ASCII-letter dominated, an implausibly low vowel ratio, AND one
    of two corroborating signals — a high fraction of words with a mid-word
    ``;``/``:`` (the virama-mapped-to-punctuation tell), OR a high no-vowel-word
    fraction *combined with* a running-prose shape (few digits, mostly lowercase)
    so ID/SKU/part-number tables and hex/base64 blobs do not qualify.
    """
    text = text or ""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_letters = [c for c in letters if c.isascii()]
    if len(ascii_letters) / len(letters) < _MIN_ASCII_DOM:
        return False            # Latin-1 accented soup → handled by _garbage_ratio

    tokens = [w for w in text.split()
              if any(c.isalpha() and c.isascii() for c in w)]
    n = len(tokens)
    if n < _MIN_WORD_TOKENS:
        return False

    vowel_ratio = sum(1 for c in ascii_letters if c in _VOWELS) / len(ascii_letters)
    if vowel_ratio >= _MAX_VOWEL_RATIO:
        return False            # plausible Latin vowel structure → not mojibake

    def _alpha(w: str) -> List[str]:
        return [c for c in w if c.isalpha()]

    inword_punct = sum(1 for w in tokens if _INWORD_PUNCT.search(w))
    novowel = sum(1 for w in tokens
                  if len(_alpha(w)) >= 2 and not any(c in _VOWELS for c in _alpha(w)))
    inword_punct_frac = inword_punct / n
    novowel_frac = novowel / n

    alnum = sum(1 for c in text if c.isalnum())
    digit_ratio = (sum(1 for c in text if c.isdigit()) / alnum) if alnum else 0.0
    upper_ratio = sum(1 for c in ascii_letters if c.isupper()) / len(ascii_letters)

    path_a = inword_punct_frac >= _MIN_INWORD_PUNCT_FRAC
    path_b = (novowel_frac >= _MIN_NOVOWEL_FRAC
              and digit_ratio < _MAX_DIGIT_RATIO
              and upper_ratio < _MAX_UPPER_RATIO)
    return path_a or path_b


def is_garbled(text: str, font_names=None) -> bool:
    """True if ``text`` looks like legacy-font mojibake rather than real text.

    Three independent signals, any one sufficient:
      1. A known legacy Indic font is present (decisive, deterministic).
      2. Heuristic: a high density of accented-Latin glyphs — the byte pattern
         a non-Unicode Latin-1-mapped Indic font produces.
      3. Generic ASCII glyph-mojibake (:func:`_looks_like_glyph_mojibake`) — the
         *disjoint* legacy-font family (TAB/TSCII/Bamini/VANAVIL …) that maps
         onto plain ASCII, where signal 2 reads 0.0. Script/language independent.

    Signals 1 and 2 target Latin-1 mojibake; signal 3 targets ASCII mojibake —
    disjoint failure modes, so they are OR-ed rather than merged.

    Deliberately conservative so clean documents never enter recovery; and even a
    false positive is self-healing — recovery only replaces the text if OCR
    yields real script-block characters (see ``MIN_SCRIPT_CONF`` verify gate).
    """
    if script_from_fonts(font_names):
        return True
    text = text or ""
    return _garbage_ratio(text) >= 0.40 or _looks_like_glyph_mojibake(text)


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

    # Installed packs bound BOTH routing and verification — filter everything to
    # what the backend can actually OCR (no point resolving to a missing pack).
    try:
        installed = backend.available_scripts()
    except Exception:
        installed = set()

    cand_packs = [p for p in (script_for_language(l) for l in (candidate_langs or []))
                  if p and p in installed]

    # When NO non-English language is declared (default ``--lang en`` ⇒
    # candidate_langs empty), fall back to the full supported set for script
    # *resolution only*: OSD needs ISO candidates to map its detected script onto,
    # and ``_detect_script_osd`` early-returns None on an empty list. This is what
    # lets a Tamil page resolve with no ``--lang`` at all. We deliberately do NOT
    # OCR-join this fallback set (no all-pack joined OCR) — if OSD cannot name the
    # script we no-op below and leave the original text. So the fix has a hard
    # dependency on OSD (``osd.traineddata``) succeeding on the rendered page.
    from vega.languages import supported_languages  # noqa: PLC0415
    resolve_iso = list(candidate_langs) if candidate_langs else supported_languages()
    resolve_iso = [l for l in resolve_iso
                   if script_for_language(l) and script_for_language(l) in installed]

    # ── resolve the OCR pack ──
    script = None
    fs = script_from_fonts(font_names)
    if fs and (not cand_packs or fs in cand_packs):
        script = fs                                    # legacy-font hint wins
    elif declared_script and (not cand_packs or declared_script in cand_packs):
        script = declared_script
    if not script and resolve_iso:
        script = _detect_script_osd(render_png, resolve_iso)   # OSD (over all
                                                               # supported when
                                                               # none declared)
    if not script and cand_packs:
        script = "+".join(dict.fromkeys(cand_packs))   # declared candidates only
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

    # Verify against whichever candidate script actually came out on top. With no
    # declared candidates, verify against every installed supported script block
    # (filtered above), so the OSD-resolved script is scored — never [script]
    # alone, which would rubber-stamp its own guess.
    verify = cand_packs or [p for p in (script_for_language(l) for l in resolve_iso) if p]
    if not verify:
        verify = [script]
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
    the caller can fall back to plain English OCR.

    FOLLOW-UP: unlike :func:`recover`, this path is NOT yet symmetric for the
    default ``--lang en`` case (empty ``candidate_langs`` ⇒ no all-supported OSD
    here), so a *scanned* non-English page under the default language still falls
    back to English OCR. Left as a follow-up (born-digital glyph mojibake, the
    reported bug, goes through :func:`recover`, which now handles it)."""
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
