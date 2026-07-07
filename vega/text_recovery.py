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
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from vega.config import resolve_ocr_window

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
    "asm": (0x0980, 0x09FF),  # Assamese (shares the Bengali–Assamese block)
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
    "as": "asm",
}

# Legacy non-Unicode font families → script. Matched as a case-insensitive
# substring of the PyMuPDF font name (after stripping the ``ABCDEE+`` subset
# prefix). Add rows as new legacy fonts are encountered.
_LEGACY_FONT_SCRIPTS: Dict[str, str] = {
    "shree-tel": "tel", "shreetel": "tel", "anu": "tel", "hemalatha": "tel",
    "shree-dev": "hin", "shree-deo": "hin", "kruti dev": "hin", "krutidev": "hin",
    "shree-tam": "tam", "shree-kan": "kan", "shree-mal": "mal",
    "shree-guj": "guj", "shree-ben": "ben", "shree-pun": "pan", "shree-ori": "ori",
    "ramdhenu": "asm",   # the standard Assamese legacy typing font
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


# ── Broken-ToUnicode-CMap detector (script/language independent) ─────────────
#
# A third, disjoint mojibake family: the page's font has a *partial or wrong*
# ToUnicode CMap, so extraction yields REAL target-script letters interleaved
# with wrong-block codepoints — e.g. malayalam.pdf (BalooChettan2) extracts
# ``കϓΎϙൽ`` (Malayalam + archaic-Greek soup). Both earlier detectors read 0.0
# here: ``_garbage_ratio`` sees no Latin-1 accents, ``_looks_like_glyph_mojibake``
# sees no ASCII dominance. This detector is a POSITIVE validity check — "is this
# plausible text in *some* writing system?" — rather than another failure
# fingerprint, so it does not need a new rule per broken font.
#
# Signals (all computed on NFC-normalised text, so decomposed-but-valid Latin
# (macOS NFD PDFs: ``é`` = e + U+0301) composes away before judgement):
#   · word-level cross-script mixing — a word whose letters span TWO OR MORE
#     distinct non-Latin scripts is not valid in any language. One non-Latin
#     script (± Latin) is always allowed: ``θ-dependence`` (Greek notation),
#     ``Twitterपर`` (Hindi case-marker on a Latin acronym), IPA, transliteration.
#   · orphaned generic combining marks — a U+0300-block mark at word start or on
#     an Indic base (Indic scripts carry their own mark ranges; a generic mark
#     there is a CMap artefact). Marks on Latin/Greek/Cyrillic bases are legit
#     (Russian dictionaries stress-mark with U+0301, which never composes).
#   · U+FFFD replacement characters — the extractor's own "no mapping" marker.
#   · Private-Use-Area codepoints — where fully-unmapped legacy fonts land.
#
# Two floors, so partial corruption is *surfaced* without forcing a bad trade:
#   · RECOVER floor — enough of the page is corrupt that whole-page OCR is a
#     clear win → ``is_garbled`` fires and the recovery cascade runs (still
#     gated by MIN_SCRIPT_CONF, so a false fire self-heals to a no-op).
#   · SUSPECT floor — corruption is real but a small fraction of the page
#     (e.g. one garbled title line on an otherwise-clean page); OCR-replacing
#     95% clean born-digital text would be a net loss, so the page is kept and
#     flagged (``garble_suspect_pages`` → chunk ``garble_suspected``) for
#     downstream filtering / reprocessing.
#
# Thresholds pinned against measured values on real pages + a hostile negative
# corpus (word-tokens / corrupt-word fraction / PUA fraction / U+FFFD count):
#   malayalam.pdf p1 (broken CMap) :  76 / 0.75 / 0.00 / 0  → RECOVER
#   malayalam.pdf p3 (broken CMap) : 132 / 0.39 / 0.00 / 0  → RECOVER
#   hindi.pdf p1 (garbled title)   :  42 / 0.10 / 0.00 / 0  → SUSPECT only
#   fully PUA-mapped legacy page   :   0 / —    / 1.00 / 0  → RECOVER
#   Greek-notation physics prose   :  45 / 0.00 / 0.00 / 0  → clean
#   NFD French / Vietnamese        :  20 / 0.00 / 0.00 / 0  → clean
#   IPA pronunciation guide        :  26 / 0.04 / 0.00 / 0  → clean (min-count guard)
#   tri-script dictionary line     :  18 / 0.00 / 0.00 / 0  → clean
#   Hindi with attached Latin      :  19 / 0.00 / 0.00 / 0  → clean
#   Russian stress-marked text     :  14 / 0.00 / 0.00 / 0  → clean
#   dingbat/PUA bullet list        :  16 / 0.00 / 0.06 / 0  → clean
# The disjoint families stay disjoint: tamil.pdf ASCII glyph mojibake reads
# 0.01 here (signal 3 owns it); Latin-1 soup reads 0.00 (signal 2 owns it).

# Letter → script class. Deliberately coarse: classes exist to detect *mixing*,
# not to identify languages. Unlisted letters (Lm modifiers, rare scripts) class
# as "other", which never corrupts on its own.
_LETTER_SCRIPT_RANGES: Tuple[Tuple[int, int, str], ...] = (
    (0x0041, 0x024F, "latin"),
    (0x0250, 0x02AF, "latin"),      # IPA extensions — phonetic notation
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x052F, "cyrl"),
    (0x0530, 0x058F, "armn"),
    (0x0590, 0x05FF, "hebr"),
    (0x0600, 0x077F, "arab"),
    (0x0900, 0x097F, "deva"),
    (0x0980, 0x09FF, "beng"),
    (0x0A00, 0x0A7F, "guru"),
    (0x0A80, 0x0AFF, "gujr"),
    (0x0B00, 0x0B7F, "orya"),
    (0x0B80, 0x0BFF, "taml"),
    (0x0C00, 0x0C7F, "telu"),
    (0x0C80, 0x0CFF, "knda"),
    (0x0D00, 0x0D7F, "mlym"),
    (0x0D80, 0x0DFF, "sinh"),
    (0x0E00, 0x0E7F, "thai"),
    (0x10A0, 0x10FF, "geor"),
    (0x1CD0, 0x1CFF, "deva"),       # Vedic Extensions — legit with Devanagari
    (0x1E00, 0x1EFF, "latin"),      # Latin Extended Additional (Vietnamese)
    (0x1F00, 0x1FFF, "greek"),      # Greek Extended (polytonic)
    (0x3040, 0x30FF, "cjk"),
    (0x3400, 0x9FFF, "cjk"),
    (0xAC00, 0xD7AF, "hang"),
    (0x1D400, 0x1D7FF, "notation"),  # math alphanumerics — never corrupts
)

# Indic classes carry their own combining-mark blocks; a *generic* U+0300-range
# mark on one of these bases is a CMap artefact, not orthography.
_OWN_MARKS_SCRIPTS = frozenset(
    {"deva", "beng", "guru", "gujr", "orya", "taml", "telu", "knda", "mlym", "sinh"})
_GENERIC_COMBINING: Tuple[Tuple[int, int], ...] = (
    (0x0300, 0x036F), (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF), (0x20D0, 0x20FF))

_CMAP_MIN_WORDS = 8            # short strings (headers, captions) never judged
_CMAP_RECOVER_FRAC = 0.15      # corrupt-word fraction → whole-page recovery …
_CMAP_RECOVER_MIN = 4          # … and never on fewer than this many bad words
_CMAP_SUSPECT_FRAC = 0.05      # corrupt-word fraction → flag, keep text …
_CMAP_SUSPECT_MIN = 2          # … needs at least two independent bad words
_PUA_RECOVER_FRAC = 0.30       # PUA share of (letters+PUA) → fully-unmapped font
_PUA_MIN_CHARS = 20            # a couple of dingbat bullets never trigger
_PUA_SUSPECT_FRAC = 0.10
_PUA_SUSPECT_MIN = 8
_FFFD_RECOVER_MIN = 4          # replacement chars: deterministic missing-CMap
_FFFD_SUSPECT_MIN = 2


def _letter_script(cp: int) -> str:
    for lo, hi, name in _LETTER_SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    return "other"


def _word_cmap_corrupted(word: str) -> bool:
    """True if a single word token cannot be valid text in any one language."""
    if "�" in word:
        return True
    scripts: set = set()
    prev_script: Optional[str] = None
    for ch in word:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if cat == "Co":                       # Private Use Area
            return True
        if cat == "Mn" and any(lo <= cp <= hi for lo, hi in _GENERIC_COMBINING):
            if prev_script is None or prev_script in _OWN_MARKS_SCRIPTS:
                return True                   # orphaned / on an Indic base
            continue
        if ch.isalpha():
            s = _letter_script(cp)
            scripts.add(s)
            prev_script = s
    non_latin = {s for s in scripts if s not in ("latin", "notation")}
    return len(non_latin) >= 2                # two non-Latin scripts in one word


def _cmap_stats(text: str) -> Tuple[int, int, int, int, int]:
    """→ (corrupt_words, words, letters, pua_chars, fffd_chars), NFC-normalised."""
    text = unicodedata.normalize("NFC", text or "")
    words = [w for w in text.split() if sum(c.isalpha() for c in w) >= 2]
    corrupt = sum(1 for w in words if _word_cmap_corrupted(w))
    letters = sum(1 for c in text if c.isalpha())
    pua = sum(1 for c in text if unicodedata.category(c) == "Co")
    fffd = text.count("�")
    return corrupt, len(words), letters, pua, fffd


def _looks_like_broken_cmap(text: str) -> bool:
    """RECOVER floor: enough corruption that whole-page OCR is a clear win."""
    corrupt, words, letters, pua, fffd = _cmap_stats(text)
    if fffd >= _FFFD_RECOVER_MIN:
        return True
    if pua >= _PUA_MIN_CHARS and pua / max(1, pua + letters) >= _PUA_RECOVER_FRAC:
        return True
    return (words >= _CMAP_MIN_WORDS and corrupt >= _CMAP_RECOVER_MIN
            and corrupt / words >= _CMAP_RECOVER_FRAC)


# ── Signal 5: legacy symbol-glyph splicing ───────────────────────────────────
#
# A second legacy-font family (old Telugu/Kannada DTP fonts, seen in the wild
# on scanned-book OCR layers) maps conjunct/vowel-sign glyphs onto ASCII
# punctuation: ్ర → "|", ఁ → "(" …  The page then decodes as REAL target-script
# letters with symbols spliced *inside* words (సుర|పభుఁబోలి, భూమిని(బోలి).
# There is no wrong-block mixing, no PUA, no U+FFFD — signals 1–4 all read
# clean — so this signature gets its own detector. Because interior ASCII
# symbols are essentially never valid Indic orthography, the floor is lower
# than the generic CMap one.
_SPLICE_OK = frozenset("-‐‑–—'’‘.,:/&_@·।॥")  # legit intraword/attached
_SPLICE_NEVER = frozenset("|\\£¥¤~^*=+<>`")   # never legit against an Indic letter
_SPLICE_MIN_WORDS = 8          # same "short strings never judged" guard
_SPLICE_RECOVER_MIN = 4        # ≥ this many spliced words …
_SPLICE_RECOVER_FRAC = 0.08    # … at ≥ this fraction → whole-page recovery
_SPLICE_SUSPECT_MIN = 2
_SPLICE_SUSPECT_FRAC = 0.03


def _indic_char(ch: str) -> bool:
    if not ch or ch.isdigit():
        return False
    if unicodedata.category(ch)[0] not in ("L", "M"):
        return False
    return _letter_script(ord(ch)) in _OWN_MARKS_SCRIPTS


def _word_symbol_spliced(word: str) -> bool:
    """True when ASCII symbols are spliced against Indic letters inside one
    word token. Paired/ordinary punctuation only counts flanked by Indic on
    BOTH sides, so a legit parenthetical like (చూడండి) stays clean; the
    _SPLICE_NEVER set counts with an Indic letter on either side."""
    for i, ch in enumerate(word):
        if ch in _SPLICE_OK or unicodedata.category(ch)[0] not in ("P", "S"):
            continue
        prev_indic = _indic_char(word[i - 1]) if i else False
        next_indic = _indic_char(word[i + 1]) if i + 1 < len(word) else False
        if ch in _SPLICE_NEVER:
            if prev_indic or next_indic:
                return True
        elif prev_indic and next_indic:
            return True
    return False


def _symbol_splice_stats(text: str) -> Tuple[int, int]:
    """→ (spliced_words, words), same word criterion as :func:`_cmap_stats`."""
    text = unicodedata.normalize("NFC", text or "")
    words = [w for w in text.split() if sum(c.isalpha() for c in w) >= 2]
    return sum(1 for w in words if _word_symbol_spliced(w)), len(words)


def _looks_like_symbol_glyphs(text: str) -> bool:
    """RECOVER floor for signal 5 (legacy symbol-glyph splicing)."""
    bad, words = _symbol_splice_stats(text)
    return (words >= _SPLICE_MIN_WORDS and bad >= _SPLICE_RECOVER_MIN
            and bad / words >= _SPLICE_RECOVER_FRAC)


def _symbol_glyphs_suspect(text: str) -> bool:
    """SUSPECT floor for signal 5 — superset of its recover floor."""
    bad, words = _symbol_splice_stats(text)
    return (words >= _SPLICE_MIN_WORDS and bad >= _SPLICE_SUSPECT_MIN
            and bad / words >= _SPLICE_SUSPECT_FRAC)


def garble_suspect(text: str) -> bool:
    """SUSPECT floor: real but sub-recovery corruption (e.g. one garbled title
    line on an otherwise clean page). The caller keeps the text and flags the
    page so downstream can filter — replacing 95%-clean born-digital text with
    OCR would be a net loss. Superset of the recover floor by construction."""
    corrupt, words, letters, pua, fffd = _cmap_stats(text)
    if fffd >= _FFFD_SUSPECT_MIN:
        return True
    if pua >= _PUA_SUSPECT_MIN and pua / max(1, pua + letters) >= _PUA_SUSPECT_FRAC:
        return True
    if (words >= _CMAP_MIN_WORDS and corrupt >= _CMAP_SUSPECT_MIN
            and corrupt / words >= _CMAP_SUSPECT_FRAC):
        return True
    return _symbol_glyphs_suspect(text)


def is_garbled(text: str, font_names=None) -> bool:
    """True if ``text`` looks like legacy-font mojibake rather than real text.

    Four independent signals, any one sufficient:
      1. A known legacy Indic font is present (decisive, deterministic).
      2. Heuristic: a high density of accented-Latin glyphs — the byte pattern
         a non-Unicode Latin-1-mapped Indic font produces.
      3. Generic ASCII glyph-mojibake (:func:`_looks_like_glyph_mojibake`) — the
         *disjoint* legacy-font family (TAB/TSCII/Bamini/VANAVIL …) that maps
         onto plain ASCII, where signal 2 reads 0.0. Script/language independent.
      4. Broken-ToUnicode-CMap corruption (:func:`_looks_like_broken_cmap`) —
         real target-script letters polluted with wrong-block codepoints /
         PUA / U+FFFD, where signals 2 and 3 both read 0.0. A positive
         "valid text in some writing system?" check, script independent.
      5. Legacy symbol-glyph splicing (:func:`_looks_like_symbol_glyphs`) —
         clean target-script letters with ASCII symbols spliced inside words
         (సుర|పభుఁబోలి: | and ( standing in for conjunct/vowel glyphs), where
         signals 1–4 all read 0.0.

    The signals target disjoint failure modes (Latin-1, ASCII, mixed-block,
    symbol-splice), so they are OR-ed rather than merged.

    Deliberately conservative so clean documents never enter recovery; and even a
    false positive is self-healing — recovery only replaces the text if OCR
    yields real script-block characters (see ``MIN_SCRIPT_CONF`` verify gate).
    """
    if script_from_fonts(font_names):
        return True
    text = text or ""
    return (_garbage_ratio(text) >= 0.40 or _looks_like_glyph_mojibake(text)
            or _looks_like_broken_cmap(text) or _looks_like_symbol_glyphs(text))


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
    engine: Optional[str] = None   # OCR engine that produced the text
                                   # (surya/easyocr/tesseract), None if no-op


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


def _ocr_png_attributed(png_bytes: bytes, script: str, backend) -> Tuple[str, Optional[str]]:
    """Like :func:`_ocr_png` but also reports which engine produced the text
    (composites surface their per-call winner; plain backends report themselves)."""
    lang = _ocr_lang(script, backend)
    fn = getattr(backend, "image_to_text_attributed", None)
    if fn is not None:
        return fn(png_bytes, lang)
    out = backend.image_to_text(png_bytes, lang)
    return out, (getattr(backend, "name", None) if out else None)


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
    force: bool = False,
) -> Recovery:
    """Rescue a page's text to clean Unicode, or return a no-op result.

    ``force=True`` (the ``--force-ocr`` flag) skips the is_garbled gate and
    re-OCRs even a clean-looking page — for corrupt text layers the detector
    misses. The verify gate below still applies, so the original text is kept
    whenever OCR can't produce confident target-script output.

    Script selection cascade (multilingual-aware): legacy **font name** →
    ``declared_script`` → **Tesseract OSD** on the rendered image (true scanned
    pages) → all ``candidate_langs`` packs joined (let the engine decide). The
    candidate set (the caller's ``languages``, as ISO codes) bounds detection so
    a mixed-language corpus routes each page to the right pack.

    ``was_recovered`` is False for every clean page — the English path is
    completely untouched, and no image is ever rendered for a clean page.
    """
    if not force and not is_garbled(text, font_names):
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
        recovered, engine = _ocr_png_attributed(render_png(), script, backend)
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
                "conf=%.2f, chars=%d, engine=%s)",
                script, win, conf, len(recovered), engine or "?")
    return Recovery(text=recovered, was_recovered=True, script=(win or script),
                    method="ocr", confidence=conf, engine=engine)


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

    With NO declared languages (default ``--lang en`` ⇒ empty ``candidate_langs``)
    the same symmetry as :func:`recover` applies: OSD runs over every supported
    language with an installed pack, so a scanned Tamil page resolves with no
    ``--lang`` at all. A Latin/``eng`` OSD answer is deliberately ignored there —
    the caller's plain-English fallback already owns that case — and, as in
    :func:`recover`, there is no all-pack joined OCR: if OSD cannot name a
    non-Latin script we no-op and the caller falls back to English OCR.

    The result is gated by the same Unicode-block confidence check as legacy-font
    recovery: if the first script's output is low-confidence it retries with the
    full candidate-pack set, and if that is still garbage it returns a no-op so
    the caller can fall back to plain English OCR."""
    if backend is None:
        return _noop()
    cand_packs = [p for p in (script_for_language(l) for l in (candidate_langs or [])) if p]

    # Resolution set for the undeclared case, bounded by installed packs (same
    # filter recover() applies — no point resolving to a pack we cannot OCR).
    try:
        installed = backend.available_scripts()
    except Exception:
        installed = set()
    from vega.languages import supported_languages  # noqa: PLC0415
    resolve_iso = [l for l in (list(candidate_langs or []) or supported_languages())
                   if script_for_language(l) and script_for_language(l) in installed]

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
    elif resolve_iso:
        # No declared languages: all-supported OSD, non-Latin results only.
        osd = _detect_script_osd(render_png, resolve_iso)
        if osd and osd != "eng":
            _add(osd)
    if not attempts:
        return _noop()

    # Never verify an OSD guess against itself alone — score the output over the
    # declared packs, else every installed supported script block.
    verify = (cand_packs
              or [p for p in (script_for_language(l) for l in resolve_iso) if p]
              or attempts)
    best: Optional[Recovery] = None
    for script in attempts:
        try:
            recovered, engine = _ocr_png_attributed(render_png(), script, backend)
        except Exception as e:
            logger.warning("text_recovery: scanned OCR failed (%s): %r", script, e)
            continue
        if not recovered:
            continue
        conf, win = _best_ratio(recovered, verify)
        cand = Recovery(text=recovered, was_recovered=True, script=(win or script),
                        method="ocr", confidence=conf, engine=engine)
        if best is None or conf > best.confidence:
            best = cand
        if conf >= MIN_SCRIPT_CONF:
            logger.info("text_recovery: OCR'd scanned page (pack=%s, detected=%s, "
                        "conf=%.2f, chars=%d, engine=%s)",
                        script, win, conf, len(recovered), engine or "?")
            return cand

    # Every attempt was low-confidence → discard so the caller falls back to
    # plain English OCR rather than emitting mis-scripted garbage.
    if best is not None:
        logger.warning("text_recovery: scanned OCR low-confidence (best=%.2f) — "
                       "discarding, caller falls back to English",
                       best.confidence)
    return _noop()


# ── batch orchestration (Phase 1 of docs/DESIGN-scale-ocr.md) ────────────────
#
# The single-page paths above stay the source of truth for *semantics*; the
# planner/executor below reproduce their resolution and verification exactly,
# but split "decide" from "OCR" so a whole file's pages can go to the GPU in
# script-grouped, windowed batches. Golden tests assert page-for-page equality
# with the single-page path — if you change recover()/ocr_scanned(), mirror it
# here (the golden tests will catch you if you forget).

@dataclass
class OCRPlan:
    """A deferred page-OCR decision: script resolution done, OCR pending."""
    png: bytes                    # page pixels rendered at OCR dpi
    kind: str                     # "recover" | "scanned"
    attempts: List[str]           # ordered scripts to try (>= 1)
    verify: List[str]             # packs the output is scored against
    original_text: str = ""       # recover: text to keep when OCR is discarded


def plan_recover(text, font_names, png: bytes, *, backend,
                 declared_script: Optional[str] = None,
                 candidate_langs: Optional[List[str]] = None,
                 force: bool = False,
                 ) -> Optional[OCRPlan]:
    """The resolution half of :func:`recover`, for deferred/batched OCR.

    Returns None exactly when recover() would no-op **without OCRing** (clean
    page, no backend, unresolvable script, missing pack) — the caller keeps the
    original text and applies the same suspect flagging as today. ``force``
    mirrors :func:`recover`: skip the clean gate, verify gate still applies
    (the plan carries ``original_text``, kept when OCR is discarded)."""
    if not force and not is_garbled(text, font_names):
        return None
    if backend is None:
        return None
    try:
        installed = backend.available_scripts()
    except Exception:
        installed = set()
    cand_packs = [p for p in (script_for_language(l) for l in (candidate_langs or []))
                  if p and p in installed]
    from vega.languages import supported_languages  # noqa: PLC0415
    resolve_iso = list(candidate_langs) if candidate_langs else supported_languages()
    resolve_iso = [l for l in resolve_iso
                   if script_for_language(l) and script_for_language(l) in installed]

    script = None
    fs = script_from_fonts(font_names)
    if fs and (not cand_packs or fs in cand_packs):
        script = fs                                    # legacy-font hint wins
    elif declared_script and (not cand_packs or declared_script in cand_packs):
        script = declared_script
    if not script and resolve_iso:
        script = _detect_script_osd(lambda: png, resolve_iso)
    if not script and cand_packs:
        script = "+".join(dict.fromkeys(cand_packs))
    if not script:
        script = fs or declared_script

    if not script:
        logger.warning("text_recovery: garbled text but no script resolved "
                       "(fonts=%s); leaving as-is", list(font_names or [])[:4])
        return None
    if not _ocr_available(script, backend):
        logger.warning("text_recovery: script %r detected but OCR pack(s) "
                       "unavailable in backend %r — leaving text as-is (mojibake).",
                       script, getattr(backend, "name", "?"))
        return None

    verify = cand_packs or [p for p in (script_for_language(l) for l in resolve_iso) if p]
    if not verify:
        verify = [script]
    return OCRPlan(png=png, kind="recover", attempts=[script], verify=verify,
                   original_text=text)


def plan_scanned(png: bytes, *, backend,
                 candidate_langs: Optional[List[str]] = None,
                 declared_script: Optional[str] = None) -> Optional[OCRPlan]:
    """The resolution half of :func:`ocr_scanned`, for deferred/batched OCR.
    None when there is nothing sensible to try — the caller falls back to
    plain English OCR exactly as it does after an ocr_scanned no-op."""
    if backend is None:
        return None
    cand_packs = [p for p in (script_for_language(l) for l in (candidate_langs or [])) if p]
    try:
        installed = backend.available_scripts()
    except Exception:
        installed = set()
    from vega.languages import supported_languages  # noqa: PLC0415
    resolve_iso = [l for l in (list(candidate_langs or []) or supported_languages())
                   if script_for_language(l) and script_for_language(l) in installed]

    attempts: List[str] = []

    def _add(s: Optional[str]) -> None:
        if s and s not in attempts and _ocr_available(s, backend):
            attempts.append(s)

    if declared_script and (not cand_packs or declared_script in cand_packs):
        _add(declared_script)
    if cand_packs:
        _add(_detect_script_osd(lambda: png, list(candidate_langs or [])))
        _add("+".join(dict.fromkeys(cand_packs)))
    elif resolve_iso:
        osd = _detect_script_osd(lambda: png, resolve_iso)
        if osd and osd != "eng":
            _add(osd)
    if not attempts:
        return None
    verify = (cand_packs
              or [p for p in (script_for_language(l) for l in resolve_iso) if p]
              or attempts)
    return OCRPlan(png=png, kind="scanned", attempts=attempts, verify=verify)


def _batch_ocr_attributed(backend, images: List[bytes], lang: str):
    fn = getattr(backend, "image_to_text_batch_attributed", None)
    try:
        if fn is not None:
            return fn(images, lang)
        texts = backend.image_to_text_batch(images, lang)
        name = getattr(backend, "name", None)
        return texts, [(name if t else None) for t in texts]
    except Exception as e:  # noqa: BLE001 - a whole window must not strand pages
        logger.warning("text_recovery: batch OCR failed (%s): %r", lang, e)
        return ["" for _ in images], [None for _ in images]


def execute_plans(plans: List[OCRPlan], backend,
                  window: Optional[int] = None) -> List[Recovery]:
    """OCR a file's deferred plans in script-grouped, windowed batches.

    Returns one Recovery per plan, positionally. Verification, retry and
    discard semantics are identical to the single-page paths: a recover plan
    whose output misses the confidence floor no-ops back to its original text;
    a scanned plan walks its attempt list and no-ops when every attempt is
    low-confidence (caller then does plain English OCR)."""
    n = len(plans)
    if backend is None or n == 0:
        return [_noop(p.original_text) for p in plans]
    # RAM knob, not VRAM (the backend's own batch sizes cap per-forward
    # memory regardless). Resolution: caller value > VEGA_OCR_WINDOW > 16.
    window = resolve_ocr_window(window)
    results: List[Optional[Recovery]] = [None] * n
    best: List[Optional[Recovery]] = [None] * n
    attempt_idx = [0] * n

    while True:
        pending = [i for i in range(n)
                   if results[i] is None and attempt_idx[i] < len(plans[i].attempts)]
        if not pending:
            break
        groups: Dict[str, List[int]] = {}
        for i in pending:
            groups.setdefault(plans[i].attempts[attempt_idx[i]], []).append(i)
        for script, idxs in groups.items():
            lang = _ocr_lang(script, backend)
            for w in range(0, len(idxs), window):
                chunk = idxs[w:w + window]
                texts, engines = _batch_ocr_attributed(
                    backend, [plans[i].png for i in chunk], lang)
                for i, txt, eng in zip(chunk, texts, engines):
                    attempt_idx[i] += 1
                    if not txt:
                        continue
                    conf, win = _best_ratio(txt, plans[i].verify)
                    cand = Recovery(text=txt, was_recovered=True,
                                    script=(win or script), method="ocr",
                                    confidence=conf, engine=eng)
                    if best[i] is None or conf > best[i].confidence:
                        best[i] = cand
                    if conf >= MIN_SCRIPT_CONF:
                        results[i] = cand
                        if plans[i].kind == "recover":
                            logger.info(
                                "text_recovery: recovered page via OCR (pack=%s, "
                                "detected=%s, conf=%.2f, chars=%d, engine=%s)",
                                script, win, conf, len(txt), eng or "?")
                        else:
                            logger.info(
                                "text_recovery: OCR'd scanned page (pack=%s, "
                                "detected=%s, conf=%.2f, chars=%d, engine=%s)",
                                script, win, conf, len(txt), eng or "?")

    out: List[Recovery] = []
    for i, plan in enumerate(plans):
        if results[i] is not None:
            out.append(results[i])
        elif plan.kind == "recover":
            logger.warning("text_recovery: OCR produced low-confidence output "
                           "(packs=%s ratio=%.2f) — discarding",
                           plan.attempts[0],
                           best[i].confidence if best[i] else 0.0)
            out.append(_noop(plan.original_text))
        else:
            if best[i] is not None:
                logger.warning("text_recovery: scanned OCR low-confidence "
                               "(best=%.2f) — discarding, caller falls back "
                               "to English", best[i].confidence)
            out.append(_noop())
    return out
