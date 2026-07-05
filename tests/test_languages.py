"""Language registry: normalization, asset resolution, script detection."""

import pytest

from vega import languages as L


def test_normalize_variants_to_iso():
    assert L.normalize_language("Telugu") == "te"
    assert L.normalize_language("tel") == "te"     # tesseract code
    assert L.normalize_language("TE") == "te"
    assert L.normalize_language("oriya") == "or"
    assert L.normalize_language("bangla") == "bn"
    assert L.normalize_language("klingon") is None


def test_normalize_languages_list_and_string():
    assert L.normalize_languages("Telugu, Hindi, English") == ["te", "hi", "en"]
    assert L.normalize_languages(["te", "te", "hi"]) == ["te", "hi"]   # dedup, order
    assert L.normalize_languages("te/en") == ["te", "en"]
    assert L.normalize_languages(None) == []


def test_all_ten_indic_plus_english_present():
    iso = set(L.supported_languages())
    assert {"en", "te", "hi", "mr", "ta", "kn", "ml", "bn", "gu", "pa", "or"} <= iso


def test_tesseract_and_block_resolution():
    assert L.to_tesseract("te") == "tel"
    assert L.to_tesseract("en") is None            # English needs no OCR pack
    assert L.script_block("te") == (0x0C00, 0x0C7F)
    assert L.script_block("en") is None


def test_language_of_text_vs_dominant():
    telugu = "పరిపాలన"          # pure Telugu
    assert L.language_of_text(telugu, ["te", "hi"]) == "te"
    # a single Telugu proper noun in an English sentence:
    mixed = "The GO was issued by పరిషత్ office"
    assert L.language_of_text(mixed, ["te"]) == "te"   # non-Latin histogram
    assert L.dominant_language(mixed, ["te"]) == "en"  # Latin dominates


def test_osd_script_disambiguation():
    # Devanagari is shared by hi and mr — the declared candidate wins.
    assert L.iso_for_osd_script("Devanagari", ["mr"]) == "mr"
    assert L.iso_for_osd_script("Devanagari", ["hi"]) == "hi"
    assert L.iso_for_osd_script("Telugu", ["hi", "te"]) == "te"


# ── Assamese: shares the Bengali block, has its own Tesseract pack ────────────

def test_assamese_normalization_and_assets():
    assert L.normalize_language("Assamese") == "as"
    assert L.normalize_language("asm") == "as"       # tesseract code
    assert L.normalize_language("asomiya") == "as"   # variant spelling
    assert "as" in L.supported_languages()
    assert L.to_tesseract("as") == "asm"
    assert L.script_block("as") == (0x0980, 0x09FF)  # Bengali–Assamese block


def test_shared_bengali_block_disambiguates_by_declaration():
    # OSD reports "Bengali" for both languages — the declared candidate wins;
    # undeclared falls back to the canonical block owner (bn).
    assert L.iso_for_osd_script("Bengali", ["as"]) == "as"
    assert L.iso_for_osd_script("Bengali", ["bn", "as"]) == "bn"
    assert L.iso_for_osd_script("Bengali", []) == "bn"
    # Block-histogram tagging cannot tell as from bn: canonical owner wins.
    assamese = "অসমীয়া ভাষা উত্তৰ-পূৱ ভাৰতৰ"
    assert L.dominant_language(assamese) == "bn"
    assert L.dominant_language(assamese, ["as"]) == "as"   # declared-only pool
