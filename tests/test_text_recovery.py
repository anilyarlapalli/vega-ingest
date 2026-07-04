"""Text recovery — mojibake detection, script routing, OCR-backed recovery.

All OCR is served by in-memory stubs; no real Tesseract pack is required.
"""

import pytest

from vega import text_recovery as tr

# Real Telugu (Unicode block 0x0C00-0x0C7F) — what a good recovery OCR yields.
TELUGU = "పరిపాలన పరిషత్తు నుండి ఉత్తర్వు జారీ చేయబడింది"
# Real Tamil (Unicode block 0x0B80-0x0BFF) — the clean recovery of GLYPH_MOJIBAKE.
TAMIL = "தமிழ்நாடு அரசு வேலைவாய்ப்பு மற்றும் பயிற்சித் துறை பொதுத் தமிழ்"
# Legacy-font mojibake: a dense run of accented-Latin glyphs, no real words.
MOJIBAKE = "Bçñüéàæþ§«ßÿîïçœ ëêÀÁÂÃ æþðøµ¶·"
CLEAN_EN = "The order was issued by the administration office in the district."

# ── generic ASCII glyph-mojibake corpus (NO font hint) ───────────────────────
# Real page-1 text of tamil.pdf as PyMuPDF extracts it from a VANAVIL/SunTommy
# (WinAnsiEncoding, no /ToUnicode) glyph font: renders தமிழ்நாடு அரசு … but the
# codepoints are plain ASCII, so _garbage_ratio (Latin-1 accented) reads 0.0.
GLYPH_MOJIBAKE = (
    "jkpo;ehL muR Ntiytha;g;G kw;Wk; gapw;rpj;Jiw gphpT : TNPSC Group II Njh;T "
    "ghlk; : nghJj;jkpo; (,yf;fzk;) gFjp : ,yf;fzf; Fwpg;gwpjy; fhg;Ghpik "
    "jkpo;ehL muRg; gzpahsh; Njh;thizak; F&g; - 2 Kjy;epiy kw;Wk; Kjd;ik "
    "Njh;TfSf;fhd fhnzhyp fhl;rp gjpTfs; xypg;gjpT ghlf;Fwpg;Gfs; khjphp Njh;T "
    "tpdhj;jhs;fs; kw;Wk; nkd;ghlf;Fwpg;Gfs; Mfpait Nghl;bj; Njh;tpw;F jahuhFk;"
)

# Negative corpus — ASCII-heavy real text that must NOT be judged mojibake.
EN_PROSE = (
    "The order was issued by the administration office in the district. "
    "Employment and training department prepares model test papers and soft "
    "study notes for candidates appearing in the competitive examination."
)
PY_SOURCE = (
    'def compute(x, y):\n    result = x + y  # add them together\n'
    '    for i in range(10):\n        result += i * 2\n'
    '    return {"total": result, "count": i}\n'
    'class Foo(Bar):\n    pass\n'
)
URL_LIST = (
    "https://example.com/path/to/resource?id=123&q=abc\n"
    "http://test.org/api/v1/users/list\n"
    "ftp://files.net/dir/sub/file.txt\n"
    "https://github.com/user/repo/blob/main/src/module/handler.py\n"
    "https://docs.python.org/3/library/functions.html\n"
    "https://en.wikipedia.org/wiki/Optical_character_recognition\n"
    "https://pypi.org/project/pytesseract/#history\n"
    "https://stackoverflow.com/questions/12345/how-to-do-a-thing"
)
ID_TABLE = (
    "SKU-4471-XZ  PART#99823  ASSY-0012-B  REF:ZX9  MTR-7781QQ  BRK-0091 "
    "INV-2231-KL  NUT-M8x20  BLT-M6x40  WSH-08  GKT-114RR  SPR-2290 CLP-773"
)
BASE64_BLOB = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk "
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg cQ2FtZXJhIHJlYWR5IHNldA "
    "TW96aWxsYS81LjAgKFgxMTsgTGludXggeDg2XzY0KSBBcHBsZVdlYktpdA "
    "SGVsbG8gV29ybGQgdGhpcyBpcyBhIHRlc3Qgc3RyaW5n"
)


# ── detection ────────────────────────────────────────────────────────────────

def test_clean_english_is_not_garbled():
    assert tr.is_garbled(CLEAN_EN) is False
    assert tr._garbage_ratio(CLEAN_EN) < 0.4


def test_accented_glyph_soup_is_garbled():
    assert tr.is_garbled(MOJIBAKE) is True
    assert tr._garbage_ratio(MOJIBAKE) >= 0.4


# ── generic ASCII glyph-mojibake detection (language/script independent) ──────

def test_ascii_glyph_mojibake_detected_without_font_hint():
    # The reported bug: born-digital Tamil in a legacy ASCII glyph font. No font
    # name is passed, and _garbage_ratio (Latin-1) is 0.0 — detection must come
    # purely from the generic heuristic.
    assert tr._garbage_ratio(GLYPH_MOJIBAKE) == 0.0
    assert tr._looks_like_glyph_mojibake(GLYPH_MOJIBAKE) is True
    assert tr.is_garbled(GLYPH_MOJIBAKE) is True          # no font_names given
    assert tr.is_garbled(GLYPH_MOJIBAKE, font_names=[]) is True


@pytest.mark.parametrize("name,text", [
    ("english_prose", EN_PROSE),
    ("python_source", PY_SOURCE),
    ("url_list", URL_LIST),
    ("id_sku_table", ID_TABLE),
    ("base64_blob", BASE64_BLOB),
])
def test_negative_corpus_not_flagged_as_glyph_mojibake(name, text):
    # Both reviewers required this negative corpus: the compound heuristic must
    # NOT fire on real ASCII-heavy content (prose / code / URLs / IDs / base64).
    assert tr._looks_like_glyph_mojibake(text) is False, name
    assert tr.is_garbled(text) is False, name


def test_glyph_heuristic_guards_short_strings():
    # Below the word-token floor, even a ';'-heavy fragment is not judged.
    assert tr._looks_like_glyph_mojibake("jkpo;ehL muR") is False


def test_legacy_font_name_is_decisive_even_on_clean_looking_text():
    # The font-name signal fires regardless of the glyph heuristic.
    assert tr.is_garbled("anything at all", ["ABCDEE+SHREE-TEL-0900"]) is True
    assert tr.script_from_fonts(["ABCDEE+SHREE-TEL-0900"]) == "tel"
    assert tr.script_from_fonts(["XYZ+Krutidev010"]) == "hin"
    assert tr.script_from_fonts(["Helvetica", "Arial-Bold"]) is None


def test_script_for_language_maps_iso_to_pack():
    assert tr.script_for_language("te") == "tel"
    assert tr.script_for_language("hi") == "hin"
    assert tr.script_for_language("en") is None      # no recovery for English
    assert tr.script_for_language(None) is None


def test_script_ratio_verifies_block():
    assert tr.script_ratio(TELUGU, "tel") > 0.9
    assert tr.script_ratio(CLEAN_EN, "tel") == 0.0


# ── orchestration ────────────────────────────────────────────────────────────

def _raise_png():
    raise AssertionError("render_png must not be called on a clean page")


def test_clean_page_is_a_noop_and_never_renders(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    rec = tr.recover(CLEAN_EN, ["Helvetica"], render_png=_raise_png,
                     backend=backend, candidate_langs=["te"])
    assert rec.was_recovered is False
    assert backend.calls == []                        # no OCR attempted


def test_font_driven_recovery_routes_to_declared_script(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    rec = tr.recover(
        MOJIBAKE, ["ABCDEE+SHREE-TEL-0900"],
        render_png=lambda: b"PNGBYTES",
        backend=backend, declared_script="tel", candidate_langs=["te"],
    )
    assert rec.was_recovered is True
    assert rec.script == "tel"
    assert rec.method == "ocr"
    assert rec.text == TELUGU
    # Bilingual co-load: Telugu pack first, English appended.
    assert backend.calls and backend.calls[0][0] == "tel+eng"


def test_recovery_degrades_when_pack_unavailable(make_ocr_stub):
    # Garbled + script resolves to 'tel', but the backend can't OCR Telugu →
    # leave the text as-is (a no-op) rather than crash or emit garbage.
    backend = make_ocr_stub(scripts=("eng",), output="whatever")
    rec = tr.recover(
        MOJIBAKE, ["ABCDEE+SHREE-TEL-0900"],
        render_png=lambda: b"PNG",
        backend=backend, declared_script="tel", candidate_langs=["te"],
    )
    assert rec.was_recovered is False
    assert backend.calls == []


def test_low_confidence_ocr_is_discarded(make_ocr_stub):
    # Backend "OCRs" but returns Latin junk — fails the script-ratio gate.
    backend = make_ocr_stub(scripts=("eng", "tel"), output="qwerty junk output")
    rec = tr.recover(
        MOJIBAKE, ["ABCDEE+SHREE-TEL-0900"],
        render_png=lambda: b"PNG",
        backend=backend, declared_script="tel", candidate_langs=["te"],
    )
    assert rec.was_recovered is False


def test_recover_noop_without_backend():
    rec = tr.recover(MOJIBAKE, ["SHREE-TEL"], render_png=lambda: b"x",
                     backend=None, candidate_langs=["te"])
    assert rec.was_recovered is False


def test_ocr_scanned_routes_declared_script(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    rec = tr.ocr_scanned(render_png=lambda: b"PNG", backend=backend,
                         candidate_langs=["te"], declared_script="tel")
    assert rec.was_recovered is True
    assert rec.script == "tel"
    assert rec.text == TELUGU


def test_ocr_scanned_noop_when_no_indic_declared(make_ocr_stub):
    # No non-English candidate ⇒ caller should fall back to plain English OCR.
    backend = make_ocr_stub(scripts=("eng",), output="x")
    rec = tr.ocr_scanned(render_png=lambda: b"PNG", backend=backend,
                         candidate_langs=[], declared_script=None)
    assert rec.was_recovered is False


# ── finding 3: scanned OCR must gate on the Unicode-block confidence ──────────

def test_ocr_scanned_discards_garbage_output(make_ocr_stub):
    # Backend returns non-empty but Latin junk for a Telugu request → conf 0.0.
    # Old behaviour accepted ANY non-empty text; now it is discarded (no-op) so
    # the caller falls back to plain English OCR.
    backend = make_ocr_stub(scripts=("eng", "tel"), output="qwerty junk output")
    rec = tr.ocr_scanned(render_png=lambda: b"PNG", backend=backend,
                         candidate_langs=["te"], declared_script="tel")
    assert rec.was_recovered is False


# ── finding 2: per-page routing must not blindly pick candidate[0] ───────────

def test_ocr_scanned_uses_osd_not_first_candidate(make_ocr_stub, monkeypatch):
    # Two declared languages, no pinned declared_script. The page is Hindi; OSD
    # detects Devanagari. Routing must pick 'hin', not the first candidate 'tel'.
    def fake_ocr(png, script):
        return "यह हिंदी पाठ है यहाँ लिखा" if "hin" in script else "zzz latin junk"
    backend = make_ocr_stub(scripts=("eng", "tel", "hin"), output="")
    backend.image_to_text = fake_ocr            # type: ignore[assignment]
    monkeypatch.setattr(tr, "_detect_script_osd", lambda rp, cl: "hin")

    rec = tr.ocr_scanned(render_png=lambda: b"PNG", backend=backend,
                         candidate_langs=["te", "hi"], declared_script=None)
    assert rec.was_recovered is True
    assert rec.script == "hin"


# ── glyph mojibake under DEFAULT --lang en (no declared candidates) ──────────

def test_recover_resolves_via_all_supported_osd_when_no_lang_declared(
        make_ocr_stub, monkeypatch):
    # The reported-bug path: default run, candidate_langs=[]. recover() must fall
    # back to all-supported OSD to resolve the script (here 'tam'), OCR, verify
    # against the supported script blocks, and replace the mojibake with Tamil.
    backend = make_ocr_stub(scripts=("eng", "tam"), output=TAMIL)
    seen = {}

    def fake_osd(render_png, iso_candidates):
        seen["iso"] = list(iso_candidates)
        return "tam"
    monkeypatch.setattr(tr, "_detect_script_osd", fake_osd)

    rec = tr.recover(
        GLYPH_MOJIBAKE, font_names=[],           # NO font hint
        render_png=lambda: b"PNG",
        backend=backend, declared_script=None, candidate_langs=[],  # NO --lang
    )
    assert rec.was_recovered is True
    assert rec.script == "tam"
    assert rec.text == TAMIL
    # OSD was offered the full supported set (filtered to installed 'tam' pack),
    # never an empty list (which would early-return None).
    assert seen["iso"] == ["ta"]
    # Bilingual co-load still applies (Tamil pack first, English appended).
    assert backend.calls and backend.calls[0][0] == "tam+eng"


def test_recover_noops_when_osd_cannot_resolve_no_lang(make_ocr_stub, monkeypatch):
    # If OSD fails (sparse page) with no declared language, there is no all-pack
    # join fallback — recover() leaves the original text untouched.
    backend = make_ocr_stub(scripts=("eng", "tam"), output=TAMIL)
    monkeypatch.setattr(tr, "_detect_script_osd", lambda rp, iso: None)
    rec = tr.recover(GLYPH_MOJIBAKE, font_names=[], render_png=lambda: b"PNG",
                     backend=backend, declared_script=None, candidate_langs=[])
    assert rec.was_recovered is False
    assert rec.text == GLYPH_MOJIBAKE            # original preserved
    assert backend.calls == []                   # no all-pack joined OCR


# ── finding 16: a no-op recovery carries the ORIGINAL text ───────────────────

def test_recover_noop_carries_original_text(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    rec = tr.recover(CLEAN_EN, ["Helvetica"], render_png=_raise_png,
                     backend=backend, candidate_langs=["te"])
    assert rec.was_recovered is False
    assert rec.text == CLEAN_EN                  # not "" — original is preserved
