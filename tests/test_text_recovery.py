"""Text recovery — mojibake detection, script routing, OCR-backed recovery.

All OCR is served by in-memory stubs; no real Tesseract pack is required.
"""

import pytest

from vega import text_recovery as tr

# Real Telugu (Unicode block 0x0C00-0x0C7F) — what a good recovery OCR yields.
TELUGU = "పరిపాలన పరిషత్తు నుండి ఉత్తర్వు జారీ చేయబడింది"
# Real Tamil (Unicode block 0x0B80-0x0BFF) — the clean recovery of GLYPH_MOJIBAKE.
TAMIL = "தமிழ்நாடு அரசு வேலைவாய்ப்பு மற்றும் பயிற்சித் துறை பொதுத் தமிழ்"
# Real Malayalam (Unicode block 0x0D00-0x0D7F) — the clean recovery of CMAP_MOJIBAKE_ML.
MALAYALAM = "ഒരിടത്ത് ഒരു കാട്ടിൽ ഒരു മരത്തിൽ ഒരു കിളിക്കൂട് ഉണ്ടായിരുന്നു കഥകൾ വളരെ"
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
# Extra negatives (review round 2) — residual false-positive shapes, each >= 8
# whitespace word-tokens so the short-string guard is NOT what saves them.
# (a) line-wrapped hex dump (offsets + hex byte columns).
HEX_DUMP = (
    "00000000  7f 45 4c 46 02 01 01 00  00 00 00 00 00 00 00 00\n"
    "00000010  02 00 3e 00 01 00 00 00  40 12 40 00 00 00 00 00\n"
    "00000020  40 00 00 00 00 00 00 00  a8 fc 01 00 00 00 00 00\n"
    "00000030  00 00 00 00 40 00 38 00  0b 00 40 00 1e 00 1d 00\n"
    "00000040  06 00 00 00 04 00 00 00  40 00 00 00 00 00 00 00"
)
# (b) line-wrapped base64 blob (newline-folded, no spaces within lines).
BASE64_WRAPPED = (
    "TW96aWxsYS81LjAgKFgxMTsgTGludXggeDg2XzY0KSBBcHBs\n"
    "ZVdlYktpdC81MzcuMzYgKEtIVE1MLCBsaWtlIEdlY2tvKSBD\n"
    "aHJvbWUvMTIwLjAuMC4wIFNhZmFyaS81MzcuMzYgRWRnLzEy\n"
    "MC4wLjAuMCBhbmQgc29tZSBtb3JlIHBhZGRpbmcgaGVyZSB0\n"
    "byBtYWtlIHRoaXMgYSBsb25nZXIgd3JhcHBlZCBibG9iIG9r"
)
# (c) lowercase consonant-heavy config/slug/CSS/YAML-ish key:value text. Real
# config keys are English-ish words (they carry vowels), so this stays clean.
CONFIG_YAML = (
    "server:\n  host: localhost\n  port: 8080\n"
    "db_url: postgres://localhost:5432/mydb\n"
    "log_level: debug\n  cache_ttl: 300\n"
    ".btn-primary { color: rgb; padding: 0; margin: 0; }\n"
    ".nav-bar { display: flex; width: 100%; gap: 8px; }\n"
    "retry_count: 3\ntmp_dir: /var/lib/app/tmp\n"
    "slug: my-cool-blog-post-title\nfeature_enabled: true"
)


# ── broken-ToUnicode-CMap corpus (third disjoint family) ─────────────────────
# Real page-1 text of malayalam.pdf as PyMuPDF extracts it: the BalooChettan2
# font's ToUnicode CMap is partial, so genuine Malayalam letters come out
# interleaved with archaic-Greek codepoints (Ͱ ϛ Ύ ϙ …). _garbage_ratio reads
# 0.0 (no Latin-1 accents) and the ASCII heuristic reads 0.0 (not ASCII); only
# the Unicode-sanity check can see it.
CMAP_MOJIBAKE_ML = (
    "കϓͩ ി͟ ിളി ഒരിടെͰാരϛ കാΎϙൽ ഒരϛ മരͰിൽ ഒരϛ അͽ͟ ിളി "
    "കϔട് കϔΎϙയϙരϛͷϛ. അതിെലാരϛ അͽ͟ ിളിയϛം അ΍ൻകിളിയϛം ഉͯ ായϙരϛͷϛ. "
    "അͽ͟ ിളി കϔΎϙൽ മϓΎയϙΎ് അതിന് അടയϙരി͟ ϛകയായϙരϛͷϛ "
    "ബϓͶϙമാനായ കാ͟ ഒരിടെͰാരിടͰ് കാർͰϛ എͷ കϓസൃതി͟ ϛΎϙ "
    "ഉͯ ായϙരϛͷϛ. കഥകൾ വളെര ഇαെΗΎϙരϛͷ കാർͰϛ കഥ േകൾ͟ ϛͷതിനായϙ"
)
# Real page-1 text of hindi.pdf (a Tamil–Hindi–English dictionary's cover): the
# title line is CMap-garbled (Tamil + Greek-Extended letters in one word) but
# the page is mostly clean Latin — corruption is real yet BELOW the recovery
# floor, so the page must be flagged suspect, not OCR-replaced.
CMAP_TITLE_PARTIAL = (
    "தமி῁ மιᾠΆ இ तिमल Simple Lear Editor: Ms. Sabitha Rani A.M. "
    "Co-Editor: Ms. Deepika K Mr. C. Karthikeyan The Publication Cell, "
    "Central University of Tamil Nadu Thiruvarur – 610 005 "
    "(Tamil–Hindi –English) இᾸதி ெமாழிையஎளியᾙை rning of Tamil and "
    "Hindi Language in most simple manner for everyone"
)
# Deterministic missing-CMap marker: the extractor emits U+FFFD per unmapped glyph.
FFFD_TEXT = ("the qui�k bro�n fox ju�ps o�er the lazy dog "
             "near the ri�er bank today")
# Fully-unmapped legacy font: every glyph lands in the Private Use Area.
PUA_PAGE = " ".join("\ue041\ue042\ue043\ue044" for _ in range(15))

# Hostile negative corpus for the CMap detector — every one of these is REAL,
# valid text that superficially resembles a corruption signal.
GREEK_PHYSICS = (
    "The decay width Γ depends on the mixing angle θ and the coupling α. "
    "For small θ-dependence the μ-term vanishes and the Δm splitting drives the "
    "oscillation. β-decay rates scale with the φ-meson mass and the Λ-baryon "
    "lifetime, while the ρ-parameter stays fixed under the ω-expansion at NLO.")
NFD_FRENCH = (   # decomposed accents (macOS-generated PDFs) — NFC must fix this
    "Le développement économique a été présenté "
    "à l'assemblée générale après la réunion "
    "du comité, malgré la controverse récente sur le budget.")
IPA_GUIDE = (
    "English /ˈɪŋɡlɪʃ/ and pronunciation /prəˌnʌnsiˈeɪʃən/ appear with "
    "schwa /ə/ and stress marks; compare water /ˈwɔːtər/ with butter /ˈbʌtər/ "
    "and thought /θɔːt/ in most dictionaries of received pronunciation.")
TRI_SCRIPT_DICT = (   # dictionary line: scripts mix on the LINE, never in a word
    "வணக்கம் नमस्ते hello வருக स्वागत welcome நன்றி धन्यवाद thanks "
    "புத்தகம் पुस्तक book தண்ணீர் पानी water வீடு घर house")
HINDI_LATIN_ATTACHED = (   # Latin acronym + Devanagari case marker in one word
    "सरकार ने Twitterपर घोषणा की और GSTकानून लागू किया गया जिससे "
    "व्यापारियों को Onlineपंजीकरण कराना होगा और बाकी नियम पहले जैसे रहेंगे")
RUSSIAN_STRESS = (   # U+0301 on Cyrillic never composes — must not read orphaned
    "ру́сский язы́к учи́ть тру́дно потому́ что ударе́ние па́дает "
    "по-ра́зному в ра́зных слова́х и фо́рмах одного́ сло́ва")
PUA_BULLETS = (   # a few dingbat bullets must not condemn a clean page
    "\uf0b7 First item in the list\n\uf0b7 Second item here\n"
    "\uf0b7 Third item as well\n\uf0b7 Fourth and final item")


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
    ("hex_dump", HEX_DUMP),
    ("base64_wrapped", BASE64_WRAPPED),
    ("config_yaml_css", CONFIG_YAML),
])
def test_negative_corpus_not_flagged_as_glyph_mojibake(name, text):
    # Both reviewers required this negative corpus: the compound heuristic must
    # NOT fire on real ASCII-heavy content (prose / code / URLs / IDs / base64).
    assert tr._looks_like_glyph_mojibake(text) is False, name
    assert tr.is_garbled(text) is False, name


def test_glyph_heuristic_guards_short_strings():
    # Below the word-token floor, even a ';'-heavy fragment is not judged.
    assert tr._looks_like_glyph_mojibake("jkpo;ehL muR") is False


# ── broken-ToUnicode-CMap detection (fourth signal, script independent) ───────

def test_cmap_mojibake_detected_without_font_hint():
    # malayalam.pdf p1: signals 2 and 3 both read ~0 — detection must come
    # purely from the Unicode-sanity (cross-script word) check.
    assert tr._garbage_ratio(CMAP_MOJIBAKE_ML) < 0.40
    assert tr._looks_like_glyph_mojibake(CMAP_MOJIBAKE_ML) is False
    assert tr._looks_like_broken_cmap(CMAP_MOJIBAKE_ML) is True
    assert tr.is_garbled(CMAP_MOJIBAKE_ML) is True        # no font hint given
    assert tr.garble_suspect(CMAP_MOJIBAKE_ML) is True    # superset by design


def test_partial_cmap_corruption_is_suspect_not_recovered():
    # hindi.pdf cover: a CMap-garbled title on a mostly-clean-Latin page. OCR-
    # replacing the whole page would trade ~90% clean born-digital text for OCR
    # output — so the page must be flagged suspect, NOT sent to recovery.
    assert tr._looks_like_broken_cmap(CMAP_TITLE_PARTIAL) is False
    assert tr.is_garbled(CMAP_TITLE_PARTIAL) is False
    assert tr.garble_suspect(CMAP_TITLE_PARTIAL) is True


def test_replacement_chars_trigger_recovery():
    # U+FFFD is the extractor's own "no mapping for this glyph" marker —
    # deterministic, so a handful is decisive.
    assert tr._looks_like_broken_cmap(FFFD_TEXT) is True
    assert tr.is_garbled(FFFD_TEXT) is True


def test_pua_mapped_page_triggers_recovery():
    # A fully-unmapped legacy font lands every glyph in the Private Use Area.
    # PUA codepoints are not isalpha(), so the word-level signals are blind
    # here — the PUA-density signal must carry it alone.
    assert tr._looks_like_broken_cmap(PUA_PAGE) is True
    assert tr.is_garbled(PUA_PAGE) is True


@pytest.mark.parametrize("name,text", [
    ("greek_physics", GREEK_PHYSICS),
    ("nfd_french", NFD_FRENCH),
    ("ipa_guide", IPA_GUIDE),
    ("tri_script_dictionary", TRI_SCRIPT_DICT),
    ("hindi_latin_attached", HINDI_LATIN_ATTACHED),
    ("russian_stress_marks", RUSSIAN_STRESS),
    ("pua_bullet_list", PUA_BULLETS),
])
def test_cmap_negative_corpus_not_flagged(name, text):
    # Not even the suspect floor may fire: these are real, valid pages, and a
    # false suspect flag would poison downstream filtering.
    assert tr._looks_like_broken_cmap(text) is False, name
    assert tr.garble_suspect(text) is False, name
    assert tr.is_garbled(text) is False, name


def test_disjoint_families_stay_disjoint():
    # Each mojibake family is owned by its own signal: the CMap detector must
    # NOT fire on ASCII glyph mojibake or Latin-1 soup (and vice versa is
    # covered above) — otherwise threshold changes in one family silently
    # shift behaviour in another.
    assert tr._looks_like_broken_cmap(GLYPH_MOJIBAKE) is False
    assert tr._looks_like_broken_cmap(MOJIBAKE) is False


def test_cmap_recovery_end_to_end(make_ocr_stub):
    # Detection → OCR → verify for the broken-CMap family: the stub returns real
    # Malayalam, which passes the script-ratio gate and replaces the garbage.
    backend = make_ocr_stub(scripts=("eng", "mal"), output=MALAYALAM)
    rec = tr.recover(CMAP_MOJIBAKE_ML, ["BalooChettan2-Regular"],
                     render_png=lambda: b"PNG", backend=backend,
                     declared_script="mal", candidate_langs=["ml"])
    assert rec.was_recovered is True
    assert rec.script == "mal"
    assert rec.text == MALAYALAM
    assert backend.calls and backend.calls[0][0] == "mal+eng"


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


def test_recover_reports_producing_engine(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU, name="stub")
    rec = tr.recover(MOJIBAKE, ["SHREE-TEL"], render_png=lambda: b"PNG",
                     backend=backend, candidate_langs=["te"])
    assert rec.was_recovered is True
    assert rec.engine == "stub"


def test_ocr_scanned_reports_producing_engine(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU, name="stub")
    rec = tr.ocr_scanned(render_png=lambda: b"PNG", backend=backend,
                         candidate_langs=["te"], declared_script="tel")
    assert rec.was_recovered is True
    assert rec.engine == "stub"


# ── scanned pages under DEFAULT --lang en (the ocr_scanned symmetry fix) ──────

def test_ocr_scanned_resolves_via_all_supported_osd_when_no_lang_declared(
        make_ocr_stub, monkeypatch):
    # A scanned Tamil page with NO --lang: ocr_scanned must OSD over the full
    # supported set (like recover), route to 'tam', and return the Tamil text —
    # not silently fall back to English OCR.
    backend = make_ocr_stub(scripts=("eng", "tam"), output=TAMIL)
    seen = {}

    def fake_osd(render_png, iso_candidates):
        seen["iso"] = list(iso_candidates)
        return "tam"
    monkeypatch.setattr(tr, "_detect_script_osd", fake_osd)

    rec = tr.ocr_scanned(render_png=lambda: b"PNG", backend=backend,
                         candidate_langs=[], declared_script=None)
    assert rec.was_recovered is True
    assert rec.script == "tam"
    assert rec.text == TAMIL
    # OSD saw the supported set filtered to installed packs ('en' has no
    # recovery script and self-filters), never an empty list.
    assert seen["iso"] == ["ta"]
    assert backend.calls and backend.calls[0][0] == "tam+eng"


def test_ocr_scanned_noops_when_osd_cannot_resolve_no_lang(
        make_ocr_stub, monkeypatch):
    # OSD failure with no declared language → no-op, caller falls back to plain
    # English OCR (no all-pack joined OCR guessing).
    backend = make_ocr_stub(scripts=("eng", "tam"), output=TAMIL)
    monkeypatch.setattr(tr, "_detect_script_osd", lambda rp, iso: None)
    rec = tr.ocr_scanned(render_png=lambda: b"PNG", backend=backend,
                         candidate_langs=[], declared_script=None)
    assert rec.was_recovered is False


def test_ocr_scanned_no_lang_discards_wrong_script_output(
        make_ocr_stub, monkeypatch):
    # OSD says Tamil but OCR emits Latin junk → the confidence gate (scored over
    # the supported blocks, not the guess alone) must discard it.
    backend = make_ocr_stub(scripts=("eng", "tam"), output="qwerty junk output")
    monkeypatch.setattr(tr, "_detect_script_osd", lambda rp, iso: "tam")
    rec = tr.ocr_scanned(render_png=lambda: b"PNG", backend=backend,
                         candidate_langs=[], declared_script=None)
    assert rec.was_recovered is False
    assert rec.text == ""                        # no-op: caller does English OCR
    # Exactly one OSD-directed attempt was made (tam, bilingual co-load) and
    # discarded by the confidence gate — never an all-pack joined guess.
    assert [c[0] for c in backend.calls] == ["tam+eng"]


def test_recover_verifies_against_installed_supported_set_not_just_script(
        make_ocr_stub, monkeypatch):
    # With no declared language, verification must score OCR output against the
    # whole INSTALLED supported script-block set — not just the OSD-guessed pack.
    # Here OSD guesses 'tam' but the backend actually emits Telugu Unicode. Only
    # if verify spans the installed set (which includes 'tel') does 'tel' win;
    # verifying against ['tam'] alone would score ~0 and DISCARD the recovery.
    backend = make_ocr_stub(scripts=("eng", "tam", "tel"), output=TELUGU)
    monkeypatch.setattr(tr, "_detect_script_osd", lambda rp, iso: "tam")
    rec = tr.recover(GLYPH_MOJIBAKE, font_names=[], render_png=lambda: b"PNG",
                     backend=backend, declared_script=None, candidate_langs=[])
    assert rec.was_recovered is True
    assert rec.script == "tel"                   # verify picked the real block
    assert rec.confidence > 0.9
    assert rec.text == TELUGU


# ── finding 16: a no-op recovery carries the ORIGINAL text ───────────────────

def test_recover_noop_carries_original_text(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    rec = tr.recover(CLEAN_EN, ["Helvetica"], render_png=_raise_png,
                     backend=backend, candidate_langs=["te"])
    assert rec.was_recovered is False
    assert rec.text == CLEAN_EN                  # not "" — original is preserved


# ── Assamese routing (shares the Bengali block, own Tesseract pack) ──────────

ASSAMESE = "অসমীয়া ভাষা উত্তৰ-পূৱ ভাৰতৰ অসম ৰাজ্যৰ চৰকাৰী ভাষা"


def test_assamese_script_mapping():
    assert tr.script_for_language("as") == "asm"
    assert tr.script_ratio(ASSAMESE, "asm") > 0.9    # same block as ben
    assert tr.script_from_fonts(["ABCDEE+Ramdhenu"]) == "asm"


def test_assamese_recovery_routes_to_asm_pack(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "asm"), output=ASSAMESE)
    rec = tr.recover(MOJIBAKE, ["Ramdhenu"], render_png=lambda: b"PNG",
                     backend=backend, declared_script="asm",
                     candidate_langs=["as"])
    assert rec.was_recovered is True
    assert rec.script == "asm"
    assert rec.text == ASSAMESE
    assert backend.calls and backend.calls[0][0] == "asm+eng"


# ── Signal 5: legacy symbol-glyph splicing (F17, docs/TEST-vast.md) ──────────
# Real corpus sample (NirvachanaRamayanam): legacy Telugu DTP font maps
# conjunct/vowel glyphs onto ASCII punctuation — | ≈ ్ర, ( ≈ ఁ — so the page
# decodes as clean-block Telugu with symbols spliced inside words. Signals 1–4
# all read 0.0 on this text.
SYMBOL_SPLICED = (
    "రాముడు వీర్యసంపద సుర|పభుఁబోలి క్షమాగుణంబునన్ భూమిని(బోలి బుద్ధి "
    "గురు(బోలి పజాభిషొతంబులున్ మహాత|పదీపుల సహ సమయూఖుని(బోలి రాజిలున్ "
    "అ|శితావనసు|వతుండై నవాని నఖిలలోకపాలోపముండైనవాని"
)
CLEAN_TE_PUNCT = (
    "రాముడు అడవికి వెళ్ళెను (చూడండి పుట 12) అక్కడ సీత లక్ష్మణుడు "
    "కుటీరము నిర్మించిరి। వారు పండ్లు కూరలు తినుచు జీవించిరి॥ "
    "రాముని ధర్మపత్ని సీత మహా పతివ్రత"
)


def test_symbol_spliced_page_is_garbled():
    assert tr._looks_like_symbol_glyphs(SYMBOL_SPLICED) is True
    assert tr.is_garbled(SYMBOL_SPLICED) is True
    # the older signals stay blind to it — signal 5 owns this family
    assert tr._looks_like_broken_cmap(SYMBOL_SPLICED) is False
    assert tr._garbage_ratio(SYMBOL_SPLICED) < 0.40


def test_clean_telugu_with_parens_and_danda_is_not_garbled():
    # paired punctuation flanking a word and daṇḍā sentence marks are legit —
    # only symbols spliced BETWEEN Indic letters count
    assert tr._looks_like_symbol_glyphs(CLEAN_TE_PUNCT) is False
    assert tr.is_garbled(CLEAN_TE_PUNCT) is False
    assert tr.garble_suspect(CLEAN_TE_PUNCT) is False


def test_latin_pipes_do_not_trigger_symbol_splice():
    row = "name | qty | price | total — see the invoice table for details " * 3
    assert tr._looks_like_symbol_glyphs(row) is False
    assert tr.is_garbled(row) is False


def test_short_spliced_header_never_judged():
    assert tr._looks_like_symbol_glyphs("సుర|పభుఁబోలి క్షమ") is False


def test_sub_recover_splice_density_flags_suspect():
    # 2 spliced words in ~30: below the recover floor, above the suspect one
    clean = "రాముడు అడవికి వెళ్ళెను అక్కడ సీత లక్ష్మణుడు కుటీరము నిర్మించిరి " * 4
    text = clean + " సుర|పభుఁబోలి భూమిని(బోలి"
    assert tr._looks_like_symbol_glyphs(text) is False
    assert tr.garble_suspect(text) is True


def test_symbol_spliced_page_recovers_via_ocr(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    rec = tr.recover(SYMBOL_SPLICED, [], render_png=lambda: b"PNG",
                     backend=backend, declared_script="tel",
                     candidate_langs=["te"])
    assert rec.was_recovered is True
    assert rec.text == TELUGU


# ── --force-ocr: skip the clean gate, keep the verify gate (F18) ─────────────

def test_force_ocr_recovers_a_clean_page(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    rec = tr.recover(TELUGU + " స్వచ్ఛమైన పాఠ్యం", [], render_png=lambda: b"PNG",
                     backend=backend, declared_script="tel",
                     candidate_langs=["te"], force=True)
    assert rec.was_recovered is True
    assert rec.text == TELUGU


def test_force_ocr_keeps_original_when_ocr_is_garbage(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output="?!.. ~~ ..")
    original = TELUGU + " స్వచ్ఛమైన పాఠ్యం"
    rec = tr.recover(original, [], render_png=lambda: b"PNG",
                     backend=backend, declared_script="tel",
                     candidate_langs=["te"], force=True)
    assert rec.was_recovered is False
    assert rec.text == original                   # verify gate still applies


def test_plan_recover_force_mirrors_recover(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    original = TELUGU + " స్వచ్ఛమైన పాఠ్యం"
    assert tr.plan_recover(original, [], b"PNG", backend=backend,
                           declared_script="tel", candidate_langs=["te"]) is None
    plan = tr.plan_recover(original, [], b"PNG", backend=backend,
                           declared_script="tel", candidate_langs=["te"],
                           force=True)
    assert plan is not None and plan.kind == "recover"
    assert plan.original_text == original
