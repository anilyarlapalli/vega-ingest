"""Text recovery — mojibake detection, script routing, OCR-backed recovery.

All OCR is served by in-memory stubs; no real Tesseract pack is required.
"""

from vega import text_recovery as tr

# Real Telugu (Unicode block 0x0C00-0x0C7F) — what a good recovery OCR yields.
TELUGU = "పరిపాలన పరిషత్తు నుండి ఉత్తర్వు జారీ చేయబడింది"
# Legacy-font mojibake: a dense run of accented-Latin glyphs, no real words.
MOJIBAKE = "Bçñüéàæþ§«ßÿîïçœ ëêÀÁÂÃ æþðøµ¶·"
CLEAN_EN = "The order was issued by the administration office in the district."


# ── detection ────────────────────────────────────────────────────────────────

def test_clean_english_is_not_garbled():
    assert tr.is_garbled(CLEAN_EN) is False
    assert tr._garbage_ratio(CLEAN_EN) < 0.4


def test_accented_glyph_soup_is_garbled():
    assert tr.is_garbled(MOJIBAKE) is True
    assert tr._garbage_ratio(MOJIBAKE) >= 0.4


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


# ── finding 16: a no-op recovery carries the ORIGINAL text ───────────────────

def test_recover_noop_carries_original_text(make_ocr_stub):
    backend = make_ocr_stub(scripts=("eng", "tel"), output=TELUGU)
    rec = tr.recover(CLEAN_EN, ["Helvetica"], render_png=_raise_png,
                     backend=backend, candidate_langs=["te"])
    assert rec.was_recovered is False
    assert rec.text == CLEAN_EN                  # not "" — original is preserved
