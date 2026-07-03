"""OCR backend protocol — the pluggable seam.

Every OCR engine (Tesseract, EasyOCR, and future PaddleOCR / Surya) implements
``OCRBackend``. Callers (the PDF/image parsers and ``text_recovery``) only ever
talk to this interface, so a new engine drops in without touching them.

The lingua franca for languages is the **Tesseract-style script code** (e.g.
``"tel"``, or ``"tel+eng"`` for a bilingual page) — that is what the crown-jewel
``text_recovery`` cascade already resolves (font-name → declared → OSD → Unicode
block). Each backend maps those codes onto its own engine codes internally and
advertises which ones it can actually handle via :meth:`available_scripts`, so
selection can route or fall back per script.
"""

from __future__ import annotations

from typing import List, Protocol, Set, runtime_checkable


@runtime_checkable
class OCRBackend(Protocol):
    """A text-OCR engine. Side-effect free; must not raise on unreadable input
    (return ``""`` instead) so the pipeline's per-file fault isolation holds."""

    name: str

    def available_scripts(self) -> Set[str]:
        """Tesseract-style codes this backend can OCR (e.g. ``{"eng","tel"}``)."""
        ...

    def image_to_text(self, image_png: bytes, script: str) -> str:
        """OCR one PNG-encoded image. ``script`` is a ``+``-joined Tesseract code
        string (``"tel+eng"``). Returns recognised text (``""`` on failure)."""
        ...

    def image_to_text_batch(self, images: List[bytes], script: str) -> List[str]:
        """OCR a batch of images (GPU backends override for batched inference)."""
        ...


class BaseOCRBackend:
    """Convenience base: a serial default for the batch variant."""

    name = "base"

    def available_scripts(self) -> Set[str]:  # pragma: no cover - abstract
        return set()

    def can_handle(self, script: str) -> bool:
        """True iff this backend can OCR **every** part of a ``+``-joined script
        request in a single call. Default: all parts are advertised. Engines with
        combination limits (e.g. EasyOCR — one non-Latin script per Reader)
        override this so the fallback router sends combos to a capable backend."""
        av = self.available_scripts()
        return all(p in av for p in script.split("+") if p)

    def cache_version(self) -> str:
        """A fingerprint of the engine + its config that affects OCR *output*.
        Included in the disk-cache key so an engine/pack upgrade doesn't serve
        stale cached text. Subclasses should incorporate their real version."""
        return f"{self.name}:v1"

    def image_to_text(self, image_png: bytes, script: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def image_to_text_batch(self, images: List[bytes], script: str) -> List[str]:
        return [self.image_to_text(im, script) for im in images]
