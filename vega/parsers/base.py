"""Parser protocol — every parser maps a file path to a ``DocumentModel``."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from vega.model import DocumentModel


class Parser(Protocol):
    """A format adapter. Implementations must be side-effect free and must
    not raise on malformed-but-readable input — degrade to whatever structure
    is recoverable. Hard IO errors (missing file, wrong type) may raise; the
    pipeline isolates them per file."""

    def parse(self, path: Path) -> DocumentModel: ...
