"""Token-aware sizing.

Chunk size should track an embedding model's *token* budget, not character
count — char thresholds over/undershoot unpredictably across dense tables vs.
flowing prose.

Uses the real model tokenizer when ``transformers`` can load it (optional extra);
otherwise falls back to a deterministic ~4-chars-per-token approximation. Either
way the call is cheap (tokenizer loaded once and cached) and import-safe.

Adapted from the AgenticAI_Manufacturing ``doc_pipeline.ingestion.tokenization``
module (the ``config.EMBEDDING_MODEL`` coupling is replaced by an env var).
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger("vega.tokenization")

# Leave headroom below a typical 512-token embedder limit for the heading
# breadcrumb + any contextual prefix prepended at chunk time.
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 60
_CHARS_PER_TOKEN = 4.0


@lru_cache(maxsize=2)
def _load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
        tok = AutoTokenizer.from_pretrained(model_name)
        logger.info("token sizing: using %s tokenizer", model_name)
        return tok
    except Exception as e:  # missing optional dep, offline, etc. — degrade
        logger.debug("token sizing: falling back to char approximation (%r)", e)
        return None


def _model_name() -> str:
    return os.environ.get("VEGA_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def count_tokens(text: str) -> int:
    """Token count via the model tokenizer, or a char approximation."""
    if not text:
        return 0
    tok = _load_tokenizer(_model_name())
    if tok is None:
        return max(1, int(len(text) / _CHARS_PER_TOKEN))
    # add_special_tokens=False — we want raw content length, not [CLS]/[SEP].
    return len(tok.encode(text, add_special_tokens=False))


def approx_chars_for_tokens(n_tokens: int) -> int:
    """Inverse of the approximation — for callers that still slice by char."""
    return int(n_tokens * _CHARS_PER_TOKEN)
