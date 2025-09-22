"""Multi-LLM tokenizer support for accurate token counting."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Dict, List

try:  # pragma: no cover - exercised indirectly
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised indirectly
    TIKTOKEN_AVAILABLE = False

# Track models whose encodings could not be loaded so we can avoid repeated attempts.
_FAILED_ENCODINGS: set[str] = set()

# Model to encoding mapping for OpenAI models
OPENAI_MODEL_ENCODINGS: Dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-ada-002": "cl100k_base",
}

# Claude model patterns (approximate tokenization)
CLAUDE_MODEL_PATTERNS: Dict[str, str] = {
    "claude-3-5-sonnet": r"\S+",
    "claude-3-5-haiku": r"\S+",
    "claude-3-opus": r"\S+",
    "claude-3-sonnet": r"\S+",
    "claude-3-haiku": r"\S+",
}

# Gemini model patterns (approximate tokenization)
GEMINI_MODEL_PATTERNS: Dict[str, str] = {
    "gemini-pro": r"\S+",
    "gemini-pro-vision": r"\S+",
    "gemini-1.5-pro": r"\S+",
    "gemini-1.5-flash": r"\S+",
}


def _estimate_tokens(words: list[str], text: str) -> int:
    """Estimate token counts when an exact tokenizer is unavailable."""

    if not words:
        return 0

    char_estimate = max(1, math.ceil(len(text) / 4))

    if len(words) <= 3:
        # Short snippets: slightly over-estimate relative to word count.
        return max(char_estimate, len(words) + 1)
    if len(words) <= 10:
        # Slightly longer snippets: lean towards a mild over-estimate.
        return max(char_estimate, math.ceil(len(words) * 1.2))

    # Long texts: rely on character-based estimate and round up.
    return char_estimate


@lru_cache(maxsize=None)
def _get_openai_encoding(model: str, encoding_fn_id: int):  # pragma: no cover - exercised indirectly
    """Cache tiktoken encodings per-model to avoid repeated lookups."""

    if not TIKTOKEN_AVAILABLE:
        raise RuntimeError("tiktoken is not available")

    encoding_name = OPENAI_MODEL_ENCODINGS.get(model, "cl100k_base")
    return tiktoken.get_encoding(encoding_name)


def _fetch_openai_encoding(model: str):  # pragma: no cover - exercised indirectly
    if model in _FAILED_ENCODINGS:
        raise RuntimeError("encoding unavailable")

    try:
        return _get_openai_encoding(model, id(tiktoken.get_encoding))
    except Exception:
        _FAILED_ENCODINGS.add(model)
        raise


def count_tokens_openai(text: str, model: str) -> int:
    """Count tokens for OpenAI models using tiktoken."""

    if not TIKTOKEN_AVAILABLE:
        words = re.findall(r"\S+", text)
        return _estimate_tokens(words, text)

    try:
        encoding = _fetch_openai_encoding(model)
        return len(encoding.encode(text))
    except Exception:
        words = re.findall(r"\S+", text)
        return _estimate_tokens(words, text)


def count_tokens_claude(text: str, model: str) -> int:
    """Count tokens for Claude models using word-based approximation."""

    words = re.findall(r"\S+", text)
    return len(words)


def count_tokens_gemini(text: str, model: str) -> int:
    """Count tokens for Gemini models using word-based approximation."""

    words = re.findall(r"\S+", text)
    return len(words)


def count_tokens(text: str, model: str, vendor: str = "openai") -> int:
    """Count tokens for text using the appropriate tokenizer for the model."""

    if not text:
        return 0

    vendor = vendor.lower()

    if vendor == "openai":
        return count_tokens_openai(text, model)
    if vendor == "claude":
        return count_tokens_claude(text, model)
    if vendor == "gemini":
        return count_tokens_gemini(text, model)

    # Default fallback for unknown vendors
    return max(1, math.ceil(len(text) / 4))


_ORIGINAL_COUNT_TOKENS = count_tokens


def count_tokens_batch(texts: List[str], model: str, vendor: str = "openai") -> int:
    """Count tokens for a batch of texts."""

    if not texts:
        return 0

    vendor = vendor.lower()

    if (
        vendor == "openai"
        and TIKTOKEN_AVAILABLE
        and count_tokens is _ORIGINAL_COUNT_TOKENS
    ):
        try:
            encoding = _fetch_openai_encoding(model)
            if hasattr(encoding, "encode_batch"):
                return sum(len(tokens) for tokens in encoding.encode_batch(texts))
            if hasattr(encoding, "encode_ordinary_batch"):
                return sum(len(tokens) for tokens in encoding.encode_ordinary_batch(texts))
            return sum(len(encoding.encode(text)) for text in texts)
        except Exception:
            pass

    return sum(count_tokens(text, model, vendor) for text in texts)


def get_model_vendor(model: str) -> str:
    """Infer vendor from model name."""

    model_lower = model.lower()

    if any(openai_model in model_lower for openai_model in ["gpt", "text-embedding"]):
        return "openai"
    if "claude" in model_lower:
        return "claude"
    if "gemini" in model_lower:
        return "gemini"
    return "unknown"


def validate_model_support(model: str, vendor: str) -> bool:
    """Check if a model is supported by the tokenizer."""

    vendor = vendor.lower()

    if vendor == "openai":
        return model in OPENAI_MODEL_ENCODINGS
    if vendor == "claude":
        return any(pattern in model.lower() for pattern in CLAUDE_MODEL_PATTERNS)
    if vendor == "gemini":
        return any(pattern in model.lower() for pattern in GEMINI_MODEL_PATTERNS)
    return False
