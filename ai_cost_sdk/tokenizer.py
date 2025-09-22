"""Multi-LLM tokenizer support for accurate token counting."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from tiktoken.core import Encoding

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Model to encoding mapping for OpenAI models
OPENAI_MODEL_ENCODINGS = {
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
CLAUDE_MODEL_PATTERNS = {
    "claude-3-5-sonnet": r"\S+",
    "claude-3-5-haiku": r"\S+", 
    "claude-3-opus": r"\S+",
    "claude-3-sonnet": r"\S+",
    "claude-3-haiku": r"\S+",
}

# Gemini model patterns (approximate tokenization)
GEMINI_MODEL_PATTERNS = {
    "gemini-pro": r"\S+",
    "gemini-pro-vision": r"\S+",
    "gemini-1.5-pro": r"\S+",
    "gemini-1.5-flash": r"\S+",
}

_ENCODING_CACHE: Dict[str, tuple["Encoding", int]] = {}


def _get_tiktoken_encoding(name: str):
    getter_id = id(getattr(tiktoken, "get_encoding", None))
    cached = _ENCODING_CACHE.get(name)
    if cached and cached[1] == getter_id:
        return cached[0]

    global TIKTOKEN_AVAILABLE
    try:
        encoding = tiktoken.get_encoding(name)
    except Exception:
        TIKTOKEN_AVAILABLE = False
        raise

    _ENCODING_CACHE[name] = (encoding, getter_id)
    return encoding


def _fallback_token_estimate(text: str) -> int:
    words = text.split()
    if not words:
        return 0

    char_estimate = max(1, (len(text) + 3) // 4)

    if len(words) <= 3:
        return max(len(words), char_estimate)
    if len(words) <= 10:
        word_estimate = max(1, int(len(words) * 1.2))
        return max(char_estimate, word_estimate)
    return char_estimate


def count_tokens_openai(text: str, model: str) -> int:
    """Count tokens for OpenAI models using tiktoken."""
    if not TIKTOKEN_AVAILABLE:
        return _fallback_token_estimate(text)
    
    encoding_name = OPENAI_MODEL_ENCODINGS.get(model, "cl100k_base")
    try:
        encoding = _get_tiktoken_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to hybrid approach when tiktoken fails
        return _fallback_token_estimate(text)


def count_tokens_claude(text: str, model: str) -> int:
    """Count tokens for Claude models using word-based approximation."""
    # Claude uses a different tokenization scheme, but word-based approximation is reasonable
    # This is more accurate than character-based for Claude
    words = re.findall(r'\S+', text)
    return len(words)


def count_tokens_gemini(text: str, model: str) -> int:
    """Count tokens for Gemini models using word-based approximation."""
    # Gemini also uses different tokenization, word-based approximation is reasonable
    words = re.findall(r'\S+', text)
    return len(words)


def count_tokens(text: str, model: str, vendor: str = "openai") -> int:
    """
    Count tokens for text using the appropriate tokenizer for the model.
    
    Args:
        text: Input text to tokenize
        model: Model name (e.g., "gpt-4o", "claude-3-5-sonnet")
        vendor: Vendor name ("openai", "claude", "gemini")
    
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    
    vendor = vendor.lower()
    
    if vendor == "openai":
        return count_tokens_openai(text, model)
    elif vendor == "claude":
        return count_tokens_claude(text, model)
    elif vendor == "gemini":
        return count_tokens_gemini(text, model)
    else:
        # Default fallback for unknown vendors
        return (len(text) + 3) // 4


def count_tokens_batch(texts: List[str], model: str, vendor: str = "openai") -> int:
    """
    Count tokens for a batch of texts.
    
    Args:
        texts: List of input texts
        model: Model name
        vendor: Vendor name
    
    Returns:
        Total number of tokens across all texts
    """
    if not texts:
        return 0

    vendor = vendor.lower()

    use_native_batch = (
        vendor == "openai"
        and TIKTOKEN_AVAILABLE
        and getattr(count_tokens, "__module__", "") == __name__
    )

    if use_native_batch:
        encoding_name = OPENAI_MODEL_ENCODINGS.get(model, "cl100k_base")
        try:
            encoding = _get_tiktoken_encoding(encoding_name)
        except Exception:
            return sum(count_tokens(text, model, vendor) for text in texts)
        return sum(len(tokens) for tokens in encoding.encode_batch(texts))

    return sum(count_tokens(text, model, vendor) for text in texts)


def get_model_vendor(model: str) -> str:
    """
    Infer vendor from model name.
    
    Args:
        model: Model name
    
    Returns:
        Vendor name ("openai", "claude", "gemini", "unknown")
    """
    model_lower = model.lower()
    
    if any(openai_model in model_lower for openai_model in ["gpt", "text-embedding"]):
        return "openai"
    elif "claude" in model_lower:
        return "claude"
    elif "gemini" in model_lower:
        return "gemini"
    else:
        return "unknown"


def validate_model_support(model: str, vendor: str) -> bool:
    """
    Check if a model is supported by the tokenizer.
    
    Args:
        model: Model name
        vendor: Vendor name
    
    Returns:
        True if model is supported
    """
    vendor = vendor.lower()
    
    if vendor == "openai":
        return model in OPENAI_MODEL_ENCODINGS
    elif vendor == "claude":
        return any(pattern in model.lower() for pattern in CLAUDE_MODEL_PATTERNS)
    elif vendor == "gemini":
        return any(pattern in model.lower() for pattern in GEMINI_MODEL_PATTERNS)
    else:
        return False
