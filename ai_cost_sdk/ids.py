"""ID hashing utilities."""

from __future__ import annotations

import hashlib


_DEF_SALT = b"ai-cost-sdk"


def hash_id(value: str) -> str:
    """Return a stable hashed identifier."""
    h = hashlib.sha256(_DEF_SALT + value.encode("utf-8"))
    return h.hexdigest()[:16]
