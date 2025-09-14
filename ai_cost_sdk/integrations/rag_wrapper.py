"""RAG helpers."""

from __future__ import annotations

from typing import Iterable

from .. import middleware


def embed(texts: Iterable[str], model: str, vendor: str = "openai"):
    tokens = sum(len(t) // 4 for t in texts)
    with middleware.rag_embed(model=model, vendor=vendor, tokens=tokens):
        # Placeholder embedding
        return [[0.0] * 3 for _ in texts]


def vector_search(
    index_id: str,
    query: str,
    k: int,
    index_version: str,
    vendor: str = "vector_db",
):
    with middleware.rag_search(index_id, index_version, k, freshness_s=0):
        # Placeholder search
        return [f"doc-{i}" for i in range(k)]
