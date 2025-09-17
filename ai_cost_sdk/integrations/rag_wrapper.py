"""RAG helpers."""

from __future__ import annotations

from typing import Iterable

from .. import middleware
from ..pricing_calc import estimate_tokens


def embed(texts: Iterable[str], model: str, vendor: str = "openai"):
    tokens = sum(estimate_tokens(t) for t in texts)
    with middleware.rag_embed(model=model, vendor=vendor, tokens=tokens):
        # Placeholder embedding
        return [[0.0] * 3 for _ in texts]


def vector_search(
    index_id: str,
    query: str,
    k: int,
    index_version: str,
    vendor: str = "vector_db",
    read_units: int = 0,
    price_per_unit: float = 0.0,
    freshness_s: int = 0,
):
    with middleware.rag_search(
        index_id,
        index_version,
        k,
        freshness_s=freshness_s,
        read_units=read_units,
        price_per_unit=price_per_unit,
    ):
        # Placeholder search
        return [f"doc-{i}" for i in range(k)]
