"""RAG helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Callable

from .. import middleware
from ..tokenizer import count_tokens_batch

EmbeddingBackend = Callable[..., Any]
SearchBackend = Callable[..., Any]


def embed(
    texts: Iterable[str],
    model: str,
    vendor: str = "openai",
    *,
    embedder: EmbeddingBackend | None = None,
    **embed_kwargs: Any,
) -> list[list[float]]:
    """Embed texts using a configurable backend with proper token counting.

    Args:
        texts: Iterable of strings to embed.
        model: Embedding model identifier.
        vendor: Vendor name for pricing/metrics.
        embedder: Callable or client responsible for generating embeddings. The
            callable will receive ``texts`` (as a list), ``model`` and
            ``vendor`` as keyword arguments along with any ``embed_kwargs``.
        **embed_kwargs: Additional keyword arguments forwarded to the backend.

    Returns:
        List of embedding vectors returned by the backend.
    """

    texts_list = list(texts)
    if embedder is None:
        raise ValueError("An embedding backend must be provided via 'embedder'.")

    tokens = count_tokens_batch(texts_list, model, vendor)

    with middleware.rag_embed(model=model, vendor=vendor, tokens=tokens, texts=texts_list) as span:
        raw_embeddings = _call_embed_backend(
            embedder,
            texts_list=texts_list,
            model=model,
            vendor=vendor,
            **embed_kwargs,
        )
        embeddings = _normalize_embeddings(raw_embeddings)

        if span is not None:
            span.set_attribute("rag.embedding.count", len(embeddings))
            if embeddings:
                span.set_attribute("rag.embedding.dim", len(embeddings[0]))

        return embeddings


def vector_search(
    index_id: str,
    query: str,
    k: int,
    index_version: str,
    *,
    vendor: str = "vector_db",
    read_units: int = 0,
    price_per_unit: float = 0.0,
    searcher: SearchBackend | None = None,
    freshness_s: int = 0,
    **search_kwargs: Any,
) -> list[Any]:
    """Perform vector search with cost tracking using a configurable backend."""

    if searcher is None:
        raise ValueError("A search backend must be provided via 'searcher'.")

    with middleware.rag_search(
        index_id,
        index_version,
        k,
        freshness_s=freshness_s,
        read_units=read_units,
        price_per_unit=price_per_unit,
        vendor=vendor,
    ) as span:
        results, used_read_units, used_price_per_unit = _call_search_backend(
            searcher,
            index_id=index_id,
            query=query,
            k=k,
            index_version=index_version,
            vendor=vendor,
            **search_kwargs,
        )

        final_read_units = used_read_units if used_read_units is not None else read_units
        final_price_per_unit = (
            used_price_per_unit if used_price_per_unit is not None else price_per_unit
        )

        if span is not None:
            span.set_attribute("rag.results.count", len(results))
            if final_read_units is not None:
                span.set_attribute("rag.read_units", final_read_units)
                setattr(span, "_rag_read_units", final_read_units)
            if final_price_per_unit is not None:
                span.set_attribute("rag.price_per_unit", final_price_per_unit)
                setattr(span, "_rag_price_per_unit", final_price_per_unit)

        return list(results)


def _call_embed_backend(
    embedder: EmbeddingBackend,
    *,
    texts_list: list[str],
    model: str,
    vendor: str,
    **embed_kwargs: Any,
) -> Any:
    """Invoke the configured embedding backend."""

    if callable(embedder):
        return embedder(texts_list, model=model, vendor=vendor, **embed_kwargs)

    embed_method = getattr(embedder, "embed", None)
    if callable(embed_method):
        return embed_method(texts_list=texts_list, model=model, vendor=vendor, **embed_kwargs)

    raise TypeError("embedder must be callable or expose an 'embed' method")


def _normalize_embeddings(raw: Any) -> list[list[float]]:
    """Normalize backend responses into a list of embedding vectors."""

    if isinstance(raw, dict):
        candidate = raw.get("data") or raw.get("embeddings") or raw
    elif hasattr(raw, "data"):
        candidate = getattr(raw, "data")
    else:
        candidate = raw

    data = _extract_sequence(candidate, default_key_candidates=("embeddings", "results"))
    embeddings: list[list[float]] = []

    for item in data:
        vector = _extract_embedding_vector(item)
        embeddings.append([float(x) for x in vector])

    return embeddings


def _call_search_backend(
    searcher: SearchBackend,
    *,
    index_id: str,
    query: str,
    k: int,
    index_version: str,
    vendor: str,
    **search_kwargs: Any,
) -> tuple[list[Any], int | None, float | None]:
    """Invoke the configured search backend and normalize the response."""

    call_kwargs = {
        "index_id": index_id,
        "query": query,
        "k": k,
        "index_version": index_version,
        "vendor": vendor,
    }
    call_kwargs.update(search_kwargs)

    if callable(searcher):
        response = searcher(**call_kwargs)
    else:
        search_method = getattr(searcher, "search", None) or getattr(searcher, "query", None)
        if not callable(search_method):
            raise TypeError("searcher must be callable or expose a 'search'/'query' method")
        response = search_method(**call_kwargs)

    results, metadata = _normalize_search_response(response)
    read_units = _extract_metric(metadata, [
        ("read_units",),
        ("usage", "read_units"),
        ("metering", "read_units"),
    ])
    price_per_unit = _extract_metric(metadata, [
        ("price_per_unit",),
        ("usage", "price_per_unit"),
        ("metering", "price_per_unit"),
    ])

    return results, read_units, price_per_unit


def _normalize_search_response(response: Any) -> tuple[list[Any], Any]:
    """Normalize various backend response shapes into results and metadata."""

    metadata: Any = {}

    if isinstance(response, tuple):
        if len(response) == 3:
            results, read_units, price_per_unit = response
            metadata = {"read_units": read_units, "price_per_unit": price_per_unit}
        elif len(response) == 2:
            results, metadata = response
            if isinstance(metadata, (int, float)):
                metadata = {"read_units": metadata}
        else:
            results = response[0]
        return list(results or []), metadata

    if isinstance(response, dict):
        metadata = response
        results = _extract_sequence(
            response,
            default_key_candidates=("results", "matches", "data", "documents"),
        )
        return list(results), metadata

    attr_results = None
    for attr in ("results", "matches", "data"):
        if hasattr(response, attr):
            attr_results = getattr(response, attr)
            metadata = response
            break

    if attr_results is not None:
        return list(attr_results or []), metadata

    return list(response or []), metadata


def _extract_sequence(raw: Any, default_key_candidates: Sequence[str] | tuple[str, ...]) -> list[Any]:
    """Extract a sequence from different response shapes."""

    if raw is None:
        return []

    if isinstance(raw, list):
        return raw

    if isinstance(raw, tuple):
        return list(raw)

    if isinstance(raw, dict):
        for key in default_key_candidates:
            if key in raw:
                value = raw[key]
                if value is None:
                    return []
                if isinstance(value, list):
                    return value
                if isinstance(value, tuple):
                    return list(value)
                return list(value)
        return list(raw.values())

    if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes)):
        return list(raw)

    return [raw]


def _extract_embedding_vector(item: Any) -> Sequence[float]:
    """Extract the embedding vector from an item."""

    if isinstance(item, dict):
        vector = item.get("embedding") or item.get("vector") or item.get("values")
        if vector is not None:
            return _ensure_sequence(vector)
        return _ensure_sequence(item)

    if hasattr(item, "embedding"):
        return _ensure_sequence(getattr(item, "embedding"))

    return _ensure_sequence(item)


def _ensure_sequence(value: Any) -> Sequence[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return list(value.tolist())
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _extract_metric(metadata: Any, paths: Sequence[Sequence[str]]) -> int | float | None:
    """Extract numeric metrics such as read units or price per unit."""

    if metadata is None:
        return None

    for path in paths:
        value = metadata
        for key in path:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = getattr(value, key, None)

            if value is None:
                break
        else:
            if isinstance(value, (int, float)):
                return value
    return None
