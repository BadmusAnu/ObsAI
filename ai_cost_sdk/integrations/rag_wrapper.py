"""RAG helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Callable

from .. import config, middleware
from ..tokenizer import count_tokens_batch, get_model_vendor

EmbeddingBackend = Callable[..., Any]
SearchBackend = Callable[..., Any]


def embed(
    texts: Iterable[str],
    model: str | None = None,
    vendor: str | None = None,
    *,
    embedder: EmbeddingBackend | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    batch_size: int | None = None,
    **backend_kwargs: Any,
) -> list[list[float]]:
    """Embed texts with proper token counting and configurable backends."""

    texts_list = list(texts)
    if not texts_list:
        return []

    sdk_config = config.load_config_permissive()

    resolved_model = model or sdk_config.embedding_model
    if not resolved_model:
        raise ValueError("An embedding model must be provided via argument or configuration")

    resolved_vendor = vendor or get_model_vendor(resolved_model)

    tokens = count_tokens_batch(texts_list, resolved_model, resolved_vendor)

    with middleware.rag_embed(
        model=resolved_model,
        vendor=resolved_vendor,
        tokens=tokens,
        texts=texts_list,
    ) as span:
        if embedder is not None:
            raw_embeddings = _call_embed_backend(
                embedder,
                texts_list=texts_list,
                model=resolved_model,
                vendor=resolved_vendor,
                **backend_kwargs,
            )
            embeddings = _normalize_embeddings(raw_embeddings)
        else:
            resolved_api_key = api_key or sdk_config.embedding_api_key
            resolved_client = client or _ensure_client(resolved_vendor, resolved_api_key)

            resolved_batch_size = batch_size or sdk_config.embedding_batch_size or len(texts_list)
            if resolved_batch_size <= 0:
                resolved_batch_size = len(texts_list)

            collected: list[Sequence[float]] = []
            for start in range(0, len(texts_list), resolved_batch_size):
                batch = texts_list[start : start + resolved_batch_size]
                response = resolved_client.embeddings.create(
                    model=resolved_model,
                    input=batch,
                    **backend_kwargs,
                )
                batch_embeddings = _extract_embeddings(response)
                if len(batch_embeddings) != len(batch):
                    raise ValueError(
                        "Embedding response length (%d) did not match input batch (%d)"
                        % (len(batch_embeddings), len(batch))
                    )
                collected.extend(batch_embeddings)

            embeddings = [list(vec) for vec in collected]

        _record_embedding_span(span, embeddings)
        return embeddings


def _ensure_client(vendor: str, api_key: str | None) -> Any:
    """Instantiate a default embeddings client for supported vendors."""

    if vendor == "openai":  # pragma: no cover - import guarded for optional dependency
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive programming
            raise RuntimeError(
                "openai package is required to instantiate the default embeddings client"
            ) from exc

        return OpenAI(api_key=api_key)

    raise ValueError(
        "An embeddings client must be supplied for vendor %s" % vendor
    )


def _extract_embeddings(response: Any) -> list[Sequence[float]]:
    """Extract embeddings from a provider response."""

    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    if data is None:
        raise ValueError("Embedding response did not contain 'data'")

    embeddings: list[Sequence[float]] = []
    for item in data:
        if hasattr(item, "embedding"):
            embeddings.append(getattr(item, "embedding"))
        elif isinstance(item, dict) and "embedding" in item:
            embeddings.append(item["embedding"])
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            embeddings.append(item)
        else:
            raise ValueError("Embedding item missing 'embedding' field")

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
    **search_kwargs: Any,
) -> tuple[list[Any], int | None, float | None]:
    """Invoke the configured search backend and normalize the response."""

    call_kwargs = {
        "index_id": index_id,
        "query": query,
        "k": k,
        "index_version": index_version,
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


def _record_embedding_span(span: Any, embeddings: list[list[float]]) -> None:
    """Attach embedding metadata to the active span when available."""

    if span is None:
        return

    span.set_attribute("rag.embedding.count", len(embeddings))
    if embeddings and embeddings[0]:
        span.set_attribute("rag.embedding.dim", len(embeddings[0]))
