"""RAG helpers."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable, Mapping
from typing import Any

from .. import middleware
from ..tokenizer import count_tokens_batch


def embed(texts: Iterable[str], model: str, vendor: str = "openai"):
    """Embed texts with proper token counting."""
    texts_list = list(texts)
    tokens = count_tokens_batch(texts_list, model, vendor)
    
    with middleware.rag_embed(model=model, vendor=vendor, tokens=tokens, texts=texts_list):
        # Placeholder embedding - replace with actual embedding logic
        return [[0.0] * 3 for _ in texts_list]


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _extract_from_container(container: Any, key: str) -> Any:
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)


def _extract_usage(response: Any) -> tuple[int | None, float | None]:
    """Pull read unit usage information from a vector store response."""

    sources = [
        _extract_from_container(response, "usage"),
        _extract_from_container(response, "cost"),
        _extract_from_container(response, "metadata"),
        response,
    ]

    read_units: int | None = None
    price_per_unit: float | None = None

    for source in sources:
        if read_units is None:
            read_units = _coerce_int(_extract_from_container(source, "read_units"))
        if price_per_unit is None:
            # Support a few common field names for price information
            price_per_unit = _coerce_float(
                _extract_from_container(source, "price_per_unit")
                or _extract_from_container(source, "unit_price")
            )
        if read_units is not None and price_per_unit is not None:
            break

    return read_units, price_per_unit


def _extract_results(response: Any) -> Any:
    for key in ("matches", "results", "documents", "data"):
        value = _extract_from_container(response, key)
        if value is not None:
            return value
    return response


def _prepare_call_kwargs(
    method: Any,
    *,
    index_id: str,
    index_version: str,
    query: str,
    k: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select keyword arguments supported by the client's query method."""

    overrides = overrides or {}
    candidate_kwargs: dict[str, Any] = {
        "index_id": index_id,
        "index": index_id,
        "query": query,
        "text": query,
        "top_k": k,
        "k": k,
        "index_version": index_version,
    }
    candidate_kwargs.update(overrides)

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # pragma: no cover - some builtins
        return candidate_kwargs

    has_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if has_var_kwargs:
        return candidate_kwargs

    supported = set(signature.parameters.keys())
    return {k: v for k, v in candidate_kwargs.items() if k in supported}


def _default_price_per_unit() -> float:
    raw = os.getenv("VECTOR_SEARCH_PRICE_PER_UNIT")
    coerced = _coerce_float(raw)
    return coerced if coerced is not None else 0.0


def vector_search(
    client: Any,
    index_id: str,
    query: str,
    k: int,
    index_version: str,
    vendor: str = "vector_db",
    read_units: int | None = None,
    price_per_unit: float | None = None,
    freshness_s: int = 0,
    method: str = "query",
    **kwargs: Any,
):
    """Perform vector search with cost tracking using a vector store client."""

    usage_state = {
        "read_units": _coerce_int(read_units),
        "price_per_unit": _coerce_float(price_per_unit),
    }
    default_price = _default_price_per_unit()

    def _read_units_value() -> int:
        value = usage_state.get("read_units")
        return int(value) if value is not None else 0

    def _price_per_unit_value() -> float:
        value = usage_state.get("price_per_unit")
        return float(value) if value is not None else default_price

    query_method = getattr(client, method, None)
    if query_method is None:
        raise AttributeError(f"Client does not provide a '{method}' method for vector search")

    call_kwargs = _prepare_call_kwargs(
        query_method,
        index_id=index_id,
        index_version=index_version,
        query=query,
        k=k,
        overrides=kwargs,
    )

    with middleware.rag_search(
        index_id,
        index_version,
        k,
        freshness_s=freshness_s,
        read_units=_read_units_value,
        price_per_unit=_price_per_unit_value,
        vendor=vendor,
    ) as span:
        response = query_method(**call_kwargs)
        usage_read_units, usage_price = _extract_usage(response)

        if usage_read_units is not None:
            usage_state["read_units"] = usage_read_units
        if usage_state.get("read_units") is None:
            usage_state["read_units"] = 0

        if usage_price is not None:
            usage_state["price_per_unit"] = usage_price
        if usage_state.get("price_per_unit") is None:
            usage_state["price_per_unit"] = default_price

        if span is not None:
            span.set_attribute("rag.read_units", usage_state["read_units"])
            span.set_attribute("rag.price_per_unit", usage_state["price_per_unit"])

    return _extract_results(response)
