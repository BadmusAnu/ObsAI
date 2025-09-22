"""RAG helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Sequence

from .. import config, middleware
from ..tokenizer import count_tokens_batch, get_model_vendor


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


def embed(
    texts: Iterable[str],
    model: str | None = None,
    vendor: str | None = None,
    *,
    client: Any | None = None,
    api_key: str | None = None,
    batch_size: int | None = None,
    **client_kwargs: Any,
):
    """Embed texts with proper token counting."""

    texts_list = list(texts)
    if not texts_list:
        return []

    sdk_config = config.load_config()

    model = model or sdk_config.embedding_model
    if not model:
        raise ValueError("An embedding model must be provided via argument or configuration")

    resolved_vendor = vendor or get_model_vendor(model)

    api_key = api_key or sdk_config.embedding_api_key

    batch_size = batch_size or sdk_config.embedding_batch_size or len(texts_list)
    if batch_size <= 0:
        batch_size = len(texts_list)

    client = client or _ensure_client(resolved_vendor, api_key)

    tokens = count_tokens_batch(texts_list, model, resolved_vendor)

    embeddings: list[Sequence[float]] = []

    with middleware.rag_embed(
        model=model,
        vendor=resolved_vendor,
        tokens=tokens,
        texts=texts_list,
    ):
        for start in range(0, len(texts_list), batch_size):
            batch = texts_list[start : start + batch_size]
            response = client.embeddings.create(model=model, input=batch, **client_kwargs)
            batch_embeddings = _extract_embeddings(response)
            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    "Embedding response length (%d) did not match input batch (%d)"
                    % (len(batch_embeddings), len(batch))
                )
            embeddings.extend(batch_embeddings)

    return [list(vec) for vec in embeddings]


def vector_search(
    index_id: str,
    query: str,
    k: int,
    index_version: str,
    vendor: str = "vector_db",
    read_units: int = 0,
    price_per_unit: float = 0.0,
):
    """Perform vector search with cost tracking."""
    with middleware.rag_search(
        index_id, 
        index_version, 
        k, 
        freshness_s=0,
        read_units=read_units,
        price_per_unit=price_per_unit,
        vendor=vendor
    ):
        # Placeholder search - replace with actual vector search logic
        # In a real implementation, you would:
        # 1. Perform the actual vector search
        # 2. Calculate actual read_units based on search results
        # 3. Get actual price_per_unit from your vector DB pricing
        # For now, we'll use the provided values
        return [f"doc-{i}" for i in range(k)]
