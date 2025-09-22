"""Pricing helpers."""

from __future__ import annotations

import os

from .pricing import load_pricing
from .config import load_config

# Cache for pricing data to avoid repeated loading
_pricing_cache: dict = {}
_current_snapshot: str = ""

def _resolve_pricing_snapshot() -> str:
    """Return the pricing snapshot ID without enforcing tenant/project config."""

    try:
        config = load_config()
    except ValueError:
        # When tenant/project values are missing we still want pricing helpers to
        # operate. Fall back to reading the snapshot directly from the
        # environment, mirroring the default used by ``load_config``.
        return os.getenv("PRICING_SNAPSHOT", "openai-2025-09")

    return config.pricing_snapshot


def _get_pricing_data():
    """Get pricing data, loading from config if needed."""
    global _pricing_cache, _current_snapshot

    snapshot_id = _resolve_pricing_snapshot()
    if _current_snapshot != snapshot_id or not _pricing_cache:
        _pricing_cache, _current_snapshot = load_pricing(snapshot_id)

    return _pricing_cache, _current_snapshot


def llm_cost(
    model: str, in_tokens: int = 0, out_tokens: int = 0, cached_tokens: int = 0, vendor: str = "openai"
) -> float:
    """Calculate LLM cost based on vendor and model."""
    pricing_table, _ = _get_pricing_data()
    prices = pricing_table.get(vendor, {}).get(model, {})
    
    # If no pricing found for this vendor/model, return 0 with a warning
    if not prices:
        # Log a warning for unknown vendor/model combinations
        import warnings
        warnings.warn(f"No pricing data found for {vendor}/{model}. Cost will be 0.", UserWarning)
        return 0.0
    
    price_in = prices.get("in", 0.0)
    price_out = prices.get("out", 0.0)
    cost = (in_tokens / 1000) * price_in + (out_tokens / 1000) * price_out
    if cached_tokens:
        cost -= (cached_tokens / 1000) * price_in
    return round(cost, 10)


def embedding_cost(model: str, tokens: int = 0) -> float:
    pricing_table, _ = _get_pricing_data()
    price = pricing_table.get("embeddings", {}).get(model, 0.0)
    cost = (tokens / 1000) * price
    return round(cost, 10)


def tool_cost(unit_price: float | None) -> float:
    return float(unit_price or 0.0)


def rag_search_cost(read_units: int, price_per_unit: float) -> float:
    return round(read_units * price_per_unit, 10)


def is_model_supported(model: str, vendor: str) -> bool:
    """Check if a model is supported in the pricing table."""
    pricing_table, _ = _get_pricing_data()
    return bool(pricing_table.get(vendor, {}).get(model))


def get_supported_models(vendor: str) -> list[str]:
    """Get list of supported models for a vendor."""
    pricing_table, _ = _get_pricing_data()
    return list(pricing_table.get(vendor, {}).keys())


def get_supported_vendors() -> list[str]:
    """Get list of supported vendors."""
    pricing_table, _ = _get_pricing_data()
    return [k for k in pricing_table.keys() if k != "embeddings"]


def get_pricing_snapshot_id() -> str:
    """Get the current pricing snapshot ID."""
    _, snapshot_id = _get_pricing_data()
    return snapshot_id
