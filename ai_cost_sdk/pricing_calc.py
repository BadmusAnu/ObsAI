"""Pricing helpers."""

from __future__ import annotations

from .pricing import load_pricing

PRICING_TABLE, PRICING_SNAPSHOT_ID = load_pricing()


def llm_cost(
    model: str, in_tokens: int = 0, out_tokens: int = 0, cached_tokens: int = 0
) -> float:
    prices = PRICING_TABLE.get("openai", {}).get(model, {})
    price_in = prices.get("in", 0.0)
    price_out = prices.get("out", 0.0)
    cost = (in_tokens / 1000) * price_in + (out_tokens / 1000) * price_out
    if cached_tokens:
        cost -= (cached_tokens / 1000) * price_in
    return round(cost, 10)


def embedding_cost(model: str, tokens: int = 0) -> float:
    price = PRICING_TABLE.get("embeddings", {}).get(model, 0.0)
    cost = (tokens / 1000) * price
    return round(cost, 10)


def tool_cost(unit_price: float | None) -> float:
    return float(unit_price or 0.0)


def rag_search_cost(read_units: int, price_per_unit: float) -> float:
    return round(read_units * price_per_unit, 10)
