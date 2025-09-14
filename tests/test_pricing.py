from ai_cost_sdk.pricing_calc import (
    PRICING_SNAPSHOT_ID,
    embedding_cost,
    llm_cost,
    rag_search_cost,
    tool_cost,
)


def test_llm_cost_and_snapshot():
    assert PRICING_SNAPSHOT_ID == "openai-2025-09"
    cost = llm_cost("gpt-4o", in_tokens=1000, out_tokens=1000)
    assert cost == 0.0125


def test_other_costs():
    assert embedding_cost("text-embedding-3-large", tokens=1000) == 0.00002
    assert rag_search_cost(10, 0.0001) == 0.001
    assert tool_cost(0.5) == 0.5
