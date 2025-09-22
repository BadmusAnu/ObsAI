import os
from unittest.mock import patch

from ai_cost_sdk.pricing_calc import (
    embedding_cost,
    get_pricing_snapshot_id,
    llm_cost,
    rag_search_cost,
    tool_cost,
)


def test_llm_cost_and_snapshot():
    assert get_pricing_snapshot_id() == "openai-2025-09"
    cost = llm_cost("gpt-4o", in_tokens=1000, out_tokens=1000)
    assert cost == 0.0125


def test_llm_cost_without_tenant_configuration():
    # Ensure pricing cache is reset so the test observes the patched env
    import ai_cost_sdk.pricing_calc as pricing_calc

    with patch.dict(
        os.environ,
        {"TENANT_ID": "", "PROJECT_ID": "", "SDK_ENABLED": "1"},
        clear=False,
    ):
        pricing_calc._pricing_cache = {}
        pricing_calc._current_snapshot = ""

        cost = llm_cost("gpt-4o", in_tokens=1000, out_tokens=1000)
        assert cost == 0.0125


def test_other_costs():
    assert embedding_cost("text-embedding-3-large", tokens=1000) == 0.00002
    assert rag_search_cost(10, 0.0001) == 0.001
    assert tool_cost(0.5) == 0.5
