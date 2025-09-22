import pytest

from ai_cost_sdk import pricing_calc


@pytest.fixture(autouse=True)
def clear_pricing_cache():
    pricing_calc._pricing_cache = {}
    pricing_calc._current_snapshot = ""
    yield
    pricing_calc._pricing_cache = {}
    pricing_calc._current_snapshot = ""


def _clear_relevant_env(monkeypatch):
    for var in ("TENANT_ID", "PROJECT_ID", "SDK_ENABLED", "PRICING_SNAPSHOT"):
        monkeypatch.delenv(var, raising=False)


def test_llm_cost_and_snapshot_without_tenant(monkeypatch):
    _clear_relevant_env(monkeypatch)

    cost = pricing_calc.llm_cost("gpt-4o", in_tokens=1000, out_tokens=1000)

    assert cost == 0.0125
    assert pricing_calc.get_pricing_snapshot_id() == "openai-2025-09"


def test_other_costs(monkeypatch):
    _clear_relevant_env(monkeypatch)

    assert pricing_calc.embedding_cost("text-embedding-3-large", tokens=1000) == 0.00002
    assert pricing_calc.rag_search_cost(10, 0.0001) == 0.001
    assert pricing_calc.tool_cost(0.5) == 0.5


def test_pricing_cache_switches_snapshots(monkeypatch):
    _clear_relevant_env(monkeypatch)

    loaded_snapshots: list[str] = []

    def fake_load_pricing(snapshot_id: str):
        loaded_snapshots.append(snapshot_id)
        price = {"first": 0.001, "second": 0.01}[snapshot_id]
        return {"openai": {"gpt-test": {"in": price, "out": price}}}, snapshot_id

    monkeypatch.setattr(pricing_calc, "load_pricing", fake_load_pricing)

    monkeypatch.setenv("PRICING_SNAPSHOT", "first")
    first_cost = pricing_calc.llm_cost("gpt-test", in_tokens=1000)

    monkeypatch.setenv("PRICING_SNAPSHOT", "second")
    second_cost = pricing_calc.llm_cost("gpt-test", in_tokens=1000)

    assert first_cost == 0.001
    assert second_cost == 0.01
    assert second_cost != first_cost
    assert pricing_calc.get_pricing_snapshot_id() == "second"
    assert loaded_snapshots == ["first", "second"]
