from types import SimpleNamespace

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_cost_sdk.config import Config
from ai_cost_sdk.integrations.openai_wrapper import chat_completion
from ai_cost_sdk.integrations.rag_wrapper import vector_search
from ai_cost_sdk.middleware import agent_turn, configure


_EXPORTER: InMemorySpanExporter | None = None


def _setup_tracer() -> InMemorySpanExporter:
    global _EXPORTER
    if _EXPORTER is None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _EXPORTER = exporter
    else:
        _EXPORTER.clear()
    return _EXPORTER


def _config(**overrides) -> Config:
    data = {
        "sdk_enabled": True,
        "tenant_id": "tenant",
        "project_id": "project",
        "route": "route",
        "export_otlp_endpoint": None,
        "export_json_path": None,
        "pricing_snapshot": "openai-2025-09",
        "redact_prompts": True,
        "tokenize_fallback": False,
        "service_name": "svc",
    }
    data.update(overrides)
    return Config(**data)


class _ChatCompletions:
    def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
            usage={"prompt_tokens": 5, "completion_tokens": 7},
        )


class _Chat:
    completions = _ChatCompletions()


class Client:
    chat = _Chat()


def test_openai_wrapper_sets_attrs():
    exporter = _setup_tracer()
    configure(_config())
    try:
        client = Client()
        with agent_turn("t1", "r1"):
            chat_completion(
                client, model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
            )
    finally:
        configure(None)

    spans = list(exporter.get_finished_spans())
    llm = [s for s in spans if s.name == "llm.call"][0]
    assert llm.attributes["ai.tokens.input"] == 5
    assert llm.attributes["ai.tokens.output"] == 7
    assert llm.attributes["cost.usd.total"] > 0


def test_openai_fallback_tokenization():
    class _NoUsageCompletions:
        def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
                usage=None,
            )

    class _NoUsageChat:
        completions = _NoUsageCompletions()

    class _NoUsageClient:
        chat = _NoUsageChat()

    exporter = _setup_tracer()
    cfg = _config(tokenize_fallback=True)
    configure(cfg)
    try:
        client = _NoUsageClient()
        with agent_turn("t1", "r1"):
            chat_completion(
                client,
                config=cfg,
                model="gpt-4o",
                messages=[{"role": "user", "content": "hello"}],
            )
    finally:
        configure(None)

    spans = list(exporter.get_finished_spans())
    llm = [s for s in spans if s.name == "llm.call"][0]
    assert llm.attributes["ai.tokens.input"] > 0
    assert llm.attributes["ai.tokens.output"] > 0


def test_rag_search_cost_propagated():
    exporter = _setup_tracer()
    configure(_config())
    try:
        with agent_turn("tenant", "route"):
            vector_search(
                "idx",
                "query",
                k=1,
                index_version="v1",
                read_units=2,
                price_per_unit=0.5,
            )
    finally:
        configure(None)

    spans = list(exporter.get_finished_spans())
    rag = [s for s in spans if s.name == "rag.search"][0]
    assert rag.attributes["cost.usd.total"] == 1.0


def test_sdk_disabled_no_spans():
    exporter = _setup_tracer()
    configure(_config(sdk_enabled=False))
    try:
        with agent_turn("tenant", "route"):
            pass
    finally:
        configure(None)

    assert list(exporter.get_finished_spans()) == []
