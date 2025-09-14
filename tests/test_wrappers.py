from types import SimpleNamespace

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_cost_sdk.integrations.openai_wrapper import chat_completion
from ai_cost_sdk.middleware import agent_turn


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
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    client = Client()
    with agent_turn("t1", "r1"):
        chat_completion(
            client, model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )

    spans = exporter.get_finished_spans()
    llm = [s for s in spans if s.name == "llm.call"][0]
    assert llm.attributes["ai.tokens.input"] == 5
    assert llm.attributes["ai.tokens.output"] == 7
    assert llm.attributes["cost.usd.total"] > 0
