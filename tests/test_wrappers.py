from types import SimpleNamespace

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_cost_sdk.integrations.openai_wrapper import chat_completion
from ai_cost_sdk.integrations.rag_wrapper import embed, vector_search
from ai_cost_sdk.integrations.tools_wrapper import priced_tool
from ai_cost_sdk.middleware import agent_turn
from ai_cost_sdk.tokenizer import count_tokens, count_tokens_batch, get_model_vendor


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


def test_rag_wrapper_embed():
    """Test RAG embedding wrapper with proper token counting."""
    texts = ["Hello world", "Test document"]
    embeddings = embed(texts, model="text-embedding-3-large", vendor="openai")
    
    assert len(embeddings) == 2
    assert all(len(emb) == 3 for emb in embeddings)


def test_rag_wrapper_search():
    """Test RAG search wrapper."""
    results = vector_search(
        index_id="test-index",
        query="test query",
        k=3,
        index_version="v1.0",
        vendor="pinecone",
        read_units=5,
        price_per_unit=0.001
    )
    
    assert len(results) == 3
    assert all(result.startswith("doc-") for result in results)


def test_tool_wrapper():
    """Test tool wrapper with cost tracking."""
    @priced_tool("test_tool", unit_price=0.01, vendor="test-vendor")
    def test_function():
        return "test result"
    
    result = test_function()
    assert result == "test result"


def test_tokenizer():
    """Test tokenizer functionality."""
    # Test single text
    tokens = count_tokens("Hello world", "gpt-4o", "openai")
    assert tokens > 0
    
    # Test batch processing
    texts = ["Hello", "World", "Test"]
    total_tokens = count_tokens_batch(texts, "gpt-4o", "openai")
    assert total_tokens > 0
    
    # Test vendor detection
    vendor = get_model_vendor("gpt-4o")
    assert vendor == "openai"
    
    vendor = get_model_vendor("claude-3-5-sonnet")
    assert vendor == "claude"
