import os

os.environ.setdefault("SDK_ENABLED", "1")
os.environ.setdefault("TENANT_ID", "test-tenant")
os.environ.setdefault("PROJECT_ID", "test-project")

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


TEST_EXPORTER = InMemorySpanExporter()
TEST_PROVIDER = TracerProvider()
TEST_PROVIDER.add_span_processor(SimpleSpanProcessor(TEST_EXPORTER))
trace.set_tracer_provider(TEST_PROVIDER)


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


class _FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts_list: list[str], model: str, vendor: str, **kwargs):
        """Return deterministic embeddings for testing."""
        self.calls.append({
            "texts": texts_list,
            "model": model,
            "vendor": vendor,
            "kwargs": kwargs,
        })
        return [[float(len(text)), float(index)] for index, text in enumerate(texts_list)]


class _FakeVectorClient:
    def __init__(self, read_units: int, price_per_unit: float):
        self.read_units = read_units
        self.price_per_unit = price_per_unit
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        matches = [
            {
                "id": f"doc-{i}",
                "score": 1.0 - (i * 0.1),
            }
            for i in range(kwargs["k"])
        ]
        return {
            "results": matches,
            "read_units": self.read_units,
            "price_per_unit": self.price_per_unit,
        }


def test_openai_wrapper_sets_attrs():
    TEST_EXPORTER.clear()

    client = Client()
    with agent_turn("t1", "r1"):
        chat_completion(
            client, model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )

    spans = TEST_EXPORTER.get_finished_spans()
    llm = [s for s in spans if s.name == "llm.call"][0]
    assert llm.attributes["ai.tokens.input"] == 5
    assert llm.attributes["ai.tokens.output"] == 7
    assert llm.attributes["cost.usd.total"] > 0


def test_rag_wrapper_embed():
    """Test RAG embedding wrapper with proper token counting."""
    TEST_EXPORTER.clear()

    texts = ["Hello world", "Test document"]
    fake_embedder = _FakeEmbedder()

    embeddings = embed(
        texts,
        model="text-embedding-3-large",
        vendor="openai",
        embedder=fake_embedder,
    )

    assert embeddings == [[11.0, 0.0], [13.0, 1.0]]
    assert fake_embedder.calls[0]["texts"] == texts

    spans = TEST_EXPORTER.get_finished_spans()
    rag_embed_spans = [s for s in spans if s.name == "rag.embed"]
    assert rag_embed_spans
    assert rag_embed_spans[0].attributes["rag.embedding.count"] == 2


def test_rag_wrapper_search():
    """Test RAG search wrapper."""
    TEST_EXPORTER.clear()

    fake_client = _FakeVectorClient(read_units=9, price_per_unit=0.002)

    results = vector_search(
        index_id="test-index",
        query="test query",
        k=3,
        index_version="v1.0",
        vendor="pinecone",
        searcher=fake_client,
    )

    assert [r["id"] for r in results] == ["doc-0", "doc-1", "doc-2"]
    assert fake_client.calls[0]["index_id"] == "test-index"

    spans = TEST_EXPORTER.get_finished_spans()
    rag_search_spans = [s for s in spans if s.name == "rag.search"]
    assert rag_search_spans
    assert rag_search_spans[0].attributes["rag.read_units"] == 9
    assert rag_search_spans[0].attributes["cost.usd.total"] == 0.018


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
