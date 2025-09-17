"""Enhanced sample app demonstrating the SDK features."""

from __future__ import annotations

import time
from types import SimpleNamespace

from ai_cost_sdk.config import load_config
from ai_cost_sdk.integrations.openai_wrapper import chat_completion
from ai_cost_sdk.integrations.rag_wrapper import embed, vector_search
from ai_cost_sdk.integrations.tools_wrapper import priced_tool
from ai_cost_sdk.middleware import agent_turn, llm_call, rag_embed, tool_call
from ai_cost_sdk.metrics import start_server
from ai_cost_sdk.observability import setup_tracer
from ai_cost_sdk.tokenizer import count_tokens, get_model_vendor


class _FakeChatCompletions:
    def create(self, **_kwargs):
        time.sleep(0.01)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello! I'm an AI assistant."))],
            usage={"prompt_tokens": 15, "completion_tokens": 8},
        )


class _FakeChat:
    completions = _FakeChatCompletions()


class FakeOpenAIClient:
    chat = _FakeChat()


class FakeClaudeClient:
    def generate(self, **kwargs):
        time.sleep(0.015)
        return SimpleNamespace(
            content="Hello! I'm Claude, an AI assistant.",
            usage={"input_tokens": 12, "output_tokens": 10}
        )


@priced_tool("fake_crm", unit_price=0.002, vendor="crm-service")
def call_crm():
    time.sleep(0.005)
    return {"status": "ok", "customer_id": "12345"}


@priced_tool("weather_api", unit_price=0.001, vendor="weather-service")
def get_weather(location: str):
    time.sleep(0.003)
    return {"location": location, "temperature": 72, "condition": "sunny"}


@priced_tool("database_query", unit_price=0.0005, vendor="database-service")
def query_database(query: str):
    time.sleep(0.008)
    return {"results": [{"id": 1, "name": "John"}, {"id": 2, "name": "Jane"}]}


def demonstrate_llm_calls():
    """Demonstrate different LLM calls with cost tracking."""
    print("\n=== LLM Calls Demo ===")
    
    # OpenAI call
    client = FakeOpenAIClient()
    with llm_call(model="gpt-4o", vendor="openai", prompt="What is the capital of France?"):
        response = chat_completion(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "What is the capital of France?"}]
        )
        print(f"OpenAI Response: {response.choices[0].message.content}")
    
    # Manual LLM call with different vendor
    with llm_call(model="claude-3-5-sonnet", vendor="claude", prompt="Explain quantum computing"):
        claude_client = FakeClaudeClient()
        response = claude_client.generate(prompt="Explain quantum computing")
        print(f"Claude Response: {response.content}")


def demonstrate_rag_operations():
    """Demonstrate RAG operations with proper token counting."""
    print("\n=== RAG Operations Demo ===")
    
    # Embedding multiple documents
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science."
    ]
    
    with rag_embed(model="text-embedding-3-large", vendor="openai", texts=documents):
        embeddings = embed(documents, model="text-embedding-3-large", vendor="openai")
        print(f"Generated {len(embeddings)} embeddings")
    
    # Vector search
    with rag_embed(model="text-embedding-3-large", vendor="openai", texts=["search query"]):
        results = vector_search(
            index_id="knowledge-base",
            query="What is machine learning?",
            k=3,
            index_version="v2.0",
            vendor="pinecone",
            read_units=5,
            price_per_unit=0.0001
        )
        print(f"Search results: {results}")


def demonstrate_tools():
    """Demonstrate custom tool usage with cost tracking."""
    print("\n=== Tools Demo ===")
    
    # CRM tool
    with tool_call("crm_lookup", vendor="crm-service", unit_price=0.002):
        crm_result = call_crm()
        print(f"CRM Result: {crm_result}")
    
    # Weather tool
    with tool_call("weather_check", vendor="weather-service", unit_price=0.001):
        weather_result = get_weather("New York")
        print(f"Weather Result: {weather_result}")
    
    # Database tool
    with tool_call("db_query", vendor="database-service", unit_price=0.0005):
        db_result = query_database("SELECT * FROM users LIMIT 2")
        print(f"Database Result: {db_result}")


def demonstrate_tokenizer():
    """Demonstrate tokenizer functionality."""
    print("\n=== Tokenizer Demo ===")
    
    text = "Hello world! This is a test of the tokenizer functionality."
    
    # Test different models and vendors
    models_vendors = [
        ("gpt-4o", "openai"),
        ("claude-3-5-sonnet", "claude"),
        ("gemini-pro", "gemini")
    ]
    
    for model, vendor in models_vendors:
        tokens = count_tokens(text, model, vendor)
        detected_vendor = get_model_vendor(model)
        print(f"{model} ({vendor}): {tokens} tokens (detected vendor: {detected_vendor})")


def demonstrate_agent_workflow():
    """Demonstrate a complete agent workflow with cost tracking."""
    print("\n=== Agent Workflow Demo ===")
    
    with agent_turn(tenant_id="demo-tenant", route="customer-support"):
        print("Starting customer support agent workflow...")
        
        # 1. Process customer query
        client = FakeOpenAIClient()
        response = chat_completion(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "I need help with my order"}]
        )
        print(f"Agent Response: {response.choices[0].message.content}")
        
        # 2. Look up customer information
        crm_result = call_crm()
        print(f"Customer Info: {crm_result}")
        
        # 3. Search knowledge base
        kb_results = vector_search(
            index_id="support-kb",
            query="order status tracking",
            k=2,
            index_version="v1.0",
            vendor="pinecone",
            read_units=3,
            price_per_unit=0.0001
        )
        print(f"Knowledge Base Results: {kb_results}")
        
        # 4. Query internal database
        db_result = query_database("SELECT order_status FROM orders WHERE customer_id = '12345'")
        print(f"Database Query Result: {db_result}")
        
        # 5. Generate final response
        final_response = chat_completion(
            client,
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Based on the information gathered, provide a helpful response"}
            ]
        )
        print(f"Final Response: {final_response.choices[0].message.content}")


def main() -> None:
    """Main function demonstrating all SDK features."""
    print("AI Cost SDK - Enhanced Demo Application")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    print(f"Configuration loaded: tenant={config.tenant_id}, route={config.route}")
    
    # Setup observability
    setup_tracer(config)
    print("OpenTelemetry tracer configured")
    
    # Start metrics server
    start_server(9108)
    print("Prometheus metrics server started on port 9108")
    print("View metrics at: http://localhost:9108/metrics")
    
    # Demonstrate various features
    demonstrate_tokenizer()
    demonstrate_llm_calls()
    demonstrate_rag_operations()
    demonstrate_tools()
    demonstrate_agent_workflow()
    
    print("\n" + "=" * 50)
    print("Demo completed! Check the metrics endpoint for detailed cost and performance data.")
    print("All operations have been tracked with:")
    print("- Accurate token counting using proper tokenizers")
    print("- Cost calculation per operation")
    print("- Latency and throughput metrics")
    print("- Error tracking and resource utilization")
    print("- OpenTelemetry traces for detailed observability")


if __name__ == "__main__":
    main()
