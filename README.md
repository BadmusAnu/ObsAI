# AI Cost SDK - Developer Guide

A comprehensive Python SDK for tracking and monitoring AI/ML operation costs with multi-vendor support, advanced observability, and detailed metrics.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Metrics & Observability](#metrics--observability)
- [Advanced Features](#advanced-features)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Installation

```bash
pip install ai-cost-sdk
```

### Optional Dependencies

For enhanced tokenization accuracy with OpenAI models:

```bash
pip install tiktoken
```

## Quick Start

### 1. Basic Setup

```python
from ai_cost_sdk import config, observability, metrics
from ai_cost_sdk.middleware import agent_turn
from ai_cost_sdk.integrations.openai_wrapper import chat_completion
from openai import OpenAI

# Load configuration
config = config.load_config()

# Setup observability
tracer = observability.setup_tracer(config)

# Start metrics server
metrics.start_server(port=9108)

# Use the SDK
client = OpenAI()
with agent_turn(tenant_id="my-tenant", route="production"):
    response = chat_completion(
        client,
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

### 2. Environment Configuration

```bash
export SDK_ENABLED=true
export TENANT_ID="your-tenant-id"
export PROJECT_ID="your-project-id"
export ROUTE="production"
export EXPORT_OTLP_ENDPOINT="https://your-observability-platform.com"
# OR for local development:
export EXPORT_JSON_PATH="/path/to/telemetry.json"
```

## Core Features

### 1. Multi-Vendor LLM Support

The SDK supports multiple LLM vendors with accurate tokenization and cost tracking:

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo, embeddings
- **Claude**: Claude-3.5-Sonnet, Claude-3.5-Haiku, Claude-3 models
- **Gemini**: Gemini-Pro, Gemini-1.5-Pro, Gemini-1.5-Flash

All vendors include accurate pricing data for proper cost calculation across different model tiers.

### 2. Accurate Token Counting

Uses proper tokenizers for each vendor with intelligent fallback:

```python
from ai_cost_sdk.tokenizer import count_tokens, count_tokens_batch

# Single text
tokens = count_tokens("Hello world", "gpt-4o", "openai")

# Batch processing
texts = ["Text 1", "Text 2", "Text 3"]
total_tokens = count_tokens_batch(texts, "gpt-4o", "openai")
```

**Fallback Strategy**: When tiktoken is unavailable, the SDK uses a hybrid approach:
- Very short texts (1-3 words): 1 token per word
- Short texts (4-10 words): ~1.2 tokens per word  
- Long texts (10+ words): Character-based estimation
- Ensures short prompts are never counted as 0 tokens

### 3. Cost Tracking

Automatic cost calculation for all AI operations:

- LLM calls (input/output/cached tokens) with multi-vendor pricing
- Embeddings
- RAG operations
- Custom tools

The SDK includes comprehensive pricing data for OpenAI, Claude, and Gemini models, ensuring accurate cost tracking across all supported vendors. Pricing data is loaded dynamically based on the `PRICING_SNAPSHOT` configuration, allowing you to use different pricing versions as needed.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SDK_ENABLED` | `true` | Enable/disable the SDK |
| `TENANT_ID` | `""` | Tenant identifier (required when SDK enabled) |
| `PROJECT_ID` | `""` | Project identifier (required when SDK enabled) |
| `ROUTE` | `"default"` | Route name for grouping |
| `SERVICE_NAME` | `"ai-cost-sdk"` | Service name for OpenTelemetry resource |
| `EXPORT_OTLP_ENDPOINT` | `None` | OTLP endpoint for traces |
| `EXPORT_JSON_PATH` | `None` | Local JSON file for traces |
| `PRICING_SNAPSHOT` | `"openai-2025-09"` | Pricing data version (dynamically loaded) |
| `REDACT_PROMPTS` | `true` | Redact sensitive prompt data |
| `TOKENIZE_FALLBACK` | `false` | Use fallback tokenization |

### Programmatic Configuration

```python
from ai_cost_sdk.config import Config

config = Config(
    sdk_enabled=True,
    tenant_id="my-tenant",
    project_id="my-project",
    route="production",
    service_name="my-ai-service",
    export_otlp_endpoint="https://api.honeycomb.io/v1/traces",
    pricing_snapshot="openai-2025-09",
    redact_prompts=True,
    tokenize_fallback=False
)
```

## Usage Examples

### 1. LLM Calls

#### OpenAI Integration

```python
from ai_cost_sdk.integrations.openai_wrapper import chat_completion
from openai import OpenAI

client = OpenAI()

# Automatic cost tracking
response = chat_completion(
    client,
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
```

#### Manual LLM Tracking

```python
from ai_cost_sdk.middleware import llm_call

with llm_call(
    model="gpt-4o",
    vendor="openai",
    usage={"prompt_tokens": 10, "completion_tokens": 5},
    prompt="Your prompt here"
):
    # Your LLM call logic
    pass
```

### 2. RAG Operations

#### Embeddings

```python
from ai_cost_sdk.integrations.rag_wrapper import embed

# Automatic token counting and cost tracking
embeddings = embed(
    texts=["Document 1", "Document 2", "Document 3"],
    model="text-embedding-3-large",
    vendor="openai"
)
```

#### Vector Search

```python
from ai_cost_sdk.integrations.rag_wrapper import vector_search

results = vector_search(
    index_id="my-index",
    query="search query",
    k=5,
    index_version="v1.0",
    vendor="pinecone",
    read_units=10,
    price_per_unit=0.0001
)
```

### 3. Custom Tools

```python
from ai_cost_sdk.integrations.tools_wrapper import priced_tool

@priced_tool(name="weather_api", unit_price=0.001, vendor="weather-service")
def get_weather(location: str):
    # Your tool implementation
    return {"temperature": 72, "condition": "sunny"}

# Usage automatically tracks costs
weather = get_weather("New York")
```

### 4. Agent Workflows

```python
from ai_cost_sdk.middleware import agent_turn

with agent_turn(tenant_id="tenant-123", route="chat-api"):
    # Multiple AI operations within a single agent turn
    response = chat_completion(client, model="gpt-4o", messages=messages)
    embeddings = embed(documents, model="text-embedding-3-large")
    results = vector_search("index", query, k=5, index_version="v1")
    weather = get_weather("New York")
    
    # All costs are aggregated and tracked
```

## Metrics & Observability

### Prometheus Metrics

The SDK exposes comprehensive metrics at `http://localhost:9108/metrics`:

#### Request Metrics
- `llm_requests_total{model, vendor}` - Total LLM requests
- `agent_requests_total{tenant, route}` - Total agent requests
- `llm_tokens_total{type, model, vendor}` - Token usage

#### Cost Metrics
- `agent_cost_usd_total{tenant, route, model, component, vendor}` - Total costs
- `llm_cost_per_request_usd{model, vendor}` - Cost per LLM request
- `agent_cost_per_request_usd{tenant, route}` - Cost per agent request

#### Latency Metrics
- `agent_turn_latency_seconds{route}` - Agent response time
- `llm_latency_seconds{model, vendor}` - LLM call latency
- `rag_latency_seconds{operation, model, vendor}` - RAG operation latency
- `tool_latency_seconds{tool_name, vendor}` - Tool call latency

#### Throughput Metrics
- `llm_throughput_tokens_per_second{model, vendor}` - LLM throughput
- `agent_throughput_requests_per_second{tenant, route}` - Agent throughput

#### Error Metrics
- `llm_errors_total{model, vendor, error_type}` - LLM errors
- `agent_errors_total{tenant, route, error_type}` - Agent errors

#### Resource Metrics
- `active_requests{tenant, route}` - Currently active requests
- `request_queue_size{tenant, route}` - Request queue size

### OpenTelemetry Traces

All operations generate detailed traces with:

- **Span Attributes**: Model, vendor, tokens, costs, latency
- **Error Tracking**: Exception types and messages
- **Context Propagation**: Tenant, route, request correlation
- **Cost Attribution**: Per-component cost breakdown

### Export Options

1. **OTLP Endpoint**: Send to observability platforms (Honeycomb, Jaeger, etc.)
2. **JSON File**: Local development and debugging
3. **Console**: Development and testing

## Advanced Features

### 1. Custom Tokenizers

```python
from ai_cost_sdk.tokenizer import count_tokens

# Custom vendor support
tokens = count_tokens("text", "custom-model", "custom-vendor")
```

### 2. Pricing Customization

```python
from ai_cost_sdk.pricing_calc import llm_cost

# Custom pricing calculation
cost = llm_cost("gpt-4o", in_tokens=1000, out_tokens=500, cached_tokens=100)
```

### 3. Error Handling

```python
from ai_cost_sdk.middleware import llm_call

try:
    with llm_call(model="gpt-4o", vendor="openai"):
        # Your LLM call
        pass
except Exception as e:
    # Errors are automatically tracked in metrics
    print(f"Error occurred: {e}")
```

### 4. Multi-tenant Support

```python
# Different tenants can be tracked separately
with agent_turn(tenant_id="tenant-a", route="production"):
    # Operations for tenant A
    pass

with agent_turn(tenant_id="tenant-b", route="staging"):
    # Operations for tenant B
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_pricing.py

# Run with coverage
python -m pytest --cov=ai_cost_sdk tests/
```

### Test Examples

```python
def test_llm_cost_calculation():
    from ai_cost_sdk.pricing_calc import llm_cost
    
    cost = llm_cost("gpt-4o", in_tokens=1000, out_tokens=500)
    assert cost == 0.0075  # $0.0025 * 1 + $0.01 * 0.5

def test_tokenizer():
    from ai_cost_sdk.tokenizer import count_tokens
    
    tokens = count_tokens("Hello world", "gpt-4o", "openai")
    assert tokens > 0
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'tiktoken'**
   ```bash
   pip install tiktoken
   ```

2. **Metrics not appearing**
   - Check if metrics server is running: `curl http://localhost:9108/metrics`
   - Verify `SDK_ENABLED=true`

3. **Cost calculations seem wrong**
   - Check pricing snapshot version
   - Verify token counts are accurate
   - Ensure model is supported

4. **Traces not exporting**
   - Verify OTLP endpoint is accessible
   - Check JSON file path permissions
   - Enable debug logging

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# SDK will log detailed information
```

### Support

- Check the examples in `/examples` directory
- Review test cases in `/tests` directory
- Open an issue for bugs or feature requests

## Best Practices

1. **Always use context managers** for proper cleanup
2. **Set meaningful tenant/route names** for better organization
3. **Monitor costs regularly** using Prometheus metrics
4. **Use proper tokenization** for accurate cost tracking
5. **Handle errors gracefully** to maintain observability
6. **Test with different vendors** to ensure compatibility

## License

MIT License - see LICENSE file for details.