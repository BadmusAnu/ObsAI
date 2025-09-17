"""Test enhanced metrics functionality."""

import time
from unittest.mock import patch

from ai_cost_sdk.metrics import (
    LLM_REQUESTS, LLM_TOKENS, AGENT_REQUESTS, AGENT_COST,
    LLM_COST_PER_REQUEST, AGENT_COST_PER_REQUEST, LLM_LATENCY,
    RAG_LATENCY, TOOL_LATENCY, LLM_THROUGHPUT, AGENT_THROUGHPUT,
    LLM_ERRORS, AGENT_ERRORS, ACTIVE_REQUESTS, QUEUE_SIZE
)
from ai_cost_sdk.middleware import agent_turn, llm_call, rag_embed, tool_call


def test_llm_metrics():
    """Test LLM-specific metrics."""
    # Test request counting
    LLM_REQUESTS.labels(model="gpt-4o", vendor="openai").inc()
    assert LLM_REQUESTS.labels(model="gpt-4o", vendor="openai")._value._value == 1
    
    # Test token counting
    LLM_TOKENS.labels(type="input", model="gpt-4o", vendor="openai").inc(100)
    LLM_TOKENS.labels(type="output", model="gpt-4o", vendor="openai").inc(50)
    assert LLM_TOKENS.labels(type="input", model="gpt-4o", vendor="openai")._value._value == 100
    assert LLM_TOKENS.labels(type="output", model="gpt-4o", vendor="openai")._value._value == 50
    
    # Test cost per request
    LLM_COST_PER_REQUEST.labels(model="gpt-4o", vendor="openai").set(0.01)
    assert LLM_COST_PER_REQUEST.labels(model="gpt-4o", vendor="openai")._value._value == 0.01


def test_agent_metrics():
    """Test agent-level metrics."""
    # Test agent requests
    AGENT_REQUESTS.labels(tenant="test-tenant", route="test-route").inc()
    assert AGENT_REQUESTS.labels(tenant="test-tenant", route="test-route")._value._value == 1
    
    # Test agent cost
    AGENT_COST.labels(
        tenant="test-tenant", 
        route="test-route", 
        model="gpt-4o", 
        component="llm",
        vendor="openai"
    ).inc(0.05)
    assert AGENT_COST.labels(
        tenant="test-tenant", 
        route="test-route", 
        model="gpt-4o", 
        component="llm",
        vendor="openai"
    )._value._value == 0.05
    
    # Test cost per request
    AGENT_COST_PER_REQUEST.labels(tenant="test-tenant", route="test-route").set(0.02)
    assert AGENT_COST_PER_REQUEST.labels(tenant="test-tenant", route="test-route")._value._value == 0.02


def test_latency_metrics():
    """Test latency metrics."""
    # Test LLM latency
    LLM_LATENCY.labels(model="gpt-4o", vendor="openai").observe(1.5)
    assert LLM_LATENCY.labels(model="gpt-4o", vendor="openai")._sum._value == 1.5
    assert LLM_LATENCY.labels(model="gpt-4o", vendor="openai")._count._value == 1
    
    # Test RAG latency
    RAG_LATENCY.labels(operation="embed", model="text-embedding-3-large", vendor="openai").observe(0.5)
    assert RAG_LATENCY.labels(operation="embed", model="text-embedding-3-large", vendor="openai")._sum._value == 0.5
    
    # Test tool latency
    TOOL_LATENCY.labels(tool_name="test-tool", vendor="test-vendor").observe(0.1)
    assert TOOL_LATENCY.labels(tool_name="test-tool", vendor="test-vendor")._sum._value == 0.1


def test_throughput_metrics():
    """Test throughput metrics."""
    # Test LLM throughput
    LLM_THROUGHPUT.labels(model="gpt-4o", vendor="openai").observe(100.0)
    assert LLM_THROUGHPUT.labels(model="gpt-4o", vendor="openai")._sum._value == 100.0
    
    # Test agent throughput
    AGENT_THROUGHPUT.labels(tenant="test-tenant", route="test-route").observe(10.0)
    assert AGENT_THROUGHPUT.labels(tenant="test-tenant", route="test-route")._sum._value == 10.0


def test_error_metrics():
    """Test error tracking metrics."""
    # Test LLM errors
    LLM_ERRORS.labels(model="gpt-4o", vendor="openai", error_type="RateLimitError").inc()
    assert LLM_ERRORS.labels(model="gpt-4o", vendor="openai", error_type="RateLimitError")._value._value == 1
    
    # Test agent errors
    AGENT_ERRORS.labels(tenant="test-tenant", route="test-route", error_type="TimeoutError").inc()
    assert AGENT_ERRORS.labels(tenant="test-tenant", route="test-route", error_type="TimeoutError")._value._value == 1


def test_resource_metrics():
    """Test resource utilization metrics."""
    # Test active requests
    ACTIVE_REQUESTS.labels(tenant="test-tenant", route="test-route").inc()
    assert ACTIVE_REQUESTS.labels(tenant="test-tenant", route="test-route")._value._value == 1
    
    ACTIVE_REQUESTS.labels(tenant="test-tenant", route="test-route").dec()
    assert ACTIVE_REQUESTS.labels(tenant="test-tenant", route="test-route")._value._value == 0
    
    # Test queue size
    QUEUE_SIZE.labels(tenant="test-tenant", route="test-route").set(5)
    assert QUEUE_SIZE.labels(tenant="test-tenant", route="test-route")._value._value == 5


def test_metrics_integration():
    """Test metrics integration with middleware."""
    # This would require a more complex setup with actual tracing
    # For now, we'll test that the metrics can be created and accessed
    assert LLM_REQUESTS is not None
    assert AGENT_COST is not None
    assert LLM_LATENCY is not None
    assert RAG_LATENCY is not None
    assert TOOL_LATENCY is not None
    assert LLM_THROUGHPUT is not None
    assert AGENT_THROUGHPUT is not None
    assert LLM_ERRORS is not None
    assert AGENT_ERRORS is not None
    assert ACTIVE_REQUESTS is not None
    assert QUEUE_SIZE is not None
