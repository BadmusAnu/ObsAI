"""Span context managers and decorators."""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import os
import time

from opentelemetry import trace

from .config import load_config, load_config_permissive, Config
from .metrics import (
    AGENT_COST, AGENT_LATENCY, LLM_REQUESTS, LLM_TOKENS, AGENT_REQUESTS,
    LLM_COST_PER_REQUEST, AGENT_COST_PER_REQUEST, LLM_LATENCY, RAG_LATENCY, 
    TOOL_LATENCY, LLM_THROUGHPUT, AGENT_THROUGHPUT, LLM_ERRORS, AGENT_ERRORS,
    ACTIVE_REQUESTS, QUEUE_SIZE
)
from .pricing_calc import (
    embedding_cost,
    llm_cost,
    rag_search_cost,
    tool_cost,
    get_pricing_snapshot_id,
)
from .tokenizer import count_tokens, get_model_vendor

_turn_cost: contextvars.ContextVar[float] = contextvars.ContextVar(
    "turn_cost", default=0.0
)
_tenant: contextvars.ContextVar[str] = contextvars.ContextVar("tenant", default="")
_route: contextvars.ContextVar[str] = contextvars.ContextVar("route", default="")
_request_count: contextvars.ContextVar[int] = contextvars.ContextVar("request_count", default=0)

# Cache config to avoid repeated loading
_config_cache: Config | None = None


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean environment variables with sensible defaults."""

    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _load_config_with_fallback() -> Config:
    """Load configuration but tolerate missing tenant/project IDs."""

    try:
        return load_config()
    except ValueError:
        return load_config_permissive()


def _get_config() -> Config:
    """Get cached config or load it."""
    global _config_cache
    if _config_cache is None:
        _config_cache = _load_config_with_fallback()
    return _config_cache


def _is_sdk_enabled() -> bool:
    """Check if SDK is enabled."""
    return _get_config().sdk_enabled


def _add_cost(cost: float, model: str | None, component: str, vendor: str = "unknown") -> None:
    total = _turn_cost.get()
    _turn_cost.set(total + cost)
    AGENT_COST.labels(
        tenant=_tenant.get(),
        route=_route.get(),
        model=model or "",
        component=component,
        vendor=vendor,
    ).inc(cost)


@contextmanager
def agent_turn(tenant_id: str, route: str):
    # Check if SDK is enabled - if not, provide no-op context manager
    if not _is_sdk_enabled():
        yield None
        return
    
    tracer = trace.get_tracer("ai_cost_sdk")
    start = time.time()
    
    # Increment active requests
    ACTIVE_REQUESTS.labels(tenant=tenant_id, route=route).inc()
    
    with tracer.start_as_current_span("gateway.request") as span:
        span.set_attribute("tenant.id", tenant_id)
        span.set_attribute("route.name", route)
        span.set_attribute("pricing_snapshot_id", get_pricing_snapshot_id())
        token_c = _turn_cost.set(0.0)
        token_t = _tenant.set(tenant_id)
        token_r = _route.set(route)
        token_rc = _request_count.set(0)
        
        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            AGENT_ERRORS.labels(tenant=tenant_id, route=route, error_type=exc.__class__.__name__).inc()
            raise
        finally:
            end_time = time.time()
            latency = end_time - start
            total_cost = _turn_cost.get()
            request_count = _request_count.get()
            
            # Update span attributes
            span.set_attribute("cost.usd.total", round(total_cost, 10))
            span.set_attribute("latency_ms", latency * 1000)
            span.set_attribute("request_count", request_count)
            
            # Update metrics
            AGENT_LATENCY.labels(route=route).observe(latency)
            AGENT_REQUESTS.labels(tenant=tenant_id, route=route).inc()
            
            # Calculate cost per request
            if request_count > 0:
                cost_per_request = total_cost / request_count
                AGENT_COST_PER_REQUEST.labels(tenant=tenant_id, route=route).set(cost_per_request)
            
            # Calculate throughput
            if latency > 0:
                throughput = request_count / latency
                AGENT_THROUGHPUT.labels(tenant=tenant_id, route=route).observe(throughput)
            
            # Decrement active requests
            ACTIVE_REQUESTS.labels(tenant=tenant_id, route=route).dec()
            
            # Reset context variables
            _turn_cost.reset(token_c)
            _tenant.reset(token_t)
            _route.reset(token_r)
            _request_count.reset(token_rc)


@contextmanager
def llm_call(model: str, vendor: str, usage: dict | None = None, prompt: str | None = None):
    # Check if SDK is enabled - if not, provide no-op context manager
    if not _is_sdk_enabled():
        yield None
        return
    
    tracer = trace.get_tracer("ai_cost_sdk")
    usage = usage if usage is not None else {}
    start = time.time()
    
    # Increment request count
    current_count = _request_count.get()
    _request_count.set(current_count + 1)
    
    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("ai.model.id", model)
        span.set_attribute("ai.vendor", vendor)
        span.set_attribute("pricing_snapshot_id", get_pricing_snapshot_id())
        
        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            LLM_ERRORS.labels(model=model, vendor=vendor, error_type=exc.__class__.__name__).inc()
            raise
        finally:
            end_time = time.time()
            latency = end_time - start
            latency_ms = latency * 1000
            
            # Get token counts from usage or calculate from prompt
            in_t = int(usage.get("prompt_tokens", 0))
            out_t = int(usage.get("completion_tokens", 0))
            cached_t = int(usage.get("cached_tokens", 0))
            
            # If no usage provided but prompt is available, calculate tokens
            if not in_t and prompt:
                in_t = count_tokens(prompt, model, vendor)
            
            # Update span attributes
            span.set_attribute("latency_ms", latency_ms)
            if in_t:
                span.set_attribute("ai.tokens.input", in_t)
            if out_t:
                span.set_attribute("ai.tokens.output", out_t)
            if cached_t:
                span.set_attribute("ai.tokens.cached", cached_t)
            
            # Calculate cost and update metrics
            cost = llm_cost(model, in_t, out_t, cached_t, vendor)
            span.set_attribute("cost.usd.total", cost)
            _add_cost(cost, model, "llm", vendor)
            
            # Update Prometheus metrics
            LLM_REQUESTS.labels(model=model, vendor=vendor).inc()
            LLM_LATENCY.labels(model=model, vendor=vendor).observe(latency)
            
            if in_t:
                LLM_TOKENS.labels(type="input", model=model, vendor=vendor).inc(in_t)
            if out_t:
                LLM_TOKENS.labels(type="output", model=model, vendor=vendor).inc(out_t)
            if cached_t:
                LLM_TOKENS.labels(type="cached", model=model, vendor=vendor).inc(cached_t)
            
            # Calculate cost per request
            LLM_COST_PER_REQUEST.labels(model=model, vendor=vendor).set(cost)
            
            # Calculate throughput (tokens per second)
            total_tokens = in_t + out_t
            if latency > 0 and total_tokens > 0:
                throughput = total_tokens / latency
                LLM_THROUGHPUT.labels(model=model, vendor=vendor).observe(throughput)


@contextmanager
def rag_embed(model: str, vendor: str, tokens: int, texts: list[str] | None = None):
    # Check if SDK is enabled - if not, provide no-op context manager
    if not _is_sdk_enabled():
        yield None
        return
    
    tracer = trace.get_tracer("ai_cost_sdk")
    start = time.time()
    
    # Calculate tokens from texts if not provided
    if not tokens and texts:
        tokens = sum(count_tokens(text, model, vendor) for text in texts)
    
    with tracer.start_as_current_span("rag.embed") as span:
        span.set_attribute("ai.model.id", model)
        span.set_attribute("ai.vendor", vendor)
        span.set_attribute("ai.tokens.input", tokens)
        span.set_attribute("pricing_snapshot_id", get_pricing_snapshot_id())
        
        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            raise
        finally:
            end_time = time.time()
            latency = end_time - start
            latency_ms = latency * 1000
            
            span.set_attribute("latency_ms", latency_ms)
            cost = embedding_cost(model, tokens)
            span.set_attribute("cost.usd.total", cost)
            _add_cost(cost, model, "rag_embed", vendor)
            
            # Update RAG latency metrics
            RAG_LATENCY.labels(operation="embed", model=model, vendor=vendor).observe(latency)


@contextmanager
def rag_search(
    index_id: str,
    index_version: str,
    k: int,
    freshness_s: int,
    read_units: int = 0,
    price_per_unit: float = 0.0,
    vendor: str = "vector_db",
):
    # Check if SDK is enabled - if not, provide no-op context manager
    if not _is_sdk_enabled():
        yield None
        return
    
    tracer = trace.get_tracer("ai_cost_sdk")
    start = time.time()
    
    with tracer.start_as_current_span("rag.search") as span:
        span.set_attribute("rag.index.id", index_id)
        span.set_attribute("rag.index.version", index_version)
        span.set_attribute("rag.k", k)
        span.set_attribute("rag.freshness_s", freshness_s)
        span.set_attribute("pricing_snapshot_id", get_pricing_snapshot_id())
        span.set_attribute("ai.vendor", vendor)
        span.set_attribute("rag.read_units", read_units)
        span.set_attribute("rag.price_per_unit", price_per_unit)
        setattr(span, "_rag_read_units", read_units)
        setattr(span, "_rag_price_per_unit", price_per_unit)

        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            raise
        finally:
            end_time = time.time()
            latency = end_time - start
            latency_ms = latency * 1000

            span.set_attribute("latency_ms", latency_ms)
            final_read_units = getattr(span, "_rag_read_units", read_units)
            final_price_per_unit = getattr(span, "_rag_price_per_unit", price_per_unit)

            if final_read_units is None:
                final_read_units = read_units
            if final_price_per_unit is None:
                final_price_per_unit = price_per_unit

            span.set_attribute("rag.read_units", final_read_units)
            span.set_attribute("rag.price_per_unit", final_price_per_unit)

            cost = rag_search_cost(float(final_read_units or 0), float(final_price_per_unit or 0))
            span.set_attribute("cost.usd.total", cost)
            _add_cost(cost, index_id, "rag_search", vendor)

            # Update RAG latency metrics
            RAG_LATENCY.labels(operation="search", model=index_id, vendor=vendor).observe(latency)


@contextmanager
def tool_call(name: str, vendor: str | None = None, unit_price: float | None = None):
    # Check if SDK is enabled - if not, provide no-op context manager
    if not _is_sdk_enabled():
        yield None
        return
    
    tracer = trace.get_tracer("ai_cost_sdk")
    start = time.time()
    vendor = vendor or "unknown"
    
    with tracer.start_as_current_span("tool.call") as span:
        span.set_attribute("tool.name", name)
        span.set_attribute("ai.vendor", vendor)
        span.set_attribute("pricing_snapshot_id", get_pricing_snapshot_id())
        
        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            raise
        finally:
            end_time = time.time()
            latency = end_time - start
            latency_ms = latency * 1000
            
            span.set_attribute("latency_ms", latency_ms)
            cost = tool_cost(unit_price)
            span.set_attribute("cost.usd.total", cost)
            _add_cost(cost, name, "tool", vendor)
            
            # Update tool latency metrics
            TOOL_LATENCY.labels(tool_name=name, vendor=vendor).observe(latency)
