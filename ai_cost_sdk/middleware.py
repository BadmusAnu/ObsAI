"""Span context managers and decorators."""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import time

from opentelemetry import trace

from .config import Config
from .metrics import AGENT_COST, AGENT_LATENCY, LLM_REQUESTS, LLM_TOKENS
from .pricing_calc import (
    PRICING_SNAPSHOT_ID,
    embedding_cost,
    llm_cost,
    rag_search_cost,
    tool_cost,
)

_turn_cost: contextvars.ContextVar[float] = contextvars.ContextVar(
    "turn_cost", default=0.0
)
_tenant: contextvars.ContextVar[str] = contextvars.ContextVar("tenant", default="")
_route: contextvars.ContextVar[str] = contextvars.ContextVar("route", default="")
_config: Config | None = None


def configure(config: Config | None) -> None:
    """Register configuration for middleware helpers."""

    global _config
    _config = config


def _is_enabled() -> bool:
    return _config.sdk_enabled if _config is not None else True


def _resolve_default(value: str | None, attribute: str) -> str:
    if value:
        return value
    if _config is None:
        return ""
    return getattr(_config, attribute)


def _add_cost(cost: float, model: str | None, component: str) -> None:
    total = _turn_cost.get()
    _turn_cost.set(total + cost)
    AGENT_COST.labels(
        tenant=_tenant.get(),
        route=_route.get(),
        model=model or "",
        component=component,
    ).inc(cost)


@contextmanager
def agent_turn(tenant_id: str | None = None, route: str | None = None):
    if not _is_enabled():
        yield None
        return

    tenant_id = _resolve_default(tenant_id, "tenant_id")
    route = _resolve_default(route, "route") or "default"
    tracer = trace.get_tracer("ai_cost_sdk")
    start = time.time()
    with tracer.start_as_current_span("gateway.request") as span:
        span.set_attribute("tenant.id", tenant_id)
        span.set_attribute("route.name", route)
        span.set_attribute("pricing_snapshot_id", PRICING_SNAPSHOT_ID)
        token_c = _turn_cost.set(0.0)
        token_t = _tenant.set(tenant_id)
        token_r = _route.set(route)
        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            raise
        finally:
            span.set_attribute("cost.usd.total", round(_turn_cost.get(), 10))
            AGENT_LATENCY.labels(route=route).observe(time.time() - start)
            _turn_cost.reset(token_c)
            _tenant.reset(token_t)
            _route.reset(token_r)


@contextmanager
def llm_call(model: str | None, vendor: str, usage: dict | None = None):
    if not _is_enabled():
        yield None
        return
    tracer = trace.get_tracer("ai_cost_sdk")
    usage = usage if usage is not None else {}
    start = time.time()
    with tracer.start_as_current_span("llm.call") as span:
        if model:
            span.set_attribute("ai.model.id", model)
        span.set_attribute("ai.vendor", vendor)
        span.set_attribute("pricing_snapshot_id", PRICING_SNAPSHOT_ID)
        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            raise
        finally:
            latency = (time.time() - start) * 1000
            span.set_attribute("latency_ms", latency)
            in_t = int(usage.get("prompt_tokens", 0))
            out_t = int(usage.get("completion_tokens", 0))
            cached_t = int(usage.get("cached_tokens", 0))
            if in_t:
                span.set_attribute("ai.tokens.input", in_t)
            if out_t:
                span.set_attribute("ai.tokens.output", out_t)
            if cached_t:
                span.set_attribute("ai.tokens.cached", cached_t)
            cost = llm_cost(model, in_t, out_t, cached_t)
            span.set_attribute("cost.usd.total", cost)
            _add_cost(cost, model, "llm")
            model_label = model or ""
            LLM_REQUESTS.labels(model=model_label).inc()
            if in_t:
                LLM_TOKENS.labels(type="input", model=model_label).inc(in_t)
            if out_t:
                LLM_TOKENS.labels(type="output", model=model_label).inc(out_t)
            if cached_t:
                LLM_TOKENS.labels(type="cached", model=model_label).inc(cached_t)


@contextmanager
def rag_embed(model: str, vendor: str, tokens: int):
    if not _is_enabled():
        yield None
        return
    tracer = trace.get_tracer("ai_cost_sdk")
    start = time.time()
    with tracer.start_as_current_span("rag.embed") as span:
        span.set_attribute("ai.model.id", model)
        span.set_attribute("ai.vendor", vendor)
        span.set_attribute("ai.tokens.input", tokens)
        span.set_attribute("pricing_snapshot_id", PRICING_SNAPSHOT_ID)
        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            raise
        finally:
            latency = (time.time() - start) * 1000
            span.set_attribute("latency_ms", latency)
            cost = embedding_cost(model, tokens)
            span.set_attribute("cost.usd.total", cost)
            _add_cost(cost, model, "rag_embed")


@contextmanager
def rag_search(
    index_id: str,
    index_version: str,
    k: int,
    freshness_s: int,
    read_units: int = 0,
    price_per_unit: float = 0.0,
):
    if not _is_enabled():
        yield None
        return
    tracer = trace.get_tracer("ai_cost_sdk")
    start = time.time()
    with tracer.start_as_current_span("rag.search") as span:
        span.set_attribute("rag.index.id", index_id)
        span.set_attribute("rag.index.version", index_version)
        span.set_attribute("rag.k", k)
        span.set_attribute("rag.freshness_s", freshness_s)
        if read_units:
            span.set_attribute("rag.read_units", read_units)
        if price_per_unit:
            span.set_attribute("rag.price_per_unit", price_per_unit)
        span.set_attribute("pricing_snapshot_id", PRICING_SNAPSHOT_ID)
        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            raise
        finally:
            latency = (time.time() - start) * 1000
            span.set_attribute("latency_ms", latency)
            cost = rag_search_cost(read_units, price_per_unit)
            span.set_attribute("cost.usd.total", cost)
            _add_cost(cost, index_id, "rag_search")


@contextmanager
def tool_call(name: str, vendor: str | None = None, unit_price: float | None = None):
    if not _is_enabled():
        yield None
        return
    tracer = trace.get_tracer("ai_cost_sdk")
    start = time.time()
    with tracer.start_as_current_span("tool.call") as span:
        span.set_attribute("tool.name", name)
        if vendor:
            span.set_attribute("ai.vendor", vendor)
        span.set_attribute("pricing_snapshot_id", PRICING_SNAPSHOT_ID)
        try:
            yield span
        except Exception as exc:  # pragma: no cover - passthrough
            span.set_attribute("error.type", exc.__class__.__name__)
            raise
        finally:
            latency = (time.time() - start) * 1000
            span.set_attribute("latency_ms", latency)
            cost = tool_cost(unit_price)
            span.set_attribute("cost.usd.total", cost)
            _add_cost(cost, name, "tool")
