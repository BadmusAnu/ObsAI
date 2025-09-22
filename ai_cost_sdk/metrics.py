"""Prometheus metrics."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server


class _HistogramCountProxy:
    """Expose a Prometheus histogram's count via the legacy attribute used in tests."""

    def __init__(self, child: Histogram):
        self._child = child

    @property
    def _value(self) -> float:
        for sample in self._child._samples():  # pragma: no branch - small helper
            if sample.name == "_count":
                return float(sample.value)
        return 0.0


def _ensure_histogram_child(child: Histogram) -> Histogram:
    if not hasattr(child, "_count"):
        child._count = _HistogramCountProxy(child)  # type: ignore[attr-defined]
    return child


def _wrap_histogram(histogram: Histogram) -> Histogram:
    original_labels = histogram.labels

    def labels_with_count(*args, **kwargs):
        return _ensure_histogram_child(original_labels(*args, **kwargs))

    histogram.labels = labels_with_count  # type: ignore[assignment]
    return histogram

# Request counters
LLM_REQUESTS = Counter("llm_requests_total", "LLM requests", ["model", "vendor"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["type", "model", "vendor"])
AGENT_REQUESTS = Counter("agent_requests_total", "Agent requests", ["tenant", "route"])

# Cost metrics
AGENT_COST = Counter(
    "agent_cost_usd_total",
    "Total cost in USD",
    ["tenant", "route", "model", "component", "vendor"],
)
LLM_COST_PER_REQUEST = Gauge(
    "llm_cost_per_request_usd",
    "Cost per LLM request in USD",
    ["model", "vendor"],
)
AGENT_COST_PER_REQUEST = Gauge(
    "agent_cost_per_request_usd", 
    "Cost per agent request in USD",
    ["tenant", "route"],
)

# Latency metrics
AGENT_LATENCY = _wrap_histogram(
    Histogram("agent_turn_latency_seconds", "Agent turn latency", ["route"])
)
LLM_LATENCY = _wrap_histogram(
    Histogram("llm_latency_seconds", "LLM call latency", ["model", "vendor"])
)
RAG_LATENCY = _wrap_histogram(
    Histogram("rag_latency_seconds", "RAG operation latency", ["operation", "model", "vendor"])
)
TOOL_LATENCY = _wrap_histogram(
    Histogram("tool_latency_seconds", "Tool call latency", ["tool_name", "vendor"])
)

# Throughput metrics
LLM_THROUGHPUT = Summary("llm_throughput_tokens_per_second", "LLM throughput in tokens per second", ["model", "vendor"])
AGENT_THROUGHPUT = Summary("agent_throughput_requests_per_second", "Agent throughput in requests per second", ["tenant", "route"])

# Error metrics
LLM_ERRORS = Counter("llm_errors_total", "LLM errors", ["model", "vendor", "error_type"])
AGENT_ERRORS = Counter("agent_errors_total", "Agent errors", ["tenant", "route", "error_type"])

# Resource utilization
ACTIVE_REQUESTS = Gauge("active_requests", "Currently active requests", ["tenant", "route"])
QUEUE_SIZE = Gauge("request_queue_size", "Request queue size", ["tenant", "route"])


def start_server(port: int = 9108) -> None:
    """Start the Prometheus metrics server."""
    start_http_server(port)
