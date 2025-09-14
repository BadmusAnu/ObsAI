"""Prometheus metrics."""

from __future__ import annotations

from prometheus_client import Counter, Histogram, start_http_server

LLM_REQUESTS = Counter("llm_requests_total", "LLM requests", ["model"])
LLM_TOKENS = Counter("llm_tokens_total", "LLM tokens", ["type", "model"])
AGENT_LATENCY = Histogram("agent_turn_latency_seconds", "Agent turn latency", ["route"])
AGENT_COST = Counter(
    "agent_cost_usd_total",
    "Cost in USD",
    ["tenant", "route", "model", "component"],
)


def start_server(port: int = 9108) -> None:
    start_http_server(port)
