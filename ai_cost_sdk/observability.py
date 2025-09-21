"""OpenTelemetry setup."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)

from .config import Config
from .exporter import FileSpanExporter
from . import __version__


def setup_tracer(config: Config) -> trace.Tracer:
    resource = Resource(
        {
            "service.name": config.service_name,
            "tenant.id": config.tenant_id,
            "project.id": config.project_id,
            "route.name": config.route,
            "sdk.version": __version__,
        }
    )
    provider = TracerProvider(resource=resource)
    exporter: SpanExporter
    if config.export_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=config.export_otlp_endpoint, timeout=1)
    elif config.export_json_path:
        exporter = FileSpanExporter(config.export_json_path)
    else:
        exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("ai_cost_sdk")
