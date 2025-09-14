"""Configuration loader for the SDK."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    sdk_enabled: bool
    tenant_id: str
    project_id: str
    route: str
    export_otlp_endpoint: str | None
    export_json_path: str | None
    pricing_snapshot: str
    redact_prompts: bool
    tokenize_fallback: bool


def load_config() -> Config:
    """Load configuration from environment variables."""
    return Config(
        sdk_enabled=_get_bool("SDK_ENABLED", True),
        tenant_id=os.getenv("TENANT_ID", ""),
        project_id=os.getenv("PROJECT_ID", ""),
        route=os.getenv("ROUTE", "default"),
        export_otlp_endpoint=os.getenv("EXPORT_OTLP_ENDPOINT"),
        export_json_path=os.getenv("EXPORT_JSON_PATH"),
        pricing_snapshot=os.getenv("PRICING_SNAPSHOT", "openai-2025-09"),
        redact_prompts=_get_bool("REDACT_PROMPTS", True),
        tokenize_fallback=_get_bool("TOKENIZE_FALLBACK", False),
    )
