# AI Cost Telemetry SDK

Agent-first, cost-native telemetry for AI apps. Emits OpenTelemetry traces and Prometheus metrics with per-span cost.

## Quickstart

```bash
export TENANT_ID=demo
export PROJECT_ID=test
export SERVICE_NAME=demo-app  # optional override for service.name
python examples/sample_app.py
```

Metrics are exposed at http://localhost:9108/metrics and traces can be sent to any OTLP collector. Set `SDK_ENABLED=false` to make the middleware a no-op without touching call sites.

## OpenAI wrapper

Wrap an existing client:

```python
from ai_cost_sdk.config import load_config
from ai_cost_sdk.middleware import configure
from ai_cost_sdk.integrations.openai_wrapper import chat_completion

config = load_config()
configure(config)
resp = chat_completion(
    client,
    config=config,
    model="gpt-4o",
    messages=[{"role": "user", "content": "hi"}],
)
```

When providers omit usage metadata, enable `TOKENIZE_FALLBACK=true` and the wrapper will approximate token counts locally to keep cost attribution intact.

## OTLP export

Set `EXPORT_OTLP_ENDPOINT=http://localhost:4318/v1/traces` to ship traces. Without it, spans are written to JSON if `EXPORT_JSON_PATH` is set.

## Pricing tables

Pricing snapshots live in `ai_cost_sdk/pricing/*.json`. Add new files and set `PRICING_SNAPSHOT` env var to switch.

## Privacy & fail-open

Bodies are not recorded and queues drop on overflow to avoid blocking the request path.

## Required configuration

`TENANT_ID` and `PROJECT_ID` must be set or `load_config()` will raise. Optional env vars include:

- `SERVICE_NAME` – overrides the exported `service.name` resource attribute (defaults to `ai-cost-sdk`).
- `EXPORT_OTLP_ENDPOINT` / `EXPORT_JSON_PATH` – choose your exporter target.
- `REDACT_PROMPTS` / `TOKENIZE_FALLBACK` – control prompt capture and token fallback behaviour.

## License

MIT
