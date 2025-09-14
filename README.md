# AI Cost Telemetry SDK

Agent-first, cost-native telemetry for AI apps. Emits OpenTelemetry traces and Prometheus metrics with per-span cost.

## Quickstart

```bash
export TENANT_ID=demo
export PROJECT_ID=test
python examples/sample_app.py
```

Metrics are exposed at http://localhost:9108/metrics and traces can be sent to any OTLP collector.

## OpenAI wrapper

Wrap an existing client:

```python
from ai_cost_sdk.integrations.openai_wrapper import chat_completion
resp = chat_completion(client, model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
```

## OTLP export

Set `EXPORT_OTLP_ENDPOINT=http://localhost:4318/v1/traces` to ship traces. Without it, spans are written to JSON if `EXPORT_JSON_PATH` is set.

## Pricing tables

Pricing snapshots live in `ai_cost_sdk/pricing/*.json`. Add new files and set `PRICING_SNAPSHOT` env var to switch.

## Privacy & fail-open

Bodies are not recorded and queues drop on overflow to avoid blocking the request path.

## License

MIT
