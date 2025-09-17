"""Minimal sample app demonstrating the SDK."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

from ai_cost_sdk.config import load_config
from ai_cost_sdk.integrations.openai_wrapper import chat_completion
from ai_cost_sdk.integrations.rag_wrapper import embed, vector_search
from ai_cost_sdk.integrations.tools_wrapper import priced_tool
from ai_cost_sdk.middleware import agent_turn, configure as configure_middleware
from ai_cost_sdk.metrics import start_server
from ai_cost_sdk.observability import setup_tracer


class _FakeChatCompletions:
    def create(self, **_kwargs):
        time.sleep(0.01)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!"))],
            usage={"prompt_tokens": 5, "completion_tokens": 7},
        )


class _FakeChat:
    completions = _FakeChatCompletions()


class FakeOpenAIClient:
    chat = _FakeChat()


@priced_tool("fake_crm", unit_price=0.002)
def call_crm():
    time.sleep(0.005)
    return {"status": "ok"}


def main() -> None:
    os.environ.setdefault("TENANT_ID", "demo-tenant")
    os.environ.setdefault("PROJECT_ID", "demo-project")
    config = load_config()
    configure_middleware(config)
    setup_tracer(config)
    start_server(9108)
    client = FakeOpenAIClient()
    with agent_turn(config.tenant_id, config.route):
        resp = chat_completion(
            client,
            config=config,
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        embed(["hello world"], model="text-embedding-3-large")
        vector_search(
            "my-index",
            "hello",
            k=2,
            index_version="v1",
            read_units=3,
            price_per_unit=0.0001,
        )
        call_crm()
        print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
