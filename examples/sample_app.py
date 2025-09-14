"""Minimal sample app demonstrating the SDK."""

from __future__ import annotations

import time
from types import SimpleNamespace

from ai_cost_sdk.config import load_config
from ai_cost_sdk.integrations.openai_wrapper import chat_completion
from ai_cost_sdk.integrations.rag_wrapper import embed, vector_search
from ai_cost_sdk.integrations.tools_wrapper import priced_tool
from ai_cost_sdk.middleware import agent_turn
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
    config = load_config()
    setup_tracer(config)
    start_server(9108)
    client = FakeOpenAIClient()
    with agent_turn(config.tenant_id, config.route):
        resp = chat_completion(
            client,
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
        )
        embed(["hello world"], model="text-embedding-3-large")
        vector_search("my-index", "hello", k=2, index_version="v1")
        call_crm()
        print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
