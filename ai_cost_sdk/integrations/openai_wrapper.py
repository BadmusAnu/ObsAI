"""OpenAI client wrapper."""

from __future__ import annotations

from .. import middleware


def chat_completion(client, **kwargs):
    model = kwargs.get("model")
    usage: dict = {}
    with middleware.llm_call(model=model, vendor="openai", usage=usage):
        response = client.chat.completions.create(**kwargs)
        resp_usage = getattr(response, "usage", {}) or {}
        usage.update(resp_usage)
        return response
