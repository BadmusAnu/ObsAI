"""OpenAI client wrapper."""

from __future__ import annotations

from typing import Any, Iterable

from .. import middleware
from ..config import Config
from ..pricing_calc import estimate_tokens


def _flatten_contents(contents: Any) -> Iterable[str]:
    if isinstance(contents, str):
        yield contents
    elif isinstance(contents, dict):
        value = contents.get("text")
        if isinstance(value, str):
            yield value
    elif isinstance(contents, Iterable):
        for item in contents:
            yield from _flatten_contents(item)


def _count_prompt_tokens(kwargs: dict[str, Any]) -> int:
    messages = kwargs.get("messages")
    if isinstance(messages, Iterable) and not isinstance(messages, (str, bytes)):
        total = 0
        for message in messages:
            if isinstance(message, dict):
                for piece in _flatten_contents(message.get("content")):
                    total += estimate_tokens(piece)
            else:
                total += estimate_tokens(str(message))
        return total
    prompt = kwargs.get("prompt")
    if isinstance(prompt, Iterable) and not isinstance(prompt, (str, bytes)):
        return sum(estimate_tokens(str(p)) for p in prompt)
    if isinstance(prompt, str):
        return estimate_tokens(prompt)
    return 0


def _count_completion_tokens(response: Any) -> int:
    choices = getattr(response, "choices", None)
    if not choices:
        return 0
    total = 0
    for choice in choices:
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", "")
        else:
            content = getattr(choice, "text", "")
        total += estimate_tokens(str(content))
    return total


def chat_completion(
    client,
    *,
    config: Config | None = None,
    **kwargs,
):
    model = kwargs.get("model")
    usage: dict = {}
    with middleware.llm_call(model=model, vendor="openai", usage=usage):
        response = client.chat.completions.create(**kwargs)
        resp_usage = getattr(response, "usage", {}) or {}
        if resp_usage:
            usage.update(resp_usage)
        elif config and config.tokenize_fallback:
            prompt_tokens = _count_prompt_tokens(kwargs)
            completion_tokens = _count_completion_tokens(response)
            if prompt_tokens:
                usage["prompt_tokens"] = prompt_tokens
            if completion_tokens:
                usage["completion_tokens"] = completion_tokens
        return response
