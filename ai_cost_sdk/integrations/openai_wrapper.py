"""OpenAI client wrapper."""

from __future__ import annotations


from typing import Any


from .. import middleware, config
from ..tokenizer import get_model_vendor, count_tokens


def _extract_text_segments(content: Any) -> list[str]:
    """Flatten rich message content into plain-text segments."""

    segments: list[str] = []

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            if value:
                segments.append(value)
            return

        if isinstance(value, list):
            for item in value:
                _collect(item)
            return

        if isinstance(value, dict):
            handled = False
            for key in ("text", "input_text", "output_text", "content", "value", "arguments"):
                if key in value:
                    handled = True
                    _collect(value[key])

            if handled:
                return

            segment_type = value.get("type")
            if segment_type in {"image_url", "input_audio", "output_audio", "audio", "tool_result"}:
                return

            fallback = value.get("message")
            if isinstance(fallback, str) and fallback:
                segments.append(fallback)
                return
            if fallback not in (None, ""):
                try:
                    fallback_str = str(fallback)
                except Exception:  # pragma: no cover - defensive
                    fallback_str = ""
                if fallback_str:
                    segments.append(fallback_str)
            return

        if value is None:
            return

        try:
            text = str(value)
        except Exception:  # pragma: no cover - defensive
            text = ""

        if text:
            segments.append(text)

    _collect(content)
    return segments


def _normalize_message_content(content: Any) -> str:
    """Normalize a message content field into plain text."""

    parts = _extract_text_segments(content)
    return " ".join(part for part in parts if part)


def _build_prompt_text(messages: list[Any]) -> str:
    """Flatten a list of chat messages into a single prompt string."""

    normalized: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text = _normalize_message_content(msg.get("content"))
        if text:
            normalized.append(text)
    return " ".join(normalized)


def _is_tokenize_fallback_enabled() -> bool:
    """Determine whether tokenization fallback should be attempted."""

    try:
        return config.load_config().tokenize_fallback
    except ValueError:
        return config.load_config_permissive().tokenize_fallback


def _coerce_messages(payload: Any) -> list[Any]:
    """Ensure chat payloads are realized as a list for reuse."""

    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    return list(payload)


def _collect_completion_text(message: Any) -> str:
    """Extract plain text from an assistant message for tokenization."""

    if message is None:
        return ""

    segments: list[str] = []

    if isinstance(message, dict):
        content = message.get("content")
        tool_calls = message.get("tool_calls")
    else:
        content = getattr(message, "content", None)
        tool_calls = getattr(message, "tool_calls", None)

    if content:
        segments.extend(_extract_text_segments(content))

    if tool_calls:
        segments.extend(_extract_text_segments(tool_calls))

    if not segments:
        try:
            fallback = str(message)
        except Exception:  # pragma: no cover - defensive
            fallback = ""
        if fallback:
            segments.append(fallback)

    return " ".join(part for part in segments if part)


def chat_completion(client, **kwargs):
    """OpenAI chat completion with enhanced cost tracking."""

    model = kwargs.get("model")
    messages = _coerce_messages(kwargs.get("messages"))
    kwargs["messages"] = messages

    tokenize_fallback_enabled = _is_tokenize_fallback_enabled()
    prompt_for_span = None
    if tokenize_fallback_enabled and messages:
        prompt_for_span = _build_prompt_text(messages)

    usage: dict = {}
    vendor = get_model_vendor(model) if model else "openai"

    with middleware.llm_call(
        model=model,
        vendor=vendor,
        usage=usage,
        prompt=prompt_for_span,
    ):
        response = client.chat.completions.create(**kwargs)
        resp_usage = getattr(response, "usage", {}) or {}
        if hasattr(resp_usage, "model_dump") and callable(resp_usage.model_dump):
            resp_usage = resp_usage.model_dump()
        usage.update(resp_usage)

        if tokenize_fallback_enabled:
            if not usage.get("prompt_tokens") and messages:
                prompt_text = prompt_for_span or _build_prompt_text(messages)
                if prompt_text:
                    usage["prompt_tokens"] = count_tokens(prompt_text, model, vendor)
            if not usage.get("completion_tokens") and response.choices:
                completion_text = _collect_completion_text(response.choices[0].message)
                if completion_text:
                    usage["completion_tokens"] = count_tokens(completion_text, model, vendor)

        return response
