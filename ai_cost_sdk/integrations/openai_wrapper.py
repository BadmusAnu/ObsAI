"""OpenAI client wrapper."""

from __future__ import annotations


import os
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


def chat_completion(client, **kwargs):
    """OpenAI chat completion with enhanced cost tracking."""
    model = kwargs.get("model")
    messages = kwargs.get("messages", [])
    
    # Extract prompt text for token counting
    prompt_text = ""
    if messages:
        normalized = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            text = _normalize_message_content(content)
            if text:
                normalized.append(text)
        prompt_text = " ".join(normalized)
    
    usage: dict = {}
    vendor = get_model_vendor(model) if model else "openai"
    
    with middleware.llm_call(
        model=model,
        vendor=vendor,
        usage=usage,
        prompt=prompt_text
    ):
        response = client.chat.completions.create(**kwargs)
        resp_usage = getattr(response, "usage", {}) or {}
        if hasattr(resp_usage, "model_dump") and callable(resp_usage.model_dump):
            resp_usage = resp_usage.model_dump()
        usage.update(resp_usage)

        # If TOKENIZE_FALLBACK is enabled and usage data is missing, calculate tokens
        try:
            tokenize_fallback_enabled = config.load_config().tokenize_fallback
        except ValueError:
            val = os.getenv("TOKENIZE_FALLBACK")
            tokenize_fallback_enabled = bool(val and val.lower() in {"1", "true", "yes", "on"})

        if tokenize_fallback_enabled:
            if not usage.get("prompt_tokens") and prompt_text:
                usage["prompt_tokens"] = count_tokens(prompt_text, model, vendor)
            if not usage.get("completion_tokens") and response.choices:
                completion_text = response.choices[0].message.content or ""
                usage["completion_tokens"] = count_tokens(completion_text, model, vendor)
        
        return response
