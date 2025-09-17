"""OpenAI client wrapper."""

from __future__ import annotations

from .. import middleware
from ..tokenizer import get_model_vendor


def chat_completion(client, **kwargs):
    """OpenAI chat completion with enhanced cost tracking."""
    model = kwargs.get("model")
    messages = kwargs.get("messages", [])
    
    # Extract prompt text for token counting
    prompt_text = ""
    if messages:
        prompt_text = " ".join(msg.get("content", "") for msg in messages if isinstance(msg, dict))
    
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
        usage.update(resp_usage)
        return response
