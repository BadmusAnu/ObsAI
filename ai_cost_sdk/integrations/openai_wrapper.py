"""OpenAI client wrapper."""

from __future__ import annotations

from .. import middleware, config
from ..tokenizer import get_model_vendor, count_tokens


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
        
        # If TOKENIZE_FALLBACK is enabled and usage data is missing, calculate tokens
        if config.load_config().tokenize_fallback:
            if not usage.get("prompt_tokens") and prompt_text:
                usage["prompt_tokens"] = count_tokens(prompt_text, model, vendor)
            if not usage.get("completion_tokens") and response.choices:
                completion_text = response.choices[0].message.content or ""
                usage["completion_tokens"] = count_tokens(completion_text, model, vendor)
        
        return response
