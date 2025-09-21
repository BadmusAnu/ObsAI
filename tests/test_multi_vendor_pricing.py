"""Test multi-vendor pricing functionality."""

import pytest
from ai_cost_sdk.pricing_calc import (
    llm_cost, 
    is_model_supported, 
    get_supported_vendors, 
    get_supported_models
)


class TestMultiVendorPricing:
    """Test multi-vendor pricing functionality."""
    
    def test_openai_pricing(self):
        """Test OpenAI model pricing."""
        # Test GPT-4o pricing
        cost = llm_cost("gpt-4o", 1000, 500, 0, "openai")
        expected = (1000 / 1000) * 0.0025 + (500 / 1000) * 0.0100  # 0.0025 + 0.005 = 0.0075
        assert cost == 0.0075
        
        # Test GPT-4o-mini pricing
        cost = llm_cost("gpt-4o-mini", 2000, 1000, 0, "openai")
        expected = (2000 / 1000) * 0.0003 + (1000 / 1000) * 0.0006  # 0.0006 + 0.0006 = 0.0012
        assert cost == 0.0012
    
    def test_claude_pricing(self):
        """Test Claude model pricing."""
        # Test Claude 3.5 Sonnet pricing
        cost = llm_cost("claude-3-5-sonnet", 1000, 500, 0, "claude")
        expected = (1000 / 1000) * 0.003 + (500 / 1000) * 0.015  # 0.003 + 0.0075 = 0.0105
        assert cost == 0.0105
        
        # Test Claude 3.5 Haiku pricing
        cost = llm_cost("claude-3-5-haiku", 2000, 1000, 0, "claude")
        expected = (2000 / 1000) * 0.0008 + (1000 / 1000) * 0.004  # 0.0016 + 0.004 = 0.0056
        assert cost == 0.0056
    
    def test_gemini_pricing(self):
        """Test Gemini model pricing."""
        # Test Gemini 1.5 Pro pricing
        cost = llm_cost("gemini-1.5-pro", 1000, 500, 0, "gemini")
        expected = (1000 / 1000) * 0.00125 + (500 / 1000) * 0.005  # 0.00125 + 0.0025 = 0.00375
        assert cost == 0.00375
        
        # Test Gemini 1.5 Flash pricing
        cost = llm_cost("gemini-1.5-flash", 2000, 1000, 0, "gemini")
        expected = (2000 / 1000) * 0.000075 + (1000 / 1000) * 0.0003  # 0.00015 + 0.0003 = 0.00045
        assert cost == 0.00045
    
    def test_cached_tokens(self):
        """Test cached token pricing (should reduce cost)."""
        # Test with cached tokens
        cost_with_cache = llm_cost("gpt-4o", 1000, 500, 200, "openai")
        cost_without_cache = llm_cost("gpt-4o", 1000, 500, 0, "openai")
        
        # Cached tokens should reduce the cost
        assert cost_with_cache < cost_without_cache
        assert cost_with_cache == 0.0075 - (200 / 1000) * 0.0025  # 0.0075 - 0.0005 = 0.007
    
    def test_unknown_vendor_model(self):
        """Test unknown vendor/model combinations return 0."""
        # Test unknown vendor
        cost = llm_cost("gpt-4o", 1000, 500, 0, "unknown-vendor")
        assert cost == 0.0
        
        # Test unknown model
        cost = llm_cost("unknown-model", 1000, 500, 0, "openai")
        assert cost == 0.0
    
    def test_model_support_checks(self):
        """Test model support checking functions."""
        # Test supported models
        assert is_model_supported("gpt-4o", "openai") == True
        assert is_model_supported("claude-3-5-sonnet", "claude") == True
        assert is_model_supported("gemini-1.5-pro", "gemini") == True
        
        # Test unsupported models
        assert is_model_supported("unknown-model", "openai") == False
        assert is_model_supported("gpt-4o", "unknown-vendor") == False
    
    def test_supported_models_lists(self):
        """Test getting lists of supported models."""
        # Test getting supported vendors
        vendors = get_supported_vendors()
        assert "openai" in vendors
        assert "claude" in vendors
        assert "gemini" in vendors
        assert "embeddings" not in vendors  # embeddings is not an LLM vendor
        
        # Test getting models for each vendor
        openai_models = get_supported_models("openai")
        assert "gpt-4o" in openai_models
        assert "gpt-4o-mini" in openai_models
        
        claude_models = get_supported_models("claude")
        assert "claude-3-5-sonnet" in claude_models
        assert "claude-3-5-haiku" in claude_models
        
        gemini_models = get_supported_models("gemini")
        assert "gemini-1.5-pro" in gemini_models
        assert "gemini-1.5-flash" in gemini_models
    
    def test_backward_compatibility(self):
        """Test that default vendor is still openai for backward compatibility."""
        # Test without specifying vendor (should default to openai)
        cost_default = llm_cost("gpt-4o", 1000, 500, 0)
        cost_explicit = llm_cost("gpt-4o", 1000, 500, 0, "openai")
        assert cost_default == cost_explicit
    
    def test_zero_tokens(self):
        """Test that zero tokens return zero cost."""
        cost = llm_cost("gpt-4o", 0, 0, 0, "openai")
        assert cost == 0.0
        
        cost = llm_cost("claude-3-5-sonnet", 0, 0, 0, "claude")
        assert cost == 0.0


if __name__ == "__main__":
    # Run a simple verification without pytest
    print("Testing multi-vendor pricing...")
    
    # Test OpenAI
    openai_cost = llm_cost("gpt-4o", 1000, 500, 0, "openai")
    print(f"OpenAI GPT-4o (1000 in, 500 out): ${openai_cost:.6f}")
    assert openai_cost == 0.0075
    
    # Test Claude
    claude_cost = llm_cost("claude-3-5-sonnet", 1000, 500, 0, "claude")
    print(f"Claude 3.5 Sonnet (1000 in, 500 out): ${claude_cost:.6f}")
    assert claude_cost == 0.0105
    
    # Test Gemini
    gemini_cost = llm_cost("gemini-1.5-pro", 1000, 500, 0, "gemini")
    print(f"Gemini 1.5 Pro (1000 in, 500 out): ${gemini_cost:.6f}")
    assert gemini_cost == 0.00375
    
    # Test unknown vendor
    unknown_cost = llm_cost("unknown-model", 1000, 500, 0, "unknown-vendor")
    print(f"Unknown vendor/model: ${unknown_cost:.6f}")
    assert unknown_cost == 0.0
    
    print("All tests passed! Multi-vendor pricing is working correctly.")
