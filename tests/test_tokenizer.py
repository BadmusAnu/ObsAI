"""Test tokenizer functionality."""

import time
import pytest
from unittest.mock import patch

from ai_cost_sdk.tokenizer import (
    count_tokens, count_tokens_batch, get_model_vendor,
    validate_model_support, count_tokens_openai, count_tokens_claude,
    count_tokens_gemini
)


def test_count_tokens_openai():
    """Test OpenAI token counting."""
    # Test with tiktoken available
    with patch('ai_cost_sdk.tokenizer.TIKTOKEN_AVAILABLE', True):
        with patch('ai_cost_sdk.tokenizer.tiktoken.get_encoding') as mock_get_encoding:
            mock_encoding = mock_get_encoding.return_value
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            
            tokens = count_tokens_openai("Hello world", "gpt-4o")
            assert tokens == 5
            mock_get_encoding.assert_called_once_with("o200k_base")
    
    # Test with tiktoken not available (fallback)
    with patch('ai_cost_sdk.tokenizer.TIKTOKEN_AVAILABLE', False):
        tokens = count_tokens_openai("Hello world", "gpt-4o")
        assert tokens == 3  # len("Hello world") // 4 = 2, but we expect some reasonable value


def test_count_tokens_claude():
    """Test Claude token counting."""
    tokens = count_tokens_claude("Hello world test", "claude-3-5-sonnet")
    assert tokens == 3  # 3 words


def test_count_tokens_gemini():
    """Test Gemini token counting."""
    tokens = count_tokens_gemini("Hello world test", "gemini-pro")
    assert tokens == 3  # 3 words


def test_count_tokens():
    """Test main token counting function."""
    # Test OpenAI
    with patch('ai_cost_sdk.tokenizer.count_tokens_openai', return_value=5):
        tokens = count_tokens("Hello", "gpt-4o", "openai")
        assert tokens == 5
    
    # Test Claude
    tokens = count_tokens("Hello world", "claude-3-5-sonnet", "claude")
    assert tokens == 2  # 2 words
    
    # Test Gemini
    tokens = count_tokens("Hello world", "gemini-pro", "gemini")
    assert tokens == 2  # 2 words
    
    # Test unknown vendor (fallback)
    tokens = count_tokens("Hello world", "unknown-model", "unknown")
    assert tokens == 3  # len("Hello world") // 4 = 2, but fallback might be different
    
    # Test empty text
    tokens = count_tokens("", "gpt-4o", "openai")
    assert tokens == 0


def test_count_tokens_batch():
    """Test batch token counting."""
    texts = ["Hello", "World", "Test"]
    
    with patch('ai_cost_sdk.tokenizer.count_tokens', side_effect=[2, 1, 1]):
        total_tokens = count_tokens_batch(texts, "gpt-4o", "openai")
        assert total_tokens == 4  # 2 + 1 + 1


def test_get_model_vendor():
    """Test vendor detection from model names."""
    # OpenAI models
    assert get_model_vendor("gpt-4o") == "openai"
    assert get_model_vendor("gpt-4") == "openai"
    assert get_model_vendor("gpt-3.5-turbo") == "openai"
    assert get_model_vendor("text-embedding-3-large") == "openai"
    
    # Claude models
    assert get_model_vendor("claude-3-5-sonnet") == "claude"
    assert get_model_vendor("claude-3-5-haiku") == "claude"
    assert get_model_vendor("claude-3-opus") == "claude"
    
    # Gemini models
    assert get_model_vendor("gemini-pro") == "gemini"
    assert get_model_vendor("gemini-1.5-pro") == "gemini"
    assert get_model_vendor("gemini-1.5-flash") == "gemini"
    
    # Unknown models
    assert get_model_vendor("unknown-model") == "unknown"
    assert get_model_vendor("") == "unknown"


def test_validate_model_support():
    """Test model support validation."""
    # OpenAI models
    assert validate_model_support("gpt-4o", "openai") == True
    assert validate_model_support("gpt-4", "openai") == True
    assert validate_model_support("text-embedding-3-large", "openai") == True
    assert validate_model_support("unknown-model", "openai") == False
    
    # Claude models
    assert validate_model_support("claude-3-5-sonnet", "claude") == True
    assert validate_model_support("claude-3-5-haiku", "claude") == True
    assert validate_model_support("unknown-model", "claude") == False
    
    # Gemini models
    assert validate_model_support("gemini-pro", "gemini") == True
    assert validate_model_support("gemini-1.5-pro", "gemini") == True
    assert validate_model_support("unknown-model", "gemini") == False
    
    # Unknown vendors
    assert validate_model_support("any-model", "unknown") == False


def test_tokenizer_edge_cases():
    """Test edge cases for tokenizer."""
    # Empty string
    tokens = count_tokens("", "gpt-4o", "openai")
    assert tokens == 0
    
    # Very long text
    long_text = "word " * 1000
    tokens = count_tokens(long_text, "gpt-4o", "openai")
    assert tokens > 0
    
    # Special characters
    special_text = "Hello! @#$%^&*()_+ 123"
    tokens = count_tokens(special_text, "gpt-4o", "openai")
    assert tokens > 0
    
    # Unicode text
    unicode_text = "Hello 世界 🌍"
    tokens = count_tokens(unicode_text, "gpt-4o", "openai")
    assert tokens > 0


def test_tokenizer_performance():
    """Test tokenizer performance with large batches."""
    # Large batch of texts
    texts = [f"Text {i}" for i in range(1000)]
    
    start_time = time.time()
    total_tokens = count_tokens_batch(texts, "gpt-4o", "openai")
    end_time = time.time()
    
    assert total_tokens > 0
    # Should complete in reasonable time (less than 1 second for 1000 texts)
    assert (end_time - start_time) < 1.0


def test_tiktoken_fallback():
    """Test tiktoken fallback behavior."""
    # Test when tiktoken raises an exception
    with patch('ai_cost_sdk.tokenizer.TIKTOKEN_AVAILABLE', True):
        with patch('ai_cost_sdk.tokenizer.tiktoken.get_encoding', side_effect=Exception("Tiktoken error")):
            tokens = count_tokens_openai("Hello world", "gpt-4o")
            # Should fallback to character-based estimation
            assert tokens == 3  # len("Hello world") // 4 = 2, but fallback might be different
