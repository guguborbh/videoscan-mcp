"""Tests for vision and transcription providers."""
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from videoscan.providers.base import VisionProvider, TranscriptionProvider, get_vision_provider
from videoscan.providers.openai import OpenAIVision, OpenAITranscription
from videoscan.providers.anthropic import AnthropicVision
from videoscan.providers.google import GoogleVision

def test_get_vision_provider_openai():
    provider = get_vision_provider("openai", api_key="test-key", model="gpt-4o")
    assert isinstance(provider, OpenAIVision)

def test_get_vision_provider_anthropic():
    provider = get_vision_provider("anthropic", api_key="test-key", model="claude-sonnet-4-20250514")
    assert isinstance(provider, AnthropicVision)

def test_get_vision_provider_google():
    provider = get_vision_provider("google", api_key="test-key", model="gemini-2.0-flash")
    assert isinstance(provider, GoogleVision)

def test_get_vision_provider_unknown():
    with pytest.raises(ValueError, match="Unknown vision provider"):
        get_vision_provider("unknown", api_key="test", model="test")

def test_detail_prompt_brief():
    provider = OpenAIVision(api_key="test", model="gpt-4o")
    prompt = provider._build_prompt("brief")
    assert "one sentence" in prompt.lower()

def test_detail_prompt_detailed():
    provider = OpenAIVision(api_key="test", model="gpt-4o")
    prompt = provider._build_prompt("detailed")
    assert "comprehensive" in prompt.lower()

@pytest.mark.asyncio
async def test_openai_vision_analyze_frame():
    provider = OpenAIVision(api_key="test", model="gpt-4o")
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="A test description"))]
    with patch.object(provider.client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.analyze_frame("base64data", "standard")
        assert result.description == "A test description"
