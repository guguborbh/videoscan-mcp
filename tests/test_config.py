"""Tests for configuration loading."""
from videoscan.utils.config import Settings

def test_default_settings():
    settings = Settings()
    assert settings.vision_provider == "openai"
    assert settings.vision_concurrency == 15
    assert settings.cache_enabled is True
    assert settings.max_video_duration == 3600
    assert settings.max_analyzed_frames == 100

def test_unlimited_mode(env_override):
    env_override(MAX_VIDEO_DURATION="0", MAX_DOWNLOAD_SIZE="0", MAX_ANALYZED_FRAMES="0", DOWNLOAD_TIMEOUT="0", FRAME_ANALYSIS_TIMEOUT="0")
    settings = Settings()
    assert settings.max_video_duration == 0
    assert settings.max_download_size == 0

def test_custom_provider(env_override):
    env_override(VISION_PROVIDER="anthropic", ANTHROPIC_API_KEY="sk-ant-test")
    settings = Settings()
    assert settings.vision_provider == "anthropic"

def test_default_vision_model_per_provider():
    s = Settings()
    assert s.get_vision_model() == "gpt-4o"
    s2 = Settings(vision_provider="anthropic")
    assert "claude" in s2.get_vision_model()
    s3 = Settings(vision_provider="google")
    assert "gemini" in s3.get_vision_model()

def test_explicit_model_overrides_default(env_override):
    env_override(VISION_PROVIDER="openai", VISION_MODEL="gpt-4o-mini")
    settings = Settings()
    assert settings.get_vision_model() == "gpt-4o-mini"
