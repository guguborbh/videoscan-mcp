"""Configuration loading from environment variables."""
from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings

_DEFAULT_VISION_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.0-flash",
}

class Settings(BaseSettings):
    vision_provider: str = "openai"
    vision_model: str = ""
    vision_concurrency: int = 15
    transcription_provider: str = "openai"
    transcription_model: str = "whisper-1"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    cache_enabled: bool = True
    cache_dir: str = "~/.videoscan/cache"
    cache_max_size_gb: int = 5
    cache_download_ttl: int = 3600
    cache_frames_ttl: int = 86400
    cache_results_ttl: int = 604800
    max_video_duration: int = 3600
    max_download_size: int = 2_147_483_648
    max_analyzed_frames: int = 100
    download_timeout: int = 300
    frame_analysis_timeout: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def get_vision_model(self) -> str:
        if self.vision_model:
            return self.vision_model
        return _DEFAULT_VISION_MODELS.get(self.vision_provider, "gpt-4o")

    def get_cache_path(self) -> Path:
        return Path(self.cache_dir).expanduser()

    def get_api_key(self, provider: str | None = None) -> str:
        p = provider or self.vision_provider
        keys = {"openai": self.openai_api_key, "anthropic": self.anthropic_api_key, "google": self.google_api_key}
        return keys.get(p, "")
