"""transcribe MCP tool — audio transcription only."""
from __future__ import annotations
import asyncio
from videoscan.core.downloader import Downloader
from videoscan.providers.base import get_transcription_provider
from videoscan.types import TranscriptResult, TranscriptSegment
from videoscan.utils.config import Settings

async def transcribe(source: str, language: str = "auto", settings: Settings | None = None) -> TranscriptResult:
    settings = settings or Settings()
    downloader = Downloader(timeout=settings.download_timeout)
    result = await asyncio.to_thread(downloader.resolve_source, source)
    audio_path = result.audio_path or result.video_path
    provider = get_transcription_provider(settings.transcription_provider, settings.get_api_key("openai"), settings.transcription_model)
    transcription = await provider.transcribe(audio_path, language)
    segments = [TranscriptSegment(text=seg["text"], start=seg["start"], end=seg["end"]) for seg in transcription.segments]
    return TranscriptResult(language=transcription.language, segments=segments)
