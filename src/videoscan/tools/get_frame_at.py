"""get_frame_at MCP tool — single frame at a timestamp."""
from __future__ import annotations
from videoscan.core.downloader import Downloader
from videoscan.core.frame_extractor import FrameExtractor
from videoscan.providers.base import get_vision_provider
from videoscan.types import FrameResult
from videoscan.utils.config import Settings

async def get_frame_at(source: str, timestamp: float, analyze: bool = True, provider: str | None = None, force_refresh: bool = False, settings: Settings | None = None) -> FrameResult:
    settings = settings or Settings()
    downloader = Downloader(timeout=settings.download_timeout)
    extractor = FrameExtractor()
    result = downloader.resolve_source(source)
    frame = extractor.extract_at(result.video_path, timestamp)
    b64 = frame.to_base64()
    description = None
    ocr_text = None
    if analyze:
        vision_provider_name = provider or settings.vision_provider
        vision = get_vision_provider(vision_provider_name, settings.get_api_key(vision_provider_name), settings.get_vision_model() if not provider else settings.vision_model or settings.get_vision_model())
        analysis = await vision.analyze_frame(b64, "standard")
        description = analysis.description
        ocr_text = analysis.ocr_text
    return FrameResult(index=0, timestamp=frame.timestamp, strategy=frame.strategy, image_base64=b64, description=description, ocr_text=ocr_text)
