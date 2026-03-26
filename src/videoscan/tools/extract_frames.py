"""extract_frames MCP tool — frame extraction with AI analysis."""
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass
from videoscan.core.deduplicator import Deduplicator
from videoscan.core.downloader import Downloader
from videoscan.core.frame_extractor import FrameExtractor
from videoscan.providers.base import get_vision_provider
from videoscan.types import FrameResult, Warning
from videoscan.utils.config import Settings

logger = logging.getLogger(__name__)

@dataclass
class ExtractFramesResult:
    frames: list[FrameResult]
    warnings: list[Warning]
    total_raw_extracted: int
    total_after_dedup: int
    vision_cost_usd: float

async def extract_frames(source: str, max_frames: int = 30, threshold: float = 0.3, strategy: str = "combined", interval: int = 5, detail: str = "standard", deduplicate: bool = True, provider: str | None = None, force_refresh: bool = False, settings: Settings | None = None) -> ExtractFramesResult:
    settings = settings or Settings()
    downloader = Downloader(timeout=settings.download_timeout)
    extractor = FrameExtractor()
    warnings: list[Warning] = []
    result = await asyncio.to_thread(downloader.resolve_source, source)
    if strategy == "scene":
        raw_frames = await asyncio.to_thread(extractor.extract_scene_changes, result.video_path, threshold, max_frames)
    elif strategy == "interval":
        raw_frames = await asyncio.to_thread(extractor.extract_interval, result.video_path, float(interval), max_frames)
    else:
        raw_frames = await asyncio.to_thread(extractor.extract_combined, result.video_path, threshold, float(interval), max_frames)
    total_raw = len(raw_frames)
    if deduplicate and raw_frames:
        dedup = Deduplicator()
        raw_frames = dedup.deduplicate(raw_frames)
    total_after_dedup = len(raw_frames)
    effective_max = max_frames
    if settings.max_analyzed_frames > 0:
        effective_max = min(effective_max, settings.max_analyzed_frames)
    raw_frames = raw_frames[:effective_max]
    vision_provider_name = provider or settings.vision_provider
    vision = get_vision_provider(vision_provider_name, settings.get_api_key(vision_provider_name), settings.get_vision_model())
    cost_per_image = getattr(vision, "cost_per_image", 0.015)
    semaphore = asyncio.Semaphore(settings.vision_concurrency)

    async def analyze_one(idx, frame):
        b64 = frame.to_base64()
        async with semaphore:
            try:
                analysis = await asyncio.wait_for(vision.analyze_frame(b64, detail), timeout=settings.frame_analysis_timeout if settings.frame_analysis_timeout > 0 else None)
                return FrameResult(index=idx, timestamp=frame.timestamp, strategy=frame.strategy, image_base64=b64, description=analysis.description, ocr_text=analysis.ocr_text)
            except Exception as e:
                logger.warning(f"Frame analysis failed at {frame.timestamp}s: {e}")
                warnings.append(Warning(code="FRAME_ANALYSIS_FAILED", message=f"Vision AI failed: {e}", timestamp=frame.timestamp))
                return FrameResult(index=idx, timestamp=frame.timestamp, strategy=frame.strategy, image_base64=b64)

    tasks = [analyze_one(i, f) for i, f in enumerate(raw_frames)]
    frame_results = await asyncio.gather(*tasks)
    analyzed_count = len([f for f in frame_results if f.description])
    vision_cost = analyzed_count * cost_per_image
    return ExtractFramesResult(frames=list(frame_results), warnings=warnings, total_raw_extracted=total_raw, total_after_dedup=total_after_dedup, vision_cost_usd=round(vision_cost, 4))
