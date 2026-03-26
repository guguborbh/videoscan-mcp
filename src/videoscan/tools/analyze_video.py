"""analyze_video MCP tool — full pipeline combining all analysis."""
from __future__ import annotations
import hashlib, logging, time
from videoscan.core.cache import Cache
from videoscan.core.downloader import Downloader
from videoscan.tools.extract_frames import extract_frames
from videoscan.tools.get_metadata import get_metadata
from videoscan.tools.transcribe import transcribe
from videoscan.types import Stats, TimelineEntry, VideoAnalysisResult, Warning
from videoscan.utils.config import Settings

logger = logging.getLogger(__name__)

def _params_hash(**kwargs) -> str:
    return hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:12]

async def analyze_video(source: str, detail: str = "standard", max_frames: int = 30, threshold: float = 0.3, strategy: str = "combined", interval: int = 5, skip_frames: bool = False, skip_audio: bool = False, language: str = "auto", provider: str | None = None, force_refresh: bool = False, settings: Settings | None = None) -> VideoAnalysisResult:
    settings = settings or Settings()
    t_start = time.time()
    warnings: list[Warning] = []
    cache = Cache(cache_dir=str(settings.get_cache_path()), max_size_gb=settings.cache_max_size_gb, enabled=settings.cache_enabled)
    downloader = Downloader(timeout=settings.download_timeout)
    if downloader.is_url(source):
        video_id = downloader.get_video_id(source)
    else:
        resolved = downloader.resolve_source(source)
        video_id = resolved.video_id
    ph = _params_hash(detail=detail, max_frames=max_frames, threshold=threshold, strategy=strategy, interval=interval, skip_frames=skip_frames, skip_audio=skip_audio, language=language, provider=provider or settings.vision_provider)
    if not force_refresh:
        cached = cache.get_result(video_id, ph)
        if cached:
            return VideoAnalysisResult(**cached)
    raw_meta = downloader.get_metadata(source)
    duration = raw_meta.get("duration", 0) or 0
    if settings.max_video_duration > 0 and duration > settings.max_video_duration:
        raise ValueError(f"Video duration ({duration}s) exceeds limit ({settings.max_video_duration}s). Use analyze_moment for specific time ranges, or set MAX_VIDEO_DURATION=0 for unlimited.")
    metadata = await get_metadata(source)
    transcript_result = None
    transcription_cost = 0.0
    if not skip_audio:
        try:
            transcript_result = await transcribe(source, language, settings)
            transcription_cost = (duration or 0) / 60.0 * 0.006
        except Exception as e:
            logger.warning(f"Transcription failed: {e}")
            warnings.append(Warning(code="TRANSCRIPTION_FAILED", message=str(e)))
    frame_results = []
    total_extracted = 0
    total_deduped = 0
    vision_cost = 0.0
    if not skip_frames:
        try:
            ef_result = await extract_frames(source=source, max_frames=max_frames, threshold=threshold, strategy=strategy, interval=interval, detail=detail, provider=provider, force_refresh=force_refresh, settings=settings)
            frame_results = ef_result.frames
            warnings.extend(ef_result.warnings)
            total_extracted = ef_result.total_raw_extracted
            total_deduped = ef_result.total_after_dedup
            vision_cost = ef_result.vision_cost_usd
        except Exception as e:
            logger.warning(f"Frame extraction failed: {e}")
            warnings.append(Warning(code="FRAME_EXTRACTION_FAILED", message=str(e)))
    timeline: list[TimelineEntry] = []
    if transcript_result:
        for seg in transcript_result.segments:
            timeline.append(TimelineEntry(type="transcript", start=seg.start, end=seg.end, content=seg.text))
    for fr in frame_results:
        timeline.append(TimelineEntry(type="frame", start=fr.timestamp, end=fr.timestamp, content=fr.description or "", frame_index=fr.index))
    if metadata.chapters:
        for ch in metadata.chapters:
            timeline.append(TimelineEntry(type="chapter", start=ch.start, end=ch.end, content=ch.title))
    timeline.sort(key=lambda e: e.start)
    vision_name = provider or settings.vision_provider
    result = VideoAnalysisResult(metadata=metadata, transcript=transcript_result, frames=frame_results, timeline=timeline, warnings=warnings, stats=Stats(total_frames_extracted=total_extracted, frames_after_dedup=total_deduped, frames_analyzed=len([f for f in frame_results if f.description]), vision_provider=vision_name, vision_model=settings.get_vision_model(), transcription_provider=settings.transcription_provider, transcription_model=settings.transcription_model, transcription_cost_usd=round(transcription_cost, 4), vision_cost_usd=round(vision_cost, 4), total_cost_usd=round(transcription_cost + vision_cost, 4), processing_time_seconds=round(time.time() - t_start, 2)))
    cache.store_result(video_id, ph, result.model_dump(), ttl=settings.cache_results_ttl)
    return result
