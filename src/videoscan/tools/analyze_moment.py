"""analyze_moment MCP tool — deep-dive on a specific time range."""
from __future__ import annotations
import asyncio, logging, time
from videoscan.core.downloader import Downloader
from videoscan.core.frame_extractor import FrameExtractor
from videoscan.providers.base import get_transcription_provider, get_vision_provider
from videoscan.types import FrameResult, MetadataResult, MomentResult, Stats, TimelineEntry, TranscriptResult, TranscriptSegment, Warning
from videoscan.utils.config import Settings

logger = logging.getLogger(__name__)

async def analyze_moment(source: str, start: float, end: float, dense: bool = True, detail: str = "detailed", provider: str | None = None, force_refresh: bool = False, settings: Settings | None = None) -> MomentResult:
    settings = settings or Settings()
    t_start = time.time()
    warnings: list[Warning] = []
    downloader = Downloader(timeout=settings.download_timeout)
    dl_result = await asyncio.to_thread(downloader.resolve_source, source)
    extractor = FrameExtractor()
    if dense:
        raw_frames = await asyncio.to_thread(extractor.extract_dense, dl_result.video_path, start, end)
    else:
        raw_frames = await asyncio.to_thread(extractor.extract_interval, dl_result.video_path, 2.0, 30)
        raw_frames = [f for f in raw_frames if start <= f.timestamp <= end]
    vision_name = provider or settings.vision_provider
    vision = get_vision_provider(vision_name, settings.get_api_key(vision_name), settings.get_vision_model())
    cost_per_image = getattr(vision, "cost_per_image", 0.015)
    semaphore = asyncio.Semaphore(settings.vision_concurrency)

    async def analyze_one(idx, frame):
        b64 = frame.to_base64()
        async with semaphore:
            try:
                analysis = await vision.analyze_frame(b64, detail)
                return FrameResult(index=idx, timestamp=frame.timestamp, strategy=frame.strategy, image_base64=b64, description=analysis.description, ocr_text=analysis.ocr_text)
            except Exception as e:
                warnings.append(Warning(code="FRAME_ANALYSIS_FAILED", message=str(e), timestamp=frame.timestamp))
                return FrameResult(index=idx, timestamp=frame.timestamp, strategy=frame.strategy, image_base64=b64)

    frame_results = await asyncio.gather(*[analyze_one(i, f) for i, f in enumerate(raw_frames)])
    transcript_result = None
    transcription_cost = 0.0
    if dl_result.audio_path:
        try:
            tp = get_transcription_provider(settings.transcription_provider, settings.get_api_key("openai"), settings.transcription_model)
            transcription = await tp.transcribe(dl_result.audio_path, "auto")
            range_segments = [s for s in transcription.segments if s["end"] >= start and s["start"] <= end]
            transcript_result = TranscriptResult(language=transcription.language, segments=[TranscriptSegment(**s) for s in range_segments])
            transcription_cost = ((end - start) / 60.0) * tp.get_cost_per_minute()
        except Exception as e:
            warnings.append(Warning(code="TRANSCRIPTION_FAILED", message=str(e)))
    timeline: list[TimelineEntry] = []
    if transcript_result:
        for seg in transcript_result.segments:
            timeline.append(TimelineEntry(type="transcript", start=seg.start, end=seg.end, content=seg.text))
    for fr in frame_results:
        timeline.append(TimelineEntry(type="frame", start=fr.timestamp, end=fr.timestamp, content=fr.description or "", frame_index=fr.index))
    timeline.sort(key=lambda e: e.start)
    analyzed_count = len([f for f in frame_results if f.description])
    vision_cost = analyzed_count * cost_per_image
    return MomentResult(metadata=MetadataResult(duration=end - start), transcript=transcript_result, frames=list(frame_results), timeline=timeline, warnings=warnings, stats=Stats(total_frames_extracted=len(raw_frames), frames_after_dedup=len(raw_frames), frames_analyzed=analyzed_count, vision_provider=vision_name, vision_model=settings.get_vision_model(), transcription_provider=settings.transcription_provider, transcription_model=settings.transcription_model, transcription_cost_usd=round(transcription_cost, 4), vision_cost_usd=round(vision_cost, 4), total_cost_usd=round(transcription_cost + vision_cost, 4), processing_time_seconds=round(time.time() - t_start, 2)), start=start, end=end)
