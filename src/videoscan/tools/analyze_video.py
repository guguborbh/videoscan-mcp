"""analyze_video MCP tool — full pipeline combining all analysis."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time

from videoscan.core.cache import Cache
from videoscan.core.deduplicator import Deduplicator
from videoscan.core.downloader import Downloader
from videoscan.core.frame_extractor import FrameExtractor
from videoscan.providers.base import get_transcription_provider, get_vision_provider
from videoscan.types import (
    FrameResult,
    MetadataResult,
    Stats,
    TimelineEntry,
    TranscriptResult,
    TranscriptSegment,
    VideoAnalysisResult,
    Warning,
)
from videoscan.utils.config import Settings

logger = logging.getLogger(__name__)


def _params_hash(**kwargs) -> str:
    return hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:12]


# Sentinel to detect when user didn't explicitly set a parameter
_AUTO = -1


def _auto_tune_params(
    duration: float,
    max_frames: int,
    interval: int,
    strategy: str,
    detail: str,
) -> tuple[int, int, str, str]:
    """Auto-tune frame extraction params based on video duration.

    Only overrides params that the user didn't explicitly set (value == _AUTO).
    Returns (max_frames, interval, strategy, detail).
    """
    if duration <= 0:
        # Unknown duration — use conservative defaults
        return (
            max_frames if max_frames != _AUTO else 20,
            interval if interval != _AUTO else 5,
            strategy,
            detail,
        )

    dur_min = duration / 60.0

    if dur_min <= 2:
        # Short video: dense, capture everything
        auto_max = int(duration)  # ~1 frame/second
        auto_interval = 1
        auto_strategy = "combined"
        auto_detail = detail if detail != "standard" else "detailed"
    elif dur_min <= 10:
        # Medium: balanced
        auto_max = 40
        auto_interval = 3
        auto_strategy = "combined"
        auto_detail = detail
    elif dur_min <= 30:
        # Long: efficient
        auto_max = 30
        auto_interval = 10
        auto_strategy = "combined"
        auto_detail = detail
    elif dur_min <= 60:
        # Very long: economical
        auto_max = 30
        auto_interval = 20
        auto_strategy = "combined"
        auto_detail = detail if detail != "standard" else "brief"
    else:
        # 60min+: light
        auto_max = 20
        auto_interval = 30
        auto_strategy = "scene"
        auto_detail = detail if detail != "standard" else "brief"

    # Only apply auto values where user didn't specify
    final_max = max_frames if max_frames != _AUTO else auto_max
    final_interval = interval if interval != _AUTO else auto_interval
    final_strategy = strategy  # strategy is always explicit (has a default of "combined")
    final_detail = auto_detail  # detail auto-adjusts unless user explicitly chose non-standard

    logger.info(
        f"Auto-tune ({dur_min:.1f}min): max_frames={final_max}, "
        f"interval={final_interval}s, strategy={final_strategy}, detail={final_detail}"
    )

    return final_max, final_interval, final_strategy, final_detail


def _build_metadata(raw: dict) -> MetadataResult:
    """Build MetadataResult from raw yt-dlp info dict."""
    return MetadataResult(
        title=raw.get("title"),
        channel=raw.get("uploader") or raw.get("channel"),
        duration=raw.get("duration"),
        url=raw.get("webpage_url") or raw.get("url"),
        platform=(raw.get("extractor_key", "") or raw.get("extractor", "")).lower(),
        chapters=[
            {"title": ch.get("title", ""), "start": ch.get("start_time", 0), "end": ch.get("end_time", 0)}
            for ch in (raw.get("chapters") or [])
        ],
        tags=raw.get("tags") or [],
        view_count=raw.get("view_count"),
        upload_date=raw.get("upload_date"),
        description=raw.get("description"),
        thumbnail=raw.get("thumbnail"),
    )


async def _transcribe_audio(
    audio_path: str, language: str, settings: Settings
) -> tuple[TranscriptResult | None, float, list[Warning]]:
    """Transcribe audio file. Returns (result, cost, warnings)."""
    warnings: list[Warning] = []
    try:
        t0 = time.time()
        provider = get_transcription_provider(
            settings.transcription_provider,
            settings.get_api_key("openai"),
            settings.transcription_model,
        )
        transcription = await provider.transcribe(audio_path, language)
        segments = [
            TranscriptSegment(text=seg["text"], start=seg["start"], end=seg["end"])
            for seg in transcription.segments
        ]
        result = TranscriptResult(language=transcription.language, segments=segments)
        logger.info(f"Transcription done in {time.time() - t0:.1f}s ({len(segments)} segments)")
        return result, 0.0, warnings  # cost calculated by caller from duration
    except Exception as e:
        logger.warning(f"Transcription failed: {e}")
        warnings.append(Warning(code="TRANSCRIPTION_FAILED", message=str(e)))
        return None, 0.0, warnings


async def _extract_and_analyze_frames(
    video_path: str,
    max_frames: int,
    threshold: float,
    strategy: str,
    interval: int,
    detail: str,
    provider: str | None,
    settings: Settings,
) -> tuple[list[FrameResult], int, int, float, list[Warning]]:
    """Extract frames and analyze with vision AI. Returns (frames, raw_count, dedup_count, cost, warnings)."""
    warnings: list[Warning] = []
    extractor = FrameExtractor()

    t0 = time.time()
    if strategy == "scene":
        raw_frames = extractor.extract_scene_changes(video_path, threshold, max_frames)
    elif strategy == "interval":
        raw_frames = extractor.extract_interval(video_path, float(interval), max_frames)
    else:
        raw_frames = extractor.extract_combined(video_path, threshold, float(interval), max_frames)

    total_raw = len(raw_frames)
    logger.info(f"Extracted {total_raw} raw frames in {time.time() - t0:.1f}s")

    # Deduplicate
    if raw_frames:
        dedup = Deduplicator()
        raw_frames = dedup.deduplicate(raw_frames)
    total_after_dedup = len(raw_frames)
    logger.info(f"After dedup: {total_after_dedup} frames")

    # Limit
    effective_max = max_frames
    if settings.max_analyzed_frames > 0:
        effective_max = min(effective_max, settings.max_analyzed_frames)
    raw_frames = raw_frames[:effective_max]

    # Vision analysis
    vision_provider_name = provider or settings.vision_provider
    vision = get_vision_provider(
        vision_provider_name,
        settings.get_api_key(vision_provider_name),
        settings.get_vision_model(),
    )
    cost_per_image = getattr(vision, "cost_per_image", 0.015)
    semaphore = asyncio.Semaphore(settings.vision_concurrency)

    async def analyze_one(idx: int, frame):
        b64 = frame.to_base64()
        async with semaphore:
            try:
                analysis = await asyncio.wait_for(
                    vision.analyze_frame(b64, detail),
                    timeout=settings.frame_analysis_timeout if settings.frame_analysis_timeout > 0 else None,
                )
                return FrameResult(
                    index=idx, timestamp=frame.timestamp, strategy=frame.strategy,
                    image_base64=b64, description=analysis.description, ocr_text=analysis.ocr_text,
                )
            except Exception as e:
                logger.warning(f"Frame analysis failed at {frame.timestamp}s: {e}")
                warnings.append(Warning(
                    code="FRAME_ANALYSIS_FAILED",
                    message=f"Vision AI failed: {e}",
                    timestamp=frame.timestamp,
                ))
                return FrameResult(
                    index=idx, timestamp=frame.timestamp, strategy=frame.strategy, image_base64=b64,
                )

    t0 = time.time()
    tasks = [analyze_one(i, f) for i, f in enumerate(raw_frames)]
    frame_results = list(await asyncio.gather(*tasks))
    analyzed_count = len([f for f in frame_results if f.description])
    vision_cost = analyzed_count * cost_per_image
    logger.info(f"Vision analysis done in {time.time() - t0:.1f}s ({analyzed_count}/{len(raw_frames)} frames)")

    return frame_results, total_raw, total_after_dedup, round(vision_cost, 4), warnings


async def analyze_video(
    source: str,
    detail: str = "standard",
    max_frames: int = _AUTO,
    threshold: float = 0.3,
    strategy: str = "combined",
    interval: int = _AUTO,
    skip_frames: bool = False,
    skip_audio: bool = False,
    language: str = "auto",
    provider: str | None = None,
    force_refresh: bool = False,
    settings: Settings | None = None,
) -> VideoAnalysisResult:
    settings = settings or Settings()
    t_start = time.time()
    warnings: list[Warning] = []

    # --- STEP 1: Single metadata fetch + cache check ---
    downloader = Downloader(timeout=settings.download_timeout)
    t0 = time.time()

    if downloader.is_url(source):
        raw_meta = downloader.get_metadata(source)  # single yt-dlp call for both metadata + video_id
        video_id = raw_meta.get("id", hashlib.md5(source.encode()).hexdigest()[:16])
    else:
        dl_result = downloader.resolve_source(source)
        video_id = dl_result.video_id
        raw_meta = {"id": video_id, "filepath": source, "duration": None}

    logger.info(f"Metadata fetch done in {time.time() - t0:.1f}s")

    duration = raw_meta.get("duration", 0) or 0
    if settings.max_video_duration > 0 and duration > settings.max_video_duration:
        raise ValueError(
            f"Video duration ({duration}s) exceeds limit ({settings.max_video_duration}s). "
            "Use analyze_moment for specific time ranges, or set MAX_VIDEO_DURATION=0 for unlimited."
        )

    # --- Auto-tune params based on video duration ---
    max_frames, interval, strategy, detail = _auto_tune_params(
        duration, max_frames, interval, strategy, detail,
    )

    # Check cache
    cache = Cache(
        cache_dir=str(settings.get_cache_path()),
        max_size_gb=settings.cache_max_size_gb,
        enabled=settings.cache_enabled,
    )
    ph = _params_hash(
        detail=detail, max_frames=max_frames, threshold=threshold,
        strategy=strategy, interval=interval, skip_frames=skip_frames,
        skip_audio=skip_audio, language=language, provider=provider or settings.vision_provider,
    )
    if not force_refresh:
        cached = cache.get_result(video_id, ph)
        if cached:
            logger.info("Cache hit — returning cached result")
            return VideoAnalysisResult(**cached)

    # Build metadata from the already-fetched raw info
    metadata = _build_metadata(raw_meta)

    # --- STEP 2: Single download (only for URLs) ---
    if downloader.is_url(source):
        t0 = time.time()
        dl_result = downloader.download_with_info(source, raw_meta)  # reuse metadata, no extra API call
        logger.info(f"Download done in {time.time() - t0:.1f}s")
    # For local files, dl_result is already set above

    local_video_path = dl_result.video_path
    local_audio_path = dl_result.audio_path or dl_result.video_path

    # --- STEP 3: Transcription + Frame extraction IN PARALLEL ---
    transcript_result = None
    transcription_cost = 0.0
    frame_results = []
    total_extracted = 0
    total_deduped = 0
    vision_cost = 0.0

    parallel_tasks = []

    if not skip_audio:
        parallel_tasks.append(("transcribe", _transcribe_audio(local_audio_path, language, settings)))

    if not skip_frames:
        parallel_tasks.append(("frames", _extract_and_analyze_frames(
            local_video_path, max_frames, threshold, strategy, interval, detail, provider, settings,
        )))

    if parallel_tasks:
        t0 = time.time()
        results = await asyncio.gather(*[task for _, task in parallel_tasks], return_exceptions=True)
        logger.info(f"Parallel processing done in {time.time() - t0:.1f}s")

        for (name, _), result in zip(parallel_tasks, results):
            if isinstance(result, Exception):
                logger.warning(f"{name} failed: {result}")
                warnings.append(Warning(code=f"{name.upper()}_FAILED", message=str(result)))
                continue

            if name == "transcribe":
                transcript_result, _, transcribe_warnings = result
                warnings.extend(transcribe_warnings)
                if duration:
                    transcription_cost = (duration / 60.0) * 0.006
            elif name == "frames":
                frame_results, total_extracted, total_deduped, vision_cost, frame_warnings = result
                warnings.extend(frame_warnings)

    # --- STEP 4: Build timeline ---
    timeline: list[TimelineEntry] = []
    if transcript_result:
        for seg in transcript_result.segments:
            timeline.append(TimelineEntry(type="transcript", start=seg.start, end=seg.end, content=seg.text))
    for fr in frame_results:
        timeline.append(TimelineEntry(
            type="frame", start=fr.timestamp, end=fr.timestamp,
            content=fr.description or "", frame_index=fr.index,
        ))
    if metadata.chapters:
        for ch in metadata.chapters:
            timeline.append(TimelineEntry(type="chapter", start=ch.start, end=ch.end, content=ch.title))
    timeline.sort(key=lambda e: e.start)

    # --- STEP 5: Build result ---
    vision_name = provider or settings.vision_provider
    result = VideoAnalysisResult(
        metadata=metadata,
        transcript=transcript_result,
        frames=frame_results,
        timeline=timeline,
        warnings=warnings,
        stats=Stats(
            total_frames_extracted=total_extracted,
            frames_after_dedup=total_deduped,
            frames_analyzed=len([f for f in frame_results if f.description]),
            vision_provider=vision_name,
            vision_model=settings.get_vision_model(),
            transcription_provider=settings.transcription_provider,
            transcription_model=settings.transcription_model,
            transcription_cost_usd=round(transcription_cost, 4),
            vision_cost_usd=round(vision_cost, 4),
            total_cost_usd=round(transcription_cost + vision_cost, 4),
            processing_time_seconds=round(time.time() - t_start, 2),
        ),
    )

    total_time = round(time.time() - t_start, 2)
    logger.info(f"Total analyze_video time: {total_time}s")

    cache.store_result(video_id, ph, result.model_dump(), ttl=settings.cache_results_ttl)
    return result
