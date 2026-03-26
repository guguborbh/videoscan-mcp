"""VideoScan MCP server — tool registration and entry point."""
from __future__ import annotations
import logging
from mcp.server.fastmcp import FastMCP
from videoscan.utils.config import Settings
from videoscan.utils.deps import check_dependencies

logger = logging.getLogger(__name__)

mcp = FastMCP("VideoScan", instructions="Comprehensive video analysis — transcription, AI-powered visual frame analysis, and metadata extraction")

@mcp.tool()
async def analyze_video(source: str, detail: str = "standard", max_frames: int = -1, threshold: float = 0.3, strategy: str = "combined", interval: int = -1, skip_frames: bool = False, skip_audio: bool = False, language: str = "auto", provider: str | None = None, force_refresh: bool = False) -> dict:
    """Analyze a video comprehensively — transcription + AI visual frame analysis + metadata.

    Frame extraction auto-tunes based on video duration when max_frames and interval are not specified:
    - Under 2 min: dense (1 frame/sec), detailed analysis
    - 2-10 min: balanced (~40 frames, 3s interval)
    - 10-30 min: efficient (~30 frames, 10s interval)
    - 30-60 min: economical (~30 frames, 20s interval)
    - Over 60 min: light (scene detection only, ~20 frames)

    Args:
        source: URL (YouTube, Vimeo, 1000+ sites) or local file path
        detail: Vision analysis level — "brief", "standard", or "detailed"
        max_frames: Maximum frames to analyze. Set to -1 (default) for auto-tuning based on video duration
        threshold: Scene change sensitivity (0.0-1.0)
        strategy: Frame extraction — "scene", "interval", or "combined"
        interval: Seconds between frames. Set to -1 (default) for auto-tuning based on video duration
        skip_frames: Skip visual analysis, transcription only
        skip_audio: Skip transcription, frames only
        language: Preferred transcription language or "auto"
        provider: Override vision provider ("openai", "anthropic", "google")
        force_refresh: Bypass cache
    """
    from videoscan.tools.analyze_video import analyze_video as _analyze
    result = await _analyze(source=source, detail=detail, max_frames=max_frames, threshold=threshold, strategy=strategy, interval=interval, skip_frames=skip_frames, skip_audio=skip_audio, language=language, provider=provider, force_refresh=force_refresh)
    return result.model_dump()

@mcp.tool()
async def transcribe(source: str, language: str = "auto") -> dict:
    """Transcribe video/audio to text with timestamps.
    Args:
        source: URL or local file path
        language: Preferred language or "auto" for detection
    """
    from videoscan.tools.transcribe import transcribe as _transcribe
    result = await _transcribe(source=source, language=language)
    return result.model_dump()

@mcp.tool()
async def extract_frames(source: str, max_frames: int = 30, threshold: float = 0.3, strategy: str = "combined", interval: int = 5, detail: str = "standard", deduplicate: bool = True, provider: str | None = None, force_refresh: bool = False) -> dict:
    """Extract frames from video and analyze them with AI vision.
    Args:
        source: URL or local file path
        max_frames: Maximum frames to extract (1-100)
        threshold: Scene change sensitivity (0.0-1.0)
        strategy: "scene", "interval", or "combined"
        interval: Seconds between frames in interval mode
        detail: Vision analysis level
        deduplicate: Remove near-duplicate frames via dHash
        provider: Override vision provider
        force_refresh: Bypass cache
    """
    from videoscan.tools.extract_frames import extract_frames as _extract
    result = await _extract(source=source, max_frames=max_frames, threshold=threshold, strategy=strategy, interval=interval, detail=detail, deduplicate=deduplicate, provider=provider, force_refresh=force_refresh)
    return {"frames": [f.model_dump() for f in result.frames], "warnings": [w.model_dump() for w in result.warnings], "total_raw_extracted": result.total_raw_extracted, "total_after_dedup": result.total_after_dedup, "vision_cost_usd": result.vision_cost_usd}

@mcp.tool()
async def analyze_moment(source: str, start: float, end: float, dense: bool = True, detail: str = "detailed", provider: str | None = None, force_refresh: bool = False) -> dict:
    """Deep-dive analysis on a specific time range of a video.
    Args:
        source: URL or local file path
        start: Start time in seconds
        end: End time in seconds
        dense: Extract 1 frame per second in the range
        detail: Vision analysis level
        provider: Override vision provider
        force_refresh: Bypass cache
    """
    from videoscan.tools.analyze_moment import analyze_moment as _analyze
    result = await _analyze(source=source, start=start, end=end, dense=dense, detail=detail, provider=provider, force_refresh=force_refresh)
    return result.model_dump()

@mcp.tool()
async def get_frame_at(source: str, timestamp: float, analyze: bool = True, provider: str | None = None, force_refresh: bool = False) -> dict:
    """Get a single frame at a specific timestamp, optionally analyzed by AI.
    Args:
        source: URL or local file path
        timestamp: Time in seconds
        analyze: Run AI vision analysis on the frame
        provider: Override vision provider
        force_refresh: Bypass cache
    """
    from videoscan.tools.get_frame_at import get_frame_at as _get_frame
    result = await _get_frame(source=source, timestamp=timestamp, analyze=analyze, provider=provider, force_refresh=force_refresh)
    return result.model_dump()

@mcp.tool()
async def get_metadata(source: str, include: list[str] | None = None) -> dict:
    """Get video metadata without downloading the full video.
    Args:
        source: URL or local file path
        include: Specific fields — "title", "duration", "channel", "description", "thumbnail", "chapters", "tags", "view_count". Returns all if omitted.
    """
    from videoscan.tools.get_metadata import get_metadata as _get_meta
    result = await _get_meta(source=source, include=include)
    return result.model_dump()

def main():
    """Entry point for the VideoScan MCP server."""
    logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

    # Pre-import heavy modules BEFORE MCP stdio transport starts.
    # cv2/numpy can deadlock if first imported after stdio is captured by MCP.
    import cv2  # noqa: F401
    import numpy  # noqa: F401

    # Also pre-import all tool modules so they're ready when called
    import videoscan.tools.analyze_video  # noqa: F401
    import videoscan.tools.extract_frames  # noqa: F401
    import videoscan.tools.analyze_moment  # noqa: F401
    import videoscan.tools.get_frame_at  # noqa: F401
    import videoscan.tools.get_metadata  # noqa: F401
    import videoscan.tools.transcribe  # noqa: F401

    deps = check_dependencies()
    for name, status in deps.items():
        if status.available:
            logger.info(f"{name}: OK ({status.path or 'available'})")
        else:
            logger.warning(f"{name}: {status.message}")
    settings = Settings()
    key = settings.get_api_key()
    if not key:
        logger.error(f"No API key found for provider '{settings.vision_provider}'. Set the appropriate key in .env (e.g., OPENAI_API_KEY).")
    mcp.run()

if __name__ == "__main__":
    main()
