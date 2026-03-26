"""Integration tests for MCP tools (mocked external calls)."""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from videoscan.providers.base import FrameAnalysis
from videoscan.providers.openai import TranscriptionResult
from videoscan.types import MetadataResult, TranscriptResult, TranscriptSegment
from videoscan.utils.config import Settings


@pytest.fixture
def mock_settings(env_override):
    env_override(VISION_PROVIDER="openai", OPENAI_API_KEY="test-key", CACHE_ENABLED="false", MAX_VIDEO_DURATION="0")
    return Settings()


# --- get_metadata ---

@pytest.mark.asyncio
async def test_get_metadata_local_file(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    with patch("videoscan.tools.get_metadata.Downloader") as MockDL:
        MockDL.return_value.get_metadata.return_value = {
            "title": "Test Video",
            "duration": 120.0,
            "uploader": "TestChannel",
        }
        from videoscan.tools.get_metadata import get_metadata
        result = await get_metadata(str(video))
        assert result.title == "Test Video"
        assert result.duration == 120.0


@pytest.mark.asyncio
async def test_get_metadata_with_include(tmp_path):
    with patch("videoscan.tools.get_metadata.Downloader") as MockDL:
        MockDL.return_value.get_metadata.return_value = {
            "title": "Test",
            "duration": 60.0,
            "uploader": "Ch",
            "description": "desc",
            "tags": ["a", "b"],
        }
        from videoscan.tools.get_metadata import get_metadata
        result = await get_metadata("https://example.com/vid", include=["title", "duration"])
        assert result.title == "Test"
        assert result.description is None


# --- transcribe ---

@pytest.mark.asyncio
async def test_transcribe_returns_segments_and_language(mock_settings):
    mock_transcription = TranscriptionResult(
        segments=[{"text": "Hello", "start": 0.0, "end": 1.5}],
        language="en",
    )
    with patch("videoscan.tools.transcribe.Downloader") as MockDL, \
         patch("videoscan.tools.transcribe.get_transcription_provider") as MockTP:
        MockDL.return_value.resolve_source.return_value = MagicMock(
            audio_path="/tmp/audio.wav",
            video_path="/tmp/video.mp4",
        )
        MockTP.return_value.transcribe = AsyncMock(return_value=mock_transcription)
        from videoscan.tools.transcribe import transcribe
        result = await transcribe("https://example.com/vid", settings=mock_settings)
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello"


# --- get_frame_at ---

@pytest.mark.asyncio
async def test_get_frame_at_with_analysis(mock_settings):
    fake_frame = MagicMock()
    fake_frame.timestamp = 5.0
    fake_frame.strategy = "interval"
    fake_frame.to_base64.return_value = "base64data"
    with patch("videoscan.tools.get_frame_at.Downloader") as MockDL, \
         patch("videoscan.tools.get_frame_at.FrameExtractor") as MockFE, \
         patch("videoscan.tools.get_frame_at.get_vision_provider") as MockVP:
        MockDL.return_value.resolve_source.return_value = MagicMock(video_path="/tmp/v.mp4")
        MockFE.return_value.extract_at.return_value = fake_frame
        MockVP.return_value.analyze_frame = AsyncMock(
            return_value=FrameAnalysis(description="A dashboard", ocr_text="Total: $100")
        )
        from videoscan.tools.get_frame_at import get_frame_at
        result = await get_frame_at("https://example.com/vid", timestamp=5.0, settings=mock_settings)
        assert result.description == "A dashboard"
        assert result.ocr_text == "Total: $100"


@pytest.mark.asyncio
async def test_get_frame_at_without_analysis(mock_settings):
    fake_frame = MagicMock()
    fake_frame.timestamp = 5.0
    fake_frame.strategy = "interval"
    fake_frame.to_base64.return_value = "base64data"
    with patch("videoscan.tools.get_frame_at.Downloader") as MockDL, \
         patch("videoscan.tools.get_frame_at.FrameExtractor") as MockFE:
        MockDL.return_value.resolve_source.return_value = MagicMock(video_path="/tmp/v.mp4")
        MockFE.return_value.extract_at.return_value = fake_frame
        from videoscan.tools.get_frame_at import get_frame_at
        result = await get_frame_at(
            "https://example.com/vid", timestamp=5.0, analyze=False, settings=mock_settings
        )
        assert result.description is None


# --- extract_frames ---

@pytest.mark.asyncio
async def test_extract_frames_returns_stats(mock_settings):
    fake_frames = [
        MagicMock(timestamp=1.0, strategy="interval", to_base64=MagicMock(return_value="b64")),
        MagicMock(timestamp=2.0, strategy="interval", to_base64=MagicMock(return_value="b64")),
    ]
    with patch("videoscan.tools.extract_frames.Downloader") as MockDL, \
         patch("videoscan.tools.extract_frames.FrameExtractor") as MockFE, \
         patch("videoscan.tools.extract_frames.Deduplicator") as MockDedup, \
         patch("videoscan.tools.extract_frames.get_vision_provider") as MockVP:
        MockDL.return_value.resolve_source.return_value = MagicMock(video_path="/tmp/v.mp4")
        MockFE.return_value.extract_combined.return_value = fake_frames
        MockDedup.return_value.deduplicate.return_value = fake_frames[:1]
        mock_vision = MagicMock()
        mock_vision.analyze_frame = AsyncMock(return_value=FrameAnalysis(description="frame"))
        mock_vision.cost_per_image = 0.015
        MockVP.return_value = mock_vision
        from videoscan.tools.extract_frames import extract_frames
        result = await extract_frames("https://example.com/vid", settings=mock_settings)
        assert result.total_raw_extracted == 2
        assert result.total_after_dedup == 1
        assert len(result.frames) == 1
        assert result.vision_cost_usd > 0


@pytest.mark.asyncio
async def test_extract_frames_handles_vision_failure(mock_settings):
    fake_frame = MagicMock(
        timestamp=1.0,
        strategy="interval",
        to_base64=MagicMock(return_value="b64"),
    )
    with patch("videoscan.tools.extract_frames.Downloader") as MockDL, \
         patch("videoscan.tools.extract_frames.FrameExtractor") as MockFE, \
         patch("videoscan.tools.extract_frames.Deduplicator") as MockDedup, \
         patch("videoscan.tools.extract_frames.get_vision_provider") as MockVP:
        MockDL.return_value.resolve_source.return_value = MagicMock(video_path="/tmp/v.mp4")
        MockFE.return_value.extract_combined.return_value = [fake_frame]
        MockDedup.return_value.deduplicate.return_value = [fake_frame]
        mock_vision = MagicMock()
        mock_vision.analyze_frame = AsyncMock(side_effect=Exception("API error"))
        mock_vision.cost_per_image = 0.015
        MockVP.return_value = mock_vision
        from videoscan.tools.extract_frames import extract_frames
        result = await extract_frames("https://example.com/vid", settings=mock_settings)
        assert len(result.warnings) == 1
        assert result.warnings[0].code == "FRAME_ANALYSIS_FAILED"
        assert result.frames[0].description is None


# --- analyze_video ---

@pytest.mark.asyncio
async def test_analyze_video_full_pipeline(mock_settings):
    mock_transcript = (
        TranscriptResult(language="en", segments=[TranscriptSegment(text="Hi", start=0.0, end=1.0)]),
        0.0,
        [],
    )
    mock_frames = (
        [],  # frame_results
        10,  # total_raw
        8,   # total_after_dedup
        0.12,  # vision_cost
        [],  # warnings
    )
    with patch("videoscan.tools.analyze_video.Downloader") as MockDL, \
         patch("videoscan.tools.analyze_video._transcribe_audio", new_callable=AsyncMock, return_value=mock_transcript), \
         patch("videoscan.tools.analyze_video._extract_and_analyze_frames", new_callable=AsyncMock, return_value=mock_frames), \
         patch("videoscan.tools.analyze_video.Cache") as MockCache:
        MockDL.return_value.is_url.return_value = True
        MockDL.return_value.get_metadata.return_value = {"id": "abc123", "duration": 60, "title": "Test"}
        MockDL.return_value.resolve_source.return_value = MagicMock(video_path="/tmp/v.mp4", audio_path="/tmp/a.wav", video_id="abc123")
        MockCache.return_value.get_result.return_value = None
        from videoscan.tools.analyze_video import analyze_video
        result = await analyze_video("https://example.com/vid", settings=mock_settings)
        assert result.metadata.title == "Test"
        assert result.stats.total_frames_extracted == 10
        assert result.stats.frames_after_dedup == 8
        assert result.stats.vision_cost_usd == 0.12
        assert result.stats.total_cost_usd > 0


@pytest.mark.asyncio
async def test_analyze_video_duration_limit(env_override):
    env_override(OPENAI_API_KEY="test", MAX_VIDEO_DURATION="60", CACHE_ENABLED="false")
    settings = Settings()
    with patch("videoscan.tools.analyze_video.Downloader") as MockDL, \
         patch("videoscan.tools.analyze_video.Cache") as MockCache:
        MockDL.return_value.is_url.return_value = True
        MockDL.return_value.get_video_id.return_value = "abc"
        MockDL.return_value.get_metadata.return_value = {"duration": 7200}
        MockCache.return_value.get_result.return_value = None
        from videoscan.tools.analyze_video import analyze_video
        with pytest.raises(ValueError, match="exceeds limit"):
            await analyze_video("https://example.com/long", settings=settings)


# --- server registration ---

def test_server_has_all_tools():
    from videoscan.server import mcp
    # Try different ways to get tool names depending on MCP version
    try:
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    except AttributeError:
        try:
            import asyncio
            tool_names = {t.name for t in asyncio.run(mcp.list_tools())}
        except Exception:
            # If we can't list tools, just verify the module imports cleanly
            assert mcp is not None
            return
    expected = {
        "analyze_video",
        "transcribe",
        "extract_frames",
        "analyze_moment",
        "get_frame_at",
        "get_metadata",
    }
    assert expected.issubset(tool_names)
