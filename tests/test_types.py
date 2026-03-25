"""Tests for Pydantic output models."""

from videoscan.types import (
    Chapter, FrameResult, MetadataResult, MomentResult, Stats,
    TimelineEntry, TranscriptResult, TranscriptSegment, VideoAnalysisResult, Warning,
)

def test_transcript_segment_validates():
    seg = TranscriptSegment(text="Hello world", start=0.0, end=2.5)
    assert seg.text == "Hello world"
    assert seg.start == 0.0
    assert seg.end == 2.5

def test_frame_result_validates():
    frame = FrameResult(index=0, timestamp=12.4, strategy="scene_change", image_base64="abc123", description="A dashboard", ocr_text="Total: $100")
    assert frame.index == 0
    assert frame.strategy == "scene_change"

def test_frame_result_nullable_fields():
    frame = FrameResult(index=0, timestamp=5.0, strategy="interval", image_base64="abc")
    assert frame.description is None
    assert frame.ocr_text is None

def test_timeline_entry_frame_has_frame_index():
    entry = TimelineEntry(type="frame", start=12.4, end=12.4, content="Dashboard", frame_index=0)
    assert entry.frame_index == 0

def test_timeline_entry_transcript_no_frame_index():
    entry = TimelineEntry(type="transcript", start=0.0, end=3.2, content="Hello")
    assert entry.frame_index is None

def test_warning_validates():
    w = Warning(code="FRAME_ANALYSIS_FAILED", message="Failed at 12.4s", timestamp=12.4)
    assert w.code == "FRAME_ANALYSIS_FAILED"
    w2 = Warning(code="FFMPEG_FALLBACK", message="No scene detect")
    assert w2.timestamp is None

def test_stats_has_provider_info():
    stats = Stats(total_frames_extracted=47, frames_after_dedup=28, frames_analyzed=28, vision_provider="openai", vision_model="gpt-4o", transcription_provider="openai", transcription_model="whisper-1", transcription_cost_usd=0.037, vision_cost_usd=0.42, total_cost_usd=0.457, processing_time_seconds=34.2)
    assert stats.vision_provider == "openai"
    assert stats.total_cost_usd == 0.457

def test_video_analysis_result_full():
    result = VideoAnalysisResult(
        metadata=MetadataResult(title="Test", duration=60.0),
        transcript=TranscriptResult(language="en", segments=[]),
        frames=[], timeline=[], warnings=[],
        stats=Stats(total_frames_extracted=0, frames_after_dedup=0, frames_analyzed=0, vision_provider="openai", vision_model="gpt-4o", transcription_provider="openai", transcription_model="whisper-1", transcription_cost_usd=0.0, vision_cost_usd=0.0, total_cost_usd=0.0, processing_time_seconds=1.0),
    )
    assert result.metadata.title == "Test"
