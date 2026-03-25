"""Pydantic models for VideoScan output schemas."""

from __future__ import annotations
from pydantic import BaseModel


class Chapter(BaseModel):
    title: str
    start: float
    end: float


class MetadataResult(BaseModel):
    title: str | None = None
    channel: str | None = None
    duration: float | None = None
    url: str | None = None
    platform: str | None = None
    chapters: list[Chapter] = []
    tags: list[str] = []
    view_count: int | None = None
    upload_date: str | None = None
    description: str | None = None
    thumbnail: str | None = None


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float


class TranscriptResult(BaseModel):
    language: str
    segments: list[TranscriptSegment]


class FrameResult(BaseModel):
    index: int
    timestamp: float
    strategy: str
    image_base64: str
    description: str | None = None
    ocr_text: str | None = None


class TimelineEntry(BaseModel):
    type: str
    start: float
    end: float
    content: str
    frame_index: int | None = None


class Warning(BaseModel):
    code: str
    message: str
    timestamp: float | None = None


class Stats(BaseModel):
    total_frames_extracted: int
    frames_after_dedup: int
    frames_analyzed: int
    vision_provider: str
    vision_model: str
    transcription_provider: str
    transcription_model: str
    transcription_cost_usd: float
    vision_cost_usd: float
    total_cost_usd: float
    processing_time_seconds: float


class VideoAnalysisResult(BaseModel):
    metadata: MetadataResult
    transcript: TranscriptResult | None = None
    frames: list[FrameResult] = []
    timeline: list[TimelineEntry] = []
    warnings: list[Warning] = []
    stats: Stats


class MomentResult(BaseModel):
    metadata: MetadataResult
    transcript: TranscriptResult | None = None
    frames: list[FrameResult] = []
    timeline: list[TimelineEntry] = []
    warnings: list[Warning] = []
    stats: Stats
    start: float
    end: float
