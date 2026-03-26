"""Tests for frame extraction."""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path
from videoscan.core.frame_extractor import FrameExtractor, ExtractedFrame

@pytest.fixture
def sample_video(tmp_path: Path) -> str:
    import cv2
    video_path = str(tmp_path / "test.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(video_path, fourcc, 10, (320, 240))
    for color in [(0, 0, 255), (0, 255, 0), (255, 0, 0)]:
        frame = np.full((240, 320, 3), color, dtype=np.uint8)
        for _ in range(10):
            out.write(frame)
    out.release()
    return video_path

def test_interval_extraction(sample_video):
    extractor = FrameExtractor()
    frames = extractor.extract_interval(sample_video, interval=1, max_frames=10)
    assert len(frames) > 0
    assert all(isinstance(f, ExtractedFrame) for f in frames)
    assert all(f.timestamp >= 0 for f in frames)

def test_frame_to_base64(sample_video):
    extractor = FrameExtractor()
    frames = extractor.extract_interval(sample_video, interval=1, max_frames=2)
    assert len(frames) > 0
    b64 = frames[0].to_base64(max_width=1280, quality=85)
    assert isinstance(b64, str)
    assert len(b64) > 0

def test_max_frames_respected(sample_video):
    extractor = FrameExtractor()
    frames = extractor.extract_interval(sample_video, interval=0.5, max_frames=3)
    assert len(frames) <= 3

def test_extracted_frame_dataclass():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    frame = ExtractedFrame(timestamp=1.5, image=img, strategy="interval")
    assert frame.timestamp == 1.5
    assert frame.strategy == "interval"
