"""Tests for frame deduplication."""
import numpy as np
import pytest
from videoscan.core.deduplicator import Deduplicator
from videoscan.core.frame_extractor import ExtractedFrame

def _make_frame(timestamp: float, color: tuple = (0, 0, 0)) -> ExtractedFrame:
    img = np.full((240, 320, 3), color, dtype=np.uint8)
    return ExtractedFrame(timestamp=timestamp, image=img, strategy="interval")

def test_identical_frames_deduplicated():
    dedup = Deduplicator()
    frames = [_make_frame(0.0, (255, 0, 0)), _make_frame(1.0, (255, 0, 0))]
    result = dedup.deduplicate(frames)
    assert len(result) == 1

def test_different_frames_kept():
    dedup = Deduplicator()
    frames = [_make_frame(0.0, (255, 0, 0)), _make_frame(1.0, (0, 255, 0))]
    result = dedup.deduplicate(frames)
    assert len(result) == 2

def test_empty_input():
    dedup = Deduplicator()
    assert dedup.deduplicate([]) == []

def test_single_frame():
    dedup = Deduplicator()
    frames = [_make_frame(0.0)]
    result = dedup.deduplicate(frames)
    assert len(result) == 1

def test_custom_threshold():
    dedup = Deduplicator(hamming_threshold=0)
    img1 = np.zeros((240, 320, 3), dtype=np.uint8)
    img1[:, 160:] = 255
    img2 = np.zeros((240, 320, 3), dtype=np.uint8)
    img2[:120, :] = 255
    f1 = ExtractedFrame(timestamp=0.0, image=img1, strategy="interval")
    f2 = ExtractedFrame(timestamp=1.0, image=img2, strategy="interval")
    result = dedup.deduplicate([f1, f2])
    assert len(result) == 2
