"""Tests for video downloader."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from videoscan.core.downloader import Downloader, DownloadResult

def test_local_file_skips_download(tmp_path: Path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake video data")
    downloader = Downloader()
    result = downloader.resolve_source(str(video))
    assert result.is_local is True
    assert result.video_path == str(video)

def test_local_file_not_found_raises():
    downloader = Downloader()
    with pytest.raises(FileNotFoundError):
        downloader.resolve_source("/nonexistent/video.mp4")

def test_url_detected_as_remote():
    downloader = Downloader()
    assert downloader.is_url("https://youtube.com/watch?v=abc123") is True
    assert downloader.is_url("C:/Users/video.mp4") is False
    assert downloader.is_url("/home/user/video.mp4") is False

def test_extract_video_id():
    downloader = Downloader()
    mock_info = {"id": "dQw4w9WgXcQ", "title": "Test"}
    with patch.object(downloader, "_extract_info", return_value=mock_info):
        vid = downloader.get_video_id("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert vid == "dQw4w9WgXcQ"

def test_download_result_model():
    result = DownloadResult(video_path="/tmp/video.mp4", audio_path="/tmp/audio.wav", video_id="abc123", is_local=False)
    assert result.video_path == "/tmp/video.mp4"
    assert result.is_local is False
