"""Tests for dependency checker."""
from unittest.mock import patch
from videoscan.utils.deps import check_ffmpeg, check_ytdlp, check_dependencies

def test_check_ffmpeg_when_available():
    with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
        result = check_ffmpeg()
        assert result.available is True
        assert result.path == "/usr/bin/ffmpeg"

def test_check_ffmpeg_when_missing():
    with patch("shutil.which", return_value=None):
        result = check_ffmpeg()
        assert result.available is False
        assert "install" in result.message.lower()

def test_check_ytdlp_when_available():
    with patch("shutil.which", return_value="/usr/bin/yt-dlp"):
        result = check_ytdlp()
        assert result.available is True

def test_check_dependencies_returns_report():
    with patch("shutil.which", return_value="/usr/bin/mock"):
        report = check_dependencies()
        assert "ffmpeg" in report
        assert "yt-dlp" in report
