"""Video download and source resolution using yt-dlp."""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    video_path: str
    audio_path: str | None = None
    video_id: str = ""
    is_local: bool = False
    metadata: dict = field(default_factory=dict)


class Downloader:
    def __init__(self, download_dir: str | None = None, timeout: int = 300):
        self.download_dir = download_dir or tempfile.mkdtemp(prefix="videoscan_")
        self.timeout = timeout

    def is_url(self, source: str) -> bool:
        return bool(re.match(r"https?://", source))

    def resolve_source(self, source: str) -> DownloadResult:
        if not self.is_url(source):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Local file not found: {source}")
            mtime = os.path.getmtime(source)
            vid = hashlib.md5(f"{source}:{mtime}".encode()).hexdigest()[:16]
            return DownloadResult(video_path=str(path), is_local=True, video_id=vid)
        return self._download_url(source)

    def get_video_id(self, url: str) -> str:
        info = self._extract_info(url)
        return info.get("id", hashlib.md5(url.encode()).hexdigest()[:16])

    def _extract_info(self, url: str) -> dict:
        opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False) or {}

    def get_metadata(self, source: str) -> dict:
        if self.is_url(source):
            return self._extract_info(source)
        return {"id": self.resolve_source(source).video_id, "filepath": source}

    def _download_url(self, url: str, info: dict | None = None) -> DownloadResult:
        """Download video once, extract audio locally with ffmpeg."""
        if info is None:
            info = self._extract_info(url)
        video_id = info.get("id", hashlib.md5(url.encode()).hexdigest()[:16])
        out_dir = Path(self.download_dir) / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        video_path = str(out_dir / "video.%(ext)s")

        # Download video ONCE — use a reasonable quality (720p max to save bandwidth)
        t0 = time.time()
        video_opts = {
            "quiet": True,
            "no_warnings": True,
            "outtmpl": video_path,
            "socket_timeout": self.timeout,
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(video_opts) as ydl:
            ydl.download([url])

        actual_video = next(out_dir.glob("video.*"), None)
        if not actual_video:
            raise RuntimeError(f"Download failed: no video file found in {out_dir}")
        logger.info(f"Video downloaded in {time.time() - t0:.1f}s: {actual_video}")

        # Extract audio LOCALLY from the downloaded file (no second download!)
        t0 = time.time()
        audio_path = str(out_dir / "audio.wav")
        actual_audio = self._extract_audio_local(str(actual_video), audio_path)
        if actual_audio:
            logger.info(f"Audio extracted locally in {time.time() - t0:.1f}s")
        else:
            logger.warning("Audio extraction failed — transcription will use video file directly")

        return DownloadResult(
            video_path=str(actual_video),
            audio_path=actual_audio,
            video_id=video_id,
            is_local=False,
            metadata=info,
        )

    def download_with_info(self, url: str, info: dict) -> DownloadResult:
        """Download using pre-fetched metadata (avoids duplicate _extract_info call)."""
        return self._download_url(url, info=info)

    def _extract_audio_local(self, video_path: str, audio_path: str) -> str | None:
        """Extract audio from local video file using ffmpeg."""
        # Use MP3 instead of WAV — much smaller file for Whisper upload (~2MB vs 25MB for 13min)
        audio_path = audio_path.replace(".wav", ".mp3")
        try:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vn",  # no video
                "-acodec", "libmp3lame",  # MP3 format
                "-ar", "16000",  # 16kHz (Whisper optimal)
                "-ac", "1",  # mono (Whisper optimal)
                "-b:a", "64k",  # 64kbps is enough for speech
                "-y",  # overwrite
                audio_path,
            ]
            subprocess.run(
                cmd, capture_output=True, timeout=120,
                check=True,
            )
            if Path(audio_path).exists() and Path(audio_path).stat().st_size > 0:
                return audio_path
            return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"ffmpeg audio extraction failed: {e}")
            return None
