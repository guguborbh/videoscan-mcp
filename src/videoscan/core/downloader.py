"""Video download and source resolution using yt-dlp."""
from __future__ import annotations
import hashlib, logging, os, re, tempfile
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

    def _download_url(self, url: str) -> DownloadResult:
        info = self._extract_info(url)
        video_id = info.get("id", hashlib.md5(url.encode()).hexdigest()[:16])
        out_dir = Path(self.download_dir) / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        video_path = str(out_dir / "video.%(ext)s")
        audio_path = str(out_dir / "audio.wav")
        video_opts = {"quiet": True, "no_warnings": True, "outtmpl": video_path, "socket_timeout": self.timeout}
        with yt_dlp.YoutubeDL(video_opts) as ydl:
            ydl.download([url])
        actual_video = next(out_dir.glob("video.*"), None)
        if not actual_video:
            raise RuntimeError(f"Download failed: no video file found in {out_dir}")
        audio_opts = {"quiet": True, "no_warnings": True, "outtmpl": audio_path, "format": "bestaudio/best", "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}], "socket_timeout": self.timeout}
        try:
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.download([url])
            actual_audio = audio_path if Path(audio_path).exists() else None
        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}")
            actual_audio = None
        return DownloadResult(video_path=str(actual_video), audio_path=actual_audio, video_id=video_id, is_local=False, metadata=info)
