"""Check and report on system dependencies."""
from __future__ import annotations
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_FFMPEG_INSTALL = {
    "win32": "choco install ffmpeg",
    "darwin": "brew install ffmpeg",
    "linux": "apt install ffmpeg",
}

@dataclass
class DepStatus:
    name: str
    available: bool
    path: str | None = None
    message: str = ""

def check_ffmpeg() -> DepStatus:
    path = shutil.which("ffmpeg")
    if path:
        return DepStatus(name="ffmpeg", available=True, path=path)
    platform = sys.platform
    hint = _FFMPEG_INSTALL.get(platform, "Install ffmpeg for your platform")
    msg = f"ffmpeg not found. Install with: {hint}"
    logger.warning(msg)
    return DepStatus(name="ffmpeg", available=False, message=msg)

def check_ytdlp() -> DepStatus:
    path = shutil.which("yt-dlp")
    if path:
        return DepStatus(name="yt-dlp", available=True, path=path)
    logger.info("yt-dlp not found, attempting pip install...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "yt-dlp"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        path = shutil.which("yt-dlp")
        if path:
            return DepStatus(name="yt-dlp", available=True, path=path, message="Auto-installed via pip")
        return DepStatus(name="yt-dlp", available=True, message="Installed as Python module")
    except subprocess.CalledProcessError:
        msg = "yt-dlp not found. Install with: pip install yt-dlp"
        logger.warning(msg)
        return DepStatus(name="yt-dlp", available=False, message=msg)

def check_dependencies() -> dict[str, DepStatus]:
    return {"ffmpeg": check_ffmpeg(), "yt-dlp": check_ytdlp()}
