"""get_metadata MCP tool — video info without full download."""
from __future__ import annotations
import asyncio
from videoscan.core.downloader import Downloader
from videoscan.types import MetadataResult

async def get_metadata(source: str, include: list[str] | None = None) -> MetadataResult:
    downloader = Downloader()
    raw = await asyncio.to_thread(downloader.get_metadata, source)
    all_fields = {
        "title": raw.get("title"),
        "channel": raw.get("uploader") or raw.get("channel"),
        "duration": raw.get("duration"),
        "url": raw.get("webpage_url") or raw.get("url"),
        "platform": raw.get("extractor_key", "").lower() or raw.get("extractor", "").lower(),
        "chapters": [{"title": ch.get("title", ""), "start": ch.get("start_time", 0), "end": ch.get("end_time", 0)} for ch in raw.get("chapters", []) or []],
        "tags": raw.get("tags") or [],
        "view_count": raw.get("view_count"),
        "upload_date": raw.get("upload_date"),
        "description": raw.get("description"),
        "thumbnail": raw.get("thumbnail"),
    }
    if include:
        all_fields = {k: v for k, v in all_fields.items() if k in include}
    return MetadataResult(**all_fields)
