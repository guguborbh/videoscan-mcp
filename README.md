# VideoScan MCP

An MCP (Model Context Protocol) server for comprehensive video analysis — AI-powered transcription, visual frame analysis, and metadata extraction from 1000+ platforms.

## Features

- **Full video analysis** — combines transcription, frame extraction, and metadata in a single call
- **AI vision analysis** — describes frames and extracts on-screen text (OCR) using GPT-4o, Claude, or Gemini
- **Audio transcription** — Whisper-based transcription with timestamps and language detection
- **Auto-tuning** — automatically adjusts frame extraction density, interval, and detail level based on video duration
- **Smart frame extraction** — scene-change detection, interval sampling, or combined strategy
- **Deduplication** — perceptual hashing removes near-duplicate frames before analysis
- **Metadata extraction** — title, duration, chapters, tags, view count, and more without full download
- **Multi-provider** — OpenAI, Anthropic, and Google vision providers with per-request override
- **Caching** — persistent cache for downloads, frames, and results to minimize repeat costs
- **1000+ platforms** — powered by yt-dlp (YouTube, Vimeo, Twitter/X, TikTok, and more)

## Installation

```bash
pip install videoscan-mcp
```

### System dependencies

VideoScan requires [ffmpeg](https://ffmpeg.org/download.html) for video processing and [yt-dlp](https://github.com/yt-dlp/yt-dlp) for downloading from URLs.

```bash
# macOS
brew install ffmpeg yt-dlp

# Ubuntu/Debian
apt install ffmpeg
pip install yt-dlp

# Windows — install ffmpeg from https://ffmpeg.org/download.html, then:
pip install yt-dlp
```

## Configuration

Copy `.env.example` to `.env` and fill in at minimum one API key:

```env
# Vision provider (frame analysis)
VISION_PROVIDER=openai          # openai | anthropic | google
VISION_MODEL=                   # optional — defaults: gpt-4o / claude-sonnet-4-20250514 / gemini-2.0-flash

# Transcription provider
TRANSCRIPTION_PROVIDER=openai   # openai only for now
TRANSCRIPTION_MODEL=whisper-1

# Concurrency
VISION_CONCURRENCY=5            # max parallel vision API calls

# API keys — only need the key for your chosen provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...

# Cache
CACHE_ENABLED=true
CACHE_DIR=~/.videoscan/cache
CACHE_MAX_SIZE_GB=5
CACHE_DOWNLOAD_TTL=3600         # 1 hour
CACHE_FRAMES_TTL=86400          # 24 hours
CACHE_RESULTS_TTL=604800        # 7 days

# Safety limits (set to 0 for unlimited)
MAX_VIDEO_DURATION=3600         # 60 minutes in seconds
MAX_DOWNLOAD_SIZE=2147483648    # 2 GB in bytes
MAX_ANALYZED_FRAMES=100
DOWNLOAD_TIMEOUT=300
FRAME_ANALYSIS_TIMEOUT=30
```

## Quick Start — Claude Code

Add VideoScan to your Claude Code `settings.json` (usually at `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "videoscan": {
      "command": "videoscan",
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Or using `uvx` without a global install:

```json
{
  "mcpServers": {
    "videoscan": {
      "command": "uvx",
      "args": ["videoscan-mcp"],
      "env": {
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Once connected, you can ask Claude things like:

- "Analyze this YouTube video: https://youtube.com/watch?v=..."
- "Transcribe the audio from this video file"
- "What's on screen at the 2:30 mark of this video?"
- "Extract frames from this video and describe what you see"

## Auto-Tuning

When `max_frames` and `interval` are not explicitly set, VideoScan automatically adjusts frame extraction parameters based on video duration to optimize cost and coverage:

| Duration | Frames | Interval | Strategy | Detail |
|---|---|---|---|---|
| **< 2 min** | ~1/sec (dense) | 1s | combined | detailed |
| **2–10 min** | ~40 | 3s | combined | standard |
| **10–30 min** | ~30 | 10s | combined | standard |
| **30–60 min** | ~30 | 20s | combined | brief |
| **> 60 min** | ~20 | 30s | scene only | brief |

Short videos get dense frame extraction for maximum detail, while longer videos use lighter sampling to keep costs down. You can always override by setting `max_frames` or `interval` explicitly.

## Tool Reference

### `analyze_video`

Full pipeline — transcription + AI frame analysis + metadata in one call. Uses [auto-tuning](#auto-tuning) by default.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | string | required | URL or local file path |
| `detail` | string | `"standard"` | Vision level: `"brief"`, `"standard"`, `"detailed"` |
| `max_frames` | int | auto | Maximum frames to analyze — set to `-1` (default) for auto-tuning based on duration |
| `threshold` | float | `0.3` | Scene change sensitivity (0.0–1.0) |
| `strategy` | string | `"combined"` | Frame extraction: `"scene"`, `"interval"`, `"combined"` |
| `interval` | int | auto | Seconds between frames — set to `-1` (default) for auto-tuning based on duration |
| `skip_frames` | bool | `false` | Skip visual analysis (transcription only) |
| `skip_audio` | bool | `false` | Skip transcription (frames only) |
| `language` | string | `"auto"` | Transcription language or `"auto"` |
| `provider` | string | `null` | Override vision provider |
| `force_refresh` | bool | `false` | Bypass cache |

---

### `transcribe`

Transcribe video or audio to text with timestamps.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | string | required | URL or local file path |
| `language` | string | `"auto"` | Preferred language or `"auto"` for detection |

---

### `extract_frames`

Extract and AI-analyze frames from a video.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | string | required | URL or local file path |
| `max_frames` | int | `30` | Maximum frames to extract (1–100) |
| `threshold` | float | `0.3` | Scene change sensitivity (0.0–1.0) |
| `strategy` | string | `"combined"` | `"scene"`, `"interval"`, or `"combined"` |
| `interval` | int | `5` | Seconds between frames in interval mode |
| `detail` | string | `"standard"` | Vision analysis level |
| `deduplicate` | bool | `true` | Remove near-duplicate frames via dHash |
| `provider` | string | `null` | Override vision provider |
| `force_refresh` | bool | `false` | Bypass cache |

---

### `analyze_moment`

Deep-dive analysis on a specific time range.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | string | required | URL or local file path |
| `start` | float | required | Start time in seconds |
| `end` | float | required | End time in seconds |
| `dense` | bool | `true` | Extract 1 frame per second in the range |
| `detail` | string | `"detailed"` | Vision analysis level |
| `provider` | string | `null` | Override vision provider |
| `force_refresh` | bool | `false` | Bypass cache |

---

### `get_frame_at`

Get a single frame at a specific timestamp, optionally analyzed by AI.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | string | required | URL or local file path |
| `timestamp` | float | required | Time in seconds |
| `analyze` | bool | `true` | Run AI vision analysis |
| `provider` | string | `null` | Override vision provider |
| `force_refresh` | bool | `false` | Bypass cache |

---

### `get_metadata`

Fetch video metadata without downloading the full video.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source` | string | required | URL or local file path |
| `include` | list | `null` | Specific fields to return — `"title"`, `"duration"`, `"channel"`, `"description"`, `"thumbnail"`, `"chapters"`, `"tags"`, `"view_count"`. Returns all if omitted. |

---

## Supported Platforms

VideoScan uses yt-dlp under the hood, which supports 1000+ video platforms including:

- YouTube, YouTube Shorts, YouTube Live
- Vimeo, Dailymotion, Twitch
- Twitter/X, Instagram, TikTok, Facebook
- Reddit, LinkedIn, Pinterest
- BBC iPlayer, CNN, NBC, CBS
- SoundCloud, Bandcamp (audio)
- And hundreds more — see the [yt-dlp supported sites list](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)

Local files in any format supported by ffmpeg (mp4, mov, avi, mkv, webm, mp3, wav, etc.) are also supported.

## Cost Estimates

Costs depend on your chosen provider and usage:

| Operation | Provider | Approx. Cost |
|---|---|---|
| Vision analysis | OpenAI GPT-4o | ~$0.015 per frame |
| Vision analysis | Anthropic Claude | ~$0.024 per frame |
| Vision analysis | Google Gemini | ~$0.002 per frame |
| Transcription | OpenAI Whisper | ~$0.006 per minute |

A typical 10-minute video analyzed with `analyze_video` (30 frames + transcription) costs approximately $0.45–$0.51 with OpenAI.

## Development

```bash
git clone https://github.com/your-org/videoscan-mcp
cd videoscan-mcp
pip install -e ".[dev]"
pytest
```

## License

MIT License — see [LICENSE](LICENSE) for details.
