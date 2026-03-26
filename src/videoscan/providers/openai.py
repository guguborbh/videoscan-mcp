"""OpenAI vision (GPT-4o) and transcription (Whisper) providers."""
from __future__ import annotations
from dataclasses import dataclass
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from videoscan.providers.base import FrameAnalysis, get_detail_prompt, parse_vision_response

_VISION_COST_PER_IMAGE = 0.015

class OpenAIVision:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.cost_per_image = _VISION_COST_PER_IMAGE

    def _build_prompt(self, detail: str) -> str:
        base = get_detail_prompt(detail)
        return f"{base}\n\nAlso extract any readable text visible in the image and return it on a separate line prefixed with 'OCR:'. If no text is visible, omit the OCR line."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    async def analyze_frame(self, image_base64: str, detail: str = "standard") -> FrameAnalysis:
        prompt = self._build_prompt(detail)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}],
            max_tokens=500,
        )
        text = response.choices[0].message.content or ""
        return parse_vision_response(text)

@dataclass
class TranscriptionResult:
    segments: list[dict]
    language: str

class OpenAITranscription:
    def __init__(self, api_key: str, model: str = "whisper-1"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def transcribe(self, audio_path: str, language: str = "auto") -> TranscriptionResult:
        kwargs = {"model": self.model, "response_format": "verbose_json", "timestamp_granularities": ["segment"]}
        if language != "auto": kwargs["language"] = language
        with open(audio_path, "rb") as f:
            response = await self.client.audio.transcriptions.create(file=f, **kwargs)
        detected_lang = getattr(response, "language", language)
        segments = []
        for seg in getattr(response, "segments", []):
            segments.append({"text": seg.get("text", "").strip() if isinstance(seg, dict) else seg.text.strip(), "start": seg.get("start", 0.0) if isinstance(seg, dict) else seg.start, "end": seg.get("end", 0.0) if isinstance(seg, dict) else seg.end})
        return TranscriptionResult(segments=segments, language=detected_lang)

    def get_cost_per_minute(self) -> float:
        return 0.006
