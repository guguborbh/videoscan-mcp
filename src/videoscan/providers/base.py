"""Provider protocol interfaces and factory."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

_DETAIL_PROMPTS = {
    "brief": "Describe this video frame in one sentence. Focus on the single most important visual element.",
    "standard": "Describe this video frame in 2-3 sentences. Cover the key visual elements, any readable text (OCR), and the context of what's shown.",
    "detailed": "Provide a comprehensive description of this video frame. Include: layout, colors, UI elements, all readable text exactly as shown, spatial relationships between elements, and any notable details. Be thorough.",
}

@dataclass
class FrameAnalysis:
    description: str
    ocr_text: str | None = None

@runtime_checkable
class VisionProvider(Protocol):
    async def analyze_frame(self, image_base64: str, detail: str) -> FrameAnalysis: ...
    def _build_prompt(self, detail: str) -> str: ...

@runtime_checkable
class TranscriptionProvider(Protocol):
    async def transcribe(self, audio_path: str, language: str) -> list[dict]: ...
    def get_cost_per_minute(self) -> float: ...

def get_vision_provider(provider: str, api_key: str, model: str) -> VisionProvider:
    if provider == "openai":
        from videoscan.providers.openai import OpenAIVision
        return OpenAIVision(api_key=api_key, model=model)
    elif provider == "anthropic":
        from videoscan.providers.anthropic import AnthropicVision
        return AnthropicVision(api_key=api_key, model=model)
    elif provider == "google":
        from videoscan.providers.google import GoogleVision
        return GoogleVision(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown vision provider: {provider}")

def get_transcription_provider(provider: str, api_key: str, model: str):
    if provider == "openai":
        from videoscan.providers.openai import OpenAITranscription
        return OpenAITranscription(api_key=api_key, model=model)
    else:
        raise ValueError(f"Unknown transcription provider: {provider}")

def get_detail_prompt(detail: str) -> str:
    return _DETAIL_PROMPTS.get(detail, _DETAIL_PROMPTS["standard"])

def parse_vision_response(text: str) -> FrameAnalysis:
    lines = text.strip().split("\n")
    ocr_lines, desc_lines = [], []
    for line in lines:
        if line.strip().startswith("OCR:"):
            ocr_lines.append(line.strip()[4:].strip())
        else:
            desc_lines.append(line)
    return FrameAnalysis(description="\n".join(desc_lines).strip(), ocr_text="\n".join(ocr_lines).strip() if ocr_lines else None)
