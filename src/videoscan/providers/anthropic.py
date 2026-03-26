"""Anthropic Claude vision provider."""
from __future__ import annotations
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from videoscan.providers.base import FrameAnalysis, get_detail_prompt, parse_vision_response

_VISION_COST_PER_IMAGE = 0.01

class AnthropicVision:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.cost_per_image = _VISION_COST_PER_IMAGE

    def _build_prompt(self, detail: str) -> str:
        base = get_detail_prompt(detail)
        return f"{base}\n\nAlso extract any readable text visible in the image and return it on a separate line prefixed with 'OCR:'. If no text is visible, omit the OCR line."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    async def analyze_frame(self, image_base64: str, detail: str = "standard") -> FrameAnalysis:
        prompt = self._build_prompt(detail)
        response = await self.client.messages.create(model=self.model, max_tokens=500, messages=[{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64}}, {"type": "text", "text": prompt}]}])
        text = response.content[0].text if response.content else ""
        return parse_vision_response(text)
