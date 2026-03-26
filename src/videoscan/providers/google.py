"""Google Gemini vision provider."""
from __future__ import annotations
import base64, io
import google.generativeai as genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential
from videoscan.providers.base import FrameAnalysis, get_detail_prompt, parse_vision_response

_VISION_COST_PER_IMAGE = 0.005

class GoogleVision:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.cost_per_image = _VISION_COST_PER_IMAGE

    def _build_prompt(self, detail: str) -> str:
        base = get_detail_prompt(detail)
        return f"{base}\n\nAlso extract any readable text visible in the image and return it on a separate line prefixed with 'OCR:'. If no text is visible, omit the OCR line."

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
    async def analyze_frame(self, image_base64: str, detail: str = "standard") -> FrameAnalysis:
        prompt = self._build_prompt(detail)
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        response = await self.model.generate_content_async([prompt, image])
        text = response.text if response.text else ""
        return parse_vision_response(text)
