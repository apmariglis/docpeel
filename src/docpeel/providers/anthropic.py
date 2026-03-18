"""
Anthropic provider implementation.

Uses forced tool_choice with PAGE_EXTRACTION_SCHEMA to guarantee structured
output — no post-hoc JSON parsing needed. Content-filter blocks surface as
anthropic.BadRequestError and are caught by the fallback chain in VisionExtractor.
"""

from docpeel.pricing import anthropic_cost
from docpeel.providers.base import PAGE_EXTRACTION_SCHEMA
from docpeel.providers.base import Usage
from docpeel.providers.base import VisionProvider
from docpeel.providers.base import _with_retry
from PIL import Image


class AnthropicProvider(VisionProvider):
    def __init__(self, model: str | None = None):
        import anthropic as _anthropic

        self._anthropic = _anthropic
        self._client = _anthropic.Anthropic()
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    def _parse_usage(self, response) -> Usage:
        u = response.usage
        cost = anthropic_cost(
            model=response.model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_creation=u.cache_creation_input_tokens or 0,
            cache_read=u.cache_read_input_tokens or 0,
        )
        return Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_creation_tokens=u.cache_creation_input_tokens or 0,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            cost_usd=cost,
        )

    def _create(self, content):
        return self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": content}],
        )

    def _create_structured(self, content) -> tuple[dict, Usage]:
        """
        Call the API with tool_use forced to page_extraction so the response
        is guaranteed to be a structured dict — no JSON parsing needed.
        """
        schema = PAGE_EXTRACTION_SCHEMA
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            tools=[
                {
                    "name": schema["name"],
                    "description": schema["description"],
                    "input_schema": {
                        "type": "object",
                        "properties": schema["properties"],
                        "required": schema["required"],
                    },
                }
            ],
            tool_choice={"type": "tool", "name": schema["name"]},
            messages=[{"role": "user", "content": content}],
        )
        # With tool_choice forced, content[0] is always a tool_use block.
        tool_block = next(b for b in resp.content if b.type == "tool_use")
        return tool_block.input, self._parse_usage(resp)

    def _image_block(self, image: Image.Image) -> dict:
        from docpeel.image_utils import to_b64_safe

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": to_b64_safe(image),
            },
        }

    def call(self, image: Image.Image, prompt: str) -> tuple[str, Usage]:
        resp = _with_retry(
            self.is_content_filter_error,
            lambda: self._create(
                [self._image_block(image), {"type": "text", "text": prompt}]
            ),
        )
        return resp.content[0].text, self._parse_usage(resp)

    def call_structured(self, image: Image.Image, prompt: str) -> tuple[dict, Usage]:
        return _with_retry(
            self.is_content_filter_error,
            lambda: self._create_structured(
                [self._image_block(image), {"type": "text", "text": prompt}]
            ),
        )

    def call_with_image_and_text_structured(
        self, image: Image.Image, text_prompt: str
    ) -> tuple[dict, Usage]:
        return _with_retry(
            self.is_content_filter_error,
            lambda: self._create_structured(
                [self._image_block(image), {"type": "text", "text": text_prompt}]
            ),
        )

    def call_structured_text(self, prompt: str) -> tuple[dict, Usage]:
        """Structured call with text-only input (no image). Used by the OCR structuring path."""
        return _with_retry(
            self.is_content_filter_error,
            lambda: self._create_structured([{"type": "text", "text": prompt}]),
        )

    def is_content_filter_error(self, exc: Exception) -> bool:
        return isinstance(exc, self._anthropic.BadRequestError)

    def resolve_model_id(self) -> None:
        self._model = self._client.models.retrieve(self._model).id
