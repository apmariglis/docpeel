"""
Provider factory: builds provider instances from CLI flag values.

Two paths:
  Vision path  — --vision-model MODEL
                 Provider inferred from model name prefix.
                 Returns a VisionProvider (AnthropicProvider or GeminiProvider).

  OCR path     — --ocr ENGINE --structure-model MODEL
                 Returns a MistralProvider (OCR engine fixed to mistral-ocr-latest;
                 structure model may be Mistral or Gemini).
"""

from docpeel.providers.anthropic import AnthropicProvider
from docpeel.providers.base import VisionProvider
from docpeel.providers.gemini import GeminiProvider
from docpeel.providers.mistral import MistralProvider
from docpeel.providers.mistral import Usage

# Model name prefixes → provider label
_VISION_PREFIXES: dict[str, str] = {
    "claude": "anthropic",
    "gemini": "gemini",
}


def build_provider(
    vision_model: str | None = None,
    ocr: str | None = None,
    structure_model: str | None = None,
) -> VisionProvider | MistralProvider:
    """
    vision_model: model ID for the vision path (e.g. claude-sonnet-4-0,
                  gemini-2.5-flash). Provider inferred from name prefix.
    ocr:          OCR engine name for the two-step path. Currently only "mistral".
    structure_model: LLM for the structuring step (requires ocr). Provider
                  inferred from name prefix. Defaults to mistral-small-latest.
    """
    if ocr:
        if ocr.lower() != "mistral":
            raise ValueError(f"Unknown OCR engine '{ocr}'. Supported: mistral")
        structure_fn = None
        if structure_model and not structure_model.startswith("mistral"):
            structure_fn = _build_structure_fn(structure_model)
            return MistralProvider(model=None, structure_fn=structure_fn)
        return MistralProvider(model=structure_model)

    # Vision path — model must be explicitly provided (enforced by cli.py)
    if not vision_model:
        raise ValueError("vision_model must be specified for the vision path.")
    provider_name = _infer_vision_provider(vision_model)
    if provider_name == "anthropic":
        return AnthropicProvider(vision_model)
    if provider_name == "gemini":
        return GeminiProvider(vision_model)
    raise ValueError(
        f"Cannot infer provider from vision model '{model}'. "
        f"Model name must start with one of: {', '.join(_VISION_PREFIXES)}"
    )


def _infer_vision_provider(model: str) -> str:
    for prefix, name in _VISION_PREFIXES.items():
        if model.startswith(prefix):
            return name
    raise ValueError(
        f"Cannot infer provider from model name '{model}'. "
        f"Expected prefix: {', '.join(_VISION_PREFIXES)}"
    )


# def _build_structure_fn(structure_model: str):
#     """
#     Build a structure callable for a non-Mistral model.

#     The callable signature matches MistralProvider.structure():
#         fn(ocr_text: str, extra_context: str = "") -> tuple[dict, Usage]

#     Currently supports: gemini (any model starting with 'gemini')
#     """
#     if structure_model.startswith("gemini"):
#         import json
#         import os

#         import google.genai as genai
#         from docpeel.pricing import gemini_cost
#         from docpeel.providers.base import Usage
#         from docpeel.providers.mistral import _MISTRAL_STRUCTURE_PROMPT

#         client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

#         def gemini_structure_fn(
#             ocr_text: str, extra_context: str = ""
#         ) -> tuple[dict, Usage]:
#             prompt = _MISTRAL_STRUCTURE_PROMPT + ocr_text
#             if extra_context:
#                 prompt += f"\n\nAdditional context (quadrant texts for stitching):\n{extra_context}"

#             response = client.models.generate_content(
#                 model=structure_model,
#                 contents=prompt,
#                 config=genai.types.GenerateContentConfig(
#                     response_mime_type="application/json",
#                     temperature=0.0,
#                     max_output_tokens=4096,
#                 ),
#             )
#             raw = (response.text or "").strip()
#             if raw.startswith("```"):
#                 raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

#             parsed = json.loads(raw)
#             result = {
#                 "skip": bool(parsed.get("skip", False)),
#                 "skip_reason": parsed.get("skip_reason"),
#                 "page_number": parsed.get("page_number"),
#                 "title": parsed.get("title"),
#                 "text": parsed.get("text", ""),
#                 "tables": parsed.get("tables") or [],
#                 "watermarks": parsed.get("watermarks") or [],
#             }

#             meta = response.usage_metadata
#             in_tok = getattr(meta, "prompt_token_count", 0) or 0
#             out_tok = getattr(meta, "candidates_token_count", 0) or 0
#             usage = Usage(
#                 input_tokens=in_tok,
#                 output_tokens=out_tok,
#                 cache_creation_tokens=0,
#                 cache_read_tokens=0,
#                 cost_usd=gemini_cost(structure_model, in_tok, out_tok),
#             )
#             return result, usage

#         gemini_structure_fn._model_id = structure_model
#         return gemini_structure_fn


#     raise ValueError(
#         f"Cannot infer structure provider from model name '{structure_model}'. "
#         "Currently supported prefixes: gemini"
#     )


def _build_structure_fn(structure_model: str):
    from docpeel.providers.mistral import _MISTRAL_STRUCTURE_PROMPT

    provider_name = _infer_vision_provider(structure_model)

    if provider_name == "gemini":
        provider = GeminiProvider(structure_model)
    elif provider_name == "anthropic":
        provider = AnthropicProvider(structure_model)
    else:
        raise ValueError(
            f"Cannot build structure function for provider '{provider_name}'."
        )

    def structure_fn(ocr_text: str, extra_context: str = "") -> tuple[dict, Usage]:
        prompt = _MISTRAL_STRUCTURE_PROMPT + ocr_text
        return provider.call_structured_text(prompt)

    structure_fn._model_id = structure_model
    structure_fn._provider = provider
    return structure_fn
