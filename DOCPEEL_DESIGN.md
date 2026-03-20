# docpeel — Design Document

> Context document for LLM-assisted development. Covers architecture, data flow, conventions, and change guides.

---

## Overview

A CLI tool that converts PDF files to structured markdown + JSON using vision LLMs. It renders each PDF page to an image, sends it to an LLM for OCR/transcription, and streams results to disk. Outputs structured JSONL and markdown; well-suited for search indexing, vector databases, RAG pipelines, or any downstream processing that needs clean per-page text and table data.

---

## Module Map

| Module | Responsibility |
|---|---|
| `cli.py` | Entry point. Parses args (`--vision-model`, `--ocr`, `--structure-model`, `--dpi`, `--pages`, `--verbose`/`--quiet`), validates flag combinations, wires modules together, prints run summary. |
| `providers/` | LLM provider package. See layout below. |
| `extraction.py` | Per-page orchestration. Two extractor classes (`VisionExtractor`, `MistralExtractor`) plus `iter_pages()` dispatcher. Yields page result dicts one at a time. |
| `output.py` | Streaming writes to disk: combined markdown, per-page markdowns, JSONL, cost report. |
| `prompts.py` | All prompt strings and the `quadrant_extract_prompt()` builder. No logic. |
| `image_utils.py` | PIL image helpers: quadrant splitting, base64 encoding, JPEG size-capping, obfuscation. |
| `pricing.py` | Cost estimation. Fetches live pricing from litellm JSON; hardcoded fallbacks. |

### `providers/` package layout

| Module | Responsibility |
|---|---|
| `providers/base.py` | `Usage` NamedTuple, `PAGE_EXTRACTION_SCHEMA`, `VisionProvider` ABC, `_with_retry` helper. |
| `providers/anthropic.py` | `AnthropicProvider` — single vision call, forced `tool_choice` for structured output. |
| `providers/gemini.py` | `GeminiProvider` — single vision call, `response_schema` for structured output. |
| `providers/mistral.py` | `MistralProvider` — standalone class (not a `VisionProvider`), two-step OCR + structuring pipeline. |
| `providers/provider_factory.py` | `build_provider()` factory function. Accepts `vision_model`, `ocr`, and `structure_model` keyword args matching the CLI flags. Infers provider from model name prefix via `_infer_vision_provider()`. Mistral-native structure models (`mistral-*`, `devstral-*`, `codestral-*`, `open-*`) are routed directly to `MistralProvider`; non-Mistral models (`gemini-*`, `claude-*`) are wrapped via `_build_structure_fn()`. |
| `providers/__init__.py` | Empty. Import directly from submodules. |

---

## Architecture & Data Flow

```
cli.py
  │
  ├─ build_provider()         → providers/provider_factory.py  (returns VisionProvider or MistralProvider)
  ├─ resolve_run_folder()     → output.py                      (determines output path)
  │
  └─ iter_pages()             → extraction.py  (Iterator[dict], one dict per page)
       │
       ├─ convert_from_path() → pdf2image       (renders one page image at a time)
       │
       ├─ [VisionProvider]  VisionExtractor.extract()          (fallback chain, see below)
       │    ├─ provider.call_structured()       → providers/anthropic.py or gemini.py
       │    ├─ split_quadrants() / obfuscate()  → image_utils.py
       │    └─ prompt strings                  → prompts.py
       │
       ├─ [MistralProvider] MistralExtractor.extract()         (two-step pipeline, see below)
       │    ├─ provider.ocr_with_retry()        → providers/mistral.py
       │    └─ provider.structure_with_retry()  → providers/mistral.py
       │
       └─ yields page result dict  (identical shape regardless of extractor)
            │
  stream_outputs()            → output.py      (consumes iterator, writes to disk)
  write_report()              → output.py      (writes markdown report from metadata)
```

**Memory model:** Only one PIL image lives in RAM at a time — each page is loaded, processed, and explicitly deleted before the next. `stream_outputs()` flushes each page to disk immediately and drops the `text` field from in-memory results after writing, keeping only metadata for the final report.

---

## Extraction: Two Paths

`iter_pages()` dispatches to one of two extractor classes based on provider type. Both return identical page result dicts.

### VisionExtractor (Anthropic, Gemini)

Implements the full fallback chain per page:

1. **Full-page verbatim** — `call_structured(page_image, PAGE_EXTRACT_PROMPT)` → structured dict
2. If content-filter blocked → **Quadrant split**: split into 4 quadrants, for each:
   - 2a. `call(quad_img, quadrant_extract_prompt(label))` → raw text
   - 2b. If blocked → `call(obfuscate(quad_img), ...)` → raw text (obfuscated retry)
   - 2c. If still blocked → `call(quad_img, PARAPHRASE_PROMPT)` → raw text (flagged)
   - If blocked entirely → quadrant skipped (content missing, flagged in output)
3. After quadrant extraction → **Stitch**: `call_with_image_and_text_structured(page_image, STITCH_PROMPT + chunk_texts)` → structured dict
4. If stitch is blocked → full-page paraphrase fallback

Content-filter errors short-circuit retries immediately and flow into the next fallback stage. Rate limit errors (HTTP 429) are also raised immediately with a human-readable diagnosis (monthly quota vs per-minute limit, from response headers) and a `step` label identifying which call hit the limit (e.g. `"OCR (mistral-ocr-latest)"`, `"structuring (claude-...)"`, or blank for vision calls). All other errors go through exponential backoff retry (4 attempts, 5s–60s, ±25% jitter) defined in `_with_retry()` in `providers/base.py`.

### MistralExtractor (Mistral)

Implements the two-step pipeline per page:

1. **OCR** — `provider.ocr_with_retry(page_image)` → raw markdown text
2. **Structure** — `provider.structure_with_retry(ocr_text)` → structured dict

No fallback chain — Mistral OCR has no content filter. The `extraction_method` field in the result dict is `"ocr+structure"` for Mistral pages.

---

## Provider Abstraction

### VisionProvider ABC (`providers/base.py`)

Implemented by `AnthropicProvider` and `GeminiProvider`. Exposes:

| Method | Used for |
|---|---|
| `call(image, prompt)` | Quadrant plain-text extraction |
| `call_structured(image, prompt)` | Full-page extraction, returns schema dict |
| `call_with_image_and_text_structured(image, text)` | Stitch call |
| `call_structured_text(prompt)` | Text-only structured call (used as structure step in OCR runs) |
| `resolve_model_id()` | Pre-flight API call to resolve any model alias to its canonical ID; called in `cli.py` before naming the output folder |

### MistralProvider (`providers/mistral.py`)

Standalone class, not a `VisionProvider` subclass. The two-step pipeline is architecturally different from a single vision call, so it does not implement the `VisionProvider` interface. Exposes:

| Method | Used for |
|---|---|
| `ocr_with_retry(image)` | OCR with exponential backoff, returns `(markdown_text, page_count)` |
| `structure_with_retry(ocr_text, extra_context)` | Structuring chat call with backoff, returns `(result_dict, usage)` |
| `ocr_page_cost(n_pages)` | OCR cost calculation for use by `MistralExtractor` |
| `drain_sanitisation_warnings()` | Returns and clears warnings from the last `structure()` call |
| `resolve_model_id()` | Resolves the chat model alias to its canonical ID; delegates to the embedded vision provider when a structure fn is injected |
| `close()` | Closes the underlying HTTP client if supported; call before process exit to avoid SDK async-cleanup errors |

**Structured output enforcement:**
- Anthropic: forced `tool_choice` with `PAGE_EXTRACTION_SCHEMA` as tool definition
- Gemini: `response_schema` + `response_mime_type=application/json`
- Mistral: prompts the chat model to return JSON; strips markdown fences + invalid control chars before parsing

---

## Shared Output Schema (`PAGE_EXTRACTION_SCHEMA`)

Defined in `providers/base.py`, used by all three providers. Fields:

| Field | Type | Notes |
|---|---|---|
| `skip` | bool | True for ToC, index, blank, illustration-only, title, part-divider pages |
| `skip_reason` | string\|null | Label if skipped |
| `page_number` | int\|null | Printed margin page number |
| `title` | string\|null | Prominent display heading only |
| `text` | string | Verbatim transcription; table bodies omitted, their heading lines preserved in place |
| `tables` | array | `{title, caption, content}` per table; appended after the page text in rendered markdown |
| `watermarks` | array of strings | Detected watermarks, excluded from text |

Quadrant calls (`call()`) return **plain text only** — no schema, no tables. The stitch call reassembles structure.

---

## State & Data Flow

**No global mutable state.** All state is local to function calls or passed explicitly.

- `Usage` is a `NamedTuple` with `__add__` for accumulation across quadrants.
- `MistralProvider` holds `_sanitisation_warnings: list[str]` as instance state, drained via `drain_sanitisation_warnings()` after each `structure()` call.
- `_fetch_litellm_pricing()` is `@lru_cache(maxsize=1)` — fetched once per process.
- Page result dicts flow: `iter_pages()` → `stream_outputs()`. The `text` field is written to disk then dropped from the in-memory copy kept for `write_report()`.

**Output folder naming:** `output/<model>__<pdf-stem>[N]/` where `<model>` is the resolved canonical model ID (alias pre-resolved via `resolve_model_id()`), `<pdf-stem>` is the PDF filename without extension, and `N` auto-increments from 0. Each run gets: `extracted.md`, `pages/page_NNN.md`, `extracted.jsonl`, `report.md`.

---

## Key Conventions

- **Images are always JPEG-encoded** for API calls (via `to_b64_safe()`), with iterative downscaling if over 5 MB. PNG is only used internally (`to_b64()`, unused in providers).
- **Prompts never describe the output schema** — that's handled by provider-native tool/function-calling. Prompts focus on content instructions only.
- **`paraphrased` field values:** `None` = verbatim, `"partial"` = some quads paraphrased, `"full"` = stitch call was paraphrased. Always `None` for Mistral.
- **`extraction_method` field values:** `"full-page"` = verbatim vision call succeeded, `"quadrant-split"` = fallback chain was used, `"ocr+structure"` = Mistral two-step pipeline.
- **Tables** are extracted into the `tables` array as `{title, caption, content}` and appended after the page prose text when writing per-page markdown files, one after the other in document order.
- **Per-page markdown** has YAML frontmatter (`pdf_page`, `book_page`, `source`, `title`, `tables`, `watermarks`, `skip`, `error`).
- **Provider instantiation** is lazy — SDK imports (`anthropic`, `google.genai`, `mistralai`) happen inside `__init__`, so unused providers add no import cost.

---

## "If You Want to Change X" Guide

| Change | Files to touch |
|---|---|
| Add a new LLM provider (vision-based) | `providers/` (new `<name>.py` implementing `VisionProvider`), `providers/provider_factory.py` (`_VISION_PREFIXES`, `build_provider()`), `cli.py` (update `--vision-model` help text), `pricing.py` (new cost function) |
| Add a new LLM provider (non-vision pipeline) | `providers/` (new `<name>.py` standalone class), `providers/provider_factory.py`, `extraction.py` (new extractor class + dispatch in `iter_pages()`), `cli.py`, `pricing.py` |
| Change extraction prompts | `prompts.py` only |
| Change the structured output schema (add/remove fields) | `providers/base.py` (`PAGE_EXTRACTION_SCHEMA`), `providers/gemini.py` (inline Gemini schema in `_generate_structured()`), `providers/mistral.py` (`_MISTRAL_STRUCTURE_PROMPT`), `extraction.py` (`_unpack()`), `output.py` (frontmatter/body rendering) |
| Change output file format or folder structure | `output.py` (`stream_outputs()`, `resolve_run_folder()`) |
| Change retry behaviour (attempts, delays) | `providers/base.py` (`_MAX_RETRIES`, `_BASE_DELAY`, `_MAX_DELAY`, `_JITTER`) |
| Change quadrant split logic | `image_utils.py` (`split_quadrants()`) |
| Change obfuscation strategy | `image_utils.py` (`obfuscate()`) |
| Change skip-page categories | `prompts.py` (all three prompts + `PAGE_EXTRACT_PROMPT`), `providers/base.py` (`PAGE_EXTRACTION_SCHEMA` `skip_reason` description), `providers/mistral.py` (`_MISTRAL_STRUCTURE_PROMPT`) |
| Change cost calculation | `pricing.py` |
| Change CLI flags | `cli.py` |
| Add a new OCR engine | `cli.py` (`choices=` on `--ocr`), `providers/provider_factory.py` (`build_provider()` OCR branch), new provider module, `pricing.py` |
| Add a new structure-step provider (for OCR path) | `providers/provider_factory.py` (`_build_structure_fn()`, extend prefix logic), `pricing.py` |
| Change how provider is inferred from model name | `providers/provider_factory.py` (`_VISION_PREFIXES`, `_infer_vision_provider()`) |
| Change what appears in the cost report | `output.py` (`write_report()`) |
| Change image encoding/size limits | `image_utils.py` (`to_b64_safe()`, `_ANTHROPIC_MAX_B64_BYTES`) |
| Change render DPI default | `cli.py` (argparse default), `extraction.py` (`iter_pages()` default param) |

---

## Non-Obvious Design Decisions

- **Provider is inferred from model name, not passed explicitly** — `_VISION_PREFIXES` in `provider_factory.py` maps name prefixes (`claude` → anthropic, `gemini` → gemini) to provider constructors. The same inference applies to the structure model in OCR runs. This means the CLI has no `--provider` flag; the model name is the single source of truth for which SDK to use.
- **`--vision-model` and `--ocr` are mutually exclusive** — they select entirely different extractor paths (`VisionExtractor` vs `MistralExtractor`). `--structure-model` is only valid alongside `--ocr`. These constraints are validated in `cli.py` before `build_provider()` is called.
- **`--ocr` is an engine name, not a model name** — because OCR engines are not interchangeable with LLMs and have no model-ID concept meaningful to the user (`mistral-ocr-latest` is the only Mistral OCR model and is never user-selectable). The `--vision-model` / `--structure-model` flags take model IDs because those are user-meaningful choices.
- **`MistralProvider` is not a `VisionProvider` subclass** — the two-step OCR + structuring pipeline is architecturally different from a single vision call. Forcing it into the `VisionProvider` interface would require stub methods that ignore their arguments. Instead, `MistralExtractor` in `extraction.py` owns the Mistral-specific orchestration and calls `MistralProvider` directly.
- **The quadrant fallback chain never runs for Mistral** — Mistral OCR has no content filter, so there is no blocked-response condition to fall back from. This is now structurally enforced: `MistralExtractor` has no fallback logic at all, rather than having it silently bypassed.
- **Stitch uses the original full-page image**, not just the quadrant texts — this lets the LLM verify reading order and correct OCR errors visually.
- **Empty quadrant detection** before the stitch call: if all four quadrants return empty strings, the stitch is skipped entirely and the page is returned as empty (image-only page).
- **Pricing is fetched live once** from litellm's public JSON and cached; Gemini and Mistral have hardcoded fallbacks for models not yet indexed there.
- **`providers/__init__.py` is empty** — callers import directly from the submodule that owns the name (e.g. `from docpeel.providers.base import VisionProvider`). This makes the source of each import explicit and avoids hiding the package structure behind re-exports.
- **`_build_structure_fn()` injects a callable rather than a provider instance into `MistralProvider`** — the structuring step in an OCR run is a plain text-in/text-out call with no image; wrapping it as a `VisionProvider` subclass would be misleading. The callable pattern keeps `MistralProvider` decoupled from whichever LLM does the structuring, and the `_model_id` attribute on the fn lets `MistralProvider.model_id` surface the actual model name without needing a back-reference to the factory.
- **Model aliases are resolved before any output is written** — `cli.py` calls `provider.resolve_model_id()` immediately after `build_provider()`. This ensures the canonical model ID (not an alias like `claude-sonnet-4-0`) appears in the output folder name, JSONL records, and cost report. For Mistral OCR runs with a vision structure model, `resolve_model_id()` delegates to the embedded provider.
- **Rate limit (429) errors are never retried** — they indicate a quota boundary, not a transient failure, so retrying wastes quota and time. Instead `_with_retry()` raises immediately after printing a plain-English diagnosis built from the response headers (e.g. "monthly token quota exhausted — 0 / 4,000,000 tokens remaining").
- **OCR cost and structure cost are tracked separately** — `MistralExtractor` yields `ocr_cost_usd` and `structure_cost_usd` as distinct fields in the page dict (in addition to the combined `cost_usd`). `write_report()` sums and displays them as separate line items only when those fields are present, so the report stays clean for non-Mistral runs.
