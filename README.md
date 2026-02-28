# PDFExtract

A Python package for extracting text and tables from PDFs using vision LLMs (Claude and Gemini). Designed for RAG ingestion pipelines where accuracy and structured output matter more than speed.

## Why vision LLMs?

Traditional PDF parsers (pdfminer, PyMuPDF, etc.) fail on scanned PDFs, multi-column layouts, complex tables, and pages that mix text with illustrations. Vision LLMs read the page as an image, handling all of these naturally. The tradeoff is cost and latency — this tool is optimised to minimise both while maximising extraction quality.

---

## Features

- **Multi-stage fallback chain** per page — escalates automatically from fast single-call extraction to quadrant splitting, obfuscation, and paraphrase fallback when content filters or layout complexity intervene
- **Image-assisted stitching** — after quadrant extraction, a final LLM call sees both the original full-page image and the four quadrant transcriptions together, correcting reading order and reassembling split tables
- **Structured JSONL output** — single source of truth for downstream RAG ingestion, with per-page metadata: `book_page`, `title`, `extraction_method`, `paraphrased`, `cost_usd`, `dpi`, and more
- **Page title extraction** — prominent display headings (chapter titles, appendix labels, etc.) are separated from body text into a dedicated `title` field
- **JPEG image encoding** — pages are sent as JPEG rather than PNG, reducing payload size 5–10× for scanned content with negligible OCR quality loss
- **Memory-efficient streaming** — one page image lives in RAM at a time; results stream to disk as they arrive
- **Live cost tracking** — per-page and total cost estimates fetched from litellm's pricing index at run time
- **Provider-agnostic** — swap between Anthropic and Gemini with a single CLI flag; the extraction logic is identical for both

---

## Fallback chain

Each page goes through the following stages, stopping as soon as a clean result is obtained:

```
Stage 1  Full-page verbatim extraction (1 API call)
         └─ if content filter blocks →

Stage 2  Quadrant split (4 API calls, one per quadrant)
         Each quadrant tries:
           a. Verbatim transcription
           b. Obfuscated retry (1.5° rotation + pixel noise)
           c. Paraphrase fallback — flagged for manual review
         └─ chunks stitched together with image reference (1 API call)
            └─ if stitch is blocked →

Stage 3  Full-page paraphrase — flagged for manual review
```

Pages that are image-only (no extractable text) are detected at the quadrant stage and short-circuited cleanly with an empty text record rather than wasting API calls.

---

## Output

For each run, four output artefacts are written under `output/<provider>_<pdf-stem>_<n>/`:

| File | Description |
|---|---|
| `extracted.jsonl` | One JSON record per page — the single source of truth |
| `extracted.md` | Combined markdown of all pages with separator comments, for visual inspection |
| `pages/page_NNN.md` | Per-page markdown with YAML frontmatter, derived from JSONL |
| `report.md` | Cost and quality report with per-page breakdown |

### JSONL record schema

```json
{
  "page":                  1,
  "model":                 "gemini-2.5-flash",
  "dpi":                   150,
  "book_page":             19,
  "title":                 "Ability Scores",
  "text":                  "...",
  "extraction_method":     "full-page",
  "paraphrased":           null,
  "chunk_warnings":        [],
  "error":                 null,
  "input_tokens":          718,
  "output_tokens":         1993,
  "cache_creation_tokens": 0,
  "cache_read_tokens":     0,
  "cost_usd":              0.002208,
  "elapsed_seconds":       24.4
}
```

**Key fields:**

- `book_page` — the printed page number from the document (may differ from PDF page index)
- `title` — prominent display heading if present (chapter title, appendix label, etc.), `null` otherwise; always plain text, no markdown
- `text` — extracted or paraphrased body text, excluding the page title; empty string for image-only pages
- `extraction_method` — `"full-page"` or `"quadrant-split"`
- `paraphrased` — `null` (verbatim), `"partial"` (some quadrants paraphrased), or `"full"` (whole page paraphrased)
- `error` — exception message if the page failed entirely, `null` otherwise

### Per-page markdown frontmatter

```yaml
---
pdf_page: 1
book_page: 19
source: Player_s_Handbook_Revised_2e.pdf
title: Ability Scores
---
```

### Combined markdown separators

```
<!-- ↓ PDF page: 1      book page: 19     extraction: full-page -->

# Ability Scores

...body text...
```

---

## Installation

Requires Python 3.11+ and [Poppler](https://poppler.freedesktop.org/) (for `pdf2image`).

```bash
# Install Poppler (macOS)
brew install poppler

# Install Poppler (Ubuntu/Debian)
apt-get install poppler-utils

# Install the package
pip install -e .
```

### Environment variables

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
```

---

## Usage

```bash
python -m pdfextract path/to/file.pdf [OPTIONS]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--provider` | `anthropic` | LLM provider: `anthropic` or `gemini` |
| `--model` | provider default | Model ID to use |
| `--dpi` | `150` | PDF render resolution. 150 is sufficient for most pages; use 200–300 for very small or dense text |

### Provider defaults

| Provider | Default model |
|---|---|
| `anthropic` | `claude-sonnet-4-20250514` |
| `gemini` | `gemini-2.5-flash-lite` |

### Examples

```bash
# Anthropic with default model
python -m pdfextract book.pdf

# Gemini Flash, higher DPI for dense tables
python -m pdfextract book.pdf --provider gemini --model gemini-2.5-flash --dpi 200

# Anthropic Haiku for lower cost
python -m pdfextract book.pdf --model claude-haiku-4-5-20251001
```

---

## Module structure

```
pdf_extractor/
├── __init__.py       Public API: iter_pages, stream_outputs, write_report, build_provider
├── cli.py            Argument parsing and run orchestration
├── extraction.py     Fallback chain, JSON response parsing, page iteration
├── image_utils.py    Quadrant splitting, JPEG encoding, obfuscation
├── output.py         JSONL, markdown, and report writing
├── pricing.py        Live cost estimation via litellm pricing index
├── prompts.py        All LLM prompt strings
└── providers.py      Anthropic and Gemini provider abstractions with retry logic
```

---

## Public API

The package can also be used programmatically:

```python
from pdf_extractor import build_provider, iter_pages, stream_outputs, write_report
from pathlib import Path

pdf = Path("book.pdf")
provider = build_provider("gemini", model="gemini-2.5-flash")

pages = iter_pages(pdf, provider, dpi=150)
saved, results = stream_outputs(pdf, pages, provider_name="gemini")
write_report(pdf, results, saved)
```

---

## Tests

```bash
python -m pytest tests/
```

The test suite (`tests/test_output_consistency.py`) validates the JSONL-as-source-of-truth invariant: every JSONL record has a corresponding markdown file, the markdown body matches the `text` field exactly, frontmatter fields agree with JSONL metadata, the combined markdown uses the correct separator format, and all required fields are present.

---

## Cost notes

- **Full-page extraction** costs 1 API call per page
- **Quadrant-split** costs 5 calls per page (4 quadrants + 1 stitch) — typically triggered by content filters on pages with artwork
- Pages are sent as JPEG at quality 85, which is 5–10× smaller than PNG for scanned content, directly reducing input token counts
- Cost estimates are fetched live from [litellm's pricing index](https://github.com/BerriAI/litellm) with hardcoded fallbacks for models not yet indexed
- The `report.md` in each run folder contains a full per-page cost and token breakdown
