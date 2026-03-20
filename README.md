# docpeel

A Python CLI tool for extracting text and tables from PDFs using vision LLMs (Claude, Gemini, and Mistral). Designed to produce clean, structured output for use in RAG (Retrieval-Augmented Generation) pipelines and search indexes — prioritising accuracy over speed.

## Capabilities and current limitations

docpeel works best on structured, book-like PDFs: printed text, tables, headings, and artwork on a standard page grid. It has been tested most thoroughly on dense, table-heavy reference books.

**Works well:**
- Pages with many tables, including tables that span multiple pages
- Mixed text-and-artwork layouts — artwork and image-only pages are detected and handled cleanly; text pages are unaffected
- Scanned PDFs where no embedded text exists
- Pages blocked by content filters (automatic fallback chain)
- Standard book boilerplate (table of contents, index, blank pages) — detected and excluded

**Known limitations:**
- **Decorative sidebars** — pages with a visually distinct margin column (coloured strip, short atmospheric phrases) may have that content mixed into the body text
- **Diagram text** — text inside figures, flow charts, or labelled illustrations is not extracted
- **Two-column dense layouts** — reading order reconstruction can fail on tightly packed two-column text
- **Embedded-text PDFs** — every page is rendered to an image; native text is not reused
- **Non-Latin scripts and low-quality scans** — untested / limited effectiveness

See [STATUS.md](STATUS.md) for a more detailed breakdown and planned improvements.

Extraction quality is highly dependent on document layout, and the tool has been tested against a limited range of formats so far. If you try it on a document it handles poorly, opening an issue with a description of the layout and what went wrong is very welcome — it's the main way the extraction logic improves.

---

## Why vision LLMs?

Traditional PDF parsers (pdfminer, PyMuPDF, etc.) fail on scanned PDFs, multi-column layouts, complex tables, and pages that mix text with illustrations. Vision LLMs read the page as an image, handling all of these naturally. The tradeoff is cost and latency — this tool is optimised to minimise both while maximising extraction quality.

---

## Features

- **Multi-stage fallback chain** per page — escalates automatically from fast single-call extraction to quadrant splitting, obfuscation, and paraphrase fallback when content filters or layout complexity intervene
- **Image-assisted stitching** — after quadrant extraction, a final LLM call sees both the original full-page image and the four quadrant transcriptions together, correcting reading order and reassembling split tables
- **Structured JSONL output** — one record per page with full metadata: `book_page`, `title`, `tables`, `extraction_method`, `paraphrased`, `cost_usd`, and more; ready for downstream processing or search indexing
- **Table extraction** — tables are lifted out of the page text into a structured `tables` array with a descriptive caption per table, appended after the page prose in rendered markdown
- **Page title extraction** — prominent display headings (chapter titles, appendix labels, etc.) are separated from body text into a dedicated `title` field
- **JPEG image encoding** — pages are sent as JPEG rather than PNG, reducing payload size 5–10× for scanned content with negligible OCR quality loss
- **Memory-efficient streaming** — one page image lives in RAM at a time; results stream to disk as they arrive
- **Live cost tracking** — per-page and total cost estimates fetched from litellm's pricing index at run time
- **Three providers** — swap between Anthropic, Gemini, and Mistral with a single CLI flag; structured output is enforced natively for each (tool-calling, response schema, and prompted JSON respectively)

---

## Fallback chain

Applies to Anthropic and Gemini providers. Each page goes through the following stages, stopping as soon as a clean result is obtained:

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

Mistral uses a two-step OCR + structuring pipeline (`mistral-ocr-latest` → cheap chat model) with no fallback chain — Mistral OCR has no content filter.

Pages that are image-only (no extractable text) are detected at the quadrant stage and short-circuited cleanly with an empty text record rather than wasting API calls.

---

## Output

For each run, four output artefacts are written under `output/<provider>_<pdf-stem>_<n>/`. The JSONL output is designed to be straightforward to ingest into search indexes, vector databases, or RAG pipelines, but the tool works equally well as a standalone PDF digitiser. [ask-the-book](https://github.com/apmariglis/ask-the-book) is a companion RAG system that ingests this JSONL format directly.

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
  "model":                 "claude-sonnet-4-0",
  "dpi":                   150,
  "skip":                  false,
  "skip_reason":           null,
  "book_page":             19,
  "title":                 "Chapter 4: Experimental Results",
  "text":                  "...",
  "tables": [
    {
      "title":   "Table 3: Annual Rainfall by Region",
      "caption": "Rainfall measurements in millimetres across six regions over a five-year period.",
      "content": "| Region | 2019 | ... |"
    }
  ],
  "watermarks":            [],
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
- `text` — verbatim body text, excluding the page title and all table bodies; empty string for skipped or image-only pages
- `tables` — array of extracted tables, each with `title` (heading label), `caption` (description for search and retrieval), and `content` (full markdown table)
- `skip` / `skip_reason` — `true` for pages excluded from the output (table of contents, index, blank, illustration-only, title page, part divider)
- `watermarks` — array of watermark strings detected and excluded from the text
- `extraction_method` — `"full-page"`, `"quadrant-split"`, or `"ocr+structure"` (Mistral)
- `paraphrased` — `null` (verbatim), `"partial"` (some quadrants paraphrased), or `"full"` (whole page paraphrased); always `null` for Mistral
- `error` — exception message if the page failed entirely, `null` otherwise

### Per-page markdown

Each `pages/page_NNN.md` file has YAML frontmatter followed by the page body. Tables are appended after the prose in document order:

```markdown
---
pdf_page: 1
book_page: 42
source: my_document.pdf
title: Chapter 4: Experimental Results
tables: 1
---

Body text here...

<!-- table: Table 3: Annual Rainfall by Region -->

| Region | 2019 | 2020 | ...
```

---

## Installation

Requires Python 3.10+.

**Install docpeel:**

```bash
git clone https://github.com/apmariglis/docpeel.git
cd docpeel
pip install -e .
```

### API keys

Create a `.env` file in the project root (see `.env.example`):

```
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
MISTRAL_API_KEY=...
```

Only the key for the provider you intend to use is required.

---

## Usage

```bash
docpeel path/to/file.pdf [OPTIONS]
```

Or equivalently:

```bash
python -m docpeel path/to/file.pdf [OPTIONS]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--vision-model MODEL` | — | Vision model for direct image-to-text extraction. Provider inferred from name prefix (`claude-*` → Anthropic, `gemini-*` → Gemini). |
| `--ocr ENGINE` | — | OCR engine for a two-step pipeline (currently: `mistral`). Must be paired with `--structure-model`. Mutually exclusive with `--vision-model`. |
| `--structure-model MODEL` | — | LLM for the structuring step (requires `--ocr`). Supports `mistral-*`, `gemini-*`, and `claude-*` models. |
| `--pages PAGES` | all | Pages to extract. Accepts comma-separated numbers and/or ranges, e.g. `3`, `1,3,5`, `2-5`, `1,3,7-10`. |
| `--dpi N` | `150` | PDF render resolution. 150 is sufficient for most pages; use 200–300 for very small or dense text. |
| `--verbose` / `-v` | — | Enable debug logging (per-chunk detail, stitch steps). |
| `--quiet` / `-q` | — | Suppress progress messages; show only warnings and errors. |

### Examples

```bash
# Anthropic Sonnet (vision path)
docpeel book.pdf --vision-model claude-sonnet-4-6

# Gemini Flash at higher DPI for dense tables
docpeel book.pdf --vision-model gemini-2.5-flash --dpi 200

# Mistral OCR + Mistral chat structuring (cheapest option)
docpeel book.pdf --ocr mistral --structure-model mistral-small-latest

# Mistral OCR + Gemini structuring
docpeel book.pdf --ocr mistral --structure-model gemini-2.5-flash-lite

# Mistral OCR + Claude structuring
docpeel book.pdf --ocr mistral --structure-model claude-haiku-4-5-20251001

# Extract only specific pages
docpeel book.pdf --vision-model claude-sonnet-4-6 --pages 1,5,10-15
```

---

## Module structure

```
docpeel/
├── cli.py                    Argument parsing and run orchestration
├── extraction.py             Per-page orchestration: VisionExtractor, MistralExtractor, iter_pages
├── image_utils.py            Quadrant splitting, JPEG encoding, obfuscation
├── output.py                 JSONL, markdown, and report writing
├── pricing.py                Live cost estimation via litellm pricing index
├── prompts.py                All LLM prompt strings
└── providers/
    ├── base.py               Usage NamedTuple, PAGE_EXTRACTION_SCHEMA, VisionProvider ABC, retry helper
    ├── anthropic.py          AnthropicProvider — tool_choice forced structured output
    ├── gemini.py             GeminiProvider — response_schema structured output
    ├── mistral.py            MistralProvider — two-step OCR + structuring pipeline
    └── provider_factory.py   build_provider() factory
```

---

## Programmatic use

The package can also be used directly in Python:

```python
from docpeel.providers.provider_factory import build_provider
from docpeel.extraction import iter_pages
from docpeel.output import stream_outputs, write_report
from pathlib import Path

pdf = Path("book.pdf")

# Vision path (Anthropic or Gemini)
provider = build_provider(vision_model="claude-sonnet-4-6")

# OCR path (Mistral OCR + any structure model)
# provider = build_provider(ocr="mistral", structure_model="mistral-small-latest")

provider.resolve_model_id()
pages = iter_pages(pdf, provider, dpi=150)
saved, results = stream_outputs(pdf, pages, provider_name=provider.model_id)
write_report(pdf, results, saved)
```

---

## Benchmarking and feedback

The `scripts/benchmark/` folder contains a script that runs docpeel against a set of pages you define, evaluates the extraction quality using a vision LLM judge, and produces a markdown report. It's useful for testing how well the tool handles a specific document before committing to a full run.

```bash
# Copy the example config, fill in your PDF path and pages, then run:
cp scripts/benchmark/benchmarks.example.yaml scripts/benchmark/my-benchmarks.yaml
# edit my-benchmarks.yaml
python scripts/benchmark/benchmark.py --config scripts/benchmark/my-benchmarks.yaml
```

If you try it on a document that the tool handles poorly, sharing the benchmark report **and the page images** (saved under `scripts/benchmark/benchmark_results/.../extractions/`) in an issue is the most useful form of feedback — it gives concrete, per-page quality scores alongside the actual source material, making it much easier to diagnose and fix the issue.

See [`scripts/benchmark/benchmarks.example.yaml`](scripts/benchmark/benchmarks.example.yaml) for the full list of options.

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

---

## Cost notes

- **Full-page extraction** costs 1 API call per page
- **Quadrant-split** costs 5 calls per page (4 quadrants + 1 stitch) — typically triggered by content filters on pages with artwork
- **Mistral** charges per OCR page plus chat tokens for the structuring step
- Pages are sent as JPEG at quality 85, which is 5–10× smaller than PNG for scanned content, directly reducing input token counts
- Cost estimates are fetched live from [litellm's pricing index](https://github.com/BerriAI/litellm) with hardcoded fallbacks for models not yet indexed
- The `report.md` in each run folder contains a full per-page cost and token breakdown
