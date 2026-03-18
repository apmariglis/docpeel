"""
Output serialisation: combined markdown, per-page markdowns, JSON, and
the cost/quality report.

All write functions accept an iterator of page result dicts and stream
to disk page-by-page, so the full results list never needs to live in RAM.
"""

import json
import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

OUTPUT_FOLDER = Path("output")


def _resolve_output_folder(pdf_path: Path, provider_name: str) -> Path:
    """
    Compute a per-run output folder.

    Format: output/<provider>__<pdf-stem>[<n>]
    where <n> is one greater than the highest existing run number for
    this provider+pdf combination, starting at 1 if none exist.
    Double underscore separates model(s) from PDF stem; brackets isolate run number.
    """
    base = OUTPUT_FOLDER
    pdf_stem = pdf_path.stem
    prefix = f"{provider_name}__{pdf_stem}["

    existing = (
        [
            d
            for d in base.iterdir()
            if d.is_dir()
            and d.name.startswith(prefix)
            and re.fullmatch(r"\d+", d.name[len(prefix):-1])
            and d.name.endswith("]")
        ]
        if base.exists()
        else []
    )

    next_n = (
        max((int(d.name[len(prefix):-1]) for d in existing), default=0) + 1
    )
    return base / f"{prefix}{next_n}]"


def _page_note_md(r: dict) -> str:
    """Build the italicised extraction note for the combined markdown."""
    note = "*[Extracted via quadrant-split"
    paraphrased = r.get("paraphrased")
    if paraphrased == "full":
        note += " — ⚠️ full page paraphrased; manual review recommended"
    elif paraphrased == "partial":
        note += " — ⚠️ some chunks paraphrased; manual review recommended"
    elif r.get("chunk_warnings"):
        note += " — ⚠️ some chunks skipped"
    note += "]*"
    return note


def _method_col(r: dict) -> str:
    """Human-readable extraction method for the report table."""
    if r.get("skip"):
        return "—"
    method = r.get("extraction_method", "full-page")
    if r.get("error"):
        # Show the correct method label even for failed pages
        return method
    return method


def _outcome_col(r: dict) -> str:
    """Human-readable outcome for the report table."""
    if r.get("skip"):
        return f"⏭ skipped ({r.get('skip_reason', 'unknown')})"
    if r.get("error"):
        return "⛔ failed"
    paraphrased = r.get("paraphrased")
    chunk_warnings = r.get("chunk_warnings", [])
    skipped_chunks = [
        w
        for w in chunk_warnings
        if "blocked" in w and "paraphrase" not in w and "skipped" in w
    ]
    if paraphrased == "full":
        return "⚠️ paraphrased"
    if paraphrased == "partial":
        if skipped_chunks:
            return "⛔ chunks missing + partial paraphrase"
        return "⚠️ partial paraphrase"
    if skipped_chunks:
        return "⛔ chunks missing"
    return "✅ ok"


def resolve_run_folder(pdf_path: Path, provider_name: str) -> Path:
    """
    Return the output folder path that will be used for a given pdf + provider
    combination, without creating it. Useful for printing the destination early
    in the CLI before extraction begins.
    """
    return _resolve_output_folder(pdf_path, provider_name)


def stream_outputs(
    pdf_path: Path,
    pages: Iterator[dict],
    provider_name: str = "",
) -> tuple[dict[str, Path], list[dict]]:
    """
    Consume the page iterator, writing each page to disk immediately.

    Writes:
      - combined markdown (appended page by page)
      - per-page markdown files
      - newline-delimited JSON (one object per line, appended page by page)

    Returns (paths_dict, results) where results is the accumulated list
    needed by write_report. The result dicts do NOT include the page text
    (dropped after writing) to keep memory low.
    """
    run_folder = (
        _resolve_output_folder(pdf_path, provider_name)
        if provider_name
        else OUTPUT_FOLDER
    )
    run_folder.mkdir(parents=True, exist_ok=True)

    combined_md = run_folder / "extracted.md"
    pages_dir = run_folder / "pages"
    json_file = run_folder / "extracted.jsonl"

    pages_dir.mkdir(exist_ok=True)

    # Write combined markdown header
    combined_md.write_text(f"# Extracted text — {pdf_path.name}\n", encoding="utf-8")

    results = []  # metadata only — no text field kept in RAM

    with (
        combined_md.open("a", encoding="utf-8") as md_f,
        json_file.open("w", encoding="utf-8") as json_f,
    ):
        for r in pages:
            tables: list[dict] = r.get("tables") or []
            is_skip = r.get("skip", False)

            # ── Per-page markdown (with YAML frontmatter) ────────────────────
            page_md = pages_dir / f"page_{r['page']:03d}.md"
            error_line = f"error: {r['error']}\n" if r.get("error") else ""
            title_line = f"title: {r['title']}\n" if r.get("title") else ""
            table_count_line = f"tables: {len(tables)}\n" if tables else ""
            skip_line = (
                f"skip: true\nskip_reason: {r['skip_reason']}\n" if is_skip else ""
            )
            watermarks = r.get("watermarks") or []
            watermark_lines = (
                "watermarks:\n" + "".join(f"  - {w}\n" for w in watermarks)
                if watermarks
                else ""
            )
            frontmatter = (
                f"---\n"
                f"pdf_page: {r['page']}\n"
                f"book_page: {r['book_page'] if r['book_page'] is not None else 'null'}\n"
                f"source: {pdf_path.name}\n"
                f"{title_line}"
                f"{table_count_line}"
                f"{watermark_lines}"
                f"{skip_line}"
                f"{error_line}"
                f"---\n\n"
            )

            if is_skip:
                # Skipped pages get a minimal stub — no body content
                page_md.write_text(frontmatter, encoding="utf-8")
            else:
                # Build body: prose text followed by all tables in order
                parts = [r["text"]]
                for tbl in tables:
                    title = tbl.get("title") or ""
                    caption = tbl.get("caption", "").strip()
                    if title and caption:
                        label = f"{title} — {caption}"
                    else:
                        label = title or caption
                    if label:
                        parts.append(f"<!-- table: {label} -->")
                    content = tbl.get("content", "").strip()
                    if content:
                        parts.append(content)
                body = "\n\n".join(filter(None, parts))
                page_md.write_text(frontmatter + body, encoding="utf-8")

            # ── Combined markdown (append) ────────────────────────────────────
            book_page_str = (
                str(r["book_page"]) if r["book_page"] is not None else "unknown"
            )
            mid_content = (
                f"PDF page: {r['page']:<6} book page: {book_page_str:<6} "
                f"method: {_method_col(r)}  outcome: {_outcome_col(r)}"
            )
            separator = f"\n<!-- ↓ {mid_content} -->\n\n"
            md_f.write(separator)

            if is_skip:
                md_f.write(f"<!-- skipped: {r.get('skip_reason', 'unknown')} -->\n")
            else:
                if r.get("title"):
                    md_f.write(f"# {r['title']}\n\n")
                if r.get("extraction_method") == "quadrant-split":
                    md_f.write(_page_note_md(r) + "\n\n")
                md_f.write(body)
                md_f.write("\n")
            md_f.flush()

            # ── Newline-delimited JSON (append) ───────────────────────────────
            json_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            json_f.flush()

            # ── Keep metadata for report, drop text to free RAM ───────────────
            r_meta = {k: v for k, v in r.items() if k != "text"}
            results.append(r_meta)

    return {
        "combined_md": combined_md,
        "pages_dir": pages_dir,
        "json": json_file,
        "run_folder": run_folder,
    }, results


def write_report(pdf_path: Path, results: list[dict], saved: dict[str, Path]) -> Path:
    """Write the markdown cost/quality report and return its path."""
    run_folder = saved.get("run_folder", OUTPUT_FOLDER)
    run_folder.mkdir(parents=True, exist_ok=True)
    report_path = run_folder / "report.md"

    n_pages = len(results)
    skipped = [r for r in results if r.get("skip")]
    failed = [
        r
        for r in results
        if r.get("error") and r.get("extraction_method") != "quadrant-split"
    ]
    partial = [
        r
        for r in results
        if r.get("error") and r.get("extraction_method") == "quadrant-split"
    ]
    chunk_missing = [
        r
        for r in results
        if not r.get("error")
        and not r.get("skip")
        and any(
            "blocked" in w and "paraphrase" not in w and "skipped" in w
            for w in r.get("chunk_warnings", [])
        )
    ]
    quad_ok = [
        r
        for r in results
        if not r.get("error") and r.get("extraction_method") == "quadrant-split"
    ]
    ocr_ok = [
        r
        for r in results
        if not r.get("error") and r.get("extraction_method") == "ocr+structure"
    ]
    successful = [r for r in results if not r.get("error") and not r.get("skip")]
    fp_ok = [
        r
        for r in results
        if r.get("extraction_method") == "full-page"
        and not r.get("error")
        and not r.get("skip")
    ]

    total_input = sum(r["input_tokens"] for r in results)
    total_output = sum(r["output_tokens"] for r in results)
    total_cw = sum(r["cache_creation_tokens"] for r in results)
    total_cr = sum(r["cache_read_tokens"] for r in results)
    total_tokens = total_input + total_output
    total_cost = sum(r["cost_usd"] for r in results)
    total_time = sum(r["elapsed_seconds"] for r in results)
    model = results[0]["model"] if results else "n/a"
    dpi = results[0]["dpi"] if results else "n/a"

    # Mistral-specific cost split (only present when ocr_cost_usd field exists)
    mistral_pages = [r for r in results if "ocr_cost_usd" in r]
    total_ocr_cost = sum(r["ocr_cost_usd"] for r in mistral_pages)
    total_structure_cost = sum(r["structure_cost_usd"] for r in mistral_pages)

    avg_time = total_time / n_pages if n_pages else 0
    cost_per_page = total_cost / n_pages if n_pages else 0
    cost_per_successful = total_cost / len(successful) if successful else 0
    tokens_per_page = total_tokens / len(successful) if successful else 0
    fastest = min(results, key=lambda r: r["elapsed_seconds"])
    slowest = max(results, key=lambda r: r["elapsed_seconds"])

    page_rows = "\n".join(
        "| {pg} | {method} | {outcome} | {inp} | {out} | {cost} | {t} |".format(
            pg=r["page"],
            method=_method_col(r),
            outcome=_outcome_col(r),
            inp=f"{r['input_tokens']:,}",
            out=f"{r['output_tokens']:,}",
            cost=f"${r['cost_usd']:.6f}",
            t=f"{r['elapsed_seconds']:.1f}s",
        )
        for r in results
    )

    cache_rows = []
    if total_cw or total_cr:
        cache_rows = [
            f"| Cache-write tokens | {total_cw:,} |",
            f"| Cache-read tokens  | {total_cr:,} |",
        ]

    # Mistral cost breakdown rows (only shown for Mistral runs)
    mistral_cost_rows = []
    if mistral_pages:
        mistral_cost_rows = [
            f"| OCR cost (mistral-ocr-latest) | ${total_ocr_cost:.6f} |",
            f"| Structure cost (chat model) | ${total_structure_cost:.6f} |",
        ]

    ocr_row = (
        [f"| Pages extracted (ocr+structure) | {len(ocr_ok)} |"]
        if ocr_ok or any(r.get("extraction_method") == "ocr+structure" for r in results)
        else []
    )

    lines = [
        "# PDF Extraction Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Source PDF:** `{pdf_path}`  ",
        f"**Model:** `{model}`  ",
        f"**Render DPI:** {dpi}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Render DPI | {dpi} |",
        f"| Pages total | {n_pages} |",
        f"| Pages skipped | {len(skipped)} |",
        f"| Pages extracted (full-page) | {len(fp_ok)} |",
        f"| Pages extracted (quadrant-split) | {len(quad_ok)} |",
        *ocr_row,
        f"| Pages partially extracted | {len(partial)} |",
        f"| Pages with missing chunks | {len(chunk_missing)} |",
        f"| Pages failed entirely | {len(failed)} |",
        f"| Total input tokens | {total_input:,} |",
        f"| Total output tokens | {total_output:,} |",
        *cache_rows,
        f"| **Total tokens** | **{total_tokens:,}** |",
        f"| **Total cost (USD)** | **${total_cost:.6f}** |",
        *mistral_cost_rows,
        f"| Cost per page (all) | ${cost_per_page:.6f} |",
        f"| Cost per successful page | ${cost_per_successful:.6f} |",
        f"| Avg tokens per successful page | {tokens_per_page:,.0f} |",
        f"| Total processing time | {total_time:.1f}s ({total_time / 60:.1f} min) |",
        f"| Avg time per page | {avg_time:.1f}s |",
        f"| Fastest page | {fastest['elapsed_seconds']:.1f}s (page {fastest['page']}) |",
        f"| Slowest page | {slowest['elapsed_seconds']:.1f}s (page {slowest['page']}) |",
        "",
        "---",
        "",
    ]

    # ── Extraction Failures section (only shown when there are failures) ──────
    if failed or partial or chunk_missing:
        all_failed = failed + partial
        failure_rows = "\n".join(
            f"| {r['page']} | {r.get('extraction_method', 'full-page')} "
            f"| {r.get('error', '—')} |"
            for r in sorted(all_failed, key=lambda r: r["page"])
        )
        missing_rows = "\n".join(
            f"| {r['page']} | quadrant-split "
            f"| {'; '.join(w for w in r.get('chunk_warnings', []) if 'skipped' in w)} |"
            for r in sorted(chunk_missing, key=lambda r: r["page"])
        )
        all_rows = "\n".join(filter(None, [failure_rows, missing_rows]))
        lines += [
            "## ⛔ Extraction Failures",
            "",
            "> **These pages have content MISSING from the output. "
            "The extracted data is incomplete. "
            "Manual intervention or a re-run is required for the pages listed below.**",
            "",
            "| Page | Method | Detail |",
            "|---|---|---|",
            all_rows,
            "",
            "---",
            "",
        ]

    lines += [
        "## Per-Page Breakdown",
        "",
        "| Page | Method | Outcome | Input tokens | Output tokens | Cost (USD) | Time |",
        "|---|---|---|---|---|---|---|",
        page_rows,
        "",
        "---",
        "",
        "## Output Files",
        "",
        "| File | Path |",
        "|---|---|",
        f"| Combined markdown | `{saved['combined_md']}` |",
        f"| Per-page markdowns | `{saved['pages_dir']}/` |",
        f"| Raw JSON (JSONL) | `{saved['json']}` |",
        f"| This report | `{report_path}` |",
        "",
        "---",
        "",
        "## Glossary",
        "",
        "*This section is static documentation, not specific to this run.*",
        "",
        "### Extraction Method",
        "",
        "| Method | Description |",
        "|---|---|",
        "| `full-page` | The page was sent as a single image to the LLM and extracted in one API call. This is the default fast path. |",
        "| `quadrant-split` | The full-page call was blocked by the content filter. The page was split into 4 quadrant images, each extracted separately, then merged by a final LLM call that also receives the original page image for reading-order verification and error correction. Costs 5 API calls instead of 1. |",
        "| `ocr+structure` | Mistral two-step pipeline: `mistral-ocr-latest` extracts raw markdown from the page image (billed per page), then a chat model converts that markdown into the structured schema (billed per token). No content filter, so no fallback chain. |",
        "| `—` | No extraction attempted (page was intentionally skipped). |",
        "",
        "### Outcome",
        "",
        "| Outcome | Description |",
        "|---|---|",
        "| ✅ ok | Extracted cleanly with no issues. Content is verbatim from the source. |",
        "| ⚠️ paraphrased | The LLM was blocked from verbatim transcription (copyright / content filter) and rewrote the entire page in different words. Factual content is preserved but exact wording differs from the original. Manual review recommended. |",
        "| ⚠️ partial paraphrase | Same as above but only for some quadrants; the remainder were transcribed verbatim. |",
        "| ⛔ chunks missing | One or more quadrants were blocked even on paraphrase and dropped entirely. Content from this page is **definitively missing** from the output — treat this the same as a failed page. |",
        "| ⛔ chunks missing + partial paraphrase | Combination of the two above — some chunks dropped, others paraphrased. |",
        "| ⏭ skipped | Page intentionally excluded from the output. Reasons: `table_of_contents` — lists chapter titles and page numbers; `index` — alphabetical term index at the back; `blank` — empty or 'intentionally left blank'; `illustration_only` — full-page artwork with no text; `title_page` — only the book or series title; `part_divider` — decorative page announcing a part or section number. |",
        "| ⛔ failed | An unrecoverable error prevented extraction. This page has **no content** in the output. The extracted data is incomplete — re-run or manually extract this page. |",
        "",
        "---",
        "",
        "*Cost figures are estimates based on public pricing fetched at run-time.*",
        "*Quadrant-split pages incur 5 API calls (4 chunks + 1 stitch) instead of 1.*",
        "*Mistral ocr+structure pages incur 2 separate charges: OCR billed per page, structuring billed per token.*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
