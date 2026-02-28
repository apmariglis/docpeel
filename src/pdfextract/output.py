"""
Output serialisation: combined markdown, per-page markdowns, JSON, and
the cost/quality report.

All write functions accept an iterator of page result dicts and stream
to disk page-by-page, so the full results list never needs to live in RAM.
"""

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

OUTPUT_FOLDER = Path("output")


def _resolve_output_folder(pdf_path: Path, provider_name: str) -> Path:
    """
    Compute a per-run output folder for debugging purposes.

    Format: output/<provider>_<pdf-stem>_<n>
    where <n> is one greater than the highest existing run number for
    this provider+pdf combination, starting at 1 if none exist.

    This is intentionally separate from the main output flow so it can
    be removed or replaced without touching anything else.
    """
    base = OUTPUT_FOLDER
    pdf_stem = pdf_path.stem
    prefix = f"{provider_name}_{pdf_stem}_"

    existing = (
        [
            d
            for d in base.iterdir()
            if d.is_dir()
            and d.name.startswith(prefix)
            and d.name[len(prefix) :].isdigit()
        ]
        if base.exists()
        else []
    )

    next_n = max((int(d.name[len(prefix) :]) for d in existing), default=0) + 1
    return base / f"{prefix}{next_n}"


def _page_note_md(r: dict) -> str:
    """Build the italicised extraction note for the combined markdown."""
    note = "*[Extracted via quadrant-split fallback"
    paraphrased = r.get("paraphrased")
    if paraphrased == "full":
        note += " — ⚠️ full page paraphrased; manual review recommended"
    elif paraphrased == "partial":
        note += " — ⚠️ some chunks paraphrased; manual review recommended"
    elif r.get("chunk_warnings"):
        note += " — some chunks skipped"
    note += "]*"
    return note


def _page_status(r: dict) -> str:
    method = r.get("extraction_method", "full-page")
    if r.get("error") and method == "quadrant-split":
        return "quadrant-split + image-stitch (partial failure)"
    if r.get("error"):
        return "full-page (failed)"
    if method == "quadrant-split":
        paraphrased = r.get("paraphrased")
        suffix = f" + {paraphrased}" if paraphrased else ""
        return f"quadrant-split + image-stitch{suffix}"
    return "full-page"


def _page_notes(r: dict) -> str:
    notes = []
    paraphrased = r.get("paraphrased")
    if paraphrased == "full":
        notes.append("⚠️ full page paraphrased")
    elif paraphrased == "partial":
        notes.append("⚠️ partial paraphrase")
    skipped = [
        w
        for w in r.get("chunk_warnings", [])
        if "blocked" in w and "paraphrase" not in w and "skipped" in w
    ]
    if skipped:
        labels = []
        for w in skipped:
            for candidate in ["top-left", "top-right", "bottom-left", "bottom-right"]:
                if candidate in w:
                    labels.append(candidate)
                    break
        notes.append(
            f"⛔ skipped: {', '.join(labels) if labels else str(len(skipped))}"
        )
    return "; ".join(notes) if notes else "—"


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
            # ── Per-page markdown (with YAML frontmatter) ────────────────────
            page_md = pages_dir / f"page_{r['page']:03d}.md"
            error_line = f"error: {r['error']}\n" if r.get("error") else ""
            title_line = f"title: {r['title']}\n" if r.get("title") else ""
            frontmatter = (
                f"---\n"
                f"pdf_page: {r['page']}\n"
                f"book_page: {r['book_page'] if r['book_page'] is not None else 'null'}\n"
                f"source: {pdf_path.name}\n"
                f"{title_line}"
                f"{error_line}"
                f"---\n\n"
            )
            page_md.write_text(frontmatter + r["text"], encoding="utf-8")

            # ── Combined markdown (append) ────────────────────────────────────
            book_page_str = (
                str(r["book_page"]) if r["book_page"] is not None else "unknown"
            )
            method_str = _page_status(r)
            mid_content = f"PDF page: {r['page']:<6} book page: {book_page_str:<6} extraction: {method_str}"
            separator = f"\n<!-- ↓ {mid_content} -->\n\n"
            md_f.write(separator)
            if r.get("title"):
                md_f.write(f"# {r['title']}\n\n")
            if r.get("extraction_method") == "quadrant-split":
                md_f.write(_page_note_md(r) + "\n\n")
            md_f.write(r["text"])
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
    quad_ok = [
        r
        for r in results
        if not r.get("error") and r.get("extraction_method") == "quadrant-split"
    ]
    successful = [r for r in results if not r.get("error")]
    fp_ok = [
        r
        for r in results
        if r.get("extraction_method") == "full-page" and not r.get("error")
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

    avg_time = total_time / n_pages if n_pages else 0
    cost_per_page = total_cost / n_pages if n_pages else 0
    cost_per_successful = total_cost / len(successful) if successful else 0
    tokens_per_page = total_tokens / len(successful) if successful else 0
    fastest = min(results, key=lambda r: r["elapsed_seconds"])
    slowest = max(results, key=lambda r: r["elapsed_seconds"])

    page_rows = "\n".join(
        "| {pg} | {st} | {inp} | {out} | {cost} | {t} | {n} |".format(
            pg=r["page"],
            st=_page_status(r),
            inp=f"{r['input_tokens']:,}",
            out=f"{r['output_tokens']:,}",
            cost=f"${r['cost_usd']:.6f}",
            t=f"{r['elapsed_seconds']:.1f}s",
            n=_page_notes(r),
        )
        for r in results
    )

    cache_rows = []
    if total_cw or total_cr:
        cache_rows = [
            f"| Cache-write tokens | {total_cw:,} |",
            f"| Cache-read tokens  | {total_cr:,} |",
        ]

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
        f"| Pages extracted (full-page) | {len(fp_ok)} |",
        f"| Pages extracted (quadrant-split) | {len(quad_ok)} |",
        f"| Pages partially extracted | {len(partial)} |",
        f"| Pages failed entirely | {len(failed)} |",
        f"| Total input tokens | {total_input:,} |",
        f"| Total output tokens | {total_output:,} |",
        *cache_rows,
        f"| **Total tokens** | **{total_tokens:,}** |",
        f"| **Total cost (USD)** | **${total_cost:.6f}** |",
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
        "## Per-Page Breakdown",
        "",
        "| Page | Status | Input tokens | Output tokens | Cost (USD) | Time | Notes |",
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
        "*Cost figures are estimates based on public pricing fetched at run-time.*",
        "*Quadrant-split pages incur 5 API calls (4 chunks + 1 stitch) instead of 1.*",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
