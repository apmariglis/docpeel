#!/usr/bin/env python3
"""
Benchmark docpeel extraction quality across all configured extraction paths.

For each benchmark entry (PDF + page range), every configured extraction path
is attempted — paths where a required API key is absent are skipped and noted
in the report. Results are evaluated by a vision LLM judge (default:
claude-sonnet-4-6) that sees the original page image alongside the anonymised
extracted content.  Anonymisation strips all fields that could reveal which
model produced the output, so the judge scores blindly.

Usage:
    python scripts/benchmark/benchmark.py --config scripts/benchmark/my-benchmarks.yaml
    python scripts/benchmark/benchmark.py --config scripts/benchmark/my-benchmarks.yaml --judge-model claude-sonnet-4-6

Copy scripts/benchmark/benchmarks.example.yaml, fill in your PDF paths and pages, then run:
    pip install -e .
    python scripts/benchmark/benchmark.py --config scripts/benchmark/my-benchmarks.yaml
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pdf2image import convert_from_path

try:
    from docpeel.providers.provider_factory import build_provider
    from docpeel.extraction import iter_pages
    from docpeel.cli import _parse_pages
    from docpeel.pricing import anthropic_cost
except ImportError as exc:
    sys.exit(
        f"Cannot import docpeel: {exc}\n"
        "Run 'pip install -e .' from the project root first."
    )

load_dotenv()

# ---------------------------------------------------------------------------
# Fields stripped before sending extracted content to the judge.
# Goal: remove anything that could identify which model/path produced the output.
# ---------------------------------------------------------------------------

_STRIP_FIELDS = {
    "model",            # direct model name
    "dpi",              # not useful to judge
    "extraction_method",  # "ocr+structure" reveals Mistral; quadrant reveals vision fallback
    "paraphrased",      # non-null only on Anthropic/Gemini vision paths
    "chunk_warnings",   # quadrant-split mentions reveal vision path
    "input_tokens",
    "output_tokens",
    "cache_creation_tokens",
    "cache_read_tokens",
    "cost_usd",
    "elapsed_seconds",
}

# ---------------------------------------------------------------------------
# Required environment variable per model-name prefix
# ---------------------------------------------------------------------------

_KEY_FOR_PREFIX = {
    "claude":  "ANTHROPIC_API_KEY",
    "gemini":  "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
DEFAULT_OUTPUT_DIR  = "scripts/benchmark/benchmark_results"
DEFAULT_DPI         = 150

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are evaluating the quality of an automated PDF text extraction tool.
You will be shown the original PDF page as an image and the extraction result
as a JSON record. Compare the extracted content against the page image and
assess accuracy.

Return a JSON object with exactly these fields:
{
  "text_score": <integer 0-10>,
  "text_notes": "<specific issues, or 'Accurate' if none>",
  "tables_score": <integer 0-10, or null if no tables on this page>,
  "tables_notes": "<table issues, or null if no tables>",
  "skip_correct": <true if the skip decision was appropriate, false otherwise>,
  "skip_notes": "<explain if incorrect, else null>",
  "title_correct": <true/false, or null if no clear title exists on the page>,
  "title_notes": "<explain if incorrect, else null>",
  "page_number_correct": <true/false, or null if no page number is visible>,
  "overall_score": <integer 0-10>,
  "summary": "<1-2 sentence summary of extraction quality and key issues>"
}

Scoring guide:
  10  Perfect — complete and verbatim
  8-9 Very good — minor formatting differences, no content missing
  6-7 Good — some omissions or errors, core content intact
  4-5 Partial — notable gaps or errors affecting usability
  2-3 Poor — significant content missing or incorrect
  0-1 Very poor — mostly wrong or missing

Return ONLY the JSON object, no explanation or markdown fences.

Extracted content:
"""


# ===========================================================================
# Pure helper functions (unit-tested in tests/test_benchmark.py)
# ===========================================================================


def _anonymize_record(record: dict) -> dict:
    """Return a copy of record with all identifying fields removed."""
    return {k: v for k, v in record.items() if k not in _STRIP_FIELDS}


def _path_label(path: dict) -> str:
    """Human-readable label for an extraction path config dict."""
    if "vision_model" in path:
        return path["vision_model"]
    return f"mistral-ocr+{path['structure_model']}"


def _required_env_keys(path: dict) -> set[str]:
    """Return the set of environment variable names required for this path."""
    keys: set[str] = set()
    if "vision_model" in path:
        model = path["vision_model"]
        for prefix, key in _KEY_FOR_PREFIX.items():
            if model.startswith(prefix):
                keys.add(key)
                break
    else:
        keys.add("MISTRAL_API_KEY")
        model = path.get("structure_model", "")
        for prefix, key in _KEY_FOR_PREFIX.items():
            if model.startswith(prefix) and key != "MISTRAL_API_KEY":
                keys.add(key)
                break
    return keys


def _is_path_available(path: dict, env: dict) -> bool:
    """Return True if all required API keys for this path are present in env."""
    return _required_env_keys(path).issubset(env.keys())


def _bench_label(pdf_path: Path, pages: str) -> str:
    """Folder-safe label for a benchmark entry, e.g. 'my_book_pdf__pp_1_3_7-10'."""
    stem = re.sub(r"[^\w]", "_", pdf_path.name)   # replace dots, spaces, etc.
    stem = re.sub(r"_+", "_", stem).strip("_")
    pages_safe = re.sub(r"[,\s]", "_", pages)      # commas and spaces → underscores
    pages_safe = re.sub(r"_+", "_", pages_safe).strip("_")
    return f"{stem}__pp_{pages_safe}"


def _safe_filename(label: str) -> str:
    """Convert a path label to a filesystem-safe folder name."""
    return re.sub(r"[^\w.+-]", "_", label)


# ===========================================================================
# Report rendering
# ===========================================================================


def _avg(scores: list) -> float | None:
    valid = [s for s in scores if s is not None]
    return round(sum(valid) / len(valid), 1) if valid else None


def _score_str(score) -> str:
    return f"{score}" if score is not None else "—"


def _bool_icon(val) -> str:
    if val is True:
        return "✅"
    if val is False:
        return "❌"
    return "—"


def _render_report(results: list[dict], run_config: dict, costs: dict | None = None) -> str:
    lines: list[str] = []

    lines.append("# Benchmark Report\n")
    lines.append("## Run Configuration\n")
    lines.append(f"- **Date**: {run_config['timestamp']}")
    lines.append(f"- **Judge model**: {run_config['judge_model']}")
    lines.append(f"- **Git commit**: {run_config['git_commit']}")
    lines.append("")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Benchmark | Path | Text | Tables | Overall |")
    lines.append("|-----------|------|------|--------|---------|")
    for bench in results:
        pdf_name = Path(bench["pdf"]).name
        pages = bench["pages"]
        label_prefix = f"{pdf_name} pp.{pages}"
        for path_result in bench["paths"]:
            path_lbl = path_result["label"]
            if "error" in path_result:
                lines.append(f"| {label_prefix} | {path_lbl} | — | — | ⚠ skipped |")
                continue
            assessments = [pa["assessment"] for pa in path_result["page_assessments"]]
            text_avg  = _avg([a.get("text_score") for a in assessments])
            table_avg = _avg([a.get("tables_score") for a in assessments])
            overall   = _avg([a.get("overall_score") for a in assessments])
            lines.append(
                f"| {label_prefix} | {path_lbl} "
                f"| {_score_str(text_avg)} "
                f"| {_score_str(table_avg)} "
                f"| {_score_str(overall)} |"
            )
    lines.append("")

    # Detailed results
    lines.append("## Detailed Results\n")
    for bench in results:
        pdf_name = Path(bench["pdf"]).name
        pages = bench["pages"]
        notes = bench.get("notes", "")
        lines.append(f"### {pdf_name} — pages {pages}\n")
        if notes:
            lines.append(f"> {notes}\n")

        for path_result in bench["paths"]:
            path_lbl = path_result["label"]
            lines.append(f"#### Path: {path_lbl}\n")

            if "error" in path_result:
                lines.append(f"⚠ **Skipped**: {path_result['error']}\n")
                continue

            assessments = [pa["assessment"] for pa in path_result["page_assessments"]]
            overall   = _avg([a.get("overall_score") for a in assessments])
            text_avg  = _avg([a.get("text_score") for a in assessments])
            table_avg = _avg([a.get("tables_score") for a in assessments])
            lines.append(
                f"**Overall: {_score_str(overall)}/10** "
                f"| Text: {_score_str(text_avg)} "
                f"| Tables: {_score_str(table_avg)}\n"
            )

            for pa in path_result["page_assessments"]:
                page_num = pa["page"]
                a = pa["assessment"]
                lines.append(f"##### Page {page_num} — {a.get('overall_score', '—')}/10\n")
                lines.append(f"- **Text ({a.get('text_score', '—')}/10)**: {a.get('text_notes', '')}")
                if a.get("tables_score") is not None:
                    lines.append(f"- **Tables ({a['tables_score']}/10)**: {a.get('tables_notes', '')}")
                lines.append(
                    f"- Skip: {_bool_icon(a.get('skip_correct'))}  "
                    f"Title: {_bool_icon(a.get('title_correct'))}  "
                    f"Page number: {_bool_icon(a.get('page_number_correct'))}"
                )
                if a.get("skip_notes"):
                    lines.append(f"  - _{a['skip_notes']}_")
                if a.get("title_notes"):
                    lines.append(f"  - _{a['title_notes']}_")
                lines.append(f"- **Summary**: {a.get('summary', '')}")
                lines.append("")

    if costs:
        lines.append("## Costs & Time\n")
        lines.append("| Path | Extraction cost | Extraction time | Judging cost | Judging time | Total cost | Total time |")
        lines.append("|------|----------------|-----------------|--------------|--------------|------------|------------|")
        grand_ext_c = grand_jud_c = grand_ext_s = grand_jud_s = 0.0
        all_labels = sorted(set(costs.get("extraction", {})) | set(costs.get("judging", {})))
        for label in all_labels:
            ext_c = costs.get("extraction", {}).get(label, 0.0)
            jud_c = costs.get("judging", {}).get(label, 0.0)
            ext_s = costs.get("extraction_seconds", {}).get(label, 0.0)
            jud_s = costs.get("judging_seconds", {}).get(label, 0.0)
            grand_ext_c += ext_c; grand_jud_c += jud_c
            grand_ext_s += ext_s; grand_jud_s += jud_s
            lines.append(
                f"| {label} | ${ext_c:.6f} | {ext_s:.1f}s"
                f" | ${jud_c:.6f} | {jud_s:.1f}s"
                f" | ${ext_c + jud_c:.6f} | {ext_s + jud_s:.1f}s |"
            )
        total_elapsed = costs.get("total_elapsed", grand_ext_s + grand_jud_s)
        lines.append(
            f"| **Total** | **${grand_ext_c:.6f}** | **{grand_ext_s:.1f}s**"
            f" | **${grand_jud_c:.6f}** | **{grand_jud_s:.1f}s**"
            f" | **${grand_ext_c + grand_jud_c:.6f}** | **{total_elapsed:.1f}s** |"
        )
        lines.append("")
        judge_note = costs.get("judge_model", "")
        if judge_note:
            lines.append(f"*Judging costs use {judge_note} pricing.*\n")

    return "\n".join(lines)


# ===========================================================================
# Extraction and judge helpers
# ===========================================================================


def _run_extraction(pdf_path: Path, pages_filter: set[int], path_config: dict, dpi: int) -> list[dict]:
    """Run docpeel extraction for one path config, printing per-page progress."""
    provider = build_provider(**path_config)
    provider.resolve_model_id()
    results = []
    n = len(pages_filter)
    for idx, record in enumerate(iter_pages(pdf_path, provider, dpi=dpi, pages=pages_filter), start=1):
        method  = record.get("extraction_method", "—")
        elapsed = record.get("elapsed_seconds")
        elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "—"
        status = "skipped" if record.get("skip") else ("error" if record.get("error") else method)
        print(f"    Page {record['page']} ({idx}/{n}) … {status}  [{elapsed_str}]")
        results.append(record)
    if hasattr(provider, "close"):
        provider.close()
    return results


def _render_page_images(pdf_path: Path, pages_filter: set[int], dpi: int) -> dict:
    """Render the requested pages to PIL images, printing per-page progress."""
    images = {}
    n = len(pages_filter)
    for idx, page_num in enumerate(sorted(pages_filter), start=1):
        print(f"    Page {page_num} ({idx}/{n}) … ", end="", flush=True)
        imgs = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_num, last_page=page_num)
        images[page_num] = imgs[0]
        print("rendered")
    return images


def _image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _judge_page(client, judge_model: str, image, record: dict) -> tuple[dict, int, int]:
    """
    Send one page image + anonymised record to the judge.
    Returns (assessment, input_tokens, output_tokens).
    """
    b64 = _image_to_b64(image)
    response = client.messages.create(
        model=judge_model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {
                        "type": "text",
                        "text": _JUDGE_PROMPT + json.dumps(record, indent=2),
                    },
                ],
            }
        ],
    )
    raw = (response.content[0].text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    in_tok  = response.usage.input_tokens
    out_tok = response.usage.output_tokens
    return json.loads(raw), in_tok, out_tok


# ===========================================================================
# Run metadata helpers
# ===========================================================================


def _git_commit() -> str:
    """Return the short git commit hash of the current HEAD, for traceability."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _create_run_dir(base: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = Path(base) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ===========================================================================
# Main pipeline
# ===========================================================================


def _run_benchmark(config: dict, judge_model: str, output_dir: Path, dpi: int) -> None:
    import anthropic

    judge_client = anthropic.Anthropic()
    env = dict(os.environ)

    run_config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "judge_model": judge_model,
        "git_commit": _git_commit(),
        "dpi": dpi,
    }

    extraction_paths = config.get("extraction_paths", [])
    available_paths = [p for p in extraction_paths if _is_path_available(p, env)]
    skipped_paths   = [p for p in extraction_paths if not _is_path_available(p, env)]

    n_benchmarks = len(config["benchmarks"])
    n_paths = len(available_paths)
    print(f"\n{n_benchmarks} benchmark(s) × {n_paths} available path(s) = {n_benchmarks * n_paths} extraction run(s)")

    print("\nExtraction paths:")
    for p in available_paths:
        print(f"  ✓ {_path_label(p)}")
    if skipped_paths:
        print("Skipped paths (missing API keys):")
        for p in skipped_paths:
            missing = _required_env_keys(p) - env.keys()
            print(f"  ✗ {_path_label(p)} — missing: {', '.join(sorted(missing))}")

    if not available_paths:
        sys.exit("No extraction paths available — check your API keys.")

    all_results: list[dict] = []
    extraction_costs:   dict[str, float] = {}  # label → total USD
    extraction_seconds: dict[str, float] = {}  # label → total elapsed seconds (from records)
    judge_in_tokens:    dict[str, int]   = {}  # label → total input tokens
    judge_out_tokens:   dict[str, int]   = {}  # label → total output tokens
    judge_seconds:      dict[str, float] = {}  # label → wall-clock seconds spent judging
    t_run_start = time.perf_counter()

    for bench_idx, bench in enumerate(config["benchmarks"], start=1):
        pdf_path = Path(bench["pdf"])
        if not pdf_path.exists():
            print(f"\n⚠ PDF not found, skipping: {pdf_path}")
            continue

        pages_filter = _parse_pages(bench["pages"])
        notes = bench.get("notes", "")
        bench_dir = output_dir / "extractions" / _bench_label(pdf_path, bench["pages"])
        bench_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{bench_idx}/{n_benchmarks}] {pdf_path.name}  —  pages {bench['pages']}  ({len(pages_filter)} page(s))")
        if notes:
            print(f"  Notes: {notes}")
        print(f"  Output folder: {bench_dir}")

        print(f"  Rendering {len(pages_filter)} page image(s) at {dpi} DPI …")
        page_images = _render_page_images(pdf_path, pages_filter, dpi)
        for page_num, img in page_images.items():
            img.save(bench_dir / f"page_{page_num:03d}.jpg", format="JPEG", quality=85)
        print(f"  ✓ {len(page_images)} image(s) rendered and saved to {bench_dir}")

        bench_result: dict = {"pdf": str(pdf_path), "pages": bench["pages"], "notes": notes, "paths": []}

        for path_idx, path_config in enumerate(available_paths, start=1):
            label = _path_label(path_config)
            path_dir = bench_dir / _safe_filename(label)
            path_dir.mkdir(exist_ok=True)
            print(f"\n  [{path_idx}/{n_paths}] Extracting with: {label}")

            try:
                records = _run_extraction(pdf_path, pages_filter, path_config, dpi)
            except Exception as exc:
                print(f"  ✗ Extraction failed: {exc}")
                bench_result["paths"].append({"label": label, "error": str(exc)})
                continue

            n_skipped  = sum(1 for r in records if r.get("skip"))
            n_errors   = sum(1 for r in records if r.get("error"))
            ext_cost   = sum(r.get("cost_usd", 0.0) for r in records)
            ext_secs   = sum(r.get("elapsed_seconds", 0.0) for r in records)
            extraction_costs[label]   = extraction_costs.get(label, 0.0)   + ext_cost
            extraction_seconds[label] = extraction_seconds.get(label, 0.0) + ext_secs
            print(f"  ✓ Extraction done — {len(records)} page(s), {n_skipped} skipped, {n_errors} error(s)  [${ext_cost:.6f}  {ext_secs:.1f}s]")

            anon_records = [_anonymize_record(r) for r in records]
            with open(path_dir / "extracted_anonymized.jsonl", "w") as fh:
                for r in anon_records:
                    fh.write(json.dumps(r) + "\n")
            print(f"    Anonymised JSONL saved: {path_dir / 'extracted_anonymized.jsonl'}")

            print(f"  Judging {len(anon_records)} page(s) with {judge_model} …")
            page_assessments: list[dict] = []
            t_judge_start = time.perf_counter()
            for record in anon_records:
                page_num = record["page"]
                img = page_images.get(page_num)
                if img is None:
                    continue
                print(f"    Page {page_num} … ", end="", flush=True)
                t_page = time.perf_counter()
                try:
                    assessment, in_tok, out_tok = _judge_page(judge_client, judge_model, img, record)
                    judge_in_tokens[label]  = judge_in_tokens.get(label, 0)  + in_tok
                    judge_out_tokens[label] = judge_out_tokens.get(label, 0) + out_tok
                    score = assessment.get("overall_score", "?")
                    summary = assessment.get("summary", "")[:72]
                    print(f"{score}/10 — {summary}  [{time.perf_counter() - t_page:.1f}s]")
                except Exception as exc:
                    print(f"✗ judge failed: {exc}")
                    assessment = {"error": str(exc)}
                page_assessments.append({"page": page_num, "assessment": assessment})

            jud_secs = time.perf_counter() - t_judge_start
            judge_seconds[label] = judge_seconds.get(label, 0.0) + jud_secs
            valid = [pa["assessment"].get("overall_score") for pa in page_assessments
                     if "error" not in pa["assessment"]]
            avg = _avg(valid)
            print(f"  ✓ Judging done — average overall score: {_score_str(avg)}/10  [{jud_secs:.1f}s]")

            bench_result["paths"].append({"label": label, "page_assessments": page_assessments})

        # Record skipped paths so they appear in the report
        for p in skipped_paths:
            missing = _required_env_keys(p) - env.keys()
            bench_result["paths"].append({
                "label": _path_label(p),
                "error": f"skipped — missing API key(s): {', '.join(sorted(missing))}",
            })

        all_results.append(bench_result)

    # Compute judge costs via pricing.py
    judge_costs: dict[str, float] = {}
    for label in judge_in_tokens:
        try:
            judge_costs[label] = anthropic_cost(
                judge_model,
                judge_in_tokens[label],
                judge_out_tokens.get(label, 0),
            )
        except Exception:
            judge_costs[label] = 0.0

    total_elapsed = time.perf_counter() - t_run_start

    costs = {
        "extraction":          extraction_costs,
        "extraction_seconds":  extraction_seconds,
        "judging":             judge_costs,
        "judging_seconds":     judge_seconds,
        "judge_model":         judge_model,
        "total_elapsed":       total_elapsed,
    }

    # Print cost + time summary
    print(f"\n{'='*60}")
    print("Cost & Time Summary")
    print(f"{'─'*60}")
    all_labels = sorted(set(extraction_costs) | set(judge_costs))
    grand_ext = grand_jud = grand_ext_s = grand_jud_s = 0.0
    for label in all_labels:
        ext   = extraction_costs.get(label, 0.0)
        jud   = judge_costs.get(label, 0.0)
        ext_s = extraction_seconds.get(label, 0.0)
        jud_s = judge_seconds.get(label, 0.0)
        grand_ext   += ext;   grand_jud   += jud
        grand_ext_s += ext_s; grand_jud_s += jud_s
        print(f"  {label}")
        print(f"    Extraction : ${ext:.6f}  {ext_s:.1f}s")
        print(f"    Judging    : ${jud:.6f}  {jud_s:.1f}s")
        print(f"    Subtotal   : ${ext + jud:.6f}  {ext_s + jud_s:.1f}s")
    print(f"{'─'*60}")
    print(f"  Extraction total : ${grand_ext:.6f}  {grand_ext_s:.1f}s")
    print(f"  Judging total    : ${grand_jud:.6f}  {grand_jud_s:.1f}s")
    print(f"  Grand total      : ${grand_ext + grand_jud:.6f}  {total_elapsed:.1f}s  ({total_elapsed / 60:.1f} min)")

    print(f"\n{'='*60}")
    print("Writing report …")
    (output_dir / "run_config.json").write_text(
        json.dumps({**run_config, "extraction_paths": extraction_paths}, indent=2)
    )
    report = _render_report(all_results, run_config, costs=costs)
    report_path = output_dir / "report.md"
    report_path.write_text(report)

    print(f"\nDone.")
    print(f"  Report      : {report_path}")
    print(f"  Extractions : {output_dir / 'extractions'}")
    print(f"  Run config  : {output_dir / 'run_config.json'}")


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark docpeel extraction quality across all configured paths."
    )
    parser.add_argument(
        "--config", required=True, metavar="FILE",
        help=(
            "Benchmark config file. Copy scripts/benchmark/benchmarks.example.yaml, "
            "fill in your PDF paths and pages, then pass it here."
        ),
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL, metavar="MODEL",
        help=f"Vision LLM used to evaluate extractions (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Root folder for benchmark results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help=f"PDF render resolution (default: {DEFAULT_DPI})",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(
            f"Config file not found: {config_path}\n"
            "Copy scripts/benchmark/benchmarks.example.yaml, fill in your PDF paths, "
            "then pass it with --config."
        )

    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    if not config.get("benchmarks"):
        sys.exit("No benchmarks defined in config.")
    if not config.get("extraction_paths"):
        sys.exit("No extraction_paths defined in config.")

    output_dir = _create_run_dir(args.output_dir)
    shutil.copy(config_path, output_dir / "benchmarks.yaml")

    print(f"Output   : {output_dir}")
    print(f"Judge    : {args.judge_model}")
    print(f"Paths    : {len(config['extraction_paths'])} configured")

    _run_benchmark(config, args.judge_model, output_dir, args.dpi)


if __name__ == "__main__":
    main()
