"""
Command-line entry point for the PDF extractor.

Usage:
    python -m docpeel [OPTIONS] PDF

    # Vision path (Anthropic or Gemini — model sees the page image directly)
    python -m docpeel PDF --vision-model claude-sonnet-4-0
    python -m docpeel PDF --vision-model gemini-2.5-flash

    # OCR + structure path (dedicated OCR engine, then an LLM structures the text)
    python -m docpeel PDF --ocr mistral --structure-model mistral-small-latest
    python -m docpeel PDF --ocr mistral --structure-model gemini-2.5-flash-lite
"""

import argparse
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from docpeel.extraction import iter_pages
from docpeel.extraction import page_count
from docpeel.output import OUTPUT_FOLDER
from docpeel.output import resolve_run_folder
from docpeel.output import stream_outputs
from docpeel.output import write_report
from docpeel.providers.provider_factory import build_provider

load_dotenv()


def _parse_pages(spec: str) -> set[int]:
    """
    Parse a page specification string into a set of 1-based page numbers.

    Accepts comma-separated pages and/or inclusive ranges:
        "5"        → {5}
        "1,3,5"    → {1, 3, 5}
        "2-5"      → {2, 3, 4, 5}
        "1,3,7-10" → {1, 3, 7, 8, 9, 10}

    Raises ValueError with a descriptive message on invalid input.
    """
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            halves = part.split("-", 1)
            if len(halves) != 2 or not halves[0].strip() or not halves[1].strip():
                raise ValueError(f"invalid page range: {part!r}")
            try:
                start, end = int(halves[0].strip()), int(halves[1].strip())
            except ValueError:
                raise ValueError(f"invalid page range: {part!r}")
            if start < 1 or end < 1:
                raise ValueError(f"page numbers must be ≥ 1, got: {part!r}")
            if start > end:
                raise ValueError(f"range start must be ≤ end, got: {part!r}")
            pages.update(range(start, end + 1))
        else:
            try:
                n = int(part)
            except ValueError:
                raise ValueError(f"invalid page number: {part!r}")
            if n < 1:
                raise ValueError(f"page numbers must be ≥ 1, got: {part!r}")
            pages.add(n)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text and tables from a PDF using a vision LLM or OCR+LLM pipeline.\n\n"
            "Two modes:\n"
            "  Vision path  — model sees the page image directly:\n"
            "    %(prog)s PDF --vision-model claude-sonnet-4-0\n"
            "    %(prog)s PDF --vision-model gemini-2.5-flash\n\n"
            "  OCR path     — dedicated OCR engine, then an LLM structures the text:\n"
            "    %(prog)s PDF --ocr mistral --structure-model mistral-small-latest\n"
            "    %(prog)s PDF --ocr mistral --structure-model gemini-2.5-flash-lite\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # pdf is nargs="?" so argparse doesn't error before our validation runs
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="?",
        metavar="PDF",
        help="Path to the PDF file to process",
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        metavar="MODEL",
        help=(
            "Vision model for direct image-to-text extraction. "
            "Examples: claude-sonnet-4-0, gemini-2.5-flash"
        ),
    )
    parser.add_argument(
        "--ocr",
        choices=["mistral"],
        default=None,
        metavar="ENGINE",
        help=(
            "OCR engine for the first step of a two-step pipeline (currently: mistral). "
            "Must be paired with --structure-model. "
            "Cannot be combined with --vision-model."
        ),
    )
    parser.add_argument(
        "--structure-model",
        default=None,
        metavar="MODEL",
        help=(
            "LLM for the structuring step (requires --ocr). "
            "Examples: mistral-small-latest, gemini-2.5-flash-lite"
        ),
    )
    parser.add_argument(
        "--pages",
        default=None,
        metavar="PAGES",
        help=(
            "Pages to extract. Accepts comma-separated page numbers and/or inclusive ranges. "
            "Examples: --pages 3  |  --pages 1,3,5  |  --pages 2-5  |  --pages 1,3,7-10. "
            "Omit to process the entire PDF."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help=(
            "Resolution for rendering PDF pages (default: 150). "
            "150 DPI is sufficient for most tasks; use 200-300 for very small or dense text."
        ),
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug logging (per-chunk detail, stitch steps).",
    )
    verbosity.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress progress messages; show only warnings and errors.",
    )
    args = parser.parse_args()

    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("docpeel").setLevel(log_level)
    logging.getLogger("docpeel").addHandler(handler)

    # ── Validate flag combinations — print full help on any error ─────────────
    errors = []
    if not args.pdf:
        errors.append("PDF file is required.")
    if not args.vision_model and not args.ocr:
        errors.append("one of --vision-model or --ocr is required.")
    if args.vision_model and args.ocr:
        errors.append("--vision-model and --ocr are mutually exclusive.")
    if args.structure_model and not args.ocr:
        errors.append("--structure-model requires --ocr.")
    if args.ocr and not args.structure_model:
        errors.append("--ocr requires --structure-model.")
    if errors:
        parser.print_help()
        print("\nerror:", "  ".join(errors), file=__import__("sys").stderr)
        raise SystemExit(2)

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Parse --pages before building the provider so we fail fast on bad input
    pages_filter: set[int] | None = None
    if args.pages is not None:
        try:
            pages_filter = _parse_pages(args.pages)
        except ValueError as exc:
            parser.print_help()
            print(f"\nerror: --pages: {exc}", file=__import__("sys").stderr)
            raise SystemExit(2)

    provider = build_provider(
        vision_model=args.vision_model,
        ocr=args.ocr,
        structure_model=args.structure_model,
    )
    provider.resolve_model_id()

    provider_name = provider.model_id.replace(":", "-")
    run_folder = resolve_run_folder(pdf_path, provider_name)

    print(f"Model    : {provider.model_id}  |  DPI: {args.dpi}")
    print(f"PDF      : {pdf_path}")
    print(f"Output   : {run_folder}")

    n_pages = page_count(pdf_path)

    if pages_filter is not None:
        out_of_range = sorted(p for p in pages_filter if p > n_pages)
        if out_of_range:
            print(
                f"\nerror: --pages: page(s) {out_of_range} exceed the PDF length ({n_pages} pages).",
                file=__import__("sys").stderr,
            )
            raise SystemExit(2)

    pages_label = (
        f"{len(pages_filter)} selected / {n_pages} total"
        if pages_filter is not None
        else str(n_pages)
    )
    print(f"Pages    : {pages_label}")
    t0 = time.perf_counter()
    pages = iter_pages(pdf_path, provider, dpi=args.dpi, pages=pages_filter)
    saved, results = stream_outputs(pdf_path, pages, provider_name=provider_name)
    report_path = write_report(pdf_path, results, saved)

    _costs = [r["cost_usd"] for r in results if r["cost_usd"] is not None]
    total_cost = sum(_costs) if _costs else None
    skipped_pages = [r for r in results if r.get("skip")]
    quad_pages = [
        r
        for r in results
        if r.get("extraction_method") == "quadrant-split" and not r.get("error")
    ]
    para_pages = [r for r in results if r.get("paraphrased")]
    failed_pages = [r for r in results if r.get("error")]
    chunk_missing_pages = [
        r
        for r in results
        if not r.get("error")
        and not r.get("skip")
        and any(
            "blocked" in w and "paraphrase" not in w and "skipped" in w
            for w in r.get("chunk_warnings", [])
        )
    ]
    all_incomplete = failed_pages + chunk_missing_pages

    print(f"\nDone. {len(results)}/{pages_label} page(s) processed.")
    if skipped_pages:
        by_reason: dict[str, list[int]] = {}
        for r in skipped_pages:
            by_reason.setdefault(r.get("skip_reason", "unknown"), []).append(r["page"])
        reasons_str = ", ".join(f"{k}: {v}" for k, v in by_reason.items())
        print(f"  {len(skipped_pages)} page(s) skipped ({reasons_str}).")
    if quad_pages:
        print(f"  {len(quad_pages)} page(s) used quadrant-split fallback.")
    if para_pages:
        pages_str = ", ".join(str(r["page"]) for r in para_pages)
        print(
            f"  ⚠️  {len(para_pages)} page(s) contain paraphrased content "
            f"(manual review recommended): page(s) {pages_str}"
        )
    if all_incomplete:
        print(f"\n  {'='*60}")
        print(f"  ⛔ WARNING: {len(all_incomplete)} page(s) have MISSING content!")
        print(f"  {'='*60}")
        for r in sorted(all_incomplete, key=lambda r: r["page"]):
            if r.get("error"):
                print(f"    Page {r['page']:>4} : FAILED — {r['error']}")
            else:
                dropped = [w for w in r.get("chunk_warnings", []) if "skipped" in w]
                print(
                    f"    Page {r['page']:>4} : CHUNKS MISSING — {len(dropped)} chunk(s) dropped"
                )
        print(f"  {'='*60}")
        print(f"  Re-run or manually extract the pages listed above.")
        print(f"  See full details in the report: {report_path}")
    print(f"\n  Combined markdown : {saved['combined_md']}")
    print(f"  Per-page folder   : {saved['pages_dir']}")
    print(f"  JSON data         : {saved['json']}")
    print(f"  Cost report       : {report_path}")
    cost_str = f"${total_cost:.6f}" if total_cost is not None else "N/A"
    print(f"  Total cost        : {cost_str}")
    elapsed = time.perf_counter() - t0
    print(f"  Total time        : {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Close the Mistral HTTP client explicitly to avoid SDK async-cleanup errors at shutdown
    if hasattr(provider, "close"):
        provider.close()


if __name__ == "__main__":
    main()
