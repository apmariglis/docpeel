"""
Command-line entry point for the PDF extractor.

Usage:
    python -m pdfextract [OPTIONS] PDF

    # Vision path (Anthropic or Gemini — model sees the page image directly)
    python -m pdfextract PDF --vision-model claude-sonnet-4-0
    python -m pdfextract PDF --vision-model gemini-2.5-flash

    # OCR + structure path (dedicated OCR engine, then an LLM structures the text)
    python -m pdfextract PDF --ocr mistral --structure-model mistral-small-latest
    python -m pdfextract PDF --ocr mistral --structure-model gemini-2.5-flash-lite
"""

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

from pdfextract.extraction import iter_pages
from pdfextract.extraction import page_count
from pdfextract.output import OUTPUT_FOLDER
from pdfextract.output import resolve_run_folder
from pdfextract.output import stream_outputs
from pdfextract.output import write_report
from pdfextract.providers.provider_factory import build_provider

load_dotenv()


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
        "--dpi",
        type=int,
        default=150,
        help=(
            "Resolution for rendering PDF pages (default: 150). "
            "150 DPI is sufficient for most tasks; use 200-300 for very small or dense text."
        ),
    )
    args = parser.parse_args()

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
    print(f"Pages    : {n_pages}")
    t0 = time.perf_counter()
    pages = iter_pages(pdf_path, provider, dpi=args.dpi)
    saved, results = stream_outputs(pdf_path, pages, provider_name=provider_name)
    report_path = write_report(pdf_path, results, saved)

    total_cost = sum(r["cost_usd"] for r in results)
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

    print(f"\nDone. {len(results)}/{n_pages} page(s) processed.")
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
    print(f"  Total cost        : ${total_cost:.6f}")
    elapsed = time.perf_counter() - t0
    print(f"  Total time        : {elapsed:.1f}s ({elapsed / 60:.1f} min)")



if __name__ == "__main__":
    main()
