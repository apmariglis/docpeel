"""
Command-line entry point for the PDF extractor.

Usage:
    python -m pdfextract [OPTIONS] PDF

    python -m pdfextract data/my.pdf --provider gemini --model gemini-2.5-flash
"""

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

from extraction import iter_pages
from extraction import page_count
from output import OUTPUT_FOLDER
from output import stream_outputs
from output import write_report
from providers import build_provider

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text and tables from a PDF using a vision LLM."
    )
    parser.add_argument("pdf", type=Path, help="Path to the PDF file to process")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        default="anthropic",
        help="LLM provider to use (default: anthropic)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help=(
            "Resolution for rendering PDF pages (default: 150). "
            "150 DPI is sufficient for most OCR tasks and keeps image sizes small. "
            "Use 200-300 for pages with very small or dense text."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model ID to use. Defaults per provider: "
            "anthropic → claude-sonnet-4-20250514, "
            "gemini → gemini-2.5-flash-lite"
        ),
    )
    args = parser.parse_args()

    pdf_path: Path = args.pdf
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    provider = build_provider(args.provider, args.model)
    print(
        f"Provider : {args.provider}  |  Model: {provider.model_id}  |  DPI: {args.dpi}"
    )
    print(f"Extracting '{pdf_path}' → output folder: '{OUTPUT_FOLDER}'")

    n_pages = page_count(pdf_path)
    print(f"Pages    : {n_pages}")
    t0 = time.perf_counter()
    pages = iter_pages(pdf_path, provider, dpi=args.dpi)
    saved, results = stream_outputs(pdf_path, pages, provider_name=args.provider)
    report_path = write_report(pdf_path, results, saved)

    total_cost = sum(r["cost_usd"] for r in results)
    quad_pages = [r for r in results if r.get("extraction_method") == "quadrant-split"]
    para_pages = [r for r in results if r.get("paraphrased")]
    failed_pages = [
        r
        for r in results
        if r.get("error") and r.get("extraction_method") != "quadrant-split"
    ]

    print(f"\nDone. {len(results)}/{n_pages} page(s) processed.")
    if quad_pages:
        print(f"  {len(quad_pages)} page(s) used quadrant-split fallback.")
    if para_pages:
        pages_str = ", ".join(str(r["page"]) for r in para_pages)
        print(
            f"  ⚠️  {len(para_pages)} page(s) contain paraphrased chunks (manual review recommended): page(s) {pages_str}"
        )
    if failed_pages:
        print(f"  {len(failed_pages)} page(s) failed entirely (see report).")
    print(f"  Combined markdown : {saved['combined_md']}")
    print(f"  Per-page folder   : {saved['pages_dir']}")
    print(f"  JSON data         : {saved['json']}")
    print(f"  Cost report       : {report_path}")
    print(f"  Total cost        : ${total_cost:.6f}")
    elapsed = time.perf_counter() - t0
    print(f"  Total time        : {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
