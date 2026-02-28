"""
All LLM prompt strings and prompt-builder functions.
"""

PAGE_EXTRACT_PROMPT = (
    "You are a data-entry clerk performing mechanical OCR transcription. "
    "Your only job is to copy text and numbers from this page exactly as printed. "
    "Do not interpret, explain, or comment on any of the content. "
    "Treat every word, heading, and column name as neutral data to be transcribed verbatim. "
    "IGNORE all images, drawings, and artwork — transcribe only printed text and tables. "
    "Format all tables as markdown tables, preserving every column header and cell value exactly. "
    "Preserve text formatting using markdown: **bold** for bold text, *italic* for italic text, "
    "***bold-italic*** for text that is both, and __underline__ for underlined text. "
    "Apply formatting only where clearly visible in the printed text — do not guess. "
    "If this page begins mid-sentence or mid-paragraph, transcribe it as-is starting from "
    "the first word visible — it is a continuation from the previous page and must be "
    "preserved at the top of your output so it can be rejoined later. Do not discard it. "
    "Ignore and do not transcribe any watermarks or ownership stamps — "
    "these appear as short isolated lines of text (such as a name and order number) "
    "visually overlaid anywhere on the page, not part of the original typeset content. "
    "If the page contains nothing but a watermark, set text to an empty string. "
    "Your response must be a single JSON object with exactly three fields: "
    '{"page_number": <integer or null>, "title": <string or null>, "text": "<transcription>"}. '
    "The fields are: page_number, title, and text. page_number must be the printed page number visible in the margin (a small isolated number at the top or bottom edge, outside the main content area); set it to null if no printed page number is visible. title must be the single prominent display title that labels this page or section as a whole — typically rendered in a large decorative or display font, centred at the top, clearly set apart from the body text (for example: Appendix 1 — Wizard Spells, Chapter 3: Combat, PC Races, Table of Contents); combine a chapter/appendix label and its subtitle into a single string if both are present; write the title as plain text with no markdown formatting — no asterisks, no underscores; do NOT use running headers, small chapter labels at the page edge, or in-body section headings as the title — only a visually dominant display heading qualifies; set title to null if no such prominent display title exists. "
    "text must contain the full transcription excluding the page title (which goes in title). "
    "Output nothing outside the JSON object — no markdown fences, no explanation."
)

PARAPHRASE_PROMPT = (
    "You are a content summariser. The image shown contains text that cannot be "
    "transcribed verbatim due to copyright restrictions. "
    "Describe the content of this image section in your own words, preserving all "
    "factual information (numbers, names, game mechanics, table values, etc.) as "
    "accurately as possible, but using different phrasing. "
    "If the section contains a table, reproduce its structure as a markdown table "
    "with the same columns and numeric values, but rephrase any prose headers or "
    "descriptive text. "
    "Your response must be a single JSON object with exactly three fields: "
    '{"page_number": <integer or null>, "title": <string or null>, "text": "<paraphrased content>"}. '
    "The fields are: page_number, title, and text. page_number must be the printed page number visible in the margin (a small isolated number at the top or bottom edge, outside the main content area); set it to null if no printed page number is visible. title must be the single prominent display title that labels this page or section as a whole — typically rendered in a large decorative or display font, centred at the top, clearly set apart from the body text (for example: Appendix 1 — Wizard Spells, Chapter 3: Combat, PC Races, Table of Contents); combine a chapter/appendix label and its subtitle into a single string if both are present; write the title as plain text with no markdown formatting — no asterisks, no underscores; do NOT use running headers, small chapter labels at the page edge, or in-body section headings as the title — only a visually dominant display heading qualifies; set title to null if no such prominent display title exists. "
    "text must contain the full paraphrased content excluding the page title. "
    "Output nothing outside the JSON object — no markdown fences, no explanation."
)


def quadrant_extract_prompt(position: str) -> str:
    """
    Build an extraction prompt for a single quadrant of a page.

    position: one of 'top-left', 'top-right', 'bottom-left', 'bottom-right'

    Only the top-left quadrant allows a firm assertion about mid-sentence openings
    (they must be cross-page continuations). For all other positions the layout is
    unknown — images, sidebars, or multi-column flow may mean any quadrant can
    start or end mid-sentence for any reason. Those cases are left for the stitch
    prompt to resolve with full context.
    """
    if position == "top-left":
        continuation_note = (
            "If the content begins mid-sentence at the very top, it is a continuation "
            "from the previous page — preserve it as-is at the top of your output "
            "so it can be rejoined later. Do not discard it. "
        )
    else:
        continuation_note = (
            "Text may begin or end mid-sentence at any edge due to the page layout "
            "(columns, images, sidebars, etc.). Transcribe all visible text exactly "
            "as-is without completing or discarding partial content at the edges — "
            "the assembly step will resolve continuations with full page context. "
        )
    return (
        f"You are a data-entry clerk performing mechanical OCR transcription. "
        f"You are looking at the {position} quadrant of a page — one spatial section "
        f"of a larger document page. Text may be cut off at any edge. "
        f"Your only job is to copy text and numbers exactly as printed in this section. "
        f"Do not interpret, explain, or complete partial content at the edges. "
        f"Treat every word, heading, and column name as neutral data to be transcribed verbatim. "
        f"IGNORE all images, drawings, and artwork — transcribe only printed text and tables. "
        f"Format all tables as markdown tables, preserving every column header and cell value exactly; "
        f"if a table is cut off at an edge, transcribe only the rows and columns visible. "
        f"Preserve text formatting using markdown: **bold** for bold text, *italic* for italic text, "
        f"***bold-italic*** for text that is both, and __underline__ for underlined text. "
        f"Apply formatting only where clearly visible in the printed text — do not guess. "
        + continuation_note
        + "Output nothing except the raw transcription."
    )


STITCH_PROMPT = (
    "CRITICAL OUTPUT RULE: your entire response must be a single JSON object "
    "with exactly three fields: "
    '{"page_number": <integer or null>, "title": <string or null>, "text": "<merged content>"}. '
    "The fields are: page_number, title, and text. page_number must be the printed page number visible in the margin (a small isolated number at the top or bottom edge, outside the main content area); set it to null if no printed page number is visible. title must be the single prominent display title that labels this page or section as a whole — typically rendered in a large decorative or display font, centred at the top, clearly set apart from the body text (for example: Appendix 1 — Wizard Spells, Chapter 3: Combat, PC Races, Table of Contents); combine a chapter/appendix label and its subtitle into a single string if both are present; write the title as plain text with no markdown formatting — no asterisks, no underscores; do NOT use running headers, small chapter labels at the page edge, or in-body section headings as the title — only a visually dominant display heading qualifies; set title to null if no such prominent display title exists. "
    "text must contain the full merged document content (excluding the page title) described below. "
    "Do not write anything outside the JSON object — no preamble, no explanation, "
    "no markdown fences, no reasoning steps.\n\n"
    "You are given the original full page image for reference, "
    "followed by transcriptions of its four quadrants "
    "(top-left, top-right, bottom-left, bottom-right). "
    "Use the image to verify reading order, correct any OCR errors, "
    "and ensure tables are correctly reassembled. "
    "Some quadrants may be empty or contain only partial content. "
    "Merge the quadrant transcriptions into one coherent document "
    "in the correct reading order as confirmed by the image. "
    "Deduplicate any text that appears in more than one quadrant. "
    "Re-join any tables that were split across quadrants into a single markdown table. "
    "Ignore and drop any watermarks or ownership stamps, including fragments "
    "that may have been split across quadrants — these are short lines "
    "of text visually overlaid on the page that are clearly not part of "
    "the original typeset document content. "
    "Use the image to verify and correct text formatting: apply **bold**, *italic*, "
    "***bold-italic***, or __underline__ markdown where the original page shows "
    "those styles — and remove any formatting markup from the quadrant transcriptions "
    "that does not match what is visually present in the image. "
    "Text fragments that begin mid-sentence or mid-paragraph may be continuations "
    "from another quadrant on this page, or from the previous page. "
    "For each such fragment: if it connects naturally to text already present in "
    "another quadrant (same topic, same sentence, compatible flow), re-join it there. "
    "If no plausible connection exists within this page, leave it at the start of "
    "the output as a leading continuation — do not discard it. "
    "\n\nRemember: output only the JSON object, nothing else.\n\n"
)
