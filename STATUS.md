# docpeel — Status & Roadmap

*A working document. Updated as gaps are identified and addressed.*

---

## What works well today

Tested against dense, table-heavy reference books and mixed text-and-artwork PDFs:

- **Tables** — extracted into a structured `tables` array with captions; multi-page tables are merged automatically, including both no-title continuations and repeated-header continuations (the publishing convention of re-printing column headers on each continuation page)
- **Image-only and mixed-artwork pages** — detected and handled cleanly; full-page illustrations are skipped without affecting surrounding text pages
- **Skip detection** — table of contents, index, blank, illustration-only, title, and part-divider pages are detected and excluded
- **Page titles** — prominent display headings separated from body text
- **Watermarks** — detected and excluded from the text field
- **Content filter fallback** — pages blocked by the provider's content filter are automatically retried via quadrant-split, obfuscation, and paraphrase stages
- **Multi-provider** — Anthropic, Gemini, and Mistral interchangeable with a single CLI flag

---

## Known gaps

- **Cross-page text flow** — a paragraph that starts on one page and ends on the next is not re-joined; each page is extracted independently. For RAG use this means chunks near page boundaries may contain incomplete sentences.

- **Secondary content columns** — some publications use a visually distinct secondary column alongside the main text (a narrow strip with a different background, containing supplementary or contextual content). The tool currently extracts this into the main body text rather than separating or ignoring it.

- **Diagram and figure text** — text embedded inside figures, flow charts, or labelled illustrations is not extracted. In documents where significant content is conveyed visually rather than as body text, this content will be absent from the output.

- **Embedded-text PDFs** — every page is rendered to an image before extraction. PDFs that already contain selectable text could bypass the render and OCR steps entirely, reducing cost and processing time.

- **Footnotes** — not separated into a distinct field; they appear appended to the body text, which is acceptable for reading but not ideal for RAG chunking.

- **Two-column dense layouts** — reading order reconstruction can fail on tightly packed two-column text where columns do not align cleanly with quadrant boundaries.

- **Non-Latin scripts and low-quality scans** — untested / limited effectiveness. No pre-processing pipeline (deskew, denoise, binarise) exists yet.

- **Skip-detection page types** — currently tuned for book-like documents. Other genres (academic papers, legal documents, technical manuals) may have different boilerplate page types not yet recognised.

---

## Potential next steps

1. Cross-page text rejoining (highest impact for RAG output quality)
2. Embedded-text fast path — skip render+OCR for native-text PDFs
3. Secondary content column exclusion (prompt or visual-layout signal)
4. Footnote field in the output schema
5. Pre-processing pipeline for low-quality scans
6. Extend skip-detection vocabulary for non-book document types
