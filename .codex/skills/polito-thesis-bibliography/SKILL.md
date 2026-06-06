---
name: polito-thesis-bibliography
description: "Create, revise, verify, or critique the Bibliography and citation consistency for this project-specific Politecnico di Torino English master's thesis on artist similarity, MIR, machine learning, multimodal embeddings, datasets, tools, APIs, and web sources. Use for BibTeX entries, numeric citation style, missing citation checks, DOI/URL cleanup, and bibliography formatting."
---

# Bibliography

## Style Anchor

Write only in polished academic English where prose is needed. Use the supplied Politecnico di Torino theses only as style and formatting references: numeric in-text citations, a final Bibliography section, and complete entries for papers, books, datasets, services, software, and web resources. Do not copy their bibliography entries unless the current thesis cites the same source and the details are verified.

Never invent bibliographic metadata. If author, title, venue, year, DOI, URL, or access date is unknown, mark it for verification.

## Citation Policy

Keep a strict one-to-one relationship between citations and bibliography:

- Every in-text citation must appear in the Bibliography.
- Every bibliography entry must be cited in the thesis body.
- Claims about prior work, datasets, models, tools, and platforms must cite specific sources.
- Web services, APIs, datasets, code libraries, and pretrained models should be cited when they are material to the thesis.

## Entry Requirements

For journal or conference papers, include authors, title, venue, pages if available, year, and DOI when available.

For books, include authors or editors, title, publisher, location if required by the chosen style, and year.

For datasets, software, models, or web platforms, include creator or organization, title/name, version or release if available, URL or DOI, year or "n.d." if no date is available, and access date if the style requires it.

## BibTeX Workflow

When the thesis uses LaTeX or BibTeX:

1. Prefer stable keys such as `authorYearShortTitle`.
2. Preserve capitalization for proper nouns with braces, such as `{Spotify}`, `{AllMusic}`, `{BERT}`, `{CLIP}`, or `{Music Information Retrieval}`.
3. Use `doi`, `url`, and `urldate` fields consistently.
4. Compile or inspect citation output when possible to detect missing fields and duplicate keys.

## Formatting and Tone

Use one citation style consistently. If the existing template uses numeric citations, do not switch to author-year style. Keep author names, capitalization, punctuation, and title casing consistent across entries.

Avoid:

- Citing Wikipedia or informal blogs for core technical definitions when primary or authoritative sources exist.
- Listing tools that are not discussed or used in the thesis.
- Leaving placeholder citations such as `[?]`, unfinished markers, or "citation needed" in final text.

## Revision Checklist

Check for duplicate entries, broken URLs, missing DOIs, inconsistent capitalization, uncited bibliography items, and cited-but-missing references. Verify that citations in Abstract are absent unless the template explicitly allows them.
