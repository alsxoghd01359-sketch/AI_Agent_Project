"""Extract clean, per-page Korean text from the D&D rulebook PDF using PyMuPDF.

pdftotext / pypdf / pdfplumber all fail to decode this PDF's embedded Korean
font correctly (they emit U+FFFD). PyMuPDF (fitz) decodes it perfectly, so
this is the only extraction path used in this project.
"""
import json

import pymupdf

from config import PDF_PATH, RAW_DIR, PAGES_JSON, BOILERPLATE_LINE


def extract_pages():
    doc = pymupdf.open(PDF_PATH)
    pages = []
    last_chapter = None

    for idx in range(doc.page_count):
        raw_lines = doc[idx].get_text().split("\n")
        lines = [l.rstrip() for l in raw_lines]
        while lines and lines[-1].strip() == "":
            lines.pop()

        page_num = None
        chapter = last_chapter
        body_lines = lines

        if lines and lines[0].strip().isdigit():
            page_num = int(lines[0].strip())
            chapter = lines[1].strip() if len(lines) > 1 else last_chapter
            body_lines = lines[2:]
            if body_lines and body_lines[0].strip() == BOILERPLATE_LINE:
                body_lines = body_lines[1:]
            if body_lines and body_lines[0].strip() == chapter:
                body_lines = body_lines[1:]
            last_chapter = chapter

        body = "\n".join(body_lines).strip()

        pages.append(
            {
                "index": idx,
                "page_num": page_num,
                "chapter": chapter,
                "text": body,
            }
        )

    return pages


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pages = extract_pages()
    with open(PAGES_JSON, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Extracted {len(pages)} pages -> {PAGES_JSON}")

    chapters = sorted(set(p["chapter"] for p in pages if p["chapter"]))
    print(f"Detected {len(chapters)} chapter labels:")
    for c in chapters:
        print(" -", c)


if __name__ == "__main__":
    main()
