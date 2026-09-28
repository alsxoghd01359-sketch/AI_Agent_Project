"""Extract clean, per-page Korean text from the D&D rulebook PDF using PyMuPDF.

pdftotext / pypdf / pdfplumber all fail to decode this PDF's embedded Korean
font correctly (they emit U+FFFD). PyMuPDF (fitz) decodes it perfectly, so
this is the only extraction path used in this project.

Besides plain text, every line keeps two style hints the entry parser relies on:
  b: the bold text the line starts with — trait/action names ("부패의 주먹.") are bold
  s: whether the line is set in the serif (Batang) font — monster flavor text is
Plain text alone can't tell where one action ends and the next begins when a
sentence ends exactly at a line break; the font can.
"""
import json

import pymupdf

from config import PDF_PATH, RAW_DIR, PAGES_JSON, BOILERPLATE_LINE


def _styled_lines(page):
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            # Keep trailing spaces: a wrapped line ending in " " broke at a word boundary,
            # one without it broke mid-word ("2배" + "가 되며" -> "2배가 되며").
            text = "".join(s["text"] for s in spans)
            bold = ""
            for s in spans:
                if "Bold" in s["font"]:
                    bold += s["text"]
                elif s["text"].strip():
                    break
            total = sum(len(s["text"].strip()) for s in spans)
            serif = sum(len(s["text"].strip()) for s in spans if "Batang" in s["font"])
            lines.append({"t": text, "b": bold.strip(), "s": total > 0 and serif * 2 > total})
    return lines


def extract_pages():
    doc = pymupdf.open(PDF_PATH)
    pages = []
    last_chapter = None

    for idx in range(doc.page_count):
        lines = _styled_lines(doc[idx])
        while lines and lines[-1]["t"].strip() == "":
            lines.pop()
        while lines and lines[0]["t"].strip() == "":
            lines.pop(0)

        page_num = None
        chapter = last_chapter
        body_lines = lines

        if lines and lines[0]["t"].strip().isdigit():
            page_num = int(lines[0]["t"].strip())
            chapter = lines[1]["t"].strip() if len(lines) > 1 else last_chapter
            body_lines = lines[2:]
            if body_lines and body_lines[0]["t"].strip() == BOILERPLATE_LINE:
                body_lines = body_lines[1:]
            if body_lines and body_lines[0]["t"].strip() == chapter:
                body_lines = body_lines[1:]
            last_chapter = chapter

        pages.append(
            {
                "index": idx,
                "page_num": page_num,
                "chapter": chapter,
                "text": "\n".join(l["t"] for l in body_lines).strip("\n"),
                "lines": body_lines,
            }
        )

    return pages


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    pages = extract_pages()
    with open(PAGES_JSON, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=1)
    print(f"Extracted {len(pages)} pages -> {PAGES_JSON}")

    chapters = sorted(set(p["chapter"] for p in pages if p["chapter"]))
    print(f"Detected {len(chapters)} chapter labels:")
    for c in chapters:
        print(" -", c)


if __name__ == "__main__":
    main()
