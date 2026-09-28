"""Turn extracted pages.json into RAG-ready chunks.

Strategy:
- Spells (ch.11), monsters/NPCs (ch.12) and magic items (ch.14) become one chunk
  per entry. Entry boundaries come from parse_entries.find_entries, the same
  detection that builds data/structured/*.json, so chunks and JSON records agree.
- Everything else (including the non-entry text inside those chapters) is first
  split at the rulebook's own subsection headings, recovered from the table of
  contents on page 2 (e.g. "갑옷과 방패류", "휴식", "환경"), so a chunk never
  mixes two topics. A subsection that's still too long is further sliced into
  ~450-token windows with a small overlap, breaking on line boundaries.
"""
import json
import re

import tiktoken

from config import PAGES_JSON, CHUNKS_JSONL
from parse_entries import load_chapter_lines, find_entries

TOC_LINE_RE = re.compile(r"^(.+?)[.\t ]{2,}(\d+)$")

# chapter prefix -> (entry type, find_entries options, section label for its entries)
ENTRY_CHAPTERS = {
    "제11장": ("spell", {}, "주문 상세"),
    "제12장": ("monster", {"stop_titles": ["논플레이어 캐릭터"]}, "괴물의 자료 상자"),
    "제14장": ("magic_item", {"start_after": "물건 상세설명"}, "물건 상세설명"),
}

ENC = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 450
OVERLAP_TOKENS = 60


def load_pages():
    with open(PAGES_JSON, encoding="utf-8") as f:
        return json.load(f)


def _pairs(lines):
    """(page, raw, style) -> (page, stripped text) for the generic chunkers."""
    return [(p, raw.strip()) for p, raw, _ in lines]


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s).rstrip(".")


def parse_toc(pages):
    """Returns {chapter_title: [subsection_title, ...]} using page 2's TOC."""
    toc_page = next(p for p in pages if p["page_num"] == 2)
    current_chapter = None
    groups = {}
    for raw in toc_page["text"].split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        m = TOC_LINE_RE.match(raw)
        if not m:
            continue
        title = m.group(1).strip().rstrip(".").strip()
        if not title:
            continue
        if re.match(r"^(제\d+장|부록\s*[A-Za-z가-힣])", title):
            current_chapter = title
            groups[current_chapter] = []
        elif current_chapter is not None:
            groups[current_chapter].append(title)
    return groups


def chapter_slug(chapter: str) -> str:
    m = re.match(r"제(\d+)장", chapter)
    if m:
        return f"ch{int(m.group(1)):02d}"
    m = re.match(r"부록\s*([A-Za-z가-힣]+)", chapter)
    if m:
        return f"appendix_{m.group(1)}"
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "_", chapter)[:20] or "misc"


def make_chunk(chunk_id, chapter, entry_type, lines, name=None, name_en=None, section=None):
    pages = [p for p, _ in lines if p is not None]
    text = "\n".join(l for _, l in lines)
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {
            "source": "던전앤드래곤 기초 룰북",
            "chapter": chapter,
            "type": entry_type,
            "section": section or "",
            "name": name or "",
            "name_en": name_en or "",
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
        },
    }


def generic_chunks(lines):
    """Sliding token-window chunking over (page, line) tuples."""
    chunks = []
    buf = []
    buf_tokens = 0

    def flush():
        if buf:
            chunks.append(list(buf))

    for page, line in lines:
        lt = len(ENC.encode(line))
        if buf and buf_tokens + lt > MAX_TOKENS:
            flush()
            overlap = []
            ot = 0
            for p, l in reversed(buf):
                lt2 = len(ENC.encode(l))
                if ot + lt2 > OVERLAP_TOKENS:
                    break
                overlap.insert(0, (p, l))
                ot += lt2
            buf = overlap
            buf_tokens = ot
        buf.append((page, line))
        buf_tokens += lt
    flush()
    return chunks


def split_by_subsections(lines, subsection_titles):
    """Find subsection heading lines (in order) and split lines at each one.
    Returns [(section_title_or_None, sub_lines), ...]."""
    positions = []
    search_from = 0
    for title in subsection_titles:
        target = norm(title)
        if not target:
            continue
        found_idx = None
        for i in range(search_from, len(lines)):
            if norm(lines[i][1]) == target:
                found_idx = i
                break
        if found_idx is not None:
            positions.append((found_idx, title))
            search_from = found_idx + 1

    if not positions:
        return [(None, lines)] if lines else []

    segments = []
    if positions[0][0] > 0:
        segments.append((None, lines[: positions[0][0]]))
    for k, (idx, title) in enumerate(positions):
        end = positions[k + 1][0] if k + 1 < len(positions) else len(lines)
        segments.append((title, lines[idx:end]))
    return segments


def chunk_with_sections(slug, chapter, lines, subsection_titles, entry_type="rule"):
    out = []
    for sec_i, (section, sec_lines) in enumerate(split_by_subsections(lines, subsection_titles)):
        sub_chunks = generic_chunks(sec_lines)
        for part_i, sub in enumerate(sub_chunks):
            cid = f"{slug}_s{sec_i}" + (f"_p{part_i}" if len(sub_chunks) > 1 else "")
            out.append(make_chunk(cid, chapter, entry_type, sub, section=section))
    return out


def build_chunks():
    pages = load_pages()
    chapters = load_chapter_lines(pages)
    toc = parse_toc(pages)
    all_chunks = []

    for chapter, lines in chapters.items():
        slug = chapter_slug(chapter)
        subsection_titles = toc.get(chapter, [])
        spec = next((v for k, v in ENTRY_CHAPTERS.items() if chapter.startswith(k)), None)

        if spec is None:
            all_chunks.extend(chunk_with_sections(slug, chapter, _pairs(lines), subsection_titles))
            continue

        entry_type, options, section = spec
        for g, seg in enumerate(find_entries(lines, entry_type, **options)):
            if seg[0] == "gap":
                if any(l.strip() == "논플레이어 캐릭터" for _, l, _ in seg[1]):
                    section = "논플레이어 캐릭터"
                all_chunks.extend(chunk_with_sections(f"{slug}_g{g}", chapter, _pairs(seg[1]), subsection_titles))
            else:
                meta = seg[2]
                cid = f"{entry_type}_{re.sub(r'[^0-9a-zA-Z가-힣]+', '_', meta['name'])}"
                all_chunks.append(
                    make_chunk(cid, chapter, entry_type, _pairs(seg[1]), meta["name"], meta["name_en"], section=section)
                )

    return all_chunks


def main():
    chunks = build_chunks()

    # De-duplicate ids (e.g. two monsters sharing a name) by suffixing.
    seen = {}
    for c in chunks:
        base = c["id"]
        if base in seen:
            seen[base] += 1
            c["id"] = f"{base}_{seen[base]}"
        else:
            seen[base] = 0

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_type = {}
    for c in chunks:
        by_type[c["metadata"]["type"]] = by_type.get(c["metadata"]["type"], 0) + 1

    print(f"Wrote {len(chunks)} chunks -> {CHUNKS_JSONL}")
    for t, n in by_type.items():
        print(f"  {t}: {n}")

    with_section = sum(1 for c in chunks if c["metadata"]["type"] == "rule" and c["metadata"]["section"])
    without_section = sum(1 for c in chunks if c["metadata"]["type"] == "rule" and not c["metadata"]["section"])
    print(f"  rule chunks with matched subsection heading: {with_section}")
    print(f"  rule chunks without a matched heading (intro/unmatched): {without_section}")


if __name__ == "__main__":
    main()
