"""Turn extracted pages.json into RAG-ready chunks.

Strategy:
- Chapter 11 (주문/Spells) and Chapter 12 (괴물/Monsters, incl. NPCs) are
  split per entry (one spell / one stat block = one chunk), detected via the
  rulebook's consistent formatting:
    spell entry header  = name line, followed by a line like "3레벨 변환계"
    monster entry header = name line, followed by a line starting with
                            초소형/소형/중형/대형/거대 (creature size)
- Everything else (including the non-entry intro text inside chapters 11/12)
  is first split at the rulebook's own subsection headings, recovered from
  the table of contents on page 2 (e.g. "갑옷과 방패류", "휴식", "환경").
  Each subsection becomes its own chunk so a chunk never mixes two topics.
  A subsection that's still too long is further sliced into ~450-token
  windows with a small overlap, breaking on line boundaries.
"""
import json
import re

import tiktoken

from config import PAGES_JSON, CHUNKS_JSONL

SPELL_HEADER_RE = re.compile(r"^(소마법|\d+레벨)\s*(\(0레벨\))?\s*\S*계\s*$")
MONSTER_HEADER_RE = re.compile(r"^(초소형|소형|중형|대형|거대)\s+[가-힣/()]{1,12},\s*[가-힣\s]{1,15}$")
NAME_SPLIT_RE = re.compile(r"^([가-힣][가-힣\s]*?)\s+([A-Za-z][A-Za-z0-9\s\-'/,.]*)$")
TOC_LINE_RE = re.compile(r"^(.+?)[.\t ]{2,}(\d+)$")

ENC = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 450
OVERLAP_TOKENS = 60


def load_pages():
    with open(PAGES_JSON, encoding="utf-8") as f:
        return json.load(f)


def load_lines_by_chapter(pages):
    chapters = {}  # chapter title -> list of (page_num, line)
    order = []  # preserve first-seen order of chapters
    for p in pages:
        chapter = p["chapter"] or "미분류"
        if chapter not in chapters:
            chapters[chapter] = []
            order.append(chapter)
        for line in p["text"].split("\n"):
            line = line.strip()
            if line:
                chapters[chapter].append((p["page_num"], line))
    return order, chapters


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


def split_entries(lines, header_re):
    """Split (page,line) tuples into [intro_lines, entry_lines, entry_lines, ...]
    using header_re matched against line i while line i-1 is treated as the
    entry's name header."""
    header_starts = []
    for i in range(1, len(lines)):
        if header_re.match(lines[i][1]):
            header_starts.append(i - 1)

    if not header_starts:
        return [lines], []

    segments = []
    intro = lines[: header_starts[0]]
    for k, start in enumerate(header_starts):
        end = header_starts[k + 1] if k + 1 < len(header_starts) else len(lines)
        segments.append(lines[start:end])
    return ([intro] if intro else []), segments


def entry_name(lines):
    raw = lines[0][1]
    m = NAME_SPLIT_RE.match(raw)
    if m:
        return raw, m.group(1).strip(), m.group(2).strip()
    return raw, raw, None


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
    order, chapters = load_lines_by_chapter(pages)
    toc = parse_toc(pages)
    all_chunks = []

    for chapter in order:
        lines = chapters[chapter]
        slug = chapter_slug(chapter)
        subsection_titles = toc.get(chapter, [])

        if chapter.startswith("제11장"):
            intros, entries = split_entries(lines, SPELL_HEADER_RE)
            for intro in intros:
                all_chunks.extend(
                    chunk_with_sections(f"{slug}_intro", chapter, intro, subsection_titles)
                )
            for entry in entries:
                raw_name, ko, en = entry_name(entry)
                cid = f"spell_{re.sub(r'[^0-9a-zA-Z가-힣]+', '_', raw_name)}"
                all_chunks.append(make_chunk(cid, chapter, "spell", entry, ko, en, section="주문 상세"))

        elif chapter.startswith("제12장"):
            intros, entries = split_entries(lines, MONSTER_HEADER_RE)
            for intro in intros:
                all_chunks.extend(
                    chunk_with_sections(f"{slug}_intro", chapter, intro, subsection_titles)
                )
            for entry in entries:
                raw_name, ko, en = entry_name(entry)
                cid = f"monster_{re.sub(r'[^0-9a-zA-Z가-힣]+', '_', raw_name)}"
                all_chunks.append(make_chunk(cid, chapter, "monster", entry, ko, en, section="괴물의 자료 상자"))

        else:
            all_chunks.extend(chunk_with_sections(slug, chapter, lines, subsection_titles))

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
