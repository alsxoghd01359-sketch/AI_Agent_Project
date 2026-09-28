"""Split the spell / monster / magic-item chapters into individual entries and
parse each one into a structured JSON record.

Entry detection is shared with chunk_rulebook.py so the RAG chunks and the JSON
records always agree on where one entry ends and the next begins.

Every entry is "name line(s) + classifier line":
  spell      : "가속 Haste"         / "3레벨 변환계" | "방출계 소마법" | "1레벨 예지계 (의식)"
               (next line always starts with "시전 시간")
  monster/NPC: "점멸견 Blink Dog"   / "중형 요정, 질서 선"  (next line starts with "방어도")
  magic item : "반지, 보호의 반지 Ring of Protection" / "반지, 고급 등급 (조율 필요)"
A long name is sometimes wrapped into a Korean line followed by an English line.
"""
import json
import re

from config import PAGES_JSON, STRUCTURED_DIR, SPELLS_JSON, MONSTERS_JSON, MAGIC_ITEMS_JSON

SPELL_TYPE_RE = re.compile(
    r"^(?:(?P<level>\d)레벨\s*(?P<school>[가-힣]+계)|(?P<school0>[가-힣]+계)\s*소마법)"
    r"\s*(?P<ritual>\(의식\))?\s*$"
)
SIZE_RE = re.compile(r"^(초소형|소형|중형|대형|초거대형|거대형|거대)\s+")
ITEM_RARITIES = ["범용", "비범", "고급", "희귀", "전설", "다양함"]
ABILITY_KEYS = ["str", "dex", "con", "int", "wis", "cha"]
MONSTER_STAT_KEYS = {
    "내성 굴림": "saving_throws",
    "기술": "skills",
    "피해 취약": "damage_vulnerabilities",
    "피해 저항": "damage_resistances",
    "피해 면역": "damage_immunities",
    "상태 면역": "condition_immunities",
    "감각능력": "senses",
    "언어": "languages",
    "도전지수": "challenge",
}
MONSTER_SECTIONS = {"행동": "actions", "반응": "reactions", "반응행동": "reactions", "전설적 행동": "legendary_actions"}

HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")


# ---------- loading ----------

def load_chapter_lines(pages=None):
    """{chapter: [(page_num, raw_line, style), ...]} — trailing spaces preserved,
    stray chapter-title footer lines removed. style = {"b": leading bold text,
    "s": serif (flavor-text font)} as produced by extract_pdf.py."""
    if pages is None:
        with open(PAGES_JSON, encoding="utf-8") as f:
            pages = json.load(f)
    chapters = {}
    for p in pages:
        chapter = p["chapter"] or "미분류"
        bucket = chapters.setdefault(chapter, [])
        for line in p["lines"]:
            text = line["t"].strip()
            if not text or text == chapter:
                continue
            bucket.append((p["page_num"], line["t"], {"b": line["b"], "s": line["s"]}))
    return chapters


# ---------- text helpers ----------

def _bold_name(entry):
    """'부패의 주먹' if the line starts with a bold '부패의 주먹.' run, else None."""
    bold = entry[2]["b"]
    if bold.endswith(".") and len(bold) > 1:
        return bold[:-1].strip()
    return None


def join_lines(lines):
    """Rebuild paragraphs from PDF-wrapped (page, raw, style) lines.
    A line starting with a bold run begins a new paragraph; otherwise
    trailing space -> word boundary, sentence end -> paragraph break,
    short line -> separate cell (tables), anything else broke mid-word."""
    paras, cur = [], ""
    for entry in lines:
        raw = entry[1]
        text = raw.strip()
        if not text:
            continue
        if entry[2]["b"] and cur.strip():
            paras.append(cur.strip())
            cur = ""
        cur += text
        if raw.endswith((" ", "\t")):
            cur += " "
        elif text.endswith((".", "!", "?", '"', "”", ":")):
            paras.append(cur.strip())
            cur = ""
        elif len(text) < 20:
            cur += " "
    if cur.strip():
        paras.append(cur.strip())
    return paras


def _continuation_sep(prev_raw):
    """Separator for gluing a wrapped field line onto the previous one."""
    return " " if prev_raw.endswith((" ", "\t")) or len(prev_raw.strip()) < 20 else ""


def split_name(raw):
    raw = raw.strip()
    for i, ch in enumerate(raw):
        if LATIN.match(ch) and not HANGUL.search(raw[i:]):
            ko, en = raw[:i].strip(), raw[i:].strip()
            return (ko or en), en
    return raw, ""


def _name_at(lines, name_idx):
    """(start_idx, name_ko, name_en), merging a Korean line + English-only line."""
    raw = lines[name_idx][1].strip()
    if not HANGUL.search(raw) and name_idx > 0:
        prev = lines[name_idx - 1][1].strip()
        if HANGUL.search(prev) and not LATIN.search(prev) and len(prev) < 25 and not prev.endswith("."):
            return name_idx - 1, prev, raw
    ko, en = split_name(raw)
    return name_idx, ko, en


def _split_outside_parens(text, seps=",."):
    depth = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in seps and depth == 0:
            return text[:i].strip(), text[i + 1:].strip()
    return text.strip(), ""


# ---------- entry detection (shared with chunk_rulebook) ----------

def _is_spell_header(lines, i):
    return (
        SPELL_TYPE_RE.match(lines[i][1].strip())
        and i + 1 < len(lines)
        and lines[i + 1][1].strip().startswith("시전 시간")
    )


def _is_monster_header(lines, i):
    return (
        SIZE_RE.match(lines[i][1].strip())
        and i + 1 < len(lines)
        and lines[i + 1][1].strip().startswith("방어도")
    )


def _is_item_header(lines, i):
    text = lines[i][1].strip()
    return (
        len(text) < 80
        and ("등급" in text or text.endswith("다양함"))
        and LATIN.search(lines[i - 1][1])
    )


def find_entries(lines, kind, stop_titles=(), start_after=None):
    """Split a chapter's (page, line) list into segments.
    Returns [("gap", lines) | ("entry", lines, meta)] in reading order.
    stop_titles: section headings (e.g. "논플레이어 캐릭터") that end the current
    entry and start a gap, so section intros don't get glued onto the last entry."""
    detector = {"spell": _is_spell_header, "monster": _is_monster_header, "magic_item": _is_item_header}[kind]
    stops = {re.sub(r"\s+", "", t) for t in stop_titles}

    begin = 0
    if start_after:
        for i, line in enumerate(lines):
            if line[1].strip() == start_after:
                begin = i + 1
                break

    boundaries = []  # (start_idx, header_idx, ko, en) or (stop_idx, None, title, None)
    for i in range(max(begin, 1), len(lines)):
        if detector(lines, i):
            start, ko, en = _name_at(lines, i - 1)
            boundaries.append((start, i, ko, en))
        elif re.sub(r"\s+", "", lines[i][1]) in stops:
            boundaries.append((i, None, lines[i][1].strip(), None))

    segments = []
    first = boundaries[0][0] if boundaries else len(lines)
    if first > 0:
        segments.append(("gap", lines[:first]))
    for k, (start, header, ko, en) in enumerate(boundaries):
        end = boundaries[k + 1][0] if k + 1 < len(boundaries) else len(lines)
        part = lines[start:end]
        if header is None:
            segments.append(("gap", part))
        else:
            segments.append(("entry", part, {"name": ko, "name_en": en, "header_offset": header - start}))
    return segments


# ---------- parsers ----------

def parse_spell(part, meta):
    lines = part[meta["header_offset"]:]
    m = SPELL_TYPE_RE.match(lines[0][1].strip())
    fields, key, desc_start = {}, None, len(lines)
    field_keys = {"시전 시간": "casting_time", "사거리": "range", "구성요소": "components", "지속시간": "duration"}
    for j in range(1, len(lines)):
        text = lines[j][1].strip()
        hit = next((k for k in field_keys if text.startswith(k + ":")), None)
        if hit:
            key = field_keys[hit]
            fields[key] = text.split(":", 1)[1].strip()
            if key == "duration":
                desc_start = j + 1
                break
        elif key:
            fields[key] += _continuation_sep(lines[j - 1][1]) + text

    paras = join_lines(lines[desc_start:])
    higher = None
    body = []
    for p in paras:
        if p.startswith("고레벨에서."):
            higher = p[len("고레벨에서."):].strip()
        elif higher is not None:
            higher += "\n" + p
        else:
            body.append(p)

    comp = fields.get("components", "")
    material = re.search(r"물질\s*\((.*)\)", comp)
    return {
        "name": meta["name"],
        "name_en": meta["name_en"],
        "level": int(m.group("level")) if m.group("level") else 0,
        "school": m.group("school") or m.group("school0"),
        "ritual": bool(m.group("ritual")),
        "casting_time": fields.get("casting_time"),
        "range": fields.get("range"),
        "components": comp,
        "verbal": "음성" in comp,
        "somatic": "동작" in comp,
        "material": material.group(1).strip() if material else ("" if "물질" not in comp else "물질"),
        "duration": fields.get("duration"),
        "concentration": (fields.get("duration") or "").startswith("집중"),
        "description": "\n".join(body),
        "at_higher_levels": higher,
        "classes": [],
        "page": part[0][0],
    }


def parse_monster(part, meta):
    lines = part[meta["header_offset"]:]
    size_line = lines[0][1].strip()
    size = SIZE_RE.match(size_line).group(1)
    ctype, alignment = _split_outside_parens(size_line[len(size):].strip())

    record = {
        "name": meta["name"],
        "name_en": meta["name_en"],
        "size": size,
        "type": ctype,
        "alignment": alignment,
        "armor_class": None,
        "armor_note": "",
        "hit_points": None,
        "hit_dice": "",
        "speed": "",
        "ability_scores": {},
        **{v: "" for v in MONSTER_STAT_KEYS.values() if v != "challenge"},
        "challenge_rating": None,
        "xp": None,
        "traits": [],
        "actions": [],
        "reactions": [],
        "legendary_actions": [],
        "legendary_intro": "",
        "description": "",
        "page": part[0][0],
    }

    j = 1
    ability_buf = []
    stat_key = None
    while j < len(lines):
        text = lines[j][1].strip()
        if text.startswith("방어도"):
            m = re.match(r"방어도\s*(\d+)\s*(?:\((.*)\))?\s*$", text)
            if m:
                record["armor_class"] = int(m.group(1))
                record["armor_note"] = (m.group(2) or "").strip()
            elif re.search(r"\d+", text):
                # e.g. 위어울프 "방어도 인간형일 때 11, 늑대나 변종형일 때 12 (자연 갑옷)"
                record["armor_class"] = int(re.search(r"\d+", text).group())
                record["armor_note"] = text[len("방어도"):].strip()
        elif text.startswith("히트 포인트"):
            m = re.match(r"히트 포인트\s*(\d+)\s*(?:\(([^)]*)\))?", text)
            if m:
                record["hit_points"] = int(m.group(1))
                record["hit_dice"] = (m.group(2) or "").replace(" ", "")
        elif text.startswith("이동속도"):
            record["speed"] = text[len("이동속도"):].strip()
        else:
            hit = next((k for k in MONSTER_STAT_KEYS if text.startswith(k)), None)
            if hit:
                stat_key = MONSTER_STAT_KEYS[hit]
                value = text[len(hit):].strip()
                if stat_key == "challenge":
                    m = re.match(r"([\d/]+)\s*\(\s*([\d,]+)\s*xp\s*\)", value)
                    if m:
                        record["challenge_rating"] = m.group(1)
                        record["xp"] = int(m.group(2).replace(",", ""))
                    j += 1
                    break
                record[stat_key] = value
            elif stat_key:
                record[stat_key] += _continuation_sep(lines[j - 1][1]) + text
            else:
                ability_buf.append(text)
        j += 1

    values = re.findall(r"(\d+)\s*\(\s*([+\-−]?\d+)\s*\)", " ".join(ability_buf))
    if len(values) >= 6:
        record["ability_scores"] = {k: int(v[0]) for k, v in zip(ABILITY_KEYS, values[:6])}

    # Traits / actions / reactions / legendary actions, then flavor text.
    # Item names are set in bold ("부패의 주먹."), flavor text in the serif font, and flavor
    # sub-headings ("레드 드래곤의 본거지") are fully bold without a trailing period.
    field, current = "traits", None
    groups, intro_lines, flavor_lines = [], [], []
    for entry in lines[j:]:
        page, raw, style = entry
        text = raw.strip()
        if flavor_lines or style["s"] or (style["b"] == text and not text.endswith(".") and text not in MONSTER_SECTIONS):
            flavor_lines.append(entry)
            continue
        if text in MONSTER_SECTIONS:
            field, current = MONSTER_SECTIONS[text], None
            continue
        name = _bold_name(entry)
        if name:
            rest = raw[raw.find(style["b"]) + len(style["b"]):].lstrip()
            current = {"field": field, "name": name, "lines": [(page, rest, {"b": "", "s": False})]}
            groups.append(current)
        elif current:
            current["lines"].append(entry)
        elif field == "legendary_actions":
            intro_lines.append(entry)
        else:
            flavor_lines.append(entry)

    for g in groups:
        record[g["field"]].append({"name": g["name"], "description": " ".join(join_lines(g["lines"]))})
    record["legendary_intro"] = " ".join(join_lines(intro_lines))
    record["description"] = "\n".join(join_lines(flavor_lines))
    return record


def parse_magic_item(part, meta):
    lines = part[meta["header_offset"]:]
    type_line = lines[0][1].strip()
    aliases = []
    if "," in meta["name"]:
        category, rest = meta["name"].split(",", 1)
        rest = rest.strip()
        if rest:
            aliases.append(rest)
    return {
        "name": meta["name"],
        "name_en": meta["name_en"],
        "aliases": aliases,
        "category": re.split(r"[,(.]", type_line)[0].strip(),
        "type_line": type_line,
        "rarity": [r for r in ITEM_RARITIES if r in type_line],
        "requires_attunement": "조율 필요" in type_line,
        "description": "\n".join(join_lines(lines[1:])),
        "page": part[0][0],
    }


# The class lists and the detail section occasionally spell the same spell differently.
CLASS_LIST_NAME_FIXES = {"치유의단어": "치료의단어"}


def parse_spell_class_lists(ch11_lines):
    """{normalized spell name: [class, ...]} from the 클레릭/위저드 주문 lists.
    (안내 and 화염 폭풍 appear in the cleric list but the book has no detail entry for them.)"""
    classes, current = {}, None
    for _, raw, _ in ch11_lines:
        text = raw.strip()
        if text in ("클레릭 주문", "위저드 주문"):
            current = text.replace(" 주문", "")
            continue
        if text == "주문 상세":
            break
        if current is None or re.match(r"^(소마법|\d레벨)", text):
            continue
        key = re.sub(r"\s+", "", text)
        classes.setdefault(CLASS_LIST_NAME_FIXES.get(key, key), []).append(current)
    return classes


def _chapter(chapters, prefix):
    return next(v for k, v in chapters.items() if k.startswith(prefix))


def build_all(chapters=None):
    chapters = chapters or load_chapter_lines()

    ch11 = _chapter(chapters, "제11장")
    class_map = parse_spell_class_lists(ch11)
    spells = []
    for seg in find_entries(ch11, "spell"):
        if seg[0] == "entry":
            spell = parse_spell(seg[1], seg[2])
            spell["classes"] = class_map.get(re.sub(r"\s+", "", spell["name"]), [])
            spells.append(spell)

    ch12 = _chapter(chapters, "제12장")
    monsters, category = [], "monster"
    for seg in find_entries(ch12, "monster", stop_titles=["논플레이어 캐릭터"]):
        if seg[0] == "gap" and any(l.strip() == "논플레이어 캐릭터" for _, l, _ in seg[1]):
            category = "npc"
        elif seg[0] == "entry":
            monster = parse_monster(seg[1], seg[2])
            monster["category"] = category
            monsters.append(monster)

    ch14 = _chapter(chapters, "제14장")
    items = [
        parse_magic_item(seg[1], seg[2])
        for seg in find_entries(ch14, "magic_item", start_after="물건 상세설명")
        if seg[0] == "entry"
    ]
    return spells, monsters, items


def validate(spells, monsters, items):
    problems = []
    for s in spells:
        for f in ("casting_time", "range", "components", "duration", "description"):
            if not s.get(f):
                problems.append(f"spell {s['name']}: missing {f}")
    for m in monsters:
        for f in ("armor_class", "hit_points", "challenge_rating", "speed"):
            if m.get(f) in (None, ""):
                problems.append(f"monster {m['name']}: missing {f}")
        if len(m["ability_scores"]) != 6:
            problems.append(f"monster {m['name']}: ability_scores={m['ability_scores']}")
        if not m["actions"] and not m["traits"]:  # 개구리·해마는 원래 행동 없이 특성만 있음
            problems.append(f"monster {m['name']}: no actions or traits")
    for it in items:
        if not it["description"] or not it["rarity"]:
            problems.append(f"item {it['name']}: missing description/rarity")
    return problems


def main():
    spells, monsters, items = build_all()
    STRUCTURED_DIR.mkdir(parents=True, exist_ok=True)
    for path, data in ((SPELLS_JSON, spells), (MONSTERS_JSON, monsters), (MAGIC_ITEMS_JSON, items)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    npcs = sum(1 for m in monsters if m["category"] == "npc")
    print(f"spells: {len(spells)}  monsters: {len(monsters) - npcs}  npcs: {npcs}  magic items: {len(items)}")
    problems = validate(spells, monsters, items)
    print(f"validation problems: {len(problems)}")
    for p in problems:
        print("  " + p)


if __name__ == "__main__":
    main()
