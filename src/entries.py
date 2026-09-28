"""Exact lookup of spells / monsters / magic items parsed by parse_entries.py.

Unlike search_rulebook (semantic RAG), this returns the structured record for one
named entry, so the DM gets exact numbers (HP, AC, damage dice...) instead of
re-reading them out of flattened stat-block text.
"""
import difflib
import json
import re

from config import SPELLS_JSON, MONSTERS_JSON, MAGIC_ITEMS_JSON

KINDS = {"spell": SPELLS_JSON, "monster": MONSTERS_JSON, "magic_item": MAGIC_ITEMS_JSON}
_cache = {}


def _norm(s):
    return re.sub(r"[\s'’.,:()\-]+", "", s or "").lower()


def _entries(kind):
    if kind not in _cache:
        with open(KINDS[kind], encoding="utf-8") as f:
            _cache[kind] = json.load(f)
    return _cache[kind]


def _names(entry):
    return [n for n in (entry["name"], entry.get("name_en"), *entry.get("aliases", [])) if n]


def find_exact(kind, name):
    target = _norm(name)
    return next((e for e in _entries(kind) if any(_norm(n) == target for n in _names(e))), None)


def lookup(kind, name):
    """{"entry": record} on a confident match, else {"error", "candidates"}."""
    if kind not in KINDS:
        return {"error": f"kind는 {list(KINDS)} 중 하나여야 합니다."}

    entry = find_exact(kind, name)
    if entry:
        return {"entry": entry}

    # partial match, e.g. "레드 드래곤" -> "레드 드래곤 원숙체", "저항의 반지" -> "반지, 저항의 반지"
    target = _norm(name)
    partial = [
        e for e in _entries(kind)
        if target and any(target in _norm(n) or (len(_norm(n)) >= 2 and _norm(n) in target) for n in _names(e))
    ]
    if len(partial) == 1:
        return {"entry": partial[0]}

    candidates = [e["name"] for e in partial] or difflib.get_close_matches(
        name, [e["name"] for e in _entries(kind)], n=5, cutoff=0.4
    )
    return {"error": f"'{name}'을(를) 정확히 찾지 못했습니다.", "candidates": candidates[:8]}


def monster_for_label(label):
    """Stat block for a combat label like "고블린A" / "고블린 B" / "늑대2".
    Exact matches only — never force stats from a fuzzy guess."""
    entry = find_exact("monster", label)
    if entry:
        return entry
    base = re.sub(r"[\s_\-]*(?:[A-Za-z]|\d+)$", "", label.strip())
    return find_exact("monster", base) if base and base != label.strip() else None
