"""Character sheet storage.

Lives in the same Chroma DB as the rulebook, but in its own collection and
accessed only by id (get/upsert/delete) — never semantic query(). A sheet
changes every round (HP, inventory, conditions...) and the DM bot always
needs the *exact current* sheet, not a "similar" one, so there is no value
in embedding it, and re-embedding on every HP tick would just waste OpenAI
calls. A fixed dummy vector is stored instead so no embedding function
(local or remote) ever has to run for this collection.
"""
import json

from config import CHARACTER_SHEET_COLLECTION
from chroma_client import get_client

_DUMMY_EMBEDDING = [0.0] * 8


def get_collection():
    return get_client().get_or_create_collection(name=CHARACTER_SHEET_COLLECTION)


def new_character_sheet(name: str, race: str, char_class: str, level: int = 1, **extra) -> dict:
    """A blank D&D 5e Basic Rules character sheet. `extra` overrides/extends any field."""
    sheet = {
        "name": name,
        "gender": "",
        "race": race,
        "class": char_class,
        "level": level,
        "background": "",
        "alignment": "",
        "experience_points": 0,
        "ability_scores": {
            "str": 10, "dex": 10, "con": 10, "int": 10, "wis": 10, "cha": 10,
        },
        "proficiency_bonus": 2,
        "saving_throw_proficiencies": [],
        "skill_proficiencies": [],
        "armor_class": 10,
        "initiative": 0,
        "speed": 30,
        "hit_points": {"max": 0, "current": 0, "temp": 0},
        "hit_dice": "",
        "death_saves": {"successes": 0, "failures": 0},
        "attacks": [],
        "equipment": [],
        "currency": {"cp": 0, "sp": 0, "ep": 0, "gp": 0, "pp": 0},
        "languages": [],
        "tool_proficiencies": [],
        "features_traits": [],
        "personality_traits": "",
        "ideals": "",
        "bonds": "",
        "flaws": "",
        "spellcasting": None,  # e.g. {"ability": "int", "save_dc": 13, "attack_bonus": 5,
                                #       "slots": {"1": 2}, "known": [...], "prepared": [...]}
        "conditions": [],  # active status effects, see 부록 A: 상태
        "notes": "",
    }
    sheet.update(extra)
    return sheet


def _metadata_for(character_id: str, sheet: dict) -> dict:
    hp = sheet.get("hit_points") or {}
    return {
        "character_id": character_id,
        "name": sheet.get("name") or "",
        "race": sheet.get("race") or "",
        "class": sheet.get("class") or "",
        "level": int(sheet.get("level") or 0),
        "hp_current": int(hp.get("current") or 0),
        "hp_max": int(hp.get("max") or 0),
    }


def save_character(character_id: str, sheet: dict) -> None:
    """Create or fully overwrite a character sheet."""
    sheet = {**sheet, "id": character_id}
    coll = get_collection()
    coll.upsert(
        ids=[character_id],
        documents=[json.dumps(sheet, ensure_ascii=False)],
        metadatas=[_metadata_for(character_id, sheet)],
        embeddings=[_DUMMY_EMBEDDING],
    )


def get_character(character_id: str) -> dict | None:
    coll = get_collection()
    res = coll.get(ids=[character_id])
    if not res["ids"]:
        return None
    return json.loads(res["documents"][0])


def _deep_merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def update_character(character_id: str, patch: dict) -> dict:
    """Deep-merge patch into the existing sheet, e.g. {"hit_points": {"current": 5}}
    updates just that field without wiping out sibling fields like max/temp."""
    current = get_character(character_id)
    if current is None:
        raise KeyError(f"캐릭터 시트를 찾을 수 없습니다: {character_id}")
    _deep_merge(current, patch)
    save_character(character_id, current)
    return current


def delete_character(character_id: str) -> None:
    get_collection().delete(ids=[character_id])


def list_characters() -> list[dict]:
    res = get_collection().get()
    return [json.loads(doc) for doc in res["documents"]]
