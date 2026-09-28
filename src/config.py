from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = ROOT_DIR / "dnd_rulebook.pdf"
SCENARIO_PATH = ROOT_DIR / "scenario.txt"

RAW_DIR = ROOT_DIR / "data" / "raw"
PAGES_JSON = RAW_DIR / "pages.json"
CHUNKS_JSONL = RAW_DIR / "chunks.jsonl"

STRUCTURED_DIR = ROOT_DIR / "data" / "structured"
SPELLS_JSON = STRUCTURED_DIR / "spells.json"
MONSTERS_JSON = STRUCTURED_DIR / "monsters.json"
MAGIC_ITEMS_JSON = STRUCTURED_DIR / "magic_items.json"

CHROMA_DIR = ROOT_DIR / "data" / "chroma_db"
RULEBOOK_COLLECTION = "rulebook"
CHARACTER_SHEET_COLLECTION = "character_sheets"

EMBEDDING_MODEL = "text-embedding-3-small"

BOILERPLATE_LINE = "D&D 기초 규칙(버전 1.0). 재판매 금지. 개인적 용도로만 출력 및 사용 가능."
