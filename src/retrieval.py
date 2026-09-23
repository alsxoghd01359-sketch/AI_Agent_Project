"""Hybrid retrieval over the rulebook collection.

Pure semantic search tends to rank generic prose ("주문이란 무엇인가요?")
above the one specific spell/monster entry a question is actually about,
because the entry's own wording ("화염구 Fireball, 3레벨 방출계, ...")
doesn't textually resemble a casual question. So on top of the embedding
query, this also does an exact substring match against every known
spell/monster name and pulls those chunks in directly by id.
"""
import os

from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import EMBEDDING_MODEL, RULEBOOK_COLLECTION
from chroma_client import get_client

load_dotenv()

_collection = None
_name_index = None  # [(name, chunk_id), ...] sorted by len(name) desc


def _get_collection():
    global _collection
    if _collection is None:
        fn = OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"], model_name=EMBEDDING_MODEL)
        _collection = get_client().get_collection(RULEBOOK_COLLECTION, embedding_function=fn)
    return _collection


def _build_name_index():
    global _name_index
    if _name_index is not None:
        return _name_index

    coll = _get_collection()
    entries = []
    for doc_type in ("spell", "monster"):
        res = coll.get(where={"type": doc_type}, limit=1000)
        for chunk_id, meta in zip(res["ids"], res["metadatas"]):
            for key in ("name", "name_en"):
                name = (meta.get(key) or "").strip()
                if len(name) >= 2:
                    entries.append((name, chunk_id))
    # Longest names first so "지연 폭발 화염구" matches before plain "화염구".
    entries.sort(key=lambda e: len(e[0]), reverse=True)
    _name_index = entries
    return _name_index


def _find_name_matches(query: str) -> list[str]:
    index = _build_name_index()
    matched_ids = []
    consumed = ""  # crude guard so "화염구" doesn't also match once "지연 폭발 화염구" already did
    for name, chunk_id in index:
        if name in query and name not in consumed:
            matched_ids.append(chunk_id)
            consumed += name
    return matched_ids


def _chunk_to_result(chunk_id, doc, meta, distance=None):
    return {
        "id": chunk_id,
        "text": doc,
        "type": meta.get("type"),
        "name": meta.get("name") or meta.get("name_en"),
        "chapter": meta.get("chapter"),
        "section": meta.get("section"),
        "page_start": meta.get("page_start"),
        "page_end": meta.get("page_end"),
        "distance": distance,
    }


def search_rulebook(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid search: exact spell/monster name matches + semantic top_k."""
    coll = _get_collection()
    results = {}

    for chunk_id in _find_name_matches(query):
        got = coll.get(ids=[chunk_id])
        if got["ids"]:
            results[chunk_id] = _chunk_to_result(chunk_id, got["documents"][0], got["metadatas"][0])

    semantic = coll.query(query_texts=[query], n_results=top_k)
    for chunk_id, doc, meta, dist in zip(
        semantic["ids"][0], semantic["documents"][0], semantic["metadatas"][0], semantic["distances"][0]
    ):
        if chunk_id not in results:
            results[chunk_id] = _chunk_to_result(chunk_id, doc, meta, dist)

    return list(results.values())[: max(top_k, len(_find_name_matches(query)))]
