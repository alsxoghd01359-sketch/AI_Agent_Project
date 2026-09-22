"""Embed data/raw/chunks.jsonl into the ChromaDB "rulebook" collection
using OpenAI embeddings.
"""
import json
import os

from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import CHUNKS_JSONL, RULEBOOK_COLLECTION, EMBEDDING_MODEL
from chroma_client import get_client

BATCH_SIZE = 100


def load_chunks():
    chunks = []
    with open(CHUNKS_JSONL, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def clean_metadata(md: dict) -> dict:
    # Chroma metadata values must be str/int/float/bool, no None.
    out = {}
    for k, v in md.items():
        out[k] = v if v is not None else ""
    return out


def main():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_JSONL}")

    client = get_client()
    embedding_fn = OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL)

    # Fresh build each run so re-chunking doesn't leave stale/duplicate docs.
    try:
        client.delete_collection(RULEBOOK_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=RULEBOOK_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[clean_metadata(c["metadata"]) for c in batch],
        )
        print(f"  embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"Done. Collection '{RULEBOOK_COLLECTION}' has {collection.count()} docs.")


if __name__ == "__main__":
    main()
