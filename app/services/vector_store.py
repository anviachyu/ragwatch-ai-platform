import json
from pathlib import Path

import faiss
import numpy as np


VECTOR_STORE_DIR = Path("data/vector_store")
INDEX_PATH = VECTOR_STORE_DIR / "document.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"


def save_vector_store(
    embeddings: list[list[float]],
    chunks: list[str],
) -> int:
    if not embeddings:
        raise ValueError("No embeddings were provided")

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The number of embeddings and chunks must match"
        )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    embedding_array = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embedding_array.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_array)

    faiss.write_index(index, str(INDEX_PATH))

    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return index.ntotal