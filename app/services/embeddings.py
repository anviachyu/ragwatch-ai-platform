from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def create_embeddings(chunks: list[str]) -> list[list[float]]:
    if not chunks:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()