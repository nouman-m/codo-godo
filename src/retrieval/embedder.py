from sentence_transformers import SentenceTransformer

from src import config

_model = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_query(query: str) -> list[float]:
    model = get_embedder()
    embedding = model.encode(query)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings.tolist()
