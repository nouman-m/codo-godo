"""
Embedding utilities using sentence transformers.

Provides functions to generate embeddings for queries and documents using
the configured sentence transformer model.
"""

from sentence_transformers import SentenceTransformer

from src import config

_model = None


def get_embedder() -> SentenceTransformer:
    """
    Get or create the singleton embedding model.

    Loads the model on first call and reuses it for subsequent calls.

    Returns:
        SentenceTransformer: The configured embedding model.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_query(query: str) -> list[float]:
    """
    Generate embedding vector for a single query string.

    Args:
        query: The text query to embed.

    Returns:
        List of floats representing the embedding vector.
    """
    model = get_embedder()
    embedding = model.encode(query)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embedding vectors for multiple text strings.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    model = get_embedder()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings.tolist()
