from dataclasses import dataclass
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from src import config
from src.retrieval import embedder


_client = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


@dataclass
class SearchResult:
    text: str
    source: str
    source_type: str
    distance: float
    metadata: dict


def search(
    query: str,
    top_k: int = None,
    filter_source_type: Optional[str] = None,
) -> list[SearchResult]:
    if top_k is None:
        top_k = config.QUERY_TOP_K

    query_embedding = embedder.embed_query(query)

    client = get_client()
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME
    )

    where_filter = None
    if filter_source_type:
        where_filter = {"source_type": filter_source_type}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
    )

    search_results = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            search_results.append(
                SearchResult(
                    text=doc,
                    source=results["metadatas"][0][i].get("source", ""),
                    source_type=results["metadatas"][0][i].get("source_type", ""),
                    distance=results["distances"][0][i],
                    metadata=results["metadatas"][0][i],
                )
            )

    return search_results


def get_context_for_query(query: str, top_k: int = None) -> str:
    results = search(query, top_k=top_k)
    context_parts = []
    for r in results:
        context_parts.append(f"[{r.source_type}] {r.text}")
    return "\n\n---\n\n".join(context_parts)
