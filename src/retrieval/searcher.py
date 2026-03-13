"""
Retrieval utilities for querying the ChromaDB knowledge base.

Provides functions to search for relevant chunks using semantic similarity
with query embeddings.
"""

from dataclasses import dataclass
from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from src import config
from src.retrieval import embedder


_client = None


def get_client():
    """
    Get or create the singleton ChromaDB client.

    Returns:
        chromadb.PersistentClient: Configured client pointing to CHROMA_DIR.
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


@dataclass
class SearchResult:
    """Represents a search result from the knowledge base."""

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
    """
    Search the knowledge base for chunks relevant to the query.

    Embeds the query and performs similarity search against stored chunks.

    Args:
        query: The search query string.
        top_k: Number of results to return. Defaults to config.QUERY_TOP_K.
        filter_source_type: Optional filter to restrict to a specific source type
                           (e.g., 'gdscript_repo', 'qa', 'godot_docs').

    Returns:
        List of SearchResult objects sorted by relevance (distance).
    """
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
    """
    Generate a formatted context string from search results.

    Combines the top search results into a single context string suitable
    for passing to an LLM.

    Args:
        query: The original search query.
        top_k: Number of results to include. Defaults to config.QUERY_TOP_K.

    Returns:
        Formatted string with source type labels and content from each result,
        separated by horizontal rules.
    """
    results = search(query, top_k=top_k)
    context_parts = []
    for r in results:
        context_parts.append(f"[{r.source_type}] {r.text}")
    return "\n\n---\n\n".join(context_parts)
