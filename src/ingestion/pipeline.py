"""
Data ingestion pipeline for the Codo-Godo RAG system.

Orchestrates downloading data from HuggingFace, chunking GDScript files and
HTML documentation, generating embeddings, and storing in ChromaDB.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src import config
from src.ingestion import downloader, chunker


def get_chroma_client():
    """
    Create and return a persistent ChromaDB client.

    Returns:
        chromadb.PersistentClient: Configured client pointing to CHROMA_DIR.
    """
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_embedding_model():
    """
    Load and return the sentence transformer embedding model.

    Returns:
        SentenceTransformer: The configured embedding model.
    """
    return SentenceTransformer(config.EMBEDDING_MODEL)


def create_chunks() -> list[chunker.Chunk]:
    """
    Process all data sources and create chunks.

    Iterates through GDScript repos, Godot Q&A, and Godot documentation,
    yielding chunks from each source.

    Returns:
        list[Chunk]: Combined list of all chunks from all data sources.
    """
    all_chunks = []
    chunk_count = 0

    print("Processing GDScript repos...")
    try:
        gdscript_dir = downloader.get_gdscript_repos_dir()
        if gdscript_dir.exists():
            repo_files = list(gdscript_dir.glob("*.txt"))
            print(f"Found {len(repo_files)} repo files")
            for repo_file in tqdm(repo_files, desc="Chunking repos"):
                try:
                    for chunk in chunker.chunk_gdscript_repo(repo_file):
                        all_chunks.append(chunk)
                        chunk_count += 1
                        if chunk_count % 500 == 0:
                            print(f"\n--- Chunk {chunk_count} sample (GDScript repo) ---")
                            print(chunk.text[:500])
                            print("--- end sample ---\n")
                except Exception as e:
                    print(f"Error chunking {repo_file}: {e}")
    except FileNotFoundError:
        print("GDScript repos not found, skipping...")

    print(f"Total GDScript chunks so far: {len(all_chunks)}")

    print("Processing Godot Q&A...")
    try:
        qa_dir = downloader.get_godot_qa_dir()
        if qa_dir.exists():
            for chunk in chunker.chunk_godot_qa(qa_dir):
                all_chunks.append(chunk)
                chunk_count += 1
                if chunk_count % 500 == 0:
                    print(f"\n--- Chunk {chunk_count} sample (Q&A) ---")
                    print(chunk.text[:500])
                    print("--- end sample ---\n")
    except FileNotFoundError:
        print("Godot Q&A not found, skipping...")

    print(f"Total chunks after Q&A: {len(all_chunks)}")

    print("Processing Godot docs HTML...")
    try:
        docs_dir = downloader.get_godot_docs_dir()
        for chunk in chunker.chunk_godot_docs(docs_dir):
            all_chunks.append(chunk)
            chunk_count += 1
            if chunk_count % 500 == 0:
                print(f"\n--- Chunk {chunk_count} sample (Godot docs) ---")
                print(chunk.text[:500])
                print("--- end sample ---\n")
    except FileNotFoundError as e:
        print(f"Godot docs not found: {e}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def embed_chunks(chunks: list[chunker.Chunk], model) -> tuple[list[str], list[dict], any]:
    """
    Generate embeddings for a list of chunks using the provided model.

    Args:
        chunks: List of Chunk objects to embed.
        model: SentenceTransformer model for encoding.

    Returns:
        Tuple containing:
            - texts: List of chunk text strings
            - metadatas: List of metadata dictionaries
            - embeddings: numpy array of embeddings
    """
    texts = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source": chunk.source,
            "source_type": chunk.source_type,
            **chunk.metadata,
        }
        for chunk in chunks
    ]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    return texts, metadatas, embeddings


def store_in_chroma(
    texts: list[str],
    metadatas: list[dict],
    embeddings,
    client,
):
    """
    Store embedded chunks in ChromaDB collection.

    Args:
        texts: List of text strings to store.
        metadatas: List of metadata dictionaries for each text.
        embeddings: numpy array of embeddings.
        client: ChromaDB client instance.
    """
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION_NAME,
    )

    ids = [f"chunk_{i}" for i in range(len(texts))]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    print(f"Stored {len(ids)} chunks in ChromaDB")


def run_ingestion(download: bool = True):
    """
    Main ingestion pipeline that orchestrates the entire data processing workflow.

    Optionally downloads data from HuggingFace, creates chunks from all sources,
    generates embeddings, and stores them in ChromaDB.

    Args:
        download: If True, downloads fresh data from HuggingFace. If False,
                 uses existing raw data in DATA_DIR.
    """
    if download:
        print("=== Downloading data ===")
        try:
            downloader.download_gdscript_repos()
        except Exception as e:
            print(f"Error downloading GDScript repos: {e}")

        try:
            downloader.download_godot_qa()
        except Exception as e:
            print(f"Error downloading Godot Q&A: {e}")

    print("\n=== Creating chunks ===")
    chunks = create_chunks()

    if not chunks:
        print("No chunks created. Exiting.")
        return

    print("\n=== Loading embedding model ===")
    model = get_embedding_model()

    print("\n=== Embedding chunks ===")
    texts, metadatas, embeddings = embed_chunks(chunks, model)

    print("\n=== Storing in ChromaDB ===")
    client = get_chroma_client()
    store_in_chroma(texts, metadatas, embeddings, client)

    print("\n=== Ingestion complete ===")


if __name__ == "__main__":
    run_ingestion(download=False)
