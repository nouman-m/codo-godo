import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src import config
from src.ingestion import downloader, chunker


def get_chroma_client():
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_embedding_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)


def create_chunks() -> list[chunker.Chunk]:
    all_chunks = []

    print("Processing GDScript repos...")
    try:
        gdscript_dir = downloader.get_gdscript_repos_dir()
        if gdscript_dir.exists():
            repo_files = list(gdscript_dir.glob("*.txt"))
            print(f"Found {len(repo_files)} repo files")
            for repo_file in tqdm(repo_files, desc="Chunking repos"):
                try:
                    all_chunks.extend(chunker.chunk_gdscript_repo(repo_file))
                except Exception as e:
                    print(f"Error chunking {repo_file}: {e}")
    except FileNotFoundError:
        print("GDScript repos not found, skipping...")

    print(f"Total GDScript chunks: {len(all_chunks)}")

    print("Processing Godot Q&A...")
    try:
        qa_dir = downloader.get_godot_qa_dir()
        if qa_dir.exists():
            qa_chunks = list(chunker.chunk_godot_qa(qa_dir))
            all_chunks.extend(qa_chunks)
            print(f"Total Q&A chunks: {len(qa_chunks)}")
    except FileNotFoundError:
        print("Godot Q&A not found, skipping...")

    print("Processing Godot docs HTML...")
    try:
        docs_dir = downloader.get_godot_docs_dir()
        doc_chunks = list(chunker.chunk_godot_docs(docs_dir))
        all_chunks.extend(doc_chunks)
        print(f"Total docs chunks: {len(doc_chunks)}")
    except FileNotFoundError as e:
        print(f"Godot docs not found: {e}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def embed_chunks(chunks: list[chunker.Chunk], model) -> tuple[list[str], list[dict], any]:
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
    run_ingestion(download=True)
