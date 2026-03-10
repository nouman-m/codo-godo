from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
GODOT_DOCS_DIR = DATA_DIR / "godot-docs"
RAW_DATA_DIR = DATA_DIR / "raw"

CHROMA_COLLECTION_NAME = "godot_rag"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHUNK_MAX_LINES = 300
CHUNK_OVERLAP_LINES = 10

QUERY_TOP_K = 5

HUGGINGFACE_DATASETS = {
    "gdscript_repos": "wallstoneai/godot-gdscript-dataset",
    "godot_qa": "glaiveai/godot_4_docs",
}

API_HOST = "0.0.0.0"
API_PORT = 8000
