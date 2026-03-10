# Codo-Godo

A RAG (Retrieval-Augmented Generation) system for GDScript coding with Godot knowledge base.

## Purpose

This system helps improve GDScript code generation by LLMs by providing context from a curated knowledge base of:
- 5,000+ Godot/GDScript repositories
- Godot 4.x Q&A pairs
- Official Godot documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Godot Documentation

1. Go to: https://docs.godotengine.org/en/stable/
2. Download the HTML documentation (updated weekly)
3. Extract to: `data/godot-docs/`

The folder structure should look like:
```
data/godot-docs/
├── index.html
├── _static/
├── classes/
├── tutorials/
└── ...
```

### 3. Run Ingestion

```bash
python -m src.ingestion.pipeline
```

This will:
- Download GDScript repos from HuggingFace
- Download Godot Q&A from HuggingFace  
- Chunk all content
- Generate embeddings
- Store in ChromaDB

### 4. Start API Server

```bash
python -m src.api.main
```

Server runs at `http://localhost:8000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check system health |
| `/query` | POST | Query the knowledge base |
| `/ingest` | POST | Re-ingest data |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to move a character in Godot 4?"}'
```

## Project Structure

```
codo-godo/
├── src/
│   ├── api/main.py           # FastAPI server
│   ├── ingestion/
│   │   ├── downloader.py     # HuggingFace download
│   │   ├── chunker.py        # GDScript + HTML chunking
│   │   └── pipeline.py       # Orchestration
│   ├── retrieval/
│   │   ├── embedder.py       # Embedding model
│   │   └── searcher.py       # ChromaDB query
│   └── config.py             # Settings
├── data/
│   ├── chroma/               # ChromaDB storage
│   ├── godot-docs/           # ← Place HTML docs here
│   └── raw/                  # Downloaded HuggingFace data
├── AGENTS.md                 # Future requirements
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- ~8GB RAM for embedding model
- ~2GB disk space for data

## Configuration

Edit `src/config.py` to customize:
- Embedding model
- Chunk sizes
- API port
- ChromaDB path
