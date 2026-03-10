# Codo-Godo - Future Requirements

This file documents future requirements for the RAG system. These are NOT implemented yet but MUST be considered when making changes.

## 1. LoRA Training Preparation

### Current State
- Chunks are stored in ChromaDB with metadata
- Metadata includes: source, source_type, file, start_line, function

### Requirements

- [ ] **Preserve metadata for training**: All metadata fields must remain available for export
  - `source`: repo name or doc path
  - `source_type`: gdscript_repo, qa, godot_docs
  - `file`: relative file path
  - `start_line`: line number in original file
  - `function`: function/class name if applicable
  - `godot_version`: tag as 4.x only (current)

- [ ] **Chunk prefixing**: Add context prefixes to chunks for better training
  ```
  ### Function: player_move
  func player_move():
      # code here
  ```
  ```
  ### Class: Enemy
  class Enemy:
  ```

- [ ] **Export format**: Support exporting to instruction-response pairs
  ```json
  {
    "instruction": "How do I move a character in Godot 4?",
    "response": "func _physics_process(delta):\n    velocity = input_direction * speed\n    move_and_slide()"
  }
  ```

- [ ] **Godot version filtering**: Currently 4.x only - maintain this separation

### Target Models
- DeepSeek-Coder
- MiniMax

---

## 2. MCP Server (Claude Desktop Integration)

### Requirements

- [ ] Implement MCP protocol
  - Expose `/query` as MCP tool
  - Support streaming responses
  - Return results in MCP-compatible format

- [ ] Tool schema:
  ```json
  {
    "name": "godot_rag_query",
    "description": "Query GDScript knowledge base for Godot 4.x",
    "input_schema": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "top_k": {"type": "integer", "default": 5}
      }
    }
  }
  ```

---

## 3. LangChain / LangGraph Integration

### Requirements

- [ ] Wrap retriever as LangChain compatible retriever
  ```python
  from langchain.schema import BaseRetriever
  class GodotRAGRetriever(BaseRetriever):
      ...
  ```

- [ ] Tool wrapper for agents
  ```python
  from langchain.tools import Tool
  godot_tool = Tool(
      name="godot_rag",
      func=query_function,
      description="Search Godot/GDScript knowledge base"
  )
  ```

---

## 4. Data Sources

### Currently Configured

| Source | Format | Status |
|--------|--------|--------|
| wallstoneai/godot-gdscript-dataset | .txt (5k repos) | Implemented |
| glaiveai/godot_4_docs | parquet (3.5k Q&A) | Implemented |
| Godot docs HTML | Manual download | Implemented |

### Notes

- Godot docs must be manually placed in `data/godot-docs/`
- Download from: https://docs.godotengine.org/en/stable/ (HTML zip)
- Currently Godot 4.x only

---

## 5. Implementation Notes

### Running the Project
- **IMPORTANT**: This project uses a virtual environment at `venv/`
- Always activate the venv before running commands:
  - Windows: `.\venv\Scripts\activate` or use full path to python: `venv\Scripts\python.exe`
- Example: `venv\Scripts\python.exe -m src.api.main`

### Chunking Strategy (Current)
- Hybrid function-level chunking
- Max 300 lines per chunk
- 10 line overlap for context preservation

### Embedding Model
- sentence-transformers/all-MiniLM-L6-v2
- 384 dimensions
- Local execution (no API keys required)

### Vector Store
- ChromaDB (local, file-based)
- Persistent storage in `data/chroma/`

---

# Agent Guidelines

Guidelines for agentic coding agents working on this project.

## Commands

### Virtual Environment
This project uses a virtual environment at `venv/`. Always activate it before running commands:
- **Windows**: `.\venv\Scripts\activate` or use full path: `venv\Scripts\python.exe`
- **macOS/Linux**: `source venv/bin/activate`

### Running the Project

```bash
# Start API server
python -m src.api.main

# Run data ingestion pipeline
python -m src.ingestion.pipeline

# Query via curl
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to move a character?"}'
```

### Testing

No tests currently exist. When adding tests, use **pytest**:
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_chunker.py

# Run a single test function
pytest tests/test_chunker.py::test_chunk_gdscript_file
```

### Linting/Formatting

Install and use **ruff** for linting and formatting:
```bash
pip install ruff
ruff check src/
ruff format src/
```

## Code Style Guidelines

### Imports

Use explicit relative imports within the `src` package:
```python
from src import config
from src.ingestion import pipeline
from src.retrieval import searcher
```

Order: standard library → third-party → local.

### Formatting

- **Indentation**: 4 spaces
- **Line length**: Max 100 characters
- **String quotes**: Double quotes

### Type Hints

Use Python 3.10+ native types:
```python
def search(query: str, top_k: int = None) -> list[SearchResult]:
```

### Naming

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Dataclasses

Use for structured data:
```python
@dataclass
class Chunk:
    text: str
    source: str
    source_type: str
    metadata: dict
```

### Generators

Use `yield` for memory-efficient processing:
```python
def chunk_gdscript_file(file_path: Path) -> Iterator[Chunk]:
    for line in lines:
        yield Chunk(...)
```

### Error Handling

For APIs, catch and raise HTTPException:
```python
try:
    results = searcher.search(...)
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

For ingestion, print errors:
```python
try:
    all_chunks.extend(chunker.chunk_gdscript_repo(repo_file))
except Exception as e:
    print(f"Error chunking {repo_file}: {e}")
```

### Paths

Use `pathlib.Path`:
```python
content = file_path.read_text(encoding="utf-8", errors="ignore")
```

### Configuration

All config in `src/config.py` as module constants.

### FastAPI Conventions

- Use Pydantic BaseModel for request/response
- Use `async def`
- Add docstrings to endpoints

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
│   ├── godot-docs/           # Godot HTML docs
│   └── raw/                  # HuggingFace data
├── tests/                    # Test files (add here)
├── requirements.txt
└── README.md
```
