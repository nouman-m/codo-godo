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
