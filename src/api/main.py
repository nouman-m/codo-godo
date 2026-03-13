"""
Codo-Godo RAG API

FastAPI application providing REST endpoints for querying a GDScript knowledge base
using Retrieval-Augmented Generation (RAG) with ChromaDB and sentence transformers.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from src import config
from src.ingestion import pipeline
from src.retrieval import searcher
from src.retrieval import llm


app = FastAPI(
    title="Codo-Godo RAG API",
    description="RAG system for GDScript coding with Godot knowledge base",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    query: str
    top_k: Optional[int] = None
    filter_source_type: Optional[str] = None


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    query: str
    results: list[dict]
    context: str


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    download: bool = True


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    collection_name: str
    chunks_count: int


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    query: str
    top_k: Optional[int] = 5
    stream: bool = True


class AskResponse(BaseModel):
    """Response body for the /ask endpoint (non-streaming)."""

    query: str
    answer: str
    sources: list[dict]


@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Check the health status of the API and ChromaDB collection.

    This endpoint verifies the ChromaDB connection and returns information
    about the stored knowledge base.

    Returns:
        HealthResponse:
            - status: "healthy" if successful, otherwise error details
            - collection_name: Name of the ChromaDB collection (e.g., "godot_rag")
            - chunks_count: Number of text chunks currently stored in the database

    Raises:
        HTTPException: 500 if ChromaDB connection fails.
    """
    try:
        client = pipeline.get_chroma_client()
        collection = client.get_or_create_collection(name=config.CHROMA_COLLECTION_NAME)
        count = collection.count()
        return HealthResponse(
            status="healthy",
            collection_name=config.CHROMA_COLLECTION_NAME,
            chunks_count=count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the GDScript knowledge base using semantic search.

    This endpoint performs vector similarity search against the embedded chunks
    in ChromaDB to find relevant code examples, documentation, or Q&A pairs.

    Request body (QueryRequest):
        - query (str, required): The search question or keyword phrase.
          Examples: "how to move a character", "signal connection", "export variables"
        - top_k (int, optional): Number of results to return. Default is 5.
          Higher values return more results but increase response time.
        - filter_source_type (str, optional): Filter results by data source type.
          Valid values: "gdscript_repo" (code from GitHub repos), "qa" (Q&A pairs),
          "godot_docs" (official documentation). If omitted, all sources are searched.

    Returns (QueryResponse):
        - query (str): The original search query
        - results (list[dict]): Array of matching chunks, each containing:
            - text: The chunk content (code or documentation)
            - source: Identifier of the source (repo name or "godot_4_docs")
            - source_type: Type of source (gdscript_repo, qa, or godot_docs)
            - distance: Similarity score (lower is better, 0 = exact match)
            - metadata: Additional info like file path, line number, function name
        - context (str): Formatted string combining all results, suitable for
          passing directly to an LLM as context

    Raises:
        HTTPException: 500 if search fails.
    """
    try:
        results = searcher.search(
            query=request.query,
            top_k=request.top_k,
            filter_source_type=request.filter_source_type,
        )

        result_dicts = [
            {
                "text": r.text,
                "source": r.source,
                "source_type": r.source_type,
                "distance": r.distance,
                "metadata": r.metadata,
            }
            for r in results
        ]

        context = searcher.get_context_for_query(
            request.query, top_k=request.top_k
        )

        return QueryResponse(
            query=request.query,
            results=result_dicts,
            context=context,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Query the GDScript knowledge base and get an LLM-generated answer.

    This endpoint performs RAG (Retrieval-Augmented Generation):
    1. Searches ChromaDB for relevant code chunks
    2. Builds an augmented prompt with the retrieved context
    3. Sends to Ollama for generation

    Request body (AskRequest):
        - query (str, required): The question to ask.
          Examples: "how to move a character", "signal connection", "export variables"
        - top_k (int, optional): Number of chunks to retrieve. Default is 5.
        - stream (bool, optional): Stream the response as it's generated. Default is true.

    Returns:
        If stream=true: Server-Sent Events (SSE) with tokens
        If stream=false: JSON with answer and sources

    Raises:
        HTTPException: 500 if search or generation fails.
    """
    try:
        top_k = request.top_k or config.QUERY_TOP_K

        results = searcher.search(
            query=request.query,
            top_k=top_k,
        )

        sources = [
            {
                "text": r.text,
                "source": r.source,
                "source_type": r.source_type,
                "distance": r.distance,
                "metadata": r.metadata,
            }
            for r in results
        ]

        context = searcher.get_context_for_query(request.query, top_k=top_k)

        prompt = f"""You are a helpful Godot 4.x GDScript programming assistant. Use the following context from the Godot knowledge base to answer the user's question.

        Context:
        {context}
        
        Question: {request.query}
        """

        client = llm.get_client()

        if request.stream:
            async def stream_generator():
                for token in client.stream_generate(prompt):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
            )
        else:
            result = client.generate(prompt, stream=False)
            answer = result.get("response", "")

            return AskResponse(
                query=request.query,
                answer=answer,
                sources=sources,
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Run the data ingestion pipeline to download, chunk, embed, and store data.

    This endpoint orchestrates the full data pipeline:
    1. Downloads data from HuggingFace (if download=true)
    2. Chunks GDScript files, Godot Q&A, and HTML docs
    3. Generates embeddings using sentence transformers
    4. Stores everything in ChromaDB for semantic search

    This is a long-running operation that may take several minutes depending
    on data volume. The API will not respond until completion.

    Request body (IngestRequest):
        - download (bool, optional): If true, fetches latest data from HuggingFace.
          If false, uses already-downloaded data in data/raw/. Default is true.

    Returns:
        dict: {"status": "completed"} on success

    Raises:
        HTTPException: 500 if ingestion fails (network error, processing error, etc.)
    """
    try:
        pipeline.run_ingestion(download=request.download)
        return {"status": "completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
