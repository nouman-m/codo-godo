from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from src import config
from src.ingestion import pipeline
from src.retrieval import searcher


app = FastAPI(
    title="Codo-Godo RAG API",
    description="RAG system for GDScript coding with Godot knowledge base",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    filter_source_type: Optional[str] = None


class QueryResponse(BaseModel):
    query: str
    results: list[dict]
    context: str


class IngestRequest(BaseModel):
    download: bool = True


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    chunks_count: int


@app.get("/health", response_model=HealthResponse)
async def health():
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


@app.post("/ingest")
async def ingest(request: IngestRequest):
    try:
        pipeline.run_ingestion(download=request.download)
        return {"status": "completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
