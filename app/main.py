# app/main.py

from fastapi import FastAPI, HTTPException
from typing import List

from app.models import QueryRequest, QueryResponse, Source
from app.rag_ollama import run_query

app = FastAPI(title="RAG API (FAISS + Ollama)", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(qreq: QueryRequest) -> QueryResponse:
    """
    RAG query endpoint.

    Request body:
      {
        "query": "your question",
        "top_k": 5  // optional
      }

    Response body:
      {
        "answer": "...",
        "sources": [
          {"chunk": "...", "metadata": {...}},
          ...
        ],
        "debug": {...}
      }
    """
    try:
        result = run_query(qreq.query, top_k=qreq.top_k)
    except FileNotFoundError as e:
        # e.g. FAISS index missing / FAISS_DIR wrong
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # generic fallback for unexpected errors
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    # Build Source objects for pydantic response_model
    sources: List[Source] = [
        Source(chunk=s["chunk"], metadata=s.get("metadata", {}))
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        debug=result.get("debug", {}),
    )
from fastapi import FastAPI, HTTPException
from typing import List

from app.models import QueryRequest, QueryResponse, Source
from app.rag_gemini import run_query

app = FastAPI(title="RAG API (FAISS + Gemini)", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(qreq: QueryRequest) -> QueryResponse:
    """
    RAG query endpoint.

    Request body:
      {
        "query": "your question",
        "top_k": 15  // optional
      }
    """
    try:
        result = run_query(qreq.query, top_k=qreq.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    sources: List[Source] = [
        Source(chunk=s["chunk"], metadata=s.get("metadata", {}))
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        answer=result.get("answer", ""),
        sources=sources,
        debug=result.get("debug", {}),
    )
