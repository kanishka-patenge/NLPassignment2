from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None

class Source(BaseModel):
    chunk: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    debug: Optional[Dict[str, Any]] = None
