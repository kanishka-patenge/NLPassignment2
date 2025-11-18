from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

from app.rag_grog import retrieve_scene_context  # reuse your retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
)

prompt = ChatPromptTemplate.from_template("""
Answer strictly using the context from Shakespeare’s *Julius Caesar*.
Your tone is academic and insightful.
You must cite your sources.
If the answer is not found in context, say:
"Context insufficient to answer this question."

<context>
{context}
</context>

Question: {question}

Answer:
""")

app = FastAPI(title="Julius Caesar RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 10


class SourceChunk(BaseModel):
    act: Optional[int]
    scene: Optional[int]
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]


def build_limited_context(docs, max_tokens: int) -> str:
    parts = []
    token_count = 0
    for d in docs:
        text = d["chunk"].replace("\n", " ")
        tokens = text.split()
        n = len(tokens)
        if token_count + n > max_tokens:
            remaining = max_tokens - token_count
            if remaining > 0:
                parts.append(" ".join(tokens[:remaining]))
            break
        parts.append(" ".join(tokens))
        token_count += n
        if token_count >= max_tokens:
            break
    return "\n\n".join(parts)


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: QueryRequest):
    docs = retrieve_scene_context(payload.question, top_k=payload.top_k)

    MAX_CONTEXT_TOKENS = 2500
    context_text = build_limited_context(docs, MAX_CONTEXT_TOKENS)

    messages = prompt.format_messages(
        context=context_text,
        question=payload.question,
    )
    result = llm.invoke(messages)

    sources = []
    for d in docs[:10]:  # just return first 10 as “sources”
        meta = d["metadata"] or {}
        preview = d["chunk"][:200].replace("\n", " ")
        sources.append(
            SourceChunk(
                act=meta.get("act"),
                scene=meta.get("scene"),
                text_preview=preview,
            )
        )

    return QueryResponse(
        answer=result.content,
        sources=sources,
    )
