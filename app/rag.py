from pathlib import Path
import os, json
from typing import List, Dict, Any, Tuple

from nomic import embed
import numpy as np

# LangChain FAISS wrapper used to load index
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

# load .env values
from dotenv import load_dotenv
load_dotenv()

FAISS_DIR = os.environ.get("FAISS_DIR", "faiss_index")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "5"))
RERANK = os.environ.get("RERANK", "false").lower() == "true"

# --- Embedding adapter for Nomic to use in retrieval loading ---
class NomicEmbeddings(Embeddings):
    def __init__(self, model: str = "nomic-embed-text-v1.5", batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            resp = embed.text(texts=batch, model=self.model)
            out.extend(resp["embeddings"])
        return out

    def embed_query(self, text: str) -> List[float]:
        resp = embed.text(texts=[text], model=self.model)
        return resp["embeddings"][0]

# --- Load FAISS index (to memory) ---
_embedding = NomicEmbeddings()
faiss_store = None

def load_faiss_index():
    global faiss_store, _embedding
    if faiss_store is None:
        if not Path(FAISS_DIR).exists():
            raise FileNotFoundError(f"FAISS directory not found: {FAISS_DIR}")
        faiss_store = FAISS.load_local(FAISS_DIR, _embedding, allow_dangerous_deserialization=True)
    return faiss_store

# --- Retrieval function ---
def retrieve(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts: {'chunk': text, 'metadata': {...}, 'score': float}
    """
    top_k = top_k or DEFAULT_TOP_K
    store = load_faiss_index()
    docs = store.similarity_search(query, k=top_k)
    results = []
    for d in docs:
        results.append({
            "chunk": d.page_content,
            "metadata": d.metadata or {},
            "score": None  # langchain FAISS wrapper may not expose raw score here
        })
    return results

# --- Simple re-ranker (optional): uses lightweight lexical scoring as fallback ---
def simple_rerank(query: str, candidates: List[Dict[str,Any]], top_m: int = 3) -> List[Dict[str,Any]]:
    """
    Cheap reranker: score by overlap of query tokens with chunk tokens.
    Replace with a cross-encoder or LLM-based re-ranker if available.
    """
    q_tokens = set(query.lower().split())
    scored = []
    for c in candidates:
        txt = c["chunk"].lower()
        score = sum(1 for t in q_tokens if t in txt)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = [c for s,c in scored[:top_m]]
    return out

# --- Generator (Gemini if available else fallback) ---
# Gemini client optional:
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "text-bison@001")

# to support both: try to import google.generativeai if present
_gemini_client = None
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_client = genai
except Exception:
    _gemini_client = None

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.joinpath("system_prompt.txt")
if SYSTEM_PROMPT_PATH.exists():
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    # fallback minimal system prompt
    SYSTEM_PROMPT = "You are a helpful assistant. Use only the provided context. Cite chunks by [[chunk_id]]. If no answer, say so."

def build_generation_prompt(query: str, retrieved: List[Dict[str,Any]]) -> str:
    # assemble contexts with chunk ids
    ctxs = []
    for idx, r in enumerate(retrieved, start=1):
        meta = r.get("metadata", {})
        chunk_id = meta.get("chunk_id") or meta.get("id") or f"c{idx}"
        header = f"[[{chunk_id}]] (act={meta.get('act')}, scene={meta.get('scene')}, speaker={meta.get('speaker')})"
        ctxs.append(f"{header}\n{r['chunk']}\n")
    ctx_block = "\n\n".join(ctxs)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{ctx_block}\n\nUser question: {query}\n\nAnswer using ONLY the context above. Cite any lines you use with the corresponding [[chunk_id]]. If the answer is not present, say 'I cannot find supporting text in the provided context.'"
    return prompt

def generate_answer(query: str, retrieved: List[Dict[str,Any]], max_tokens: int = 512, temperature: float = 0.0) -> Tuple[str, Dict[str,Any]]:
    """
    Returns (answer_text, debug_info)
    Uses Gemini if configured; else fallback returns deterministic summary composed from contexts.
    """
    prompt = build_generation_prompt(query, retrieved)
    debug = {"used_model": None}

    # If Gemini client is configured, call it
    if _gemini_client:
        debug["used_model"] = f"gemini:{GEMINI_MODEL}"
        # Using Google generative ai chat completion
        resp = _gemini_client.generate_text(model=GEMINI_MODEL, prompt=prompt, max_output_tokens=max_tokens)
        # structure may vary depending on genai version; take main text
        text = ""
        try:
            # newer library returns resp.text
            text = resp.text if hasattr(resp, "text") else resp.output[0].content[0].text
        except Exception:
            text = str(resp)
        return text, debug

    # fallback deterministic generator: produce short synthetic answer from top retrieved chunks
    debug["used_model"] = "fallback_concat"
    if not retrieved:
        return ("I cannot find supporting text in the provided context.", debug)

    # naive summarization: pick top chunk(s) and return small synth
    top = retrieved[:3]
    joined = "\n\n".join([f"[[{r.get('metadata',{}).get('chunk_id','') }]] {r['chunk']}" for r in top])
    answer = f"Based on the provided context (chunks {', '.join([str(r.get('metadata',{}).get('chunk_id')) for r in top])}):\n\n{joined}"
    return answer, debug
