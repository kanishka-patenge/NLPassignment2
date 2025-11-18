"""
RAG pipeline using:
- FAISS (LangChain vector store) for retrieval
- Nomic embeddings for query encoding
- Gemini (Google Generative AI) for generation via API

Retrieval strategy (scene-aware):
1. Use FAISS to retrieve top-k most similar chunks.
2. From those, find the majority (act, scene) pair.
3. Load ALL chunks from that (act, scene) from the FAISS store
   to build the final context for generation.

Public functions:
- retrieve(query, top_k)
- generate_answer(query, retrieved)
- run_query(query, top_k)
"""

from pathlib import Path
import os
from typing import List, Dict, Any, Tuple, Optional

from dotenv import load_dotenv
from nomic import embed

from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document

import google.generativeai as genai

# ----------------- ENV & GLOBAL CONFIG -----------------

load_dotenv()

FAISS_DIR = os.environ.get("FAISS_DIR", "faiss_index")
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "15"))

SCENE_RETRIEVAL = os.environ.get("SCENE_RETRIEVAL", "true").lower() == "true"
SCENE_CONTEXT_MAX_CHUNKS = int(os.environ.get("SCENE_CONTEXT_MAX_CHUNKS", "80"))
RERANK = os.environ.get("RERANK", "false").lower() == "true"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Use a very widely supported model name
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-pro")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in environment or .env file.")

genai.configure(api_key=GEMINI_API_KEY)
_gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.joinpath("system_prompt.txt")
if SYSTEM_PROMPT_PATH.exists():
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
else:
    SYSTEM_PROMPT = (
        "You are an expert Shakespearean tutor helping a Class 10 ICSE student understand "
        'the play "Julius Caesar". Use only the provided context chunks. '
        "Cite chunks with their [[chunk_id]] labels. "
        "If the answer is not clearly supported by the context, say so explicitly. "
        "Write answers in a clear, exam-style manner."
    )

# ----------------- EMBEDDINGS & FAISS -----------------


class NomicEmbeddings(Embeddings):
    """
    Adapter for Nomic embeddings to be used with LangChain FAISS.
    """

    def __init__(self, model: str = "nomic-embed-text-v1.5", batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = embed.text(texts=batch, model=self.model)
            out.extend(resp["embeddings"])
        return out

    def embed_query(self, text: str) -> List[float]:
        resp = embed.text(texts=[text], model=self.model)
        return resp["embeddings"][0]


_embedding = NomicEmbeddings()
_faiss_store: Optional[FAISS] = None


def load_faiss_index() -> FAISS:
    """
    Load the FAISS index from disk once and cache it.
    """
    global _faiss_store
    if _faiss_store is None:
        idx_path = Path(FAISS_DIR)
        if not idx_path.exists():
            raise FileNotFoundError(f"FAISS directory not found: {FAISS_DIR}")
        _faiss_store = FAISS.load_local(
            str(idx_path),
            _embedding,
            allow_dangerous_deserialization=True,
        )
    return _faiss_store


# ----------------- RETRIEVAL HELPERS -----------------


def simple_rerank(query: str, candidates: List[Dict[str, Any]], top_m: int = 3) -> List[Dict[str, Any]]:
    """
    Cheap reranker based on lexical overlap.
    Returns top_m candidates.
    """
    q_tokens = set(query.lower().split())
    scored: List[Tuple[int, Dict[str, Any]]] = []

    for c in candidates:
        txt = c["chunk"].lower()
        score = sum(1 for t in q_tokens if t in txt)
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = [c for s, c in scored[:top_m]]
    return out


def _get_majority_scene(docs: List[Document]) -> Optional[Tuple[int, int]]:
    """
    From a list of retrieved Document objects, find the (act, scene) pair
    that appears most frequently. Tie-break on earliest rank.
    """
    scene_counts: Dict[Tuple[int, int], Dict[str, int]] = {}
    for rank, d in enumerate(docs):
        meta = d.metadata or {}
        act = meta.get("act")
        scene = meta.get("scene")
        if act is None or scene is None:
            continue
        key = (int(act), int(scene))
        if key not in scene_counts:
            scene_counts[key] = {"count": 0, "best_rank": rank}
        scene_counts[key]["count"] += 1
        scene_counts[key]["best_rank"] = min(scene_counts[key]["best_rank"], rank)

    if not scene_counts:
        return None

    majority_key = sorted(
        scene_counts.items(),
        key=lambda kv: (-kv[1]["count"], kv[1]["best_rank"]),
    )[0][0]
    return majority_key


def _collect_scene_docs(store: FAISS, act: int, scene: int) -> List[Document]:
    """
    Collect all Document objects from FAISS docstore with the given (act, scene),
    sorted by (start_page, chunk_id).
    """
    all_docs: List[Document] = list(store.docstore._dict.values())  # type: ignore[attr-defined]
    filtered: List[Document] = []
    for d in all_docs:
        meta = d.metadata or {}
        if meta.get("act") == act and meta.get("scene") == scene:
            filtered.append(d)

    filtered.sort(
        key=lambda d: (
            d.metadata.get("start_page") or 0,
            d.metadata.get("chunk_id") or 0,
        )
    )

    if len(filtered) > SCENE_CONTEXT_MAX_CHUNKS:
        filtered = filtered[:SCENE_CONTEXT_MAX_CHUNKS]

    return filtered


def _docs_to_results(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Convert a list of LangChain Document objects into the RAG result dict format.
    """
    results: List[Dict[str, Any]] = []
    for d in docs:
        results.append(
            {
                "chunk": d.page_content,
                "metadata": d.metadata or {},
                "score": None,
            }
        )
    return results


# ----------------- MAIN RETRIEVAL -----------------


def retrieve(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Scene-aware retrieval:

    1) Use FAISS similarity_search to get top_k chunks.
    2) If SCENE_RETRIEVAL is True:
         - Determine majority (act, scene) among those chunks.
         - Load all chunks from that scene as final context.
       Else:
         - Return only top_k chunks (optionally reranked).
    """
    top_k = top_k or DEFAULT_TOP_K
    store = load_faiss_index()

    docs: List[Document] = store.similarity_search(query, k=top_k)
    if not docs:
        return []

    if not SCENE_RETRIEVAL:
        base_results = _docs_to_results(docs)
        if RERANK:
            base_results = simple_rerank(query, base_results, top_m=min(len(base_results), top_k))
        return base_results

    majority_scene = _get_majority_scene(docs)
    if majority_scene is None:
        base_results = _docs_to_results(docs)
        if RERANK:
            base_results = simple_rerank(query, base_results, top_m=min(len(base_results), top_k))
        return base_results

    act, scene = majority_scene
    scene_docs = _collect_scene_docs(store, act, scene)
    scene_results = _docs_to_results(scene_docs)
    return scene_results


# ----------------- PROMPT BUILDING -----------------


def build_generation_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
    """
    Build the prompt text that will be passed (together with SYSTEM_PROMPT) to Gemini.
    """
    ctxs: List[str] = []
    for idx, r in enumerate(retrieved, start=1):
        meta = r.get("metadata", {}) or {}
        chunk_id = meta.get("chunk_id") or meta.get("id") or f"c{idx}"
        header = (
            f"[[{chunk_id}]] "
            f"(act={meta.get('act')}, scene={meta.get('scene')}, speaker={meta.get('speaker')})"
        )
        ctxs.append(f"{header}\n{r['chunk']}\n")

    ctx_block = "\n\n".join(ctxs)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Context:\n"
        f"{ctx_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using ONLY the context above. "
        "Give a short, exam-style answer in 2–4 sentences. "
        "Cite any lines you use with the corresponding [[chunk_id]]. "
        "If the answer is not clearly supported, say so explicitly."
    )
    return prompt


# ----------------- GEMINI CALL -----------------


def call_gemini(full_prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Call Gemini with the full text prompt using GenerativeModel.
    """
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }

    resp = _gemini_model.generate_content(
        full_prompt,
        generation_config=generation_config,
    )

    return resp.text


# ----------------- GENERATION WRAPPER -----------------


def generate_answer(
    query: str,
    retrieved: List[Dict[str, Any]],
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Tuple[str, Dict[str, Any]]:
    """
    RAG generation step using Gemini.
    Returns (answer_text, debug_info).
    """
    prompt = build_generation_prompt(query, retrieved)
    debug: Dict[str, Any] = {
        "used_model": f"gemini:{GEMINI_MODEL_NAME}",
        "num_context_chunks": len(retrieved),
    }

    try:
        answer = call_gemini(prompt, max_tokens=max_tokens, temperature=temperature)
        debug["generation_success"] = True
        return answer, debug
    except Exception as e:
        debug["generation_success"] = False
        debug["generation_error"] = repr(e)

    # Fallback: show chunks if Gemini fails
    if not retrieved:
        return "I cannot find supporting text in the provided context.", debug

    top = retrieved[:3]
    joined = "\n\n".join(
        [
            f"[[{r.get('metadata', {}).get('chunk_id', '')}]] {r['chunk']}"
            for r in top
        ]
    )
    answer = (
        "I could not generate an LLM-based answer, "
        "but here are the most relevant chunks from the text:\n\n"
        f"{joined}"
    )
    return answer, debug


# ----------------- ONE-SHOT PIPELINE -----------------


def run_query(query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
    """
    One-shot RAG pipeline:

    1. Scene-aware retrieval.
    2. Generation via Gemini.
    3. Return:
       {
         "answer": <str>,
         "sources": [{"chunk": <text>, "metadata": {...}}, ...],
         "debug": {...}
       }
    """
    retrieved = retrieve(query, top_k=top_k)
    answer, debug = generate_answer(query, retrieved)

    sources = [
        {
            "chunk": r["chunk"],
            "metadata": r.get("metadata", {}),
        }
        for r in retrieved
    ]

    return {
        "answer": answer,
        "sources": sources,
        "debug": debug,
    }


# ----------------- CLI TEST -----------------

if __name__ == "__main__":
    print("RAG + Gemini CLI. Type a question (or 'exit'):")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        resp = run_query(q, top_k=DEFAULT_TOP_K)
        print("\nANSWER:\n", resp["answer"])
        print("\nDEBUG:\n", resp.get("debug"))
        print("\nUSED CHUNKS:")
        for s in resp["sources"]:
            cid = s["metadata"].get("chunk_id", "N/A")
            print("-", cid)
        print("\n---\n")
