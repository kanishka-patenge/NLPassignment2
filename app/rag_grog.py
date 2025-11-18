from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
from collections import Counter

from dotenv import load_dotenv
from nomic import embed
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

load_dotenv()

FAISS_DIR = os.environ.get("FAISS_DIR", "faiss_index")
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------
# Embeddings wrapper compatible with your FAISS version
# ---------------------------------------------------------------------
class NomicEmbeddings(Embeddings):
    """
    Same style as your working rag_ollama.py, but ALSO callable,
    so FAISS can do embedding_function(text) without TypeError.
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

    # 🔑 This lets FAISS call embedding_function(text)
    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)


_embedding = NomicEmbeddings()
_faiss_store: Optional[FAISS] = None


def load_faiss_index() -> FAISS:
    """
    Load the FAISS index from disk once and cache it.
    Must match the index you built earlier in Phase 2.
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


# ---------------------------------------------------------------------
# Majority Act + Scene expansion
# ---------------------------------------------------------------------
def _expand_to_scene_with_topk(
    store: FAISS, base_docs: List[Any]
) -> Tuple[List[Any], Tuple[Any, Any]]:
    """
    Given top-K docs from similarity search:
      - Count (act, scene) pairs in base_docs
      - Choose the majority (act, scene)
      - Get ALL docs from that act+scene
      - Return: [base_docs first, then remaining scene docs], deduplicated
    """
    counts: Counter = Counter()

    for d in base_docs:
        meta = d.metadata or {}
        act = meta.get("act")
        scene = meta.get("scene")
        if act is not None and scene is not None:
            counts[(act, scene)] += 1

    # If no act/scene info, just return the base docs
    if not counts:
        return base_docs, (None, None)

    # Majority vote
    (best_act, best_scene), _ = max(counts.items(), key=lambda kv: kv[1])

    # Collect ALL chunks belonging to that act+scene
    full_scene_docs: List[Any] = []
    for d in store.docstore._dict.values():
        meta = d.metadata or {}
        if meta.get("act") == best_act and meta.get("scene") == best_scene:
            full_scene_docs.append(d)

    # Sort scene docs logically for reading (page, then chunk_id)
    full_scene_docs.sort(
        key=lambda d: (
            (d.metadata or {}).get("start_page", 0),
            (d.metadata or {}).get("chunk_id", 0),
        )
    )

    # ---- Combine: base_docs FIRST, then remaining scene docs (no duplicates) ----
    combined_docs: List[Any] = []
    seen_keys = set()

    def key_for(d_any: Any) -> Tuple[Any, Any]:
        m = (d_any.metadata or {})
        return (m.get("chunk_id"), m.get("start_page"))

    # 1) Add base_docs in their similarity order
    for d in base_docs:
        k = key_for(d)
        if k not in seen_keys:
            seen_keys.add(k)
            combined_docs.append(d)

    # 2) Add scene docs not already included
    for d in full_scene_docs:
        k = key_for(d)
        if k not in seen_keys:
            seen_keys.add(k)
            combined_docs.append(d)

    return combined_docs, (best_act, best_scene)


def retrieve_scene_context(query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    - Does FAISS similarity search with top_k
    - Uses majority (act, scene) to expand to full scene
    - Keeps original top_k docs in front of the context
    - Returns a list of {"chunk": text, "metadata": {...}}
    """
    top_k = top_k or DEFAULT_TOP_K
    store = load_faiss_index()

    # Step 1: initial semantic search
    base_docs = store.similarity_search(query, k=top_k)

    # Step 2: expand (majority act+scene + keep base top-k)
    combined_docs, (act, scene) = _expand_to_scene_with_topk(store, base_docs)

    print(f"\n📌 Selected Act {act}, Scene {scene} for context")
    print(f"🔎 Base top_k: {len(base_docs)} | Final combined docs: {len(combined_docs)}\n")

    # Step 3: convert to plain dicts
    context: List[Dict[str, Any]] = []
    for d in combined_docs:
        context.append(
            {
                "chunk": d.page_content,
                "metadata": d.metadata or {},
            }
        )

    return context


# ---------------------------------------------------------------------
# Simple CLI: test retrieval only
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("📚 Julius Caesar Retrieval System (Top-K + Scene-level context)")
    print("Ask a question (or type 'exit'):\n")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q or q.lower() in {"exit", "quit"}:
            break

        ctx = retrieve_scene_context(q, top_k=10)

        print("\n📄 Retrieved Context Chunks (showing first ~5):\n")
        for c in ctx[:5]:
            m = c["metadata"]
            cid = m.get("chunk_id", "N/A")
            act = m.get("act")
            scene = m.get("scene")
            page = m.get("start_page")
            preview = c["chunk"][:160].replace("\n", " ")
            print(f"  [chunk_id={cid}, act={act}, scene={scene}, page={page}]")
            print(f"    {preview}...\n")

        print("-" * 80)
