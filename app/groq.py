import os
import time
from dotenv import load_dotenv

from app.rag_grog import retrieve_scene_context
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# Use a currently supported Groq model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
)

# System Prompt — context-strict academic answer
prompt = ChatPromptTemplate.from_template("""
Answer strictly using the context from Shakespeare’s *Julius Caesar*.
If the answer is not found in context, say:
"Context insufficient to answer this question."

<context>
{context}
</context>

Question: {question}

Answer:
""")


def build_limited_context(docs, max_tokens: int) -> str:
    """
    Build a context string from docs, but stop once we hit ~max_tokens
    (approximate tokens using whitespace split).
    """
    parts = []
    token_count = 0

    for d in docs:
        # Clean and tokenize
        text = d["chunk"].replace("\n", " ")
        tokens = text.split()
        n = len(tokens)

        # If adding this chunk would exceed budget, add partial then stop
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


def rag_answer(question: str, k: int = 10):
    # Retrieve scene context from your FAISS model
    docs = retrieve_scene_context(question, top_k=k)

    # Groq free on_demand tier: 6000 TPM.
    # Reserve budget for system prompt + question + output.
    # So keep context around ~2500 tokens to stay safely under the limit.
    MAX_CONTEXT_TOKENS = 2500
    context_text = build_limited_context(docs, MAX_CONTEXT_TOKENS)

    # Build LangChain chat messages
    final_messages = prompt.format_messages(
        context=context_text,
        question=question,
    )

    # Run LLM
    start = time.time()
    response = llm.invoke(final_messages)
    end = time.time()

    print("\n📌 RAG Answer:\n")
    print(response.content)
    print(f"\n⏱ Response time: {end - start:.3f}s")
'''
    print("\n📚 Retrieved Context (original docs, even if LLM saw truncated version):")
    for i, d in enumerate(docs):
        act = d["metadata"].get("act")
        scene = d["metadata"].get("scene")
        print(f"\n--- Chunk {i+1} (Act {act}, Scene {scene}) ---")
        print(d["chunk"][:300].replace("\n", " ") + "...")
'''

if __name__ == "__main__":
    print("📖 Julius Caesar RAG System — Groq (CLI)")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("\nAsk a question: ").strip()
        if question.lower() == "exit":
            break
        rag_answer(question, k=10)
