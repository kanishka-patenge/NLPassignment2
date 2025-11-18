# Julius Caesar RAG Retrieval System using Groq API

This repository contains a scene-aware Retrieval-Augmented Generation (RAG) system built over Shakespeare's **_Julius Caesar_** (Folger Edition PDF) as part of:

- **Course**: AID 849 – Advances in Natural Language Processing  
- **Institute**: International Institute of Information Technology Bangalore  
- **Instructors**: Dr. Tulika Saha  

The system answers exam-style questions (ICSE-level) about the play using **only** the original text of *Julius Caesar*, without any external knowledge or fine-tuning.

---

## 1. Project Structure

```text
NLP-assignment-2/
├── app/
│   ├── api.py              # FastAPI RAG endpoint
│   ├── groq.py             # CLI RAG tester using Groq LLM
│   ├── main.py             # Optional entrypoint
│   ├── models.py           # Pydantic models / schemas
│   ├── rag.py              # Base RAG logic (if used)
│   ├── rag_gemini.py       # (Optional) Gemini-based RAG
│   ├── rag_grog.py         # Groq-based retrieval + scene retriever
│   ├── rag_ollama.py       # (Optional) Local Ollama-based RAG
│   └── ...
├── data/
│   ├── julius-caesar.pdf   # Folger Edition PDF (source text)
│   ├── raw_pages.json      # Raw extracted pages
│   ├── chunked_data.json   # Dialogue-level chunks with metadata
│   ├── step3_structured_pages.json
│   ├── step4.json
│   ├── step5.json
│   └── step6.json
├── faiss_index/
│   ├── index.faiss         # FAISS vector index (Nomic embeddings)
│   └── index.pkl           # Metadata/docstore for FAISS
├── notebooks/
│   ├── etl.ipynb           # ETL + cleaning pipeline
│   └── phas2.ipynb         # Chunking + FAISS index creation
├── logs/                   # Intermediate logs for each ETL step
├── evaluation.json         # 35-question golden testbed (25 factual + 10 analytical)
├── evaluation_results.json # RAG evaluation outputs (relevancy + faithfulness)
├── requirements.txt        # Python dependencies
├── system_prompt.txt       # LLM system prompt used for RAG
├── report.pdf              # Final project report (RAG system)
└── README.md               # This file
