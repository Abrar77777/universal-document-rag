📚 Universal Document Intelligence System (RAG + Groq)

An end-to-end AI system that allows users to upload any document (PDF, DOCX, CSV, TXT) and ask natural language questions, with answers grounded in document content, optionally enhanced by live web search, and generated using Groq’s LLaMA-3 model.

This project demonstrates production-grade Retrieval Augmented Generation (RAG) with fine-tuning, multi-document reasoning, chat memory, source citations, and confidence scoring.

🚀 Key Features

✅ Upload multiple documents (PDF / DOCX / CSV / TXT)
✅ Semantic search using FAISS vector database
✅ RAG pipeline with document-first answering
✅ Groq API (llama-3.1-8b-instant) for fast inference
✅ Optional internet search (ON / OFF toggle)
✅ Chat history per session (memory-efficient, capped)
✅ Source citations (file name + chunk ID)
✅ Confidence score per answer
✅ FastAPI backend + Streamlit UI
✅ No paid vector DBs, no proprietary storage

🧠 Architecture Overview
User Uploads Documents
        ↓
Text Extraction & Chunking
        ↓
FAISS Vector Index (per file)
        ↓
User Question
        ↓
Semantic Retrieval (Multi-Document)
        ↓
(Optional) Web Search
        ↓
Groq LLaMA-3 (Answer Generation)
        ↓
Answer + Citations + Confidence

🧰 Tech Stack
Layer	Technology
LLM	Groq – llama-3.1-8b-instant
Embeddings	SentenceTransformers
Vector DB	FAISS
Backend	FastAPI
Frontend	Streamlit
Web Search	DuckDuckGo (no API key)
Secrets	.env + python-dotenv
