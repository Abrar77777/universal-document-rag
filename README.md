# Universal Document Intelligence System
To watch this in action visit--->https://universal-document-rag-ur2fwvaw7zguvmtysadtvq.streamlit.app/

An end-to-end AI system that lets users upload documents and ask natural-language questions with answers grounded in document content. The app supports multi-document RAG, Groq-powered answer generation, source citations, confidence scoring, chat memory, optional web search, and numeric analysis for CSV/Excel files.

This is built as a portfolio-ready project with a FastAPI backend, Streamlit frontend, FAISS retrieval, evaluation metrics, a Windows demo launcher, and a GitHub Pages presentation site.

## Features

- Upload and index PDF, DOCX, TXT, Markdown, CSV, Excel, and JSON files.
- Smart chunking with overlap and sentence-aware boundaries.
- Multi-document FAISS retrieval with top-k control and similarity scores.
- Conversational RAG API using Groq-compatible OpenAI chat completions.
- Local fallback answer mode when `GROQ_API_KEY` is not configured.
- Optional internet search fallback using DuckDuckGo.
- Numeric profiling for CSV and Excel: rows, columns, missing values, summary statistics, and chart suggestions.
- Streamlit frontend for upload, question answering, citations, confidence, and charts.
- Evaluation script for precision@k, recall@k, MRR, and grounding proxy.
- No paid vector database or proprietary storage required.

## Architecture

```text
User uploads files
        |
Universal file loader
        |
Clean and chunk text
        |
Embedding backend
        |
FAISS vector index
        |
Top-k retrieval
        |
Groq LLM prompt with retrieved context
        |
Answer + citations + confidence
```

For CSV and Excel files, the tabular analyzer also creates numeric summaries and chart-ready data for the frontend.

## Tech Stack

| Layer | Technology |
| --- | --- |
| LLM | Groq, `llama-3.1-8b-instant` |
| Embeddings | Local hashing backend by default, optional SentenceTransformers |
| Vector DB | FAISS |
| Backend | FastAPI |
| Frontend | Streamlit |
| Web Search | DuckDuckGo Search |
| Evaluation | Custom retrieval and grounding metrics |

## Project Structure

```text
api/              FastAPI app and request schemas
docs/             GitHub Pages static presentation site
embeddings/       Index-building and embedding backend
evaluation/       RAG metrics runner and sample dataset
rag/              Retriever, RAG pipeline, optional web search
scripts/          Demo process supervisor
ui/               Streamlit frontend
utils/            Loaders, cleaners, chunking, tabular analysis
data/             Raw files, uploads, and generated indices
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt.txt
```

Add your Groq key to `.env`:

```text
GROQ_API_KEY=your_key_here
```

Without a key, or if the key is invalid, the app still retrieves relevant chunks and returns an extractive fallback answer.

Embeddings use a local hashing backend by default so the demo works without downloading a model. To use Sentence Transformers instead:

```powershell
set RAG_EMBEDDING_BACKEND=sentence-transformers
set RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Run

Recommended Windows demo command:

```powershell
.\run_app.ps1
```

If PowerShell blocks scripts:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_app.ps1
```

Or double-click:

```text
run_app.bat
```

Then open:

```text
http://127.0.0.1:8501
```

Manual run:

```powershell
uvicorn api.main:app --reload
streamlit run ui/app.py
```

## API

- `POST /upload?session_id=<id>` uploads and indexes one file.
- `POST /ask` asks a question using uploaded files for that session.
- `GET /analytics/{session_id}` returns tabular profiles.
- `GET /health` checks backend status.

Example request:

```json
{
  "question": "Summarize the main risks and cite the source chunks.",
  "session_id": "demo",
  "use_web": false,
  "top_k": 5
}
```

## Evaluation

Create a JSON dataset like:

```json
[
  {
    "question": "What is the document mainly about?",
    "expected_sources": ["company_policy.pdf"]
  }
]
```

Run:

```powershell
python evaluation/evaluate_rag.py --dataset evaluation/sample_eval_dataset.json --index-dir data/indices/<index_id> --top-k 5
```

The output includes precision@k, recall@k, MRR, and a grounding proxy that estimates how much of the answer is supported by retrieved text.

## Advanced RAG Techniques Included

- Top-k dense retrieval over document chunks.
- Multi-document retrieval with per-chunk metadata and citations.
- Similarity-based confidence scoring.
- Conversation memory per session.
- Numeric-aware prompting for tabular files.
- Optional web-search fallback.
- Evaluation metrics for retrieval and answer grounding.

## GitHub Pages

GitHub Pages hosts the static project presentation in `docs/index.html`.

Important: GitHub Pages cannot run FastAPI or Streamlit. Use Pages as the portfolio landing page, and run the actual app locally with `run_app.ps1` or deploy the backend/frontend later to Render, Railway, Hugging Face Spaces, or Streamlit Community Cloud.

## Free Cloud Deployment

The easiest free deployment is Streamlit Community Cloud.

Use these settings:

```text
Repository: Abrar77777/universal-document-rag
Branch: main
Main file path: streamlit_app.py
```

Add this secret in Streamlit Cloud:

```toml
GROQ_API_KEY = "your_groq_key_here"
```

The cloud app uses `streamlit_app.py`, which runs the RAG pipeline directly inside Streamlit. The local demo still supports the FastAPI + Streamlit split through `run_app.ps1`.

## Portfolio Talking Points

- Explain why chunk size and overlap matter for retrieval recall.
- Show how citations reduce hallucination risk.
- Demonstrate CSV/Excel upload and visual numeric profiling.
- Run evaluation metrics before and after changing top-k or chunk size.
- Discuss future upgrades: hybrid BM25+dense retrieval, reranking, query rewriting, user auth, async indexing, and a managed vector database.
