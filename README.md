# Universal Document Intelligence System
Recruiter overview and codebase review: [APP_HIGHLIGHTS.md](APP_HIGHLIGHTS.md).

The primary app is now a four-tab workspace: **Ask & evidence**, **Data explorer**, **Feedback intelligence**, and **Retrieval lab**. It starts empty and uses only your uploads. Charts support selectable axes/aggregations, feedback analysis uses a chosen response column, and answers persist across control changes.

To watch this in action visit--->https://universal-document-rag-ur2fwvaw7zguvmtysadtvq.streamlit.app/

An end-to-end AI system for user-uploaded documents, source-backed questions, and interactive data analysis. It supports hybrid retrieval, Gemini/Groq answer generation, source-labeled passages, scoped conversation history, optional web evidence, and numeric analysis for CSV/Excel files. Evidence counts and timings are shown instead of a purported confidence probability.

This is built as a portfolio-ready project with a FastAPI backend, Streamlit frontend, FAISS retrieval, evaluation metrics, a Windows demo launcher, and a GitHub Pages presentation site.

## Features

- Upload and index PDF, DOCX, TXT, Markdown, CSV, Excel, and JSON files.
- Smart chunking with overlap and sentence-aware boundaries.
- Multi-document FAISS + TF-IDF retrieval with reciprocal rank fusion, top-k control, and inspectable ranks.
- Conversational RAG API using Groq-compatible OpenAI chat completions.
- Clearly labeled extractive evidence when generation is disabled or unavailable.
- Optional internet search fallback using DuckDuckGo.
- Numeric profiling for CSV and Excel: rows, columns, missing values, summary statistics, and chart suggestions.
- Customer feedback analytics: sentiment distribution, keyword extraction, themes, emotion signals, pain points, and praise.
- Exploratory data analysis for spreadsheets: missing values, category distributions, correlations, outliers, histograms, bar charts, line charts, and scatter plots.
- Streamlit workspace for persistent answers, evidence, interactive charts, feedback filtering, and exports.
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
Gemini / Groq prompt with source-labeled context
        |
Answer + evidence ledger + timings
```

For CSV and Excel files, the tabular analyzer also creates numeric summaries and chart-ready data for the frontend.

## Tech Stack

| Layer | Technology |
| --- | --- |
| LLM | Configurable Gemini, then Groq when configured |
| Embeddings | Local hashing backend by default, optional SentenceTransformers |
| Retrieval | FAISS + scikit-learn TF-IDF + reciprocal rank fusion |
| Feedback | VADER sentiment, row-level results, keyword themes |
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
pip install -r requirements.txt
```

Add your Groq key to `.env`:

```text
GROQ_API_KEY=your_key_here
```

Without a key, or if the key is invalid, the app still retrieves relevant chunks and returns an extractive fallback answer.

Embeddings use a local hashing backend by default so the demo works without downloading a model. To use Sentence Transformers instead:

```powershell
$env:RAG_EMBEDDING_BACKEND="sentence-transformers"
$env:RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
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
streamlit run streamlit_app.py
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
python -m evaluation.evaluate_rag --dataset evaluation/sample_eval_dataset.json --index-dir data/indices/<index_id> --top-k 5 --retrieval-only
```

The output includes precision@k, source recall@k, and MRR. Remove `--retrieval-only` to call configured providers and compute lexical overlap for generated answers. This grounding proxy does not establish factual correctness. Set expected source names to your real files before evaluation.

## Advanced RAG Techniques Included

- Top-k vector retrieval (hashing by default; optional semantic embeddings).
- TF-IDF keyword retrieval fused with vector ranks.
- Multi-document retrieval with per-chunk metadata and citations.
- Inspectable ranking signals and observed latency, not calibrated confidence.
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
GEMINI_API_KEY = "your_gemini_key_here"
GEMINI_MODEL = "gemini-3.5-flash-lite"
RAG_ALLOW_GROQ_FALLBACK = false
```

The cloud app and Windows launcher use `streamlit_app.py`, which runs the RAG pipeline directly inside Streamlit. The launcher also starts FastAPI for API inspection. `ui/app.py` remains a legacy API-client example.

After saving secrets, reboot the Streamlit app. If you explicitly set `GEMINI_MODEL`, make sure it matches the current supported model shown above.

The default is Gemini 3.5 Flash-Lite, a low-latency model with a free API tier as of 2026-10-25. Only the approved Flash-Lite models are selectable; other configured names are replaced with the default. This is a model-selection policy, **not a billing guarantee**. Keep the API key's project on the Free tier with no paid API billing. Google AI Pro subscriptions do not cover direct API usage. Free-tier quotas and data-use terms still apply. See [Google's pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite) and [subscription/API distinction](https://ai.google.dev/gemini-api/docs/google-ai-plans).

Groq fallback is disabled by default. Only enable it after reviewing that provider's account billing and limits:

```toml
GROQ_API_KEY = "your_groq_key_here"
GROQ_MODEL = "llama-3.1-8b-instant"
RAG_ALLOW_GROQ_FALLBACK = true
```

## Portfolio Talking Points

- Explain why chunk size and overlap matter for retrieval recall.
- Show how citations reduce hallucination risk.
- Demonstrate CSV/Excel upload and visual numeric profiling.
- Run evaluation metrics before and after changing top-k or chunk size.
- Discuss remaining upgrades: BM25/cross-encoder comparisons, query rewriting, authentication, async ingestion, retention cleanup, and durable storage.

## Automated Checks

```powershell
python -m unittest discover -s tests -v
```

Tests use isolated local fixtures and mocked providers. Live cloud credentials and quotas require separate deployment checks. See [APP_HIGHLIGHTS.md](APP_HIGHLIGHTS.md) for the architecture review, privacy considerations, known limitations, and recruiter walkthrough.
