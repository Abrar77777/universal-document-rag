import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from rag.retriever import Retriever
from rag.web_search import web_search

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# -----------------------------
# In-memory chat history store
# -----------------------------
CHAT_HISTORY = {}
MAX_HISTORY = 6  # last 3 user-assistant pairs


class RAGPipeline:
    def __init__(self, index_dirs: list[str]):
        self.retriever = Retriever(index_dirs)

    def _update_history(self, session_id, role, content):
        CHAT_HISTORY.setdefault(session_id, [])
        CHAT_HISTORY[session_id].append({"role": role, "content": content})
        CHAT_HISTORY[session_id] = CHAT_HISTORY[session_id][-MAX_HISTORY:]

    def _get_history(self, session_id):
        return CHAT_HISTORY.get(session_id, [])

    def _confidence_score(self, retrieved_chunks, web_used):
        score = 0.4
        if len(retrieved_chunks) >= 3:
            score += 0.3
        if web_used:
            score += 0.2
        if len(retrieved_chunks) > 0:
            score += 0.1
        return round(min(score, 1.0), 2)

    def _fallback_answer(self, question: str, doc_context: str, analytics_context: str):
        if not doc_context and not analytics_context:
            return (
                "I do not have enough document context yet. Upload a PDF, DOCX, TXT, CSV, "
                "or Excel file, then ask again."
            )
        context = "\n".join([doc_context, analytics_context]).strip()
        return (
            "Based on the retrieved context, here are the most relevant details:\n\n"
            f"{context[:1400]}\n\n"
            "Set GROQ_API_KEY in your environment to enable a fully generated analytical answer."
        )

    def _build_analytics_context(self, analytics_profiles: list[dict[str, Any]] | None):
        if not analytics_profiles:
            return ""
        lines = []
        for profile in analytics_profiles:
            if profile.get("summary_text"):
                lines.append(f"Numeric profile for {profile.get('file', 'uploaded file')}:\n{profile['summary_text']}")
            for sheet in profile.get("sheets", []):
                for col, stats in sheet.get("numeric_summary", {}).items():
                    lines.append(
                        f"{sheet['name']}.{col}: min={stats.get('min')}, max={stats.get('max')}, "
                        f"mean={stats.get('mean')}, median={stats.get('median')}."
                    )
        return "\n".join(lines)

    def answer(
        self,
        question: str,
        session_id: str,
        use_web: bool,
        top_k: int = 5,
        analytics_profiles: list[dict[str, Any]] | None = None,
    ):
        retrieved = self.retriever.search(question, top_k=top_k)

        doc_context = "\n\n".join([r["text"] for r in retrieved])
        citations = [
            f'{r["source"]} (chunk {r["chunk_id"]}, score {r.get("score", 0)})'
            for r in retrieved
        ]

        web_context = web_search(question) if use_web else ""
        analytics_context = self._build_analytics_context(analytics_profiles)

        history = self._get_history(session_id)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent multi-document RAG analyst. Use retrieved document "
                    "context first, cite sources naturally, and be honest when context is weak. "
                    "When numeric profiles are available, explain trends, outliers, and business meaning."
                )
            },
            *history,
            {
                "role": "user",
                "content": f"""
DOCUMENT CONTEXT:
{doc_context}

NUMERIC DATA PROFILE:
{analytics_context}

WEB CONTEXT:
{web_context}

QUESTION:
{question}
"""
            }
        ]

        llm_error = None
        if os.getenv("GROQ_API_KEY"):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.25
                )
                answer = response.choices[0].message.content
            except Exception as exc:
                llm_error = str(exc)
                answer = self._fallback_answer(question, doc_context, analytics_context)
        else:
            answer = self._fallback_answer(question, doc_context, analytics_context)

        # update history
        self._update_history(session_id, "user", question)
        self._update_history(session_id, "assistant", answer)

        return {
            "answer": answer,
            "citations": list(dict.fromkeys(citations)),
            "retrieved_chunks": retrieved,
            "web_used": use_web,
            "confidence": self._confidence_score(retrieved, use_web),
            "llm_error": llm_error,
            "chart_specs": [
                chart
                for profile in (analytics_profiles or [])
                for chart in profile.get("chart_specs", [])
            ],
        }


