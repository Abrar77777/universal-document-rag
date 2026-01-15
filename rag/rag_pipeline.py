import os
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

    def answer(self, question: str, session_id: str, use_web: bool):
        retrieved = self.retriever.search(question, top_k=5)

        doc_context = "\n\n".join([r["text"] for r in retrieved])
        citations = [
            f'{r["source"]} (chunk {r["chunk_id"]})'
            for r in retrieved
        ]

        web_context = web_search(question) if use_web else ""

        history = self._get_history(session_id)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an airline recovery analyst. "
                    "Use document context first. "
                    "Be concise, analytical, and factual."
                )
            },
            *history,
            {
                "role": "user",
                "content": f"""
DOCUMENT CONTEXT:
{doc_context}

WEB CONTEXT:
{web_context}

QUESTION:
{question}
"""
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3
        )

        answer = response.choices[0].message.content

        # update history
        self._update_history(session_id, "user", question)
        self._update_history(session_id, "assistant", answer)

        return {
            "answer": answer,
            "citations": list(set(citations)),
            "web_used": use_web,
            "confidence": self._confidence_score(retrieved, use_web)
        }


