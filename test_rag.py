from rag.rag_pipeline import RAGPipeline

rag = RAGPipeline(["embeddings/faiss_index"])

question = "What this document says?"
answer = rag.answer(question=question, session_id="smoke-test", use_web=False, top_k=3)

print("\n--- ANSWER ---\n")
print(answer["answer"])
print("\n--- CITATIONS ---\n")
print(answer["citations"])
print("\n--- END ANSWER ---\n")
