from rag.rag_pipeline import RAGPipeline

rag = RAGPipeline()

question = "What this document says?"
answer = rag.answer(question)

print("\n--- ANSWER ---\n")
print(answer)
print("\n--- END ANSWER ---\n")