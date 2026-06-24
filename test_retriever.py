from rag.retriever import Retriever

retriever = Retriever(["embeddings/faiss_index"])

query = "What this data analyst job role says?"
results = retriever.search(query, top_k=3)

print("\n--- Retrieved Chunks ---\n")
for i, chunk in enumerate(results, 1):
    print(f"\nChunk {i} ({chunk.get('source')}):\n{chunk.get('text', '')[:400]}")
print("\n--- End Retrieved Chunks ---\n")
