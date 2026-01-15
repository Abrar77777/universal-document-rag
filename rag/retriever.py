import pickle
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_dirs: list[str]):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.indices = []
        self.metadata = []

        for d in index_dirs:
            self.indices.append(faiss.read_index(f"{d}/index.faiss"))
            with open(f"{d}/chunks.pkl", "rb") as f:
                self.metadata.extend(pickle.load(f))

    def search(self, query: str, top_k: int = 3):
        query_emb = self.model.encode([query])

        results = []
        for index in self.indices:
            distances, ids = index.search(query_emb, top_k)
            for idx in ids[0]:
                results.append(self.metadata[idx])

        return results[:top_k]

