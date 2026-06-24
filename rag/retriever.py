import pickle

import faiss

from embeddings.embedding_backend import EmbeddingBackend


class Retriever:
    def __init__(self, index_dirs: list[str]):
        self.model = EmbeddingBackend()
        self.stores = []

        for d in index_dirs:
            index = faiss.read_index(f"{d}/index.faiss")
            with open(f"{d}/chunks.pkl", "rb") as f:
                metadata = pickle.load(f)
            self.stores.append({"index": index, "metadata": metadata, "path": d})

    def search(self, query: str, top_k: int = 5):
        if not self.stores:
            return []

        query_emb = self.model.encode([query])
        results = []

        for store in self.stores:
            scores, ids = store["index"].search(query_emb, top_k)
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(store["metadata"]):
                    continue
                raw_item = store["metadata"][idx]
                if isinstance(raw_item, dict):
                    item = dict(raw_item)
                else:
                    item = {
                        "chunk_id": int(idx),
                        "source": store["path"],
                        "text": str(raw_item),
                    }
                raw_score = float(score)
                if getattr(store["index"], "metric_type", None) == faiss.METRIC_L2:
                    relevance = 1 / (1 + max(raw_score, 0))
                else:
                    relevance = max(min(raw_score, 1.0), 0.0)
                item["score"] = round(relevance, 4)
                item["raw_score"] = round(raw_score, 4)
                item["index_dir"] = store["path"]
                results.append(item)

        results.sort(key=lambda item: item.get("score", 0), reverse=True)
        return results[:top_k]

