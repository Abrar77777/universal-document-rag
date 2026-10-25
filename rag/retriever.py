import pickle

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from embeddings.embedding_backend import EmbeddingBackend


class Retriever:
    """FAISS + corpus TF-IDF, combined with reciprocal rank fusion (RRF)."""

    def __init__(self, index_dirs: list[str]):
        self.stores = []
        self.items = []
        self.models = {}
        for directory in dict.fromkeys(index_dirs):
            index = faiss.read_index(f"{directory}/index.faiss")
            with open(f"{directory}/chunks.pkl", "rb") as handle:
                metadata = pickle.load(handle)
            if not metadata or not isinstance(metadata[0], dict):
                raise ValueError("Legacy index has no embedding metadata. Re-upload the source file.")
            identity = (metadata[0].get("embedding_backend", "hashing"),
                        metadata[0].get("embedding_model", "all-MiniLM-L6-v2"))
            if identity not in self.models:
                self.models[identity] = EmbeddingBackend(*identity)
            offset = len(self.items)
            self.items.extend(dict(item, index_dir=directory) for item in metadata)
            self.stores.append((index, offset, len(metadata), identity))

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self.lexical_matrix = None
        if self.items:
            try:
                self.lexical_matrix = self.vectorizer.fit_transform([item["text"] for item in self.items])
            except ValueError:
                pass  # Empty vocabulary, e.g. punctuation-only documents.

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid"):
        if mode not in {"hybrid", "vector"}:
            raise ValueError("Retrieval mode must be hybrid or vector.")
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
        if not self.items or not query.strip():
            return []
        pool_size = min(max(top_k * 4, 20), len(self.items))
        vector_scores = {}
        encoded = {identity: model.encode([query]) for identity, model in self.models.items()}
        for index, offset, count, identity in self.stores:
            query_vector = encoded[identity]
            if index.d != query_vector.shape[1]:
                raise ValueError("Embedding dimensions changed. Re-upload the source file.")
            scores, ids = index.search(query_vector, min(pool_size, count))
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0:
                    continue
                relevance = 1 / (1 + max(float(score), 0)) if index.metric_type == faiss.METRIC_L2 else float(score)
                if relevance > 0:
                    vector_scores[offset + int(idx)] = relevance

        lexical_scores = {}
        if self.lexical_matrix is not None:
            query_vector = self.vectorizer.transform([query])
            scores = (self.lexical_matrix @ query_vector.T).toarray().ravel()
            lexical_scores = {int(idx): float(scores[idx]) for idx in np.argsort(-scores)[:pool_size] if scores[idx] > 0}

        # Hash collisions alone are not evidence of a lexical match.
        if all(identity[0] == "hashing" for identity in self.models):
            vector_scores = {idx: score for idx, score in vector_scores.items() if idx in lexical_scores}
        vector_ranks = {idx: rank for rank, idx in enumerate(sorted(vector_scores, key=vector_scores.get, reverse=True), 1)}
        lexical_ranks = {idx: rank for rank, idx in enumerate(lexical_scores, 1)}
        candidates = set(vector_ranks) | set(lexical_ranks) if mode == "hybrid" else set(vector_ranks)
        results = []
        for idx in candidates:
            fused = sum(1 / (60 + ranks[idx]) for ranks in (vector_ranks, lexical_ranks) if idx in ranks)
            item = dict(self.items[idx])
            item.update(score=round(fused if mode == "hybrid" else vector_scores[idx], 6),
                        vector_score=round(vector_scores.get(idx, 0), 4),
                        lexical_score=round(lexical_scores.get(idx, 0), 4),
                        vector_rank=vector_ranks.get(idx), lexical_rank=lexical_ranks.get(idx),
                        retrieval_method=mode)
            results.append(item)
        results.sort(key=lambda item: (-item["score"], item["source"], item["chunk_id"]))
        unique, seen = [], set()
        for item in results:
            identity = (item["index_dir"], item["text"])
            if identity not in seen:
                seen.add(identity)
                unique.append(item)
            if len(unique) == top_k:
                break
        return unique

