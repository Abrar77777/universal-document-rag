from __future__ import annotations

import os

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class EmbeddingBackend:
    def __init__(self, backend: str | None = None):
        self.backend = backend or os.getenv("RAG_EMBEDDING_BACKEND", "hashing")
        self.model_name = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._sentence_model = None
        self._hashing_model = HashingVectorizer(
            n_features=384,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            stop_words="english",
        )

        if self.backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._sentence_model = SentenceTransformer(self.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._sentence_model is not None:
            embeddings = self._sentence_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return np.asarray(embeddings, dtype="float32")

        matrix = self._hashing_model.transform(texts)
        matrix = normalize(matrix, norm="l2", copy=False)
        return matrix.astype("float32").toarray()
