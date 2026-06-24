import os
import pickle

import faiss

from embeddings.embedding_backend import EmbeddingBackend
from utils.file_loader import load_file
from utils.tabular_analyzer import analyze_tabular_file
from utils.text_cleaner import clean_text
from utils.text_splitter import split_text


def build_index(file_path: str, index_name: str, source_name: str | None = None):
    raw_text = load_file(file_path)
    cleaned = clean_text(raw_text)
    chunks = split_text(cleaned)
    if not chunks:
        raise ValueError("No readable text found in the uploaded file.")

    model = EmbeddingBackend()
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_dir = f"data/indices/{index_name}"
    os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(index, f"{index_dir}/index.faiss")

    display_source = source_name or os.path.basename(file_path)
    metadata = [
        {
            "chunk_id": i,
            "source": display_source,
            "text": chunk,
            "char_length": len(chunk),
            "embedding_backend": model.backend,
        }
        for i, chunk in enumerate(chunks)
    ]

    with open(f"{index_dir}/chunks.pkl", "wb") as f:
        pickle.dump(metadata, f)

    analytics = analyze_tabular_file(file_path)
    if analytics:
        analytics["file"] = display_source
        with open(f"{index_dir}/analytics.pkl", "wb") as f:
            pickle.dump(analytics, f)

    return index_dir

