import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

from utils.file_loader import load_file
from utils.text_cleaner import clean_text
from utils.text_splitter import split_text


def build_index(file_path: str, index_name: str):
    raw_text = load_file(file_path)
    cleaned = clean_text(raw_text)
    chunks = split_text(cleaned)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    index_dir = f"data/indices/{index_name}"
    os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(index, f"{index_dir}/index.faiss")

    metadata = [
        {"chunk_id": i, "source": os.path.basename(file_path), "text": chunk}
        for i, chunk in enumerate(chunks)
    ]

    with open(f"{index_dir}/chunks.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return index_dir

