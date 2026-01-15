import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

from utils.pdf_loader import load_pdf
from utils.text_cleaner import clean_text
from utils.text_splitter import split_text


DATA_PATH = "data/raw/company_policy.pdf"
INDEX_PATH = "embeddings/faiss_index"


def build_faiss_index():
    # 1. Load and prepare text
    raw_text = load_pdf(DATA_PATH)
    cleaned_text = clean_text(raw_text)
    chunks = split_text(cleaned_text)

    print(f"Total chunks: {len(chunks)}")

    # 2. Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Create embeddings
    embeddings = model.encode(chunks, show_progress_bar=True)

    # 4. Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 5. Save index + chunks
    os.makedirs(INDEX_PATH, exist_ok=True)

    faiss.write_index(index, f"{INDEX_PATH}/index.faiss")

    with open(f"{INDEX_PATH}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("FAISS index built and saved successfully")


if __name__ == "__main__":
    build_faiss_index()
