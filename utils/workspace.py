"""Ingestion and exports shared by the Streamlit workspace."""
import hashlib
import pickle
import uuid
import shutil
from pathlib import Path

from embeddings.build_index_from_file import build_index
from utils.file_loader import load_file
from utils.tabular_analyzer import _read_tables

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".xls", ".json"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def discard_documents(documents):
    for document in documents:
        path = Path(document["path"]).resolve()
        index = Path(document["index_dir"]).resolve()
        if path.is_relative_to((ROOT / "data" / "uploads").resolve()):
            path.unlink(missing_ok=True)
        if index.parent == (ROOT / "data" / "indices").resolve() and index.exists():
            shutil.rmtree(index)


def fingerprint(name: str, content: bytes) -> str:
    return hashlib.sha256(name.encode() + b"\0" + content).hexdigest()


def ingest(name: str, content: bytes) -> dict:
    name = Path(name.replace("\\", "/")).name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise ValueError("Unsupported file type.")
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Files must be nonempty and no larger than 20 MB.")
    file_id = uuid.uuid4().hex
    directory = ROOT / "data" / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{file_id}_{name}"
    path.write_bytes(content)
    try:
        index_dir = build_index(str(path), file_id, source_name=name)
        with open(Path(index_dir) / "chunks.pkl", "rb") as handle:
            chunks = pickle.load(handle)
        analytics_path = Path(index_dir) / "analytics.pkl"
        profile = {}
        if analytics_path.exists():
            with analytics_path.open("rb") as handle:
                profile = pickle.load(handle)
        tables = _read_tables(str(path))
        for frame in tables.values():
            frame.columns = frame.columns.map(str)
        return {"id": fingerprint(name, content), "name": name, "path": str(path),
                "index_dir": index_dir, "chunks": len(chunks), "bytes": len(content),
                "profile": profile, "tables": tables,
                "text": "" if tables else load_file(str(path)),
                "embedding_backend": chunks[0]["embedding_backend"]}
    except Exception:
        path.unlink(missing_ok=True)
        index = ROOT / "data" / "indices" / file_id
        if index.exists():
            shutil.rmtree(index)
        raise


def answer_report(question: str, answer: dict) -> str:
    lines = ["# Document Intelligence Report", "", "## Question", question, "",
             "## Answer", answer.get("answer", ""), "", "## Retrieval",
             f"- Mode: {answer.get('retrieval_mode', 'unknown')}",
             f"- Generation: {answer.get('generation_status', 'unknown')}",
             f"- Total time: {answer.get('total_ms', 0)} ms", "", "## Evidence"]
    for chunk in answer.get("retrieved_chunks", []):
        lines.extend([f"### [{chunk.get('citation_id', '')}] {chunk['source']} / chunk {chunk['chunk_id'] + 1}",
                      chunk["text"], ""])
    return "\n\n".join(lines)
