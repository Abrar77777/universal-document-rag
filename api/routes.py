import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File  # pyright: ignore[reportMissingImports]

from api.schemas import QueryRequest

router = APIRouter()

SESSION_INDEXES = {}
SESSION_ANALYTICS = {}
UPLOAD_DIR = Path("data/uploads")


@router.post("/upload")
def upload_file(file: UploadFile = File(...), session_id: str = "default"):
    try:
        from embeddings.build_index_from_file import build_index

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_id = str(uuid.uuid4())
        safe_name = Path(file.filename or "uploaded_file").name
        file_path = UPLOAD_DIR / f"{file_id}_{safe_name}"

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        index_dir = build_index(str(file_path), file_id, source_name=safe_name)

        SESSION_INDEXES.setdefault(session_id, []).append(index_dir)

        analytics_path = Path(index_dir) / "analytics.pkl"
        analytics = None
        if analytics_path.exists():
            import pickle
            with open(analytics_path, "rb") as f:
                analytics = pickle.load(f)
            SESSION_ANALYTICS.setdefault(session_id, []).append(analytics)

        return {"file_id": file_id, "index_dir": index_dir, "analytics": analytics}

    except Exception as e:
        return {"error": str(e)}


@router.post("/ask")
def ask_question(request: QueryRequest):
    try:
        from rag.rag_pipeline import RAGPipeline

        pipeline = RAGPipeline(SESSION_INDEXES.get(request.session_id, []))

        result = pipeline.answer(
            question=request.question,
            session_id=request.session_id,
            use_web=request.use_web,
            top_k=request.top_k,
            analytics_profiles=SESSION_ANALYTICS.get(request.session_id, []),
        )

        return result

    except Exception as e:
        return {"error": str(e)}


@router.get("/analytics/{session_id}")
def get_analytics(session_id: str):
    return {"analytics": SESSION_ANALYTICS.get(session_id, [])}


@router.get("/health")
def health():
    return {"status": "ok", "uploaded_sessions": len(SESSION_INDEXES)}
