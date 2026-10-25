import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from api.schemas import QueryRequest

router = APIRouter()

SESSION_INDEXES = {}
SESSION_ANALYTICS = {}
SESSION_TABLES = {}
UPLOAD_DIR = Path("data/uploads")
logger = logging.getLogger(__name__)


@router.post("/upload")
def upload_file(file: UploadFile = File(...), session_id: str = Query(..., min_length=1, max_length=128)):
    try:
        from embeddings.build_index_from_file import build_index
        from utils.workspace import SUPPORTED, MAX_UPLOAD_BYTES
        from utils.tabular_analyzer import _read_tables

        if not session_id.strip():
            raise HTTPException(422, "Session ID must not be blank.")
        if len(SESSION_INDEXES.get(session_id, [])) >= 10:
            raise HTTPException(413, "Workspace document limit reached.")
        if Path(file.filename or "").suffix.lower() not in SUPPORTED:
            raise HTTPException(415, "Unsupported file type.")

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_id = str(uuid.uuid4())
        safe_name = Path((file.filename or "uploaded_file").replace("\\", "/")).name
        file_path = UPLOAD_DIR / f"{file_id}_{safe_name}"

        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not content or len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Files must be nonempty and no larger than 20 MB.")
        file_path.write_bytes(content)

        index_dir = build_index(str(file_path), file_id, source_name=safe_name)

        SESSION_INDEXES.setdefault(session_id, []).append(index_dir)
        for name, frame in _read_tables(str(file_path)).items():
            SESSION_TABLES.setdefault(session_id, []).append({"source": safe_name, "sheet": name, "data": frame})

        analytics_path = Path(index_dir) / "analytics.pkl"
        analytics = None
        if analytics_path.exists():
            import pickle
            with open(analytics_path, "rb") as f:
                analytics = pickle.load(f)
            SESSION_ANALYTICS.setdefault(session_id, []).append(analytics)

        return {"file_id": file_id, "index_dir": index_dir, "analytics": analytics}

    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Upload failed (%s).", type(exc).__name__)
        raise HTTPException(422, "The document could not be indexed. Check its format and readable content.") from None


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
            retrieval_mode=request.retrieval_mode,
            generate=request.generate,
            tabular_sources=SESSION_TABLES.get(request.session_id, []),
        )

        return result

    except Exception as exc:
        logger.warning("Question failed (%s).", type(exc).__name__)
        raise HTTPException(500, "The request could not be completed.") from None


@router.get("/analytics/{session_id}")
def get_analytics(session_id: str):
    return {"analytics": SESSION_ANALYTICS.get(session_id, [])}


@router.get("/health")
def health():
    return {"status": "ok", "uploaded_sessions": len(SESSION_INDEXES)}
