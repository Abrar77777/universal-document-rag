import uuid
import shutil
from fastapi import APIRouter, UploadFile, File

from embeddings.build_index_from_file import build_index
from rag.rag_pipeline import RAGPipeline
from api.schemas import QueryRequest

router = APIRouter()

SESSION_INDEXES = {}


@router.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = f"data/uploads/{file.filename}"

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    index_dir = build_index(file_path, file_id)

    SESSION_INDEXES.setdefault("default", []).append(index_dir)

    return {"file_id": file_id}


@router.post("/ask")
def ask_question(request: QueryRequest):
    pipeline = RAGPipeline(SESSION_INDEXES.get("default", []))
    return pipeline.answer(
        question=request.question,
        session_id=request.session_id,
        use_web=request.use_web
    )

