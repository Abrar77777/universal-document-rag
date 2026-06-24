from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Intelligent Universal Document RAG Assistant",
    description="Multi-format RAG API with retrieval, citations, numeric profiling, and evaluation support",
    version="1.0.0"
)

app.include_router(router)
