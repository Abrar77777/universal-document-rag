from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="RAG Airline Recovery Assistant",
    description="RAG + Fine-tuned LLM API for airline recovery analysis",
    version="1.0.0"
)

app.include_router(router)
