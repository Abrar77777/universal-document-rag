from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    session_id: str
    use_web: bool = True
    top_k: int = 5
