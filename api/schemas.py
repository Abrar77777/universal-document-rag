from typing import Literal
from pydantic import BaseModel, Field, field_validator

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128)
    use_web: bool = False
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: Literal["hybrid", "vector"] = "hybrid"
    generate: bool = True

    @field_validator("question", "session_id")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Must not be blank.")
        return value.strip()
