from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    documents: str  # URL to the PDF
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]
